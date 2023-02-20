import copy
import functools
from typing import Any, Dict

import torch
import numpy as np
from torch import nn

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone
import pickle as pk
import sentencepiece as spm
def build_mask(seq):
    seq_length = seq.shape[1]
    mask = np.fromfunction(lambda i,j: j > i, shape=(seq_length, seq_length))
    return torch.as_tensor(mask) 

def build_key_padding_mask(seq, pad_idx):
    seq_key_padding_mask = (seq == pad_idx)
    return seq_key_padding_mask

class CaptioningModel(nn.Module):
    r"""
    A model to perform image captioning (in both forward and backward directions
    independently, only in forward direction). nó bao gồm một
    :class:`~virtex.modules.visual_backbones.VisualBackbone` và một
    :class:`~virtex.modules.textual_heads.TextualHead` on top of it.

    trong quá trình train, nó tối đa khả năng của một caption đúng điều kiện dựa trên
    các feature hình ảnh. trong quá trình suy luận, nó dự đoán 1 caption cho
    một hình ảnh đầu vào thông qua beam search decoding.

    Args:
        visual: A :class:`~virtex.modules.visual_backbones.VisualBackbone` mà
            tính toán visual features từ hình ảnh đầu vào
        textual: A :class:`~virtex.modules.textual_heads.TextualHead` which
            đưa ra các dự đoán cuối cùng dựa trên các visual features.
        sos_index:vị trí bắt đầu của token (``[SOS]``) trong vocabulary.
        eos_index: vị trí cuối của token (``[EOS]``) trong vocabulary.
        caption_backward: Whether to *also* perform captioning in backward
            direction. mặc định là ``False`` -- chỉ forward captioning is
            performed. khi có giá trị là ``True``, tạo ra 1 clone textual head, nó
            không chỉ chia sẻ weights với mô hình "forward" ngoại trừ input/output embeddings.
        decoder: A :class:`~virtex.utils.beam_search.AutoRegressiveBeamSearch`
            or :class:`~virtex.utils.nucleus_sampling.AutoRegressiveNucleusSampling`
            object for decoding captions during inference (không sử dụng trong quá trình training).
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        caption_backward: bool = False,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx
        self.caption_backward = caption_backward
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        if self.caption_backward:
            self.backward_textual = copy.deepcopy(self.textual)

        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        cho 1 batch hình ảnh và caption, tính toán ghi lại khả năng xẩy ra loss mỗi
        caption token trong quá trình training. trong quá trình suy luận (with images), dự đoán
        một caption thông qua 1 trong 2 beam search decoding hoặc nucleus sampling.

        Args:
            batch: A batch of images and (optionally) ground truth caption tokens.
                dạng có thể có của set of keys: ``{"image_id", "image", "caption_tokens",
                "noitpac_tokens", "caption_lengths"}``.

        Returns:
            1 dict với cấu trúc sau, chứa loss để optimization,
            loss components để log directly to tensorboard, và optionally
            predictions.

            .. code-block::

                {
                    "loss": torch.Tensor,
                    "loss_components": {
                        "captioning_forward": torch.Tensor,
                        "captioning_backward": torch.Tensor, (optional)
                    },
                    "predictions": torch.Tensor
                }
        """

        # shape: (batch_size, channels, height, width)
        visual_features = batch["image"]
        # đặc điểm của visual là gì
        #mới
        batch_size = visual_features.shape[1] # batch size = 1
        #end mới
        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"].to(self.device)
            caption_lengths = batch["caption_lengths"].to(self.device)
            visual_features = visual_features.to(self.device)

            # shape: (batch_size, max_caption_length, vocab_size)
            tgt_input = caption_tokens[:, :-1]
            tgt_output = caption_tokens[:, 1:]
            # memory = self.textual(input_image=visual_features.to(self.device)) # đưa dữ liệu vào encode và nhận được memory là ma trận 
            output = self.textual(
                use_t5_encoder=True,
                src=visual_features, 
                tgt=tgt_input
            ) # chuyển dữ liệu vào decoder và nhận output


            # mới  đổi cách tính loss
            loss = output['loss']

            #end mới 
            output_dict: Dict[str, Any] = {
                "loss": loss,
                # Single scalar per batch for logging in training script.
                "loss_components": {"captioning_forward": loss}, # loss.clone.detach là 1 tensor
            }
            del output
            # Do captioning in backward direction if specified.
            if self.caption_backward:
                backward_caption_tokens = batch["noitpac_tokens"].to(self.device)

                backward_tgt_input = backward_caption_tokens[:, :-1].to(self.device)
                backward_tgt_output = backward_caption_tokens[:, 1:].to(self.device)      

                output = self.backward_textual(
                    use_t5_encoder=True,
                    src=visual_features, 
                    tgt=tgt_input
                ) # chuyển dữ liệu vào decoder và nhận output     

                output_dict["loss"] += output['loss']

                # Single scalar per batch for logging in training script.
                output_dict["loss_components"].update(
                    captioning_backward=output['loss']
                )

                del output
                # end mới

            # if not self.training: # cái nào thêm vô 
            #     # During validation (while pretraining), get best prediction
            #     # at every timestep.
            #     a = 1
        else:
            if self.decoder is None:
                raise ValueError("Decoder for predicting captions is missing!")
            # response_tracker = self.decoding_step(visual_features=visual_features.unsqueeze(0),partial_captions=torch.tensor(1,dtype=float))
            response_tracker = self.decoder.search(visual_features.unsqueeze(0),self.textual)
            output_dict = {"predictions": response_tracker } 
        return output_dict


    def decoding_step(
        self, visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
         # T5: 0
        eos_token_id = 1
        # T5: 1 (same as padding)
        decoder_start_token_id = 0

        input_ids = torch.full(
            (visual_features.shape[0], 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        # First pass, outside loop
        image_features = self.textual.transformer.encode_image(visual_features.to(self.device))
        input_embeds = self.textual.transformer.t5.get_encoder()(inputs_embeds=image_features)[0]


        past = None
        cur_len = 1
        
        while cur_len < 64:
            outputs = self.textual(
                input_embeds=input_embeds,
                use_t5_encoder=False, # possible t5 encoder use already done
                decoder_input_ids=input_ids,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Avoids generation restarting after the first eos
            next_token[input_ids.eq(eos_token_id).any(-1)] = eos_token_id

            cur_len = cur_len + 1
            input_ids = torch.cat([input_ids,next_token.unsqueeze(-1)], dim=-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token, eos_token_id).all():
                break
            
            # if model has past, then set the past variable to speed up
            # decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

        return input_ids


    def log_predictions(
        self, batch: Dict[str, torch.Tensor], tokenizer: SentencePieceBPETokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            predictions_str += f"""
                Caption tokens : {" ".join(tokens.tolist())}
                Predictions (f): {" ".join(preds.tolist())}

                """
        return predictions_str


class ForwardCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=False`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=False,
            decoder=decoder,
        )


class BidirectionalCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=True`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=True,
            decoder=decoder,
        )


# Convenient handle for our main model.
VirTexModel = BidirectionalCaptioningModel
