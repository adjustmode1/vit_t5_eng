import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import timm

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializes Decoder + LM Head
        self.t5 = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")
        
        self.vit = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0)

        self.tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")


    def encode_image(self, x):
        """Encodes image and matches dimension to T5 embeddings.
        
        """
        # O que fazer com o cls token?
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = self.vit.fc_norm(x)
        return x

    def forward(
        self,
        use_t5_encoder:bool,
        input_image=None,
        input_embeds=None,
        labels=None,
        decoder_input_ids=None,
        past_key_values=None
        ):
        """Teacher forcing traning.
        For T5 details, please refer to
        `https://github.com/huggingface/transformers/blob/504ff7bb1234991eb07595c123b264a8a1064bd3/src/transformers/modeling_t5.py#L1136`
        """
        input_embeds = input_embeds if input_embeds is not None else self.encode_image(input_image)

        # print("labels fill: ",labels.masked_fill(labels==0,-100).shape)
        if labels is not None:
          labels = labels.masked_fill(labels==0,-100)

        output = self.t5(
            encoder_outputs=None if use_t5_encoder else (input_embeds,),
            inputs_embeds=input_embeds,
            labels = labels,
            return_dict=True,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values
        )
        return output
    
    @torch.no_grad()
    def greedy_generate(self, input_image, max_length, use_t5_encoder:bool):
        """Greedy token generation.
        """
        # T5: 0
        eos_token_id = self.tokenizer.eos_token_id
        # T5: 1 (same as padding)
        decoder_start_token_id = self.t5.decoder.config.decoder_start_token_id

        print('eos_token_id: ',eos_token_id)
        print("decoder_start_token_id: ",decoder_start_token_id)
        input_ids = torch.full(
            (input_image.shape[0], 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=input_image.device
        )
        print("input_ids: ",input_ids)
        # First pass, outside loop
        image_features = self.encode_image(input_image)
        if use_t5_encoder:
            input_embeds = self.t5.get_encoder()(inputs_embeds=image_features)[0]
        else:
            input_embeds = image_features


        past = None
        cur_len = 1
        
        while cur_len < max_length:
            outputs = self(
                input_embeds=input_embeds,
                use_t5_encoder=False, # possible t5 encoder use already done
                decoder_input_ids=input_ids,
                # past_key_values=past
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
    
    def decode_token_ids(self, token_ids):
        """Decodifica tokens id e transforma em texto
        """
        decoded_text = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return decoded_text
    # --------------------------------------------------------------------------
    # Daqui pra baixo era pra estar sozinho em outra classe, mas não consegui
    # fazer
    # --------------------------------------------------------------------------

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate, eps=1e-08
            )
        return optimizer

    def _base_eval_step(self, batch):
        """Base function for eval/test steps.
        """
        true_captions = batch['raw_captions']

        generated_tokens = self.greedy_generate(
            input_image=batch['transformed_image'],
            max_length=self.hparams.max_tokens_captions_gen,
            use_t5_encoder=self.hparams.use_t5_encoder
        )
        generated_text = self.decode_token_ids(generated_tokens)
        rets = {'trues': true_captions,'preds': generated_text}
        return rets
    
    def _base_eval_epoch_end(self, outputs, prefix):
        """Base function for eval/test epoch ends.
            The following metrics are calculated:
            - BLEU: BLEU score, bleu-1, ... bleu-4
        """
        trues = sum([x['trues'] for x in outputs], [])
        preds = sum([x['preds'] for x in outputs], [])

        # Bleu score
        # bleu = sacrebleu.corpus_bleu(preds, [trues])
        bleu = sacrebleu.corpus_bleu(preds, trues)

        #         Some random examples
        idx_sample = random.choice(range(len(preds)))
        sample_trues = trues[idx_sample]
        sample_preds = preds[idx_sample]
        # sample_image = 
        print(
            80 * "-",
            f"\nSample predictions epoch {self.current_epoch} '{prefix}':",
            f"\nTrues:\n {sample_trues}",
            f"\nPreds:\n {sample_preds}"
        )
        log_dict = {
            f"{prefix}_bleu_score": bleu.score,
            f"{prefix}_bleu-1": bleu.precisions[0],
            f"{prefix}_bleu-2": bleu.precisions[1],
            f"{prefix}_bleu-3": bleu.precisions[2],
            f"{prefix}_bleu-4": bleu.precisions[3]
        }
        return log_dict
        
    def training_step(self, batch, batch_idx):
        labels = batch['tokenized_target_caption']
        loss = self(
            use_t5_encoder=self.hparams.use_t5_encoder, 
            input_image=batch['transformed_image'],
            labels=labels.masked_fill(labels == 0, -100),
            )['loss']

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        output = self._base_eval_step(batch)
        return output

    def test_step(self, batch, batch_idx):        
        output = self._base_eval_step(batch)
        return output

    def validation_epoch_end(self, outputs):
        output = self._base_eval_epoch_end(outputs, 'val')
        for k,v in output.items():
            self.log(k, v, prog_bar=True)

    def test_epoch_end(self,outputs):
        output = self._base_eval_epoch_end(outputs, 'test')
        for k,v in output.items():
            self.log(k, v, prog_bar=True)
        return output