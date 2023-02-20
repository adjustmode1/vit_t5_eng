r"""
This Beam Search implementation is adapted with minor modifications from
`AllenNLP <https://github.com/allenai/allennlp/blob/master/allennlp/nn/beam_search.py>`_.

Thanks to the developers of AllenNLP!

**Update (v1.2):** The "backpointer" trick in Beam Search (as implemented in
AllenNLP) does not work well with autoregressive models (transformers). It is
now removed and it improves qualitative predictions and captioning metrics
(CIDEr/SPICE) for VirTex. Updated captioning results are on ArXiv v3. Refer
`CHANGELOG <https://github.com/kdexd/virtex/blob/master/CHANGELOG.md>`_ and
`Release Page <https://github.com/kdexd/virtex/releases/tag/v1.2>`_ for more
details.

Huge thanks to Nicolas Carion (@alcinos) and Aishwarya Kamath (@ashkamath) for
helping me fix this bug!
"""
from typing import Callable, Tuple
import warnings

import torch
from torch.nn import functional as F


class AutoRegressiveBeamSearch:
    r"""
    Implements the beam search algorithm for decoding the most likely captions.

    Args:
        eos_index: The index of the end token (``[EOS]``) in vocabulary.
        max_steps: The maximum number of decoding steps.
        beam_size: The width of the beam used.
        per_node_beam_size: The maximum number of candidates to consider per node,
            at each step in the search. Setting this parameter to a number smaller
            than ``beam_size`` may give better results, as it can introduce more
            diversity into the search. See `Beam Search Strategies for Neural
            Machine Translation. Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(
        self,
        eos_index: int = 3,
        max_steps: int = 64,
        beam_size: int = 7,
        per_node_beam_size: int = 2,
        sos_index: int = 2,
        alpha: int = 0.7
    ) -> None:
        self.eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.alpha = alpha
        self.sos_index = sos_index
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def search(
        self,
        visual_features: torch.Tensor,
        textual: any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            with torch.no_grad():
                image_features = textual.transformer.encode_image(visual_features.to(self.device))
                input_embeds = textual.transformer.t5.get_encoder()(inputs_embeds=image_features)[0]
                outputs = textual(
                  input_embeds=input_embeds,
                  use_t5_encoder=False, # possible t5 encoder use already done
                  decoder_input_ids=input_ids,
                )
                logits = outputs.logits[:, -1, :]

            scaled_logits = torch.log_softmax(logits, dim=1).cpu().squeeze(0) # over vocab size 
            weights, candidates = torch.topk(input=scaled_logits, k=self.beam_size, largest=True)
            
            response_tracker = []  # for valid final sequence 
            sequence_tracker = []  # for current active sequence
            for idx in candidates:
                option = torch.tensor([[idx]])  # a new option into the search tree 
                sequence = torch.cat([input_ids, option], dim=1)
                sequence_tracker.append(sequence)
            keep_generating = True 
            while keep_generating:
                input_batch = torch.vstack(sequence_tracker)
                with torch.no_grad():
                    input_memory = input_embeds.repeat(input_batch.shape[0], 1, 1)
                    outputs = textual(
                      input_embeds=input_memory,
                      use_t5_encoder=False, # possible t5 encoder use already done
                      decoder_input_ids=input_batch.to(self.device),
                    )
                    logits = outputs.logits[:,-1,:]
                    
                scaled_logits = torch.log_softmax(logits, dim=1).cpu()
                
                # bị cắt
                length = input_batch.shape[1] # input_batch
                vocab_size = scaled_logits.shape[1] # scaled_logits
                weighted_logits = (scaled_logits + weights[:, None]) / length ** self.alpha  
                weights, candidates = torch.topk(torch.flatten(weighted_logits), k=self.beam_size, largest=True) # beam_width
                weights = weights * length ** self.alpha  # denormalize

                weights_tmp = []
                sequence_tmp = []
                for idx, pos in enumerate(candidates):
                    row = torch.div(pos, vocab_size, rounding_mode='floor') # get relative position over nb_sequences 
                    col = pos % vocab_size  # get relative position over vocab_size 
                    sequence = torch.cat([sequence_tracker[row], torch.tensor([[col]])], dim=1)
                    if col == eos_token_id:
                        flattened_sequence = torch.flatten(sequence).tolist()
                        sequence_score = weights[idx] / len(flattened_sequence) ** self.alpha
                        response_tracker.append((flattened_sequence, sequence_score))  # a sentence was built ##### response_tracker
                        if len(response_tracker) == self.beam_size:
                            keep_generating = False 
                            break  # end the for loop over candidates
                    elif sequence.shape[1] < self.max_steps - 1:
                        weights_tmp.append(weights[row])
                        sequence_tmp.append(sequence)
                # end for loop over candidates ...!
                if len(sequence_tmp) == 0: 
                    keep_generating = False 
                else:               
                    weights = torch.tensor(weights_tmp)
                    sequence_tracker = sequence_tmp
            return response_tracker
        # end while search loop ...! 