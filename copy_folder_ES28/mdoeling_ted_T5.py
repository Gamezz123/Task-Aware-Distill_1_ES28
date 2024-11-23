# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch TEDT5 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    T5Stack,
)

@dataclass
class TEDBaseModelOutput(BaseModelOutput):
    filter_states: Tuple[torch.FloatTensor] = None

@dataclass
class TEDQuestionAnsweringModelOutput(QuestionAnsweringModelOutput):
    filter_states: Tuple[torch.FloatTensor] = None

@dataclass
class TEDSequenceClassifierOutput(SequenceClassifierOutput):
    filter_states: Tuple[torch.FloatTensor] = None

class TEDT5Layer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer = T5Stack(config)
        
        assert layer_idx is not None
        self.should_add_filter = (
            (layer_idx + 1) % config.filter_interval == 0
        ) and not config.filter_disabled
        
        if self.should_add_filter:
            filter_output_dim = config.filter_output_dim if config.filter_output_dim else config.d_model
          
            if config.filter_nonlinear:
                self.filter = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model), 
                    ACT2FN[config.hidden_act],
                    nn.Linear(config.d_model, filter_output_dim),
                )
            else:
                self.filter = nn.Linear(config.d_model, filter_output_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
    ):
        layer_output = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        filter_layer_output = None
        if self.should_add_filter:
            filter_layer_output = self.filter(layer_output[0])

        if output_attentions:
            return ((layer_output[0], filter_layer_output), layer_output[1])
        else:
            return (layer_output[0], filter_layer_output)

class TEDT5Encoder(T5Stack):
    
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([TEDT5Layer(config, layer_idx) for layer_idx in range(config.num_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        all_filter_states = ()
        filter_states = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if filter_states is not None:
                all_filter_states = all_filter_states + (filter_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                hidden_states, att_m = hidden_states
            
            hidden_states, filter_states = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if filter_states is not None:
            all_filter_states = all_filter_states + (filter_states,)

        if not return_dict:
            return tuple(v for v in [
                hidden_states, all_hidden_states, all_attentions] if v is not None)
        return TEDBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions, 
            filter_states=all_filter_states,
        )

class TEDT5Model(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = TEDT5Encoder(config)
        self.decoder = T5Stack(config, self.shared)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return TEDBaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            filter_states=encoder_outputs.filter_states,
        )

class TEDT5ForQuestionAnswering(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.t5 = TEDT5Model(config)
        if config.train_filters:
            self.num_filters = config.num_layers // config.filter_interval
            filter_output_dim = config.filter_output_dim if config.filter_output_dim else config.d_model
            self.filter_head = nn.ModuleList([
                nn.Linear(filter_output_dim, config.num_labels) for _ in range(self.num_filters)
            ])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.config.train_filters:
            sequence_output = outputs[0]
            logits = self.lm_head(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
        else:
            start_logits, end_logits = None, None
            filter_start_end_logits = []
            for filter_idx in range(self.num_filters):
                filter_logits = self.filter_head[filter_idx](outputs['filter_states'][filter_idx])
                filter_start_logits, filter_end_logits = filter_logits.split(1, dim=-1)
                filter_start_logits = filter_start_logits.squeeze(-1).contiguous()
                filter_end_logits = filter_end_logits.squeeze(-1).contiguous()
                filter_start_end_logits.append((filter_start_logits, filter_end_logits))

        total_loss = None
        if labels is not None: 
            # If we are on multi-GPU, split add a dimension
            if len(labels.size()) > 1:
                labels = labels.squeeze(-1)

            if not self.config.train_filters:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                labels = labels.clamp(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, labels)
                end_loss = loss_fct(end_logits, labels)
                total_loss = (start_loss + end_loss) / 2
            else:
                total_loss = 0.0
                for filter_start_logits, filter_end_logits in filter_start_end_logits:
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = filter_start_logits.size(1)
                    _labels = labels.clamp(0, ignored_index)
                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(filter_start_logits, _labels)
                    end_loss = loss_fct(filter_end_logits, _labels)
                    total_loss += (start_loss + end_loss) / 2
                total_loss /= self.num_filters

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TEDQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            filter_states=outputs.filter_states,
        )

class TEDContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_output_dim = config.filter_output_dim if config.filter_output_dim is not None else config.d_model
        self.dense = nn.Linear(pooler_output_dim, pooler_output_dim)
    
    @property
    def output_dim(self):
        return self.config.filter_output_dim if self.config.filter_output_dim is not None else self.config.d_model

class TEDT5ForSequenceClassification(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.t5 = TEDT5Model(config)
        if config.train_filters:
            self.num_filters = config.num_layers // config.filter_interval
            self.filter_pooler = nn.ModuleList([
                TEDContextPooler(config) for _ in range(self.num_filters)
            ])
            filter_output_dim = self.filter_pooler[0].output_dim
            self.filter_head = nn.ModuleList([
                nn.Linear(filter_output_dim, config.num_labels) for _ in range(self.num_filters)
            ])
            drop_out = getattr(config, "cls_dropout", None)
            drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
            self.filter_dropout = nn.ModuleList([
                nn.Dropout(drop_out) for _ in range(self.num_filters)
            ])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.config.train_filters:
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = None
            filter_logits = []
            for filter_idx in range(self.num_filters):
                filter_pooled_output = self.filter_pooler[filter_idx](outputs['filter_states'][filter_idx])
                filter_pooled_output = self.filter_dropout[filter_idx](filter_pooled_output)
                filter_logits.append(self.filter_head[filter_idx](filter_pooled_output))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    if not self.config.train_filters:
                        logits = logits.view(-1).to(labels.dtype)
                        loss = loss_fn(logits, labels.view(-1))
                    else:
                        filter_logits = [logits.view(-1).to(labels.dtype) for logits in filter_logits]
                        loss = sum([loss_fn(logits, labels.view(-1)) for logits in filter_logits]) / self.num_filters
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        loss_fct = CrossEntropyLoss()
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        if not self.config.train_filters:
                            labeled_logits = torch.gather(
                                logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                            )
                            loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                        else:
                            labeled_logits = [torch.gather(
                                logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                            ) for logits in filter_logits]
                            loss = sum([loss_fct(
                                logits.view(-1, self.num_labels).float(), labels.view(-1)
                            ) for logits in labeled_logits]) / self.num_filters
                    else:
                        loss = torch.tensor(0).to(logits)   
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    if not self.config.train_filters:
                        loss = -((log_softmax(logits) * labels).sum(-1)).mean()
                    else:
                        loss = sum([-((log_softmax(logits) * labels).sum(-1)).mean() for logits in filter_logits]) / self.num_filters 
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    if not self.config.train_filters:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = sum([loss_fct(logits.squeeze(), labels.squeeze()) for logits in filter_logits]) / self.num_filters
                else:
                    if not self.config.train_filters:
                        loss = loss_fct(logits, labels)
                    else:
                        loss = sum([loss_fct(logits, labels) for logits in filter_logits]) / self.num_filters
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                if not self.config.train_filters:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    loss = sum([loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) for logits in filter_logits]) / self.num_filters
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                if not self.config.train_filters:
                    loss = loss_fct(logits, labels)
                else:
                    loss = sum([loss_fct(logits, labels) for logits in filter_logits]) / self.num_filters

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TEDSequenceClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            filter_states=outputs.filter_states,
        )