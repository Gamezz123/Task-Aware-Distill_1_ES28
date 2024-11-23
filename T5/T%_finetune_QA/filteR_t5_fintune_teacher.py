import torch.nn as nn
from transformers import T5ForConditionalGeneration

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mlp_filters = nn.ModuleList([MLPFilter(config.d_model) for _ in range(len(self.encoder.block))])

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None,
                past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        
        # Encoder forward pass
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Apply MLP filters to encoder outputs
        encoder_hidden_states = encoder_outputs[0]
        for i, mlp_filter in enumerate(self.mlp_filters):
            encoder_hidden_states = mlp_filter(encoder_hidden_states)
            if i < len(self.encoder.block) - 1:
                encoder_hidden_states = self.encoder.block[i + 1](encoder_hidden_states, attention_mask)[0]
        
        # Decoder forward pass
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]
        
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        
        lm_logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class MLPFilter(nn.Module):
    def __init__(self, hidden_size):
        super(MLPFilter, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x