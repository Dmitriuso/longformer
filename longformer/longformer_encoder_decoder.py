from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor
from longformer import LongformerSelfAttention
from transformers.modeling_bart import BartConfig, BartForConditionalGeneration


class LongformerEncoderDecoderForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


class LongformerEncoderDecoderConfig(BartConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']


class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,

    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        assert list(hidden_states.size()) == [tgt_len, bsz, embed_dim]
        assert attention_mask is None

        outputs = self.longformer_self_attn(
            hidden_states.transpose(0, 1),  # LongformerSelfAttention expects (bsz, seqlen, embd_dim)
            attention_mask=key_value_states.unsqueeze(dim=1).unsqueeze(dim=1) * -1,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )

        attn_output = self.output(outputs[0].transpose(0, 1))

        return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None)
