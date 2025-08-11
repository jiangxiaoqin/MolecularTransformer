"""
GPT-5 style decoder-only Transformer.
This implements masked self-attention layers without encoder context attention,
while keeping the standard OpenNMT decoder interface for compatibility.
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.decoders.transformer import TransformerDecoderState
from onmt.modules.position_ffn import PositionwiseFeedForward

MAX_SIZE = 5000


class GPT5DecoderLayer(nn.Module):
    """
    A single decoder-only Transformer layer with masked self-attention and FFN.

    Args:
      d_model (int): model dimension
      heads (int): number of attention heads
      d_ff (int): inner feed-forward hidden size
      dropout (float): dropout probability
      self_attn_type (str): "scaled-dot" or "average"
    """

    def __init__(self, d_model, heads, d_ff, dropout, self_attn_type="scaled-dot"):
        super(GPT5DecoderLayer, self).__init__()

        self.self_attn_type = self_attn_type

        if self_attn_type == "scaled-dot":
            self.self_attn = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout
            )
        elif self_attn_type == "average":
            self.self_attn = onmt.modules.AverageAttention(
                d_model, dropout=dropout
            )
        else:
            raise ValueError("Unsupported self_attn_type: %s" % self_attn_type)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = onmt.modules.LayerNorm(d_model)
        self.layer_norm_2 = onmt.modules.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        self.register_buffer("mask", mask)

    def forward(self, inputs, tgt_pad_mask, previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): [batch x cur_len x model_dim]
            tgt_pad_mask (LongTensor): [batch x 1 x 1] or broadcastable
            previous_input (FloatTensor or None): cached inputs from previous steps
            layer_cache (dict or None): cached self-keys/values for fast decoding
            step (int or None): decoding step for average attention

        Returns:
            output (FloatTensor): [batch x 1 x model_dim]
            all_input (FloatTensor): [batch x cur_len x model_dim]
        """
        dec_mask = torch.gt(
            tgt_pad_mask + self.mask[:, : tgt_pad_mask.size(1), : tgt_pad_mask.size(1)], 0
        )

        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        if self.self_attn_type == "scaled-dot":
            query, _ = self.self_attn(
                all_input, all_input, input_norm, mask=dec_mask, layer_cache=layer_cache, type="self"
            )
        else:
            query, _ = self.self_attn(
                input_norm, mask=dec_mask, layer_cache=layer_cache, step=step
            )

        query = self.drop(query) + inputs
        output = self.feed_forward(self.drop(self.layer_norm_2(query)) + query)
        return output, all_input

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class GPT5Decoder(nn.Module):
    """
    Decoder-only Transformer stack compatible with OpenNMT's decoder API.

    It ignores encoder memory and produces a dummy attention map aligned to
    the source length to satisfy downstream code expectations.

    Args:
        num_layers (int)
        d_model (int)
        heads (int)
        d_ff (int)
        self_attn_type (str)
        dropout (float)
        embeddings (onmt.modules.Embeddings)
    """

    def __init__(self, num_layers, d_model, heads, d_ff, self_attn_type, dropout, embeddings):
        super(GPT5Decoder, self).__init__()

        self.decoder_type = "gpt5"
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.self_attn_type = self_attn_type

        self.layers = nn.ModuleList(
            [
                GPT5DecoderLayer(
                    d_model=d_model,
                    heads=heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    self_attn_type=self_attn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, tgt, memory_bank, state, memory_lengths=None, step=None, cache=None):
        # Follow Transformer decoder I/O shapes
        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        emb = self.embeddings(tgt, step=step)
        output = emb.transpose(0, 1).contiguous()  # [batch x len x dim]

        padding_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input = self.layers[i](
                output,
                tgt_pad_mask,
                previous_input=prev_layer_input,
                layer_cache=state.cache["layer_{}".format(i)] if state.cache is not None else None,
                step=step,
            )
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        outputs = output.transpose(0, 1).contiguous()

        # Dummy attention over source to keep downstream code happy
        attn = outputs.new_zeros((tgt_len, tgt_batch, src_len))
        attns = {"std": attn}

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden, with_cache=False):
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers, self.self_attn_type)
        return state