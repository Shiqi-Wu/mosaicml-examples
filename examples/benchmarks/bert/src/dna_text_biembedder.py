import torch
import torch.nn as nn
from transformers import BertTokenizer
import math

import math
import torch
import torch.nn as nn

# Alibi Attention Block
class AlibiAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads, max_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.max_len = max_len

        self.register_buffer('slopes', self._get_alibi_slopes(n_heads), persistent=False)
        self.alibi_bias = None  # Lazy init

    def _get_alibi_slopes(self, n_heads):
        # From ALiBi paper: https://github.com/ofirpress/attention_with_linear_biases
        def get_slopes_power_of_2(n):
            start = 2**(-2**-(math.log2(n) - 3))
            return [start**i for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power = 2**math.floor(math.log2(n_heads))
            base = get_slopes_power_of_2(closest_power)
            extra = self._get_alibi_slopes(2 * closest_power)[0::2][:n_heads - closest_power]
            slopes = base + extra

        return torch.tensor(slopes).view(1, n_heads, 1, 1)  # shape: (1, H, 1, 1)

    def _build_alibi_bias(self, seq_len: int, device):
        pos = torch.arange(seq_len, device=device)
        rel_pos = pos.view(1, -1) - pos.view(-1, 1)
        bias = -rel_pos.abs().float()  # (L, L)
        bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        return bias  # will be scaled by slopes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*T, L, D)
        """
        B, L, D = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, L, H, D_head)
        q = q.permute(0, 2, 1, 3)  # (B, H, L, D_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        # ALiBi bias
        if self.alibi_bias is None or self.alibi_bias.shape[-1] < L:
            self.alibi_bias = self._build_alibi_bias(L, device=x.device)  # (1,1,L,L)

        bias = self.slopes.to(x.device) * self.alibi_bias[:, :, :L, :L]  # (1,H,L,L)
        attn_scores = attn_scores + bias

        attn_probs = attn_scores.softmax(dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, H, L, D_head)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)

        return self.out_proj(attn_out)

class RoPEAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B*T, L, D)
        """
        B, L, D = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply RoPE to q/k
        q, k = apply_rope(q, k, seq_dim=2)  # RoPE on sequence dim

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, n_heads, L, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        return self.out_proj(attn_out)


class BasewiseTransformerEmbedder(nn.Module):
    def __init__(self, base_vocab_size=4, base_dim=64, hidden_dim=128, n_heads=4, n_layers=2, max_len=512, PositionalEmbedding="ALiBi"):
        super().__init__()
        self.base_embedding = nn.Embedding(base_vocab_size, base_dim)
        if PositionalEmbedding == "ALiBi":
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    AlibiAttentionBlock(base_dim, n_heads, max_len=max_len),
                    nn.LayerNorm(base_dim),
                    nn.Linear(base_dim, base_dim),
                    nn.ReLU(),
                    nn.LayerNorm(base_dim)
                ) for _ in range(n_layers)
            ])
        elif PositionalEmbedding == "RoPE":
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    RoPEAttentionBlock(base_dim, n_heads),
                    nn.LayerNorm(base_dim),
                    nn.Linear(base_dim, base_dim),
                    nn.ReLU(),
                    nn.LayerNorm(base_dim)
                ) for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unsupported positional embedding: {PositionalEmbedding}")

        self.proj = nn.Linear(base_dim, hidden_dim)
        self.max_len = max_len

    def forward(self, base_ids: torch.Tensor) -> torch.Tensor:
        """
        base_ids: (B, T, L)
        Returns: (B, T, hidden_dim)
        """
        B, T, L = base_ids.shape
        x = self.base_embedding(base_ids.view(B * T, L))  # (B*T, L, D)
        for block in self.blocks:
            x = block(x)  # Each block includes AlibiAttentionBlock + FFN
        x_cls = x[:, -1, :]  # use last token in base sequence
        return self.proj(x_cls).view(B, -1)  # (B, hidden_dim)


class DNATextBiEmbedding(nn.Module):
    """Construct the embeddings for words, ignoring position.

    There are no positional embeddings since we use ALiBi and token_type
    embeddings.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`, but is
    modified as part of Mosaic BERT's ALiBi implementation. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens.

    This module ignores the `position_ids` input to the `forward` method.
    """

    def __init__(self, config, use_dna_embedder=False):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        # ALiBi doesn't use position embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('token_type_ids',
                             torch.zeros(config.max_position_embeddings,
                                         dtype=torch.long),
                             persistent=False)
        self.use_dna_embedder = use_dna_embedder
        if use_dna_embedder:
            self.dna_embedder = BasewiseTransformerEmbedder(
                base_vocab_size=4,  # A, C, G, T
                base_dim=config.dna_base_size,
                hidden_dim=config.hidden_size,
                n_heads=config.dna_num_attention_heads,
                n_layers=config.dna_num_hidden_layers,
                max_len=config.dna_max_position_embeddings,
                PositionalEmbedding=config.dna_positional_embedding_type,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        dna_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        has_text = input_ids is not None or inputs_embeds is not None
        has_dna = dna_ids is not None and getattr(self, 'use_dna_embedder', False)

        if not has_text and not has_dna:
            raise ValueError(
                "forward() requires at least one of `input_ids`, `inputs_embeds`, or `dna_base_ids`."
            )

        embeddings_list = []
        token_types_list = []
        B = None

        # Text embedding
        if has_text:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("Specify only one of `input_ids` or `inputs_embeds`, not both.")

            if input_ids is not None:
                text_embeds = self.word_embeddings(input_ids)
                B, T_text = input_ids.shape
            else:
                text_embeds = inputs_embeds
                B, T_text = inputs_embeds.shape[:2]

            assert text_embeds.ndim == 3, f"`text_embeds` should be (B, T, H), but got {text_embeds.shape}"
            embeddings_list.append(text_embeds)
            token_types_list.append(torch.zeros((B, T_text), dtype=torch.long, device=text_embeds.device))

    # --- DNA embedding ---
    if has_dna:
        assert hasattr(self, 'dna_embedder'), (
            "DNA embedding requested but `self.dna_embedder` is not defined. "
            "Please set `self.use_dna_embedder = True` and define `self.dna_embedder` in __init__."
        )

        assert dna_ids.ndim == 3, (
            f"`dna_ids` must have shape (B, T, L), but got {dna_ids.shape}"
        )

        dna_embeds = self.dna_embedder(dna_ids)  # (B, T_dna, H)
        B_dna, T_dna = dna_embeds.shape[:2]

        if B is not None:
            assert B_dna == B, (
                f"Batch size mismatch between text and DNA embeddings: text B={B}, dna B={B_dna}"
            )
        else:
            B = B_dna

        embeddings_list.append(dna_embeds)
        token_types_list.append(torch.ones((B_dna, T_dna), dtype=torch.long, device=dna_embeds.device))

    # --- Concatenate ---
    inputs_embeds = torch.cat(embeddings_list, dim=1)  # (B, T_total, H)
    token_type_ids = torch.cat(token_types_list, dim=1)  # (B, T_total)

    assert inputs_embeds.shape[:2] == token_type_ids.shape, (
        f"Mismatched embedding and token_type shapes: "
        f"{inputs_embeds.shape[:2]} vs {token_type_ids.shape}"
    )

    # --- Final embedding ---
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    assert inputs_embeds.shape == token_type_embeddings.shape, (
        f"Shape mismatch between `inputs_embeds` ({inputs_embeds.shape}) "
        f"and `token_type_embeddings` ({token_type_embeddings.shape})"
    )

    embeddings = inputs_embeds + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings



def apply_rope(q, k, seq_dim=2):
    """
    Apply rotary embedding to q, k
    q, k: (..., seq_len, dim)
    """
    sin, cos = get_sin_cos(q.shape[seq_dim], q.shape[-1], q.device)
    q_rot = rotate_half(q)
    k_rot = rotate_half(k)
    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin
    return q, k


def get_sin_cos(seq_len, dim, device):
    # dim should be even
    theta = 10000 ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    position = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", position, theta)  # (seq_len, dim/2)
    emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (seq_len, dim)
    sin = emb.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,dim)
    cos = emb.unsqueeze(0).unsqueeze(0)
    return sin, cos


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)
