import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config import Config

class RoPEAwareEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer that supports Rotary Positional Embeddings (RoPE).
    Uses torch.nn.functional.scaled_dot_product_attention for efficiency (FlashAttention compatible).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Pre-Norm Architecture (Standard for LLMs/RoPE)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = dropout
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def apply_rotary_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE using polar coordinates (efficient standard implementation).
        x: [Batch, Seq_Len, Num_Heads, Head_Dim]
        freqs_cis: [1, Seq_Len, 1, Head_Dim/2]
        """
        # Reshape x to pairs for complex representation
        # [B, S, H, D] -> [B, S, H, D/2, 2]
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        # Apply rotation: x * e^(i*theta)
        # Broadcasting freqs_cis across batch and heads
        x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
        return x_out.type_as(x)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: [Batch, Seq_Len, Dim]
        freqs_cis: Precomputed complex frequencies
        """
        B, S, D = x.shape
        residual = x
        x = self.norm1(x)
        
        # Projections [B, S, D] -> [B, S, H, Head_Dim]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = self.apply_rotary_emb(q, freqs_cis)
        k = self.apply_rotary_emb(k, freqs_cis)
        
        # Transpose for SDPA: [B, H, S, Head_Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        # key_padding_mask in PyTorch is usually True for Padding (ignore). 
        # SDPA expects attention_mask (additive, float) OR a boolean mask.
        # Constructing boolean mask for SDPA: True means KEEP, False means IGNORE?
        # Actually PyTorch SDPA docs say: "Binary mask where True indicates that the corresponding position is not allowed to attend." (similar to attn_mask)
        # BUT key_padding_mask usually is True for Padding.
        # Let's rely on functional automatic handling or expand it.
        
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, S]. True = Padding.
            # Expand to [B, 1, 1, S] for broadcasting over Q
            # SDPA expects broadcastable mask.
            # We want to mask OUT padding.
            # If mask is boolean: True = mask out (don't attend).
            attn_mask = key_padding_mask.view(B, 1, 1, S).expand(B, self.num_heads, S, S)
            
        # Scaled Dot Product Attention (FlashAttention compatible)
        # dropout_p is handled internally
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # [B, H, S, Head_Dim] -> [B, S, H, Head_Dim] -> [B, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        
        # Residual 1
        x = residual + out
        
        # FFN + Residual 2
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class ProteinSetTransformer(nn.Module):
    """
    Models interactions between a set of proteins using a Transformer Encoder with RoPE.
    Treats all tokens from all proteins in the set as a single sequence but handles
    positional IDs to respect protein boundaries (Pos IDs reset for each protein).
    """
    def __init__(self, config: Config, embed_dim: int):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = config.pp2d.set_transformer_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"set_transformer embed_dim ({self.embed_dim}) must be divisible by set_transformer_heads ({self.num_heads})"
            )
        self.head_dim = embed_dim // self.num_heads
        
        # Role Embeddings: 0 = Inactive, 1 = Active. 
        self.role_embedding = nn.Embedding(2, embed_dim)
        
        # RoPE Setup
        # Inverse frequencies: 1 / (10000 ^ (2i / d))
        theta = float(getattr(config.pp2d, 'rope_theta', 10000.0))
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Stack of RoPE Aware Layers
        self.layers = nn.ModuleList([
            RoPEAwareEncoderLayer(embed_dim, self.num_heads)
            for _ in range(config.pp2d.set_transformer_layers)
        ])
        
    def _precompute_freqs_cis(self, max_len: int, device: torch.device):
        """
        Precompute complex exponentials for RoPE.
        Returns: [1, max_len, 1, head_dim/2]
        """
        # t = position indices [0, 1, 2, ..., max_len-1]
        t = torch.arange(max_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product: t * inv_freq
        # shape: [max_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Turn into complex numbers: e^(i * freqs) = cos + i*sin
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # Reshape for broadcasting: [1, max_len, 1, head_dim/2]
        # 2nd dim corresponds to Sequence Length
        return freqs_cis.unsqueeze(0).unsqueeze(2)

    def forward(self, batch_tokens: torch.Tensor, role_ids: torch.Tensor, protein_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            batch_tokens: [Batch, Num_Proteins, Seq_Len, Dim]
            role_ids: [Batch, Num_Proteins] (-1: Padding, 0: Inactive, 1: Active)
            protein_masks: [Batch, Num_Proteins, Seq_Len] (1/True=Valid, 0/False=Padding).
            
        Returns:
            Processed tokens: [Batch, Num_Proteins, Seq_Len, Dim]
        """
        B, Num_P, S, D = batch_tokens.shape
        device = batch_tokens.device
        
        # 1. Inject Role Embeddings
        valid_mask = role_ids != -1
        valid_role_ids = role_ids.clone()
        valid_role_ids[~valid_mask] = 0
        
        role_vectors = self.role_embedding(valid_role_ids).unsqueeze(2) # [B, Num_P, 1, D]
        batch_tokens = batch_tokens + role_vectors
        
        # 2. Flatten for Transformer: [Batch, Num_Proteins * Seq_Len, Dim]
        # Total sequence length = Num_P * S
        flat_tokens = batch_tokens.view(B, Num_P * S, D)
        
        # 3. Handle Masking
        # protein_masks: [B, N, S]. 1=Valid, 0=Padding.
        # role_ids: [B, N]. -1=Padding.
        
        # Mask for missing proteins (True = Padding/Ignore)
        protein_level_padding_mask = (role_ids == -1) # [B, N]
        combined_padding_mask = protein_level_padding_mask.unsqueeze(-1).expand(-1, -1, S) # [B, N, S]
        
        if protein_masks is not None:
            # protein_masks: 1=Valid, 0=Padding. Convert to True=Padding
            residue_padding_mask = (protein_masks == 0)
            combined_padding_mask = combined_padding_mask | residue_padding_mask
            
        # Flatten mask: [B, Num_P * S]
        flat_padding_mask = combined_padding_mask.reshape(B, Num_P * S)
        
        # 4. Prepare RoPE Frequencies
        # Key Concept: Positional IDs should respect protein boundaries.
        # For a Set of Proteins, the absolute position 0..N*S doesn't make sense.
        # The internal residue position 0..S makes sense.
        # So we generate pos_ids that repeat 0..S-1 for each protein.
        
        # pos_ids: [0, 1, ... S-1, 0, 1, ... S-1, ...]
        pos_ids_single = torch.arange(S, device=device)
        pos_ids_flat = pos_ids_single.repeat(Num_P) # [N*S]
        
        # Precompute freqs for S (max length we care about for rotation is S)
        # Note: We compute up to S, but we need to map them to the flattened sequence.
        # We can precompute 'cis' for 0..S-1, then gather them using pos_ids_flat.
        
        full_freqs_cis = self._precompute_freqs_cis(S, device) # [1, S, 1, D/2]
        
        # Gather for the flat sequence
        # We need [1, N*S, 1, D/2]
        # pos_ids_flat is [N*S]
        batch_freqs_cis = full_freqs_cis[:, pos_ids_flat, :, :] # [1, N*S, 1, D/2]
        
        # 5. Pass through Layers
        x = flat_tokens
        for layer in self.layers:
            x = layer(x, batch_freqs_cis, key_padding_mask=flat_padding_mask)
            
        # 6. Unflatten
        output = x.view(B, Num_P, S, D)
        
        return output
