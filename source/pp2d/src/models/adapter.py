import torch
import torch.nn as nn
from typing import Optional
from ..config import Config

class ProteinAdapter(nn.Module):
    """
    Adapts protein features to the LLM's embedding space using either
    a Standard Q-Former (TransformerDecoder) or a simple Linear MLP projection.
    
    Standard Q-Former Implementation:
    - Learnable Query Embeddings (Target)
    - Bidirectional Self-Attention on Queries
    - Cross-Attention (Queries attend to Protein Features)
    - FFN (Feed Forward Network)
    """
    def __init__(self, config: Config, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        self.mode = config.pp2d.protein_adapter_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.mode == "qformer":
            self.num_queries = config.pp2d.num_queries
            
            # Learnable queries [1, Num_Queries, Output_Dim]
            self.query_embeddings = nn.Parameter(torch.randn(1, self.num_queries, output_dim))
            
            # Use PyTorch's Standard TransformerDecoder for robust Q-Former implementation
            # This includes: Self-Attn -> Norm -> Cross-Attn -> Norm -> FFN -> Norm
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=output_dim,
                nhead=config.pp2d.set_transformer_heads,
                dim_feedforward=output_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            
            # Note: We likely want a few layers for better extraction, not just one.
            # Defaulting to 2 layers if not specified, or use set_transformer_layers as proxy
            num_layers = getattr(config.pp2d, 'adapter_layers', 2)
            self.qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            
            # Project input protein features to match Q-Former dimension if needed
            if input_dim != output_dim:
                self.input_proj = nn.Linear(input_dim, output_dim)
            else:
                self.input_proj = nn.Identity()
            
        elif self.mode == "linear":
            # MLP: Linear -> GELU -> Linear
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)
            )
            
        else:
            raise ValueError(f"Unknown protein_adapter_type: {self.mode}")

    def forward(self, protein_feats: torch.Tensor, protein_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            protein_feats: [Batch, Seq_Len, Input_Dim]
            protein_mask: [Batch, Seq_Len] (1 for valid, 0 for padding)
            
        Returns:
            Adapted features.
            - qformer: [Batch, Num_Queries, Output_Dim]
            - linear: [Batch, Seq_Len, Output_Dim]
        """
        if self.mode == "qformer":
            B = protein_feats.shape[0]
            
            # 1. Expand queries: [Batch, Num_Queries, Dim]
            queries = self.query_embeddings.expand(B, -1, -1)
            
            # 2. Prepare Memory (Protein Features)
            # Project if dims don't match
            memory = self.input_proj(protein_feats) # [Batch, Seq_Len, Dim]
            
            # 3. Prepare Masks
            # memory_key_padding_mask needs to be True for Padding (Ignored positions)
            memory_key_padding_mask = None
            if protein_mask is not None:
                # Assuming protein_mask is 1 for valid, 0 for padding
                if protein_mask.dtype != torch.bool:
                    memory_key_padding_mask = (protein_mask == 0)
                else:
                    memory_key_padding_mask = ~protein_mask
            
            # 4. Run Transformer Decoder
            # tgt=queries, memory=protein_features
            output = self.qformer(
                tgt=queries,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            return output

        elif self.mode == "linear":
            return self.mlp(protein_feats)
            
        return protein_feats
