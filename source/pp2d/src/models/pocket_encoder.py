import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import random

from egnn_pytorch import EGNN_Sparse_Network
from torch_geometric.nn import global_mean_pool # Import efficient global pooling

from ..config import PocketEncoderConfig

# Use unified logger - will use root logger configuration from TrainingRecorder
logger = logging.getLogger(__name__)

# --- Helper Functions (Unchanged and Correct) ---
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extracts values from a 1D tensor `a` at specified indices `t` and reshapes it."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def gaussian_expansion(values: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Expands a tensor of scalar distances into a feature vector using Gaussian basis functions."""
    return torch.exp(-((values - offsets) / widths)**2)

class SinusoidalPosEmb(nn.Module):
    """Converts a scalar (e.g., diffusion time step) into a vector embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# Note: The original `AttentionPooling` class was removed as it required an inefficient
# Python loop for batch processing. It has been replaced by `torch_geometric.nn.global_mean_pool`
# for efficient, vectorized graph-level pooling.

# ==============================================================================
# Advanced Pretraining Components (Corrected and Refactored)
# ==============================================================================

class ContrastiveLearning(nn.Module):
    """
    InfoNCE-based contrastive learning module (Corrected).
    This module pulls the representations of an original graph (view i) and its
    augmented version (view j) closer in the embedding space, while pushing them
    away from the representations of other graphs in the batch.
    """
    def __init__(self, embed_dim, temperature=0.1, projection_dim=128):
        super().__init__()
        self.temperature = temperature
        # Projection head maps graph embeddings to a latent space for contrastive loss calculation.
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, original_embeddings, augmented_embeddings):
        """
        Args:
            original_embeddings: [num_graphs, embed_dim] - Embeddings of the original graphs.
            augmented_embeddings: [num_graphs, embed_dim] - Embeddings of the augmented graphs.
        Returns:
            contrastive_loss: A scalar tensor.
        """
        # 1. Project embeddings of both views into the contrastive space.
        z_i = self.projection_head(original_embeddings)
        z_j = self.projection_head(augmented_embeddings)
        
        # 2. L2-normalize the projections.
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # 3. Compute the similarity matrix.
        # sim_matrix[i, j] is the similarity between the i-th original view and j-th augmented view.
        sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature
        
        # 4. Create labels. The diagonal elements are the positive pairs.
        num_graphs = original_embeddings.size(0)
        labels = torch.arange(num_graphs, device=z_i.device)
        
        # 5. Compute the symmetric cross-entropy loss.
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_i + loss_j) / 2

class MaskedNodePrediction(nn.Module):
    """
    Masked node feature prediction task (Corrected).
    This task forces the model to learn rich contextual representations by masking
    a subset of input nodes and then predicting their original features based on
    the surrounding context.
    """
    def __init__(self, embed_dim, num_atom_features, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        # A learnable [MASK] token, with the same dimension as projected features.
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))
        # The predictor maps the encoded representation back to the original feature dimension.
        self.predictor = nn.Linear(embed_dim, num_atom_features)
        
    def forward(self, model: 'PocketEncoder', pocket_batch):
        """
        Args:
            model: An instance of PocketEncoder to call its internal encoder function.
            pocket_batch: A PyG-formatted batch of data.
        Returns:
            mask_loss: A scalar tensor.
        """
        clean_feats_raw = pocket_batch.x
        clean_coords = pocket_batch.pos
        edge_index = pocket_batch.edge_index
        static_edge_attr = pocket_batch.edge_attr
        batch_map = pocket_batch.batch
        device = clean_feats_raw.device

        num_nodes = clean_feats_raw.size(0)
        
        # 1. Randomly generate a mask.
        perm = torch.randperm(num_nodes, device=device)
        num_mask = int(self.mask_ratio * num_nodes)
        if num_mask == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        mask_indices = perm[:num_mask]
        
        # 2. Apply the mask at the input stage.
        # First, project the raw features into the embedding space.
        projected_feats = model.input_proj(clean_feats_raw)
        # Replace the selected node features with the [MASK] token.
        masked_projected_feats = projected_feats.clone()
        masked_projected_feats[mask_indices] = self.mask_token.expand(num_mask, -1)
        
        # 3. Run the encoder on the masked input (this is the key correction).
        # We do this at t=0, as it's a representation learning task.
        t_zero = torch.zeros(pocket_batch.num_graphs, device=device).long()
        t_expanded = t_zero[batch_map]
        
        encoded_feats, _ = model._encode_graph(
            masked_projected_feats, clean_coords, edge_index, static_edge_attr, batch_map,
            t_expanded, feats_already_projected=True
        )
        
        # 4. Predict and compute loss only on the masked nodes.
        masked_output_feats = encoded_feats[mask_indices]
        predicted_raw_feats = self.predictor(masked_output_feats)
        target_raw_feats = clean_feats_raw[mask_indices]
        
        loss = F.mse_loss(predicted_raw_feats, target_raw_feats)
        return loss

class GraphPropertyPrediction(nn.Module):
    """Auxiliary task to predict graph-level properties (Unchanged and Correct)."""
    def __init__(self, embed_dim, num_properties=5):
        super().__init__()
        self.property_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_properties)
        )
        
    def forward(self, graph_embeddings, properties=None):
        predicted = self.property_predictor(graph_embeddings)
        if properties is not None:
            loss = F.mse_loss(predicted, properties)
            return predicted, loss
        return predicted, None

class DataAugmentation:
    """Graph data augmentation for contrastive learning (Unchanged and Correct)."""
    @staticmethod
    def node_dropout(features, coords, edge_index, edge_attr, batch_map, dropout_ratio=0.1):
        num_nodes = features.size(0)
        keep_mask = torch.rand(num_nodes, device=features.device) > dropout_ratio
        
        if keep_mask.sum() == 0: # Avoid dropping all nodes
            return features, coords, edge_index, edge_attr, batch_map
        
        # Create a mapping from old node indices to new ones.
        node_mapping = torch.full((num_nodes,), -1, device=features.device, dtype=torch.long)
        node_mapping[keep_mask] = torch.arange(keep_mask.sum().item(), device=features.device)
        
        # Filter nodes
        features = features[keep_mask]
        coords = coords[keep_mask]
        batch_map = batch_map[keep_mask]
        
        # Filter edges where both source and destination nodes are kept.
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]
        
        # Remap edge indices to the new node indices.
        edge_index = node_mapping[edge_index]
        
        return features, coords, edge_index, edge_attr, batch_map
    
    @staticmethod
    def edge_dropout(edge_index, edge_attr, dropout_ratio=0.1):
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > dropout_ratio
        return edge_index[:, keep_mask], edge_attr[keep_mask]
    
    @staticmethod
    def feature_noise(features, noise_std=0.1):
        noise = torch.randn_like(features) * noise_std
        return features + noise

# ==============================================================================
# Enhanced Main Encoder Class (Corrected and Refactored)
# ==============================================================================

class PocketEncoder(nn.Module):
    """
    Enhanced Pocket Encoder integrating advanced pretraining techniques:
    - Diffusion-based denoising
    - Contrastive learning
    - Masked node prediction
    - Multi-task property prediction
    """
    def __init__(self, config: PocketEncoderConfig):
        super().__init__()
        self.config = config
        
        # --- Geometry and Feature Dimensions ---
        self.radius_cutoff = config.radius_cutoff
        self.dist_feat_dim = config.dist_feat_dim
        self.dir_feat_dim = config.dir_feat_dim
        self.seq_feat_dim = config.seq_feat_dim
        total_edge_feat_dim = self.dist_feat_dim + self.dir_feat_dim + self.seq_feat_dim
        
        # --- Diffusion-related Parameters ---
        betas = torch.linspace(1e-4, 0.02, config.diffusion_num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # --- Feature and Timestep Processing Layers ---
        self.input_proj = nn.Linear(config.num_atom_features, config.egnn_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.egnn_dim),
            nn.Linear(config.egnn_dim, config.egnn_dim * 4), nn.GELU(),
            nn.Linear(config.egnn_dim * 4, config.egnn_dim)
        )
        
        # --- Core Denoising Network: EGNN ---
        self.denoiser_network = EGNN_Sparse_Network(
            n_layers = config.egnn_depth,
            feats_dim = config.egnn_dim,
            pos_dim = 3,
            edge_attr_dim = total_edge_feat_dim,
            update_coors = False # In denoising, typically only features are updated.
        )

        # --- Noise Prediction Layers ---
        self.predict_feat_noise = nn.Linear(config.egnn_dim, config.egnn_dim)
        self.predict_coord_noise = nn.Linear(config.egnn_dim, 3)
        
        # --- Final Projection Layer for the `encode` method (before pooling) ---
        self.feat_proj_to_embed = nn.Linear(config.egnn_dim, config.embedding_dim)
        
        # --- Instantiate Advanced Pretraining Components ---
        self.contrastive_learning = ContrastiveLearning(
            embed_dim=config.embedding_dim,
            temperature=getattr(config, 'contrastive_temperature', 0.1),
            projection_dim=getattr(config, 'contrastive_projection_dim', 128)
        )
        
        self.masked_prediction = MaskedNodePrediction(
            embed_dim=config.egnn_dim,
            num_atom_features=config.num_atom_features,
            mask_ratio=getattr(config, 'mask_ratio', 0.15)
        )
        
        self.property_prediction = GraphPropertyPrediction(
            embed_dim=config.embedding_dim,
            num_properties=getattr(config, 'num_graph_properties', 5)
        )
        
        # --- Loss Weights for Multi-task Learning ---
        self.diffusion_weight = getattr(config, 'diffusion_loss_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_loss_weight', 0.1)
        self.mask_weight = getattr(config, 'mask_loss_weight', 0.2)
        self.property_weight = getattr(config, 'property_loss_weight', 0.05)
            
        logger.info("Enhanced PocketEncoder initialized with corrected advanced pretraining techniques!")
    
    def _get_dynamic_edge_features(self, coords, edge_index):
        """Helper function: dynamically computes edge features (distance and direction) based on current coordinates."""
        relative_coords = coords[edge_index[0]] - coords[edge_index[1]]
        edge_distances = torch.norm(relative_coords, p=2, dim=-1).unsqueeze(-1)
        dist_offsets = torch.linspace(0, self.radius_cutoff, self.dist_feat_dim, device=coords.device)
        dist_widths = torch.full_like(dist_offsets, (self.radius_cutoff / self.dist_feat_dim) * 1.5)
        dist_attr = gaussian_expansion(edge_distances, dist_offsets, dist_widths)
        dir_attr = F.normalize(relative_coords, p=2, dim=-1)
        
        return torch.cat([dist_attr, dir_attr], dim=-1)

    def _encode_graph(self, feats, coords, edge_index, static_edge_attr, batch_map, t_expanded_per_atom, feats_already_projected=False):
        """Helper method to encode a graph through the EGNN network."""
        feats_proj = self.input_proj(feats) if not feats_already_projected else feats
            
        t_emb = self.time_mlp(t_expanded_per_atom)
        input_feats = feats_proj + t_emb
        
        dynamic_edge_attr = self._get_dynamic_edge_features(coords, edge_index)
        edge_attr = torch.cat([dynamic_edge_attr, static_edge_attr], dim=-1)
        
        # EGNN input is a concatenation of coordinates and features.
        encoder_input = torch.cat([coords, input_feats], dim=-1)
        
        encoder_output = self.denoiser_network(
            x=encoder_input,
            edge_index=edge_index,
            batch=batch_map,
            edge_attr=edge_attr
        )
        
        # Return the updated node features (without coordinates) and the projected input features.
        return encoder_output[:, 3:], feats_proj

    def forward(self, pocket_batch, use_augmentation=True) -> dict:
        """Enhanced forward pass with multi-task learning."""
        self.train()

        # --- Prepare data from the PyG batch ---
        clean_feats_raw = pocket_batch.x
        clean_coords = pocket_batch.pos
        edge_index = pocket_batch.edge_index
        static_edge_attr = pocket_batch.edge_attr
        batch_map = pocket_batch.batch
        num_graphs = pocket_batch.num_graphs
        device = clean_coords.device
        
        # ==================== 1. Diffusion Denoising Task ====================
        t = torch.randint(0, self.config.diffusion_num_timesteps, (num_graphs,), device=device).long()
        t_expanded_per_atom = t[batch_map]
        
        # Add noise to the clean data
        clean_feats_proj = self.input_proj(clean_feats_raw)
        actual_feat_noise = torch.randn_like(clean_feats_proj)
        actual_coord_noise = torch.randn_like(clean_coords)
        
        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t_expanded_per_atom, clean_coords.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t_expanded_per_atom, clean_coords.shape)
        
        noised_coords = sqrt_alpha_bar * clean_coords + sqrt_one_minus_alpha_bar * actual_coord_noise
        noised_feats_proj = sqrt_alpha_bar * clean_feats_proj + sqrt_one_minus_alpha_bar * actual_feat_noise
        
        # Run the denoiser network
        refined_feats, _ = self._encode_graph(
            noised_feats_proj, noised_coords, edge_index, static_edge_attr, batch_map, 
            t_expanded_per_atom, feats_already_projected=True
        )
        
        # Calculate diffusion loss
        predicted_feat_noise = self.predict_feat_noise(refined_feats)
        predicted_coord_noise = self.predict_coord_noise(refined_feats)
        loss_feat = F.mse_loss(predicted_feat_noise, actual_feat_noise)
        loss_coord = F.mse_loss(predicted_coord_noise, actual_coord_noise)
        diffusion_loss = loss_feat + self.config.diffusion_coord_loss_weight * loss_coord

        # ==================== 2. Representation Learning Tasks (at t=0) ====================
        
        # --- Data Augmentation for Contrastive Learning ---
        if use_augmentation and random.random() < 0.8: # Increase probability of applying augmentation
            aug_feats, aug_coords, aug_edge_index, aug_edge_attr, aug_batch_map = DataAugmentation.node_dropout(
                clean_feats_raw.clone(), clean_coords.clone(), edge_index.clone(), static_edge_attr.clone(), batch_map.clone()
            )
            if aug_edge_index.numel() > 0: # Only perform edge dropout if edges exist
                aug_edge_index, aug_edge_attr = DataAugmentation.edge_dropout(aug_edge_index, aug_edge_attr)
            aug_feats = DataAugmentation.feature_noise(aug_feats)
        else:
            aug_feats, aug_coords, aug_edge_index, aug_edge_attr, aug_batch_map = (
                clean_feats_raw, clean_coords, edge_index, static_edge_attr, batch_map
            )
            
        # --- Encode Original and Augmented Graphs (at t=0) ---
        t_zero = torch.zeros_like(t)
        t_zero_expanded = t_zero[batch_map]
        
        # Encode original graph
        orig_node_features, _ = self._encode_graph(
            clean_feats_raw, clean_coords, edge_index, static_edge_attr, batch_map, 
            t_zero_expanded, feats_already_projected=False
        )
        
        # Handle case where augmented graph might be empty
        if aug_feats.shape[0] > 0:
            t_zero_aug_expanded = t_zero[aug_batch_map]
            aug_node_features, _ = self._encode_graph(
                aug_feats, aug_coords, aug_edge_index, aug_edge_attr, aug_batch_map,
                t_zero_aug_expanded, feats_already_projected=False
            )
        else: # Fallback to original graph if augmentation results in an empty graph
             aug_node_features = orig_node_features
             aug_batch_map = batch_map
        
        # --- Project to final embedding space ---
        orig_node_embed = self.feat_proj_to_embed(orig_node_features)
        aug_node_embed = self.feat_proj_to_embed(aug_node_features)
        
        # --- Efficiently pool to get graph-level embeddings ---
        orig_graph_embed = global_mean_pool(orig_node_embed, batch_map)
        aug_graph_embed = global_mean_pool(aug_node_embed, aug_batch_map)
        
        # Ensure the number of graphs in both views is consistent
        # (node dropout might remove all nodes of a graph)
        if orig_graph_embed.shape[0] != aug_graph_embed.shape[0]:
            min_graphs = min(orig_graph_embed.shape[0], aug_graph_embed.shape[0])
            orig_graph_embed = orig_graph_embed[:min_graphs]
            aug_graph_embed = aug_graph_embed[:min_graphs]

        # --- Contrastive Learning Loss ---
        contrastive_loss = self.contrastive_learning(orig_graph_embed, aug_graph_embed)
        
        # --- Masked Node Prediction Loss (Corrected Logic) ---
        # This task requires its own forward pass on masked inputs.
        mask_loss = self.masked_prediction(self, pocket_batch)
        
        # --- Graph Property Prediction Loss ---
        property_loss = torch.tensor(0.0, device=device)
        if hasattr(pocket_batch, 'graph_properties') and pocket_batch.graph_properties is not None:
            _, property_loss = self.property_prediction(orig_graph_embed, pocket_batch.graph_properties)
        
        # --- Total Multi-task Loss ---
        total_loss = (
            self.diffusion_weight * diffusion_loss +
            self.contrastive_weight * contrastive_loss +
            self.mask_weight * mask_loss +
            self.property_weight * property_loss
        )
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'contrastive_loss': contrastive_loss,
            'mask_loss': mask_loss,
            'property_loss': property_loss,
            'loss_breakdown': {
                'feature_noise_loss': loss_feat.detach(),
                'coordinate_noise_loss': loss_coord.detach()
            }
        }

    @torch.no_grad()
    def encode(self, pocket_batch, pooling=True) -> torch.Tensor:
        """Encodes a batch of clean pockets into fixed-size global embeddings."""
        self.eval()
        clean_feats_raw = pocket_batch.x
        clean_coords = pocket_batch.pos
        edge_index = pocket_batch.edge_index
        static_edge_attr = pocket_batch.edge_attr
        batch_map = pocket_batch.batch
        num_graphs = pocket_batch.num_graphs
        
        # For inference, the time step is always 0.
        t = torch.zeros(num_graphs, device=clean_coords.device).long()
        t_expanded_per_atom = t[batch_map]
        
        # Use the helper method for encoding.
        final_node_features, _ = self._encode_graph(
            clean_feats_raw, clean_coords, edge_index, static_edge_attr, batch_map, 
            t_expanded_per_atom, feats_already_projected=False
        )
        
        # Project to the final embedding dimension.
        node_features_embed = self.feat_proj_to_embed(final_node_features)
        
        if pooling:
            # Use efficient global pooling (replaces the old for-loop).
            return global_mean_pool(node_features_embed, batch_map)
        else:
            return node_features_embed
