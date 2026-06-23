# src/features.py

import torch
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.sequence import ProteinSequence
import re
import os
from typing import Tuple, Optional

# --- Amino Acid Physicochemical Feature Table ---
# Features: Hydrophobicity (Kyte-Doolittle), Charge (+1, 0, -1), Aromatic (0/1), Polar (0/1)
AMINO_ACID_FEATURES = {
    'A': [1.8, 0, 0, 0], 'R': [-4.5, 1, 0, 1], 'N': [-3.5, 0, 0, 1], 'D': [-3.5, -1, 0, 1],
    'C': [2.5, 0, 0, 1], 'Q': [-3.5, 0, 0, 1], 'E': [-3.5, -1, 0, 1], 'G': [-0.4, 0, 0, 0],
    'H': [-3.2, 0, 1, 1], 'I': [4.5, 0, 0, 0], 'L': [3.8, 0, 0, 0], 'K': [-3.9, 1, 0, 1],
    'M': [1.9, 0, 0, 0], 'F': [2.8, 0, 1, 0], 'P': [-1.6, 0, 0, 0], 'S': [-0.8, 0, 0, 1],
    'T': [-0.7, 0, 0, 1], 'W': [-0.9, 0, 1, 0], 'Y': [-1.3, 0, 1, 1], 'V': [4.2, 0, 0, 0],
    'X': [0.0, 0, 0, 0], # Unknown/Other
}
STANDARD_AMINO_ACIDS = sorted(AMINO_ACID_FEATURES.keys())
RES_MAP = {res: i for i, res in enumerate(STANDARD_AMINO_ACIDS)}

# --- Define Feature Dimensions ---
NODE_FEATURE_DIM = len(STANDARD_AMINO_ACIDS) + 4 + 4 # 21 (one-hot) + 4 (dihedrals) + 4 (physicochem) = 29
STATIC_EDGE_FEATURE_DIM = 8

def get_pocket_file_path(pocket_id: str, pdb_root_dir: str) -> str:
    """
    Translates a pocket_id into an absolute path for the corresponding PDB file.
    """
    match = re.match(r'^([a-zA-Z0-9]{4})(_\w+)?$', pocket_id)
    if not match:
        raise ValueError(f"Invalid pocket_id format: {pocket_id}")
    pdb_id = match.group(1)
    suffix = match.group(2) if match.group(2) else ''
    nested_dir = os.path.join(pdb_root_dir, pdb_id[:2].lower(), pdb_id.lower())
    pocket_pdb_filename = f"{pdb_id.lower()}_pocket{suffix}.pdb"
    pocket_pdb_path = os.path.join(nested_dir, pocket_pdb_filename)
    if not os.path.exists(pocket_pdb_path):
        raise FileNotFoundError(f"Pocket PDB file not found: {pocket_pdb_path}")
    return pocket_pdb_path

def gaussian_expansion(values: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """
    Expands a tensor of values using Gaussian basis functions.
    """
    return torch.exp(-((values - offsets) / widths)**2)

def build_graph_and_static_edge_features(
    coords: torch.Tensor, 
    residue_indices: torch.Tensor,
    radius_cutoff: float
):
    """
    Builds a radius graph and pre-computes only the static (coordinate-independent)
    edge features, such as sequence separation.
    """
    # 1. Build graph connectivity (edge_index)
    dist_matrix = torch.cdist(coords, coords, p=2)
    adj_matrix = (dist_matrix > 0) & (dist_matrix < radius_cutoff)
    edge_index = torch.stack(torch.where(adj_matrix))
    
    # 2. Compute sequence separation features
    seq_separation = (residue_indices[edge_index[0]] - residue_indices[edge_index[1]]).abs().unsqueeze(-1)
    seq_sep_offsets = torch.linspace(0, 32, STATIC_EDGE_FEATURE_DIM, device=coords.device)
    seq_sep_widths = torch.full_like(seq_sep_offsets, (32.0 / STATIC_EDGE_FEATURE_DIM) * 1.5)
    static_edge_attr = gaussian_expansion(seq_separation.float(), seq_sep_offsets, seq_sep_widths)
    
    return edge_index, static_edge_attr

def parse_pdb_to_features(pdb_file_path: str, radius_cutoff: float = 10.0):
    """
    Enhanced version: Parses a PDB file to extract node features and static edge features.
    """
    try:
        structure = pdb.get_structure(pdb.PDBFile.read(pdb_file_path), model=1, altloc='occupancy')
    except Exception:
        return None
    
    phi, psi, omega = struc.dihedral_backbone(structure)
    coords_list, sequence_list, residue_indices_list = [], [], []
    
    for i, res_atoms in enumerate(struc.residue_iter(structure)):
        try:
            res_name = ProteinSequence.convert_letter_3to1(res_atoms[0].res_name)
            if res_name not in RES_MAP: continue
        except KeyError:
            continue
        
        ca_mask = res_atoms.atom_name == "CA"
        if not np.any(ca_mask): continue
        ca_coord = res_atoms[ca_mask][0].coord
        residue_indices_list.append(res_atoms.res_id[0][1])

        # Node feature assembly
        one_hot = np.zeros(len(STANDARD_AMINO_ACIDS))
        one_hot[RES_MAP[res_name]] = 1
        dihedrals = np.nan_to_num([np.sin(phi[i]), np.cos(phi[i]), np.sin(psi[i]), np.cos(psi[i])])
        physicochem = np.array(AMINO_ACID_FEATURES.get(res_name, AMINO_ACID_FEATURES['X']))
        full_feature_vec = np.concatenate([one_hot, dihedrals, physicochem])
        
        coords_list.append(ca_coord)
        sequence_list.append(full_feature_vec)

    if not coords_list: return None
    
    coords = torch.tensor(np.array(coords_list), dtype=torch.float32)
    coords -= coords.mean(dim=0) # Center coordinates
    residue_indices = torch.tensor(residue_indices_list, dtype=torch.long)
    
    # Call the updated graph building function
    edge_index, static_edge_attr = build_graph_and_static_edge_features(coords, residue_indices, radius_cutoff)

    return {
        'coords': coords,
        'sequence': torch.tensor(np.array(sequence_list), dtype=torch.float32),
        'edge_index': edge_index,
        'edge_attr': static_edge_attr, # Return the pre-computed static edge features
    }
    