# src/data.py

import os
import logging
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Sequence, cast, Literal, Set
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
import lmdb
import pickle
import zlib

from .config import Config

# Use unified logger
logger = logging.getLogger(__name__)


def infer_mol_token_ids(tokenizer: PreTrainedTokenizerBase) -> Tuple[Optional[int], Optional[int]]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else getattr(tokenizer, "vocab", None)
    mol_token_id = vocab.get('<mol>') if isinstance(vocab, dict) else None
    end_mol_token_id = vocab.get('</mol>') if isinstance(vocab, dict) else None

    if mol_token_id is None or end_mol_token_id is None:
        try:
            ids = tokenizer.encode('<mol></mol>', add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) >= 2:
            if mol_token_id is None:
                mol_token_id = int(ids[0])
            if end_mol_token_id is None:
                end_mol_token_id = int(ids[-1])

    return mol_token_id, end_mol_token_id


def has_complete_mol_span(input_ids: torch.Tensor, mol_token_id: Optional[int], end_mol_token_id: Optional[int]) -> bool:
    if mol_token_id is None or end_mol_token_id is None:
        return False

    mol_positions = (input_ids == mol_token_id).nonzero(as_tuple=False)
    if mol_positions.numel() == 0:
        return False

    end_positions = (input_ids == end_mol_token_id).nonzero(as_tuple=False)
    if end_positions.numel() == 0:
        return False

    return int(end_positions[-1].item()) > int(mol_positions[0].item())

# --- Helper Functions ---

def efficient_pad_sequences(
    sequences: List[torch.Tensor], 
    padding_value: float = 0.0, 
    padding_side: Literal['left', 'right'] = 'right',
    batch_first: bool = True
) -> torch.Tensor:
    """
    Highly efficient padding function.
    """
    if not sequences:
        return torch.empty(0)

    if padding_side == 'right':
        return pad_sequence(
            sequences, 
            batch_first=batch_first, 
            padding_value=padding_value
        )
    else: # padding_side == 'left'
        reversed_seqs = [s.flip(0) for s in sequences]
        padded_reversed = pad_sequence(
            reversed_seqs, 
            batch_first=batch_first, 
            padding_value=padding_value
        )
        return padded_reversed.flip(1) if batch_first else padded_reversed.flip(0)

def find_first_occurrence(tensor: torch.Tensor, value: int) -> torch.Tensor:
    """
    Efficiently finds placeholders in a left-padded tensor.
    """
    mask = (tensor == value)
    seq_len = tensor.shape[1]
    indices = torch.arange(seq_len, device=tensor.device).unsqueeze(0)
    masked_indices = torch.where(mask, indices, seq_len + 1)
    result = torch.min(masked_indices, dim=1).values
    result[result == seq_len + 1] = -1
    return result

# --- Base Dataset ---

class BaseProteinDataset(Dataset):
    """
    Base dataset with shared LMDB and tokenization helpers.
    """
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerBase, split: str = 'train'):
        self.config = config
        self.tokenizer = tokenizer
        # Normalize split names: map 'val' -> 'valid'
        self.split = 'valid' if split == 'val' else split
        self.use_precomputed = config.pp2d.use_precomputed_features

        self.lmdb_path = config.data.lmdb_path
        if not self.lmdb_path:
            raise ValueError("LMDB path must be provided in config.data.lmdb_path")
        self.lmdb_env = None
        self.all_protein_ids: List[str] = []

    def _init_lmdb_and_keys(self):
        """
        Shared LMDB key initialization for datasets using protein features.
        """
        env = None
        try:
            env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            logger.info("Scanning LMDB keys...")
            temp_set = set()
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    temp_set.add(key.decode('utf-8'))
            self.all_protein_ids = list(temp_set)
            self.protein_ids_set = temp_set
            logger.info(f"Found {len(self.all_protein_ids)} proteins in LMDB.")
        except Exception as e:
            logger.error(f"Failed to open LMDB at {self.lmdb_path}: {e}")
            raise
        finally:
            if env:
                env.close()
            self.lmdb_env = None

    def _apply_subset(self):
        """
        Shared subset logic based on split configuration.
        """
        if not getattr(self, "data", None):
            return
        cfg_data = self.config.data
        subset_config = {
            'train': (cfg_data.train_subset, cfg_data.train_subset_size),
            'valid': (cfg_data.valid_subset, cfg_data.valid_subset_size),
            'test':  (cfg_data.test_subset,  cfg_data.test_subset_size)
        }
        subset_key = self._resolve_subset_key()
        do_subset, target_size = subset_config.get(subset_key, (False, 0))
        if do_subset:
            original_len = len(self.data)
            actual_size = min(original_len, target_size)
            logger.info(f"Subsetting enabled for split '{self.split}': Sampling {actual_size} items from {original_len} total.")
            rng = random.Random(self.config.training.seed)
            rng.shuffle(self.data)
            self.data = self.data[:actual_size]

    def _get_protein_features(self, pid: str) -> Optional[torch.Tensor]:
        """
        Shared protein feature loader from LMDB.
        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb_env.begin(write=False) as txn:
            data_bytes = txn.get(pid.encode('utf-8'))

        if not data_bytes:
            return None

        try:
            data = pickle.loads(data_bytes)
        except Exception:
            try:
                data = pickle.loads(zlib.decompress(data_bytes))
            except Exception:
                return None

        if self.use_precomputed:
            if isinstance(data, dict) and 'esm_hidden' in data:
                feats = data['esm_hidden']
                if not isinstance(feats, torch.Tensor):
                    feats = torch.from_numpy(feats).float()
                return feats
            if torch.is_tensor(data):
                return data.float()
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
        else:
            if isinstance(data, dict) and 'sequence' in data:
                return data['sequence']

        return None

    def _tokenize_segment(self, text: str) -> torch.Tensor:
        if not text:
            return torch.tensor([], dtype=torch.long)
        out = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')

        res = torch.tensor([], dtype=torch.long)
        if isinstance(out, dict) and 'input_ids' in out:
            res = out['input_ids']
        elif hasattr(out, 'input_ids'):
            res = out.input_ids
        elif isinstance(out, torch.Tensor):
            res = out
        elif isinstance(out, list):
            res = torch.tensor(out)
        else:
            try:
                res = torch.tensor(out)
            except Exception:
                if hasattr(out, 'ids'):
                    res = torch.tensor(out.ids)

        if not isinstance(res, torch.Tensor):
            res = torch.tensor(res) if res is not None else torch.tensor([], dtype=torch.long)

        if res.numel() > 0:
            return res.squeeze(0)
        return torch.tensor([], dtype=torch.long)

    def _resolve_split_key(self, splits: Dict[str, Any]) -> str:
        """
        Resolve split key with val/valid compatibility.
        """
        if self.split == 'valid':
            if 'valid' in splits:
                return 'valid'
            if 'val' in splits:
                return 'val'
        if self.split == 'test':
            if 'test' in splits:
                return 'test'
            if 'test_simple' in splits:
                return 'test_simple'
            if 'test_hard' in splits:
                return 'test_hard'
        return self.split

    def _resolve_subset_key(self) -> str:
        """
        Resolve subset key to base split category.
        """
        if self.split.startswith('test_'):
            return 'test'
        return self.split

# --- Dataset ---

class LigandProfileDataset(BaseProteinDataset):
    """
    Listwise Dataset for Ligand Profiling.
    """
    def __init__(self, 
                 config: Config,
                 tokenizer: PreTrainedTokenizerBase, 
                 split: str = 'train'):
        super().__init__(config, tokenizer, split)
        
        # Config parameters
        self.max_actives = config.training.max_active
        self.max_inactives = config.training.max_inactive
        self.min_active = config.training.min_active
        self.prob_no_inactives = getattr(config.training, 'prob_no_inactives', 0.5)
        self.prob_single_active = getattr(config.training, 'prob_single_active', 0.5)
        
        # 1. Load Data (Raw)
        self.data = self._load_raw_data()
        
        # 2. LMDB Setup (Required for filtering)
        self._init_lmdb_and_keys()
        
        # 3. Filter Data (Full dataset filtering)
        # We do this BEFORE subsetting to ensure the subset contains valid items only.
        self._filter_data()

        # 4. Apply Subset (Debugging)
        # Now applied on valid data, so if you ask for 50, you get 50 (if available).
        self._apply_subset()
        
        # 5. Tokenizer & Templates Setup
        self.sep_token_id = None
        for token in [",", ", "]:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                self.sep_token_id = ids[0]
                break
        
        if self.sep_token_id is None:
            self.sep_token_id = self.tokenizer.encode(",", add_special_tokens=False)[0]
        self.templates_listwise = config.pp2d.prompt_templates
        self.templates_active_only = config.pp2d.prompt_templates_active_only
        self.protein_placeholder_id = -1 
        
    def _load_raw_data(self) -> List[Any]:
        """Handles the logic for loading data from the master file or compatibility paths."""
        cfg_data = self.config.data
        
        # Mode 1: Master File + Split Indices
        if cfg_data.split_path and cfg_data.data_path:
            logger.info(f"Loading data from master file: {cfg_data.data_path}")
            if not os.path.exists(cfg_data.data_path):
                raise FileNotFoundError(f"Master data file not found: {cfg_data.data_path}")
            if not os.path.exists(cfg_data.split_path):
                raise FileNotFoundError(f"Split indices file not found: {cfg_data.split_path}")

            with open(cfg_data.data_path, 'rb') as f:
                all_data = pickle.load(f)
            with open(cfg_data.split_path, 'rb') as f:
                splits = pickle.load(f)
            
            # Mapping valid aliases
            split_key = self._resolve_split_key(splits)
            
            if split_key not in splits:
                logger.warning(f"Split '{self.split}' not found in split file keys. Using empty dataset.")
                return []
                
            indices = splits[split_key]
            logger.info(f"Loaded {len(indices)} items for split '{self.split}' from master file.")
            return [all_data[i] for i in indices if i < len(all_data)]

        # Mode 2: compatibility separate files
        else:
            path_map = {
                'train': cfg_data.train_data,
                'valid': cfg_data.valid_data,
                'test':  cfg_data.test_data
            }
            data_paths = path_map.get(self.split, [])
            
            loaded_data = []
            for path in data_paths:
                if path and os.path.exists(path):
                    logger.info(f"Loading profile data from {path}...")
                    with open(path, 'rb') as f:
                        loaded_data.extend(pickle.load(f))
                else:
                    logger.warning(f"Data path not found or empty: {path}")
            
            logger.info(f"Loaded {len(loaded_data)} items for split '{self.split}'.")
            return loaded_data

    def _filter_data(self):
        """Filter data to ensure min_active requirement is met."""
        if not self.data:
            return
            
        logger.info(f"Filtering dataset (min_active={self.min_active})...")
        initial_len = len(self.data)
        valid_data = []
        
        # Ensure protein_ids_set exists
        if not hasattr(self, 'protein_ids_set'):
            self.protein_ids_set = set(self.all_protein_ids)
            
        for item in self.data:
            all_actives = item.get('Active_Protein_IDs', [])
            # Check how many actives are actually available in LMDB
            valid_actives_count = sum(1 for p in all_actives if p in self.protein_ids_set)
            
            if valid_actives_count >= self.min_active:
                valid_data.append(item)
                
        self.data = valid_data
        removed = initial_len - len(self.data)
        logger.info(f"Filtered {removed} items. Valid dataset size: {len(self.data)}")

    def _apply_subset(self):
        """Applies granular subset logic based on split configuration."""
        super()._apply_subset()

    def __len__(self) -> int:
        return len(self.data)

    def _select_template(self, templates: List[str], idx: int) -> str:
        if not templates:
            return ""
        if self.split in ['test', 'valid']:
            return templates[idx % len(templates)]
        return random.choice(templates)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.data[idx]
        smiles = item.get('SMILES', "")
        
        all_actives = item.get('Active_Protein_IDs', [])
        all_inactives = item.get('Inactive_Protein_IDs', [])
        
        # Filter availability in LMDB using set for O(1) lookup
        # Note: self.protein_ids_set is created in _init_lmdb_and_keys
        if not hasattr(self, 'protein_ids_set'):
            self.protein_ids_set = set(self.all_protein_ids)
            
        all_actives = [p for p in all_actives if p in self.protein_ids_set]
        all_inactives = [p for p in all_inactives if p in self.protein_ids_set]
        
        # Double check min_active (should be handled by _filter_data, but keep as safety)
        if len(all_actives) < self.min_active:
            return None
            
        
        # 1. Sampling Logic
        is_deterministic = self.split in ['test', 'valid']
        
        sampled_actives_ids = []
        sampled_inactives_ids = []
        
        if is_deterministic:
            # 1.1 Deterministic Logic (Test/Val)
            
            # Actives: Take all available (up to limit), deterministic order (slice)
            n_act = min(len(all_actives), self.max_actives)
            # Use slicing instead of random.sample to ensure order is fixed
            sampled_actives_ids = all_actives[:n_act] 
            
            # Inactives: Prioritize annotated inactives, slice them
            if all_inactives:
                n_inact = min(len(all_inactives), self.max_inactives)
                sampled_inactives_ids = all_inactives[:n_inact]
                        
        else:
            # 1.2 Stochastic Logic (Train)
            
            # Actives: Dynamic Sampling
            if random.random() < self.prob_single_active:
                n_act_target = 1
            else:
                n_act_target = random.randint(self.min_active, self.max_actives)
                
            n_act = min(len(all_actives), n_act_target)
            sampled_actives_ids = random.sample(all_actives, n_act)
            
            # Inactives: Dynamic Sampling
            if random.random() < self.prob_no_inactives:
                n_inact_target = 0
            else:
                n_inact_target = random.randint(0, self.max_inactives)
            
            if all_inactives:
                n_inact = min(len(all_inactives), n_inact_target)
                sampled_inactives_ids = random.sample(all_inactives, n_inact)
            else:
                # Negative Sampling from Global List
                sampled_inactives_ids = []
                max_attempts = n_inact_target * 2
                attempts = 0
                active_set = set(all_actives)
                
                while len(sampled_inactives_ids) < n_inact_target and attempts < max_attempts:
                    pid = random.choice(self.all_protein_ids)
                    if pid not in active_set and pid not in sampled_inactives_ids:
                        sampled_inactives_ids.append(pid)
                    attempts += 1
        
        # Update counts
        num_actives = len(sampled_actives_ids)
        num_inactives = len(sampled_inactives_ids)

        # 2. Load Features
        active_feats = []
        for pid in sampled_actives_ids:
            f = self._get_protein_features(pid)
            if f is not None: active_feats.append(f)
            
        inactive_feats = []
        for pid in sampled_inactives_ids:
            f = self._get_protein_features(pid)
            if f is not None: inactive_feats.append(f)
            
        num_actives = len(active_feats)
        num_inactives = len(inactive_feats)
        
        # 3. Construct Prompt with manual placeholder injection
        def create_separated_placeholders(count):
            if count <= 0:
                return torch.tensor([], dtype=torch.long)
            if count == 1:
                return torch.full((1,), self.protein_placeholder_id, dtype=torch.long)
            
            # Length = N proteins + (N-1) separators
            total_len = 2 * count - 1
            out = torch.full((total_len,), self.sep_token_id, dtype=torch.long)
            # Even positions fill protein placeholders
            out[0::2] = self.protein_placeholder_id
            return out
        
        # Select Template
        if num_inactives > 0 and self.templates_listwise:
            template = self._select_template(self.templates_listwise, idx)
        elif num_inactives == 0 and self.templates_active_only:
            template = self._select_template(self.templates_active_only, idx)
        else:
            if num_inactives > 0:
                template = "Generate a molecule that binds to {ACT_BLOCK} and avoids {INACT_BLOCK}."
            else:
                template = "Generate a molecule that binds to {ACT_BLOCK}."

        # Parse and Assemble
        parts = []
        
        # 1. Handle ACT_BLOCK
        if "{ACT_BLOCK}" in template:
            pre_act, post_act = template.split("{ACT_BLOCK}", 1)
            parts.append(self._tokenize_segment(pre_act))
            # Inject placeholder ID for actives
            parts.append(create_separated_placeholders(num_actives))
            remaining = post_act
        else:
            remaining = template
            
        # 2. Handle INACT_BLOCK
        if "{INACT_BLOCK}" in remaining:
            pre_inact, post_inact = remaining.split("{INACT_BLOCK}", 1)
            parts.append(self._tokenize_segment(pre_inact))
            # Inject placeholder ID for inactives
            parts.append(create_separated_placeholders(num_inactives))
            parts.append(self._tokenize_segment(post_inact))
        else:
            parts.append(self._tokenize_segment(remaining))
            
        # 3. Add Target (SMILES)
        mol_text = f" <mol>{smiles}</mol>"
        parts.append(self._tokenize_segment(mol_text))
        
        # 4. Concatenate
        input_ids = torch.cat([p for p in parts if p.numel() > 0], dim=0)
        
        # 5. Add BOS/EOS and Truncate
        max_len = self.config.pp2d.max_length
        
        # Prepend BOS
        if self.tokenizer.bos_token_id is not None:
            bos = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
            input_ids = torch.cat([bos, input_ids])
            
        # Truncate if needed (reserve 1 for EOS)
        if input_ids.shape[0] > max_len - 1:
            input_ids = input_ids[:max_len - 1]
            
        # Append EOS
        if self.tokenizer.eos_token_id is not None:
            eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            input_ids = torch.cat([input_ids, eos])

        # Drop samples where truncation removes any part of the supervised molecule span.
        mol_token_id, end_mol_token_id = infer_mol_token_ids(self.tokenizer)
        if not has_complete_mol_span(input_ids, mol_token_id, end_mol_token_id):
            return None
        
        return {
            "input_ids": input_ids,
            "active_feats": active_feats,
            "inactive_feats": inactive_feats,
            "num_actives": num_actives,
            "num_inactives": num_inactives,
            "smiles": smiles,
            "active_ids": sampled_actives_ids,
            "inactive_ids": sampled_inactives_ids
        }

# --- Collate ---

class ListwiseCollate:
    """
    Collates batch for Listwise/Set-based Protein Injection.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        # We use -1 (the value we set in Dataset) as the placeholder ID for proteins.
        # Note: This must match what is set in LigandProfileDataset.__getitem__
        self.protein_token_id = -1
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch: List[Dict]) -> Dict:
        # Filter None
        batch = [x for x in batch if x is not None]
        if not batch:
            return {}
            
        # Collect Metadata (IDs)
        active_ids_batch = [item['active_ids'] for item in batch]
        inactive_ids_batch = [item['inactive_ids'] for item in batch]
        smiles_batch = [item['smiles'] for item in batch]

        # 1. Protein Flattening
        flat_protein_feats = []
        flat_protein_mask = []
        
        batch_indices_flat = [] 
        
        max_p = 0
        for item in batch:
            total_p = item['num_actives'] + item['num_inactives']
            if total_p > max_p: max_p = total_p
            
        batch_size = len(batch)
        # We use -1 for padding in role_ids and gather_indices
        role_ids_tensor = torch.full((batch_size, max_p), -1, dtype=torch.long)
        gather_indices_tensor = torch.full((batch_size, max_p), -1, dtype=torch.long)
        
        current_flat_idx = 0
        
        for i, item in enumerate(batch):
            p_idx = 0
            
            # Actives (Role=1)
            for feat in item['active_feats']:
                flat_protein_feats.append(feat)
                mask = torch.ones(feat.shape[0], dtype=torch.bool)
                flat_protein_mask.append(mask)
                
                batch_indices_flat.append(i)
                
                role_ids_tensor[i, p_idx] = 1
                gather_indices_tensor[i, p_idx] = current_flat_idx
                
                current_flat_idx += 1
                p_idx += 1
                
            # Inactives (Role=0)
            for feat in item['inactive_feats']:
                flat_protein_feats.append(feat)
                mask = torch.ones(feat.shape[0], dtype=torch.bool)
                flat_protein_mask.append(mask)
                
                batch_indices_flat.append(i)
                
                role_ids_tensor[i, p_idx] = 0
                gather_indices_tensor[i, p_idx] = current_flat_idx
                
                current_flat_idx += 1
                p_idx += 1
        
        # Stack flat features
        if flat_protein_feats:
            # Pad sequences (features might have different lengths if graph node sequences)
            flat_protein_feats_tensor = efficient_pad_sequences(flat_protein_feats, padding_side='right')
            flat_protein_mask_tensor = efficient_pad_sequences(flat_protein_mask, padding_value=0.0, padding_side='right') 
        else:
            flat_protein_feats_tensor = torch.empty(0)
            flat_protein_mask_tensor = torch.empty(0)

        batch_indices_flat_tensor = torch.tensor(batch_indices_flat, dtype=torch.long)
        
        # 2. Text Padding (Left Padding)
        input_ids_list = [item['input_ids'] for item in batch]
        
        # Ensure pad_token_id is float for efficient_pad_sequences
        pad_val = 0.0
        if self.pad_token_id is not None:
            try:
                pad_val = float(self.pad_token_id)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid pad_token_id={self.pad_token_id}, fallback to 0.0: {e}")
                pad_val = 0.0
                
        input_ids_padded = efficient_pad_sequences(input_ids_list, padding_value=pad_val, padding_side='left')
        
        # Attention Mask
        attention_mask = (input_ids_padded != self.pad_token_id).to(torch.long)
        
        # 3. Labels
        labels = input_ids_padded.clone()
        labels[labels == self.pad_token_id] = -100
        
        # Mask Prompts
        mol_token_id: Optional[int] = None
        
        # Robustly find the EOS token ID
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else self.tokenizer.vocab
        if '<mol>' in vocab:
            mol_token_id = vocab['<mol>']
        
        if mol_token_id is None:
            try:
                test_pair = f"<mol></mol>"
                ids = self.tokenizer.encode(test_pair, add_special_tokens=False)
                mol_token_id = ids[0]
            except Exception as e:
                logger.error(f"Failed to infer <mol> token ID from tokenizer: {e}")
                raise ValueError("Failed to find <mol> token ID") from e
        
        mol_indices = find_first_occurrence(input_ids_padded, mol_token_id)
        
        seq_len = input_ids_padded.shape[1]
        col_indices = torch.arange(seq_len, device=input_ids_padded.device).unsqueeze(0)
        
        # Mask everything strictly BEFORE <mol>
        prompt_mask = col_indices < mol_indices.unsqueeze(1)
        labels[prompt_mask] = -100
        
        # 4. Protein Token Mask
        # Identify where we put the placeholder (-1) so the model knows where to inject features
        protein_token_mask = (input_ids_padded == self.protein_token_id)
        placeholder_counts = protein_token_mask.sum(dim=1)
        protein_counts = (gather_indices_tensor != -1).sum(dim=1)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels,
            "flat_protein_feats": flat_protein_feats_tensor,
            "flat_protein_mask": flat_protein_mask_tensor,
            "role_ids": role_ids_tensor, # [B, Max_P]
            "gather_indices": gather_indices_tensor, # [B, Max_P]
            "protein_token_mask": protein_token_mask,
            "batch_indices_flat": batch_indices_flat_tensor,
            "placeholder_counts": placeholder_counts,
            "protein_counts": protein_counts,
            "active_ids": active_ids_batch,
            "inactive_ids": inactive_ids_batch,
            "reference_smiles": smiles_batch
        }
