import numpy as np
import math
import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial
try:
    from transformers import BertTokenizerFast
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    _text_scorer_import_error = None
except Exception as e:
    BertTokenizerFast = None
    corpus_bleu = None
    meteor_score = None
    rouge_scorer = None
    _text_scorer_import_error = e
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, QED
from copy import deepcopy
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import sascorer
import npscorer
import logging

from .dock_utils import dock

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 配置日志 - 只配置当前模块的logger，不影响全局
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TextScorer:
    def __init__(
        self,
        references: List[str],          # List of reference sentences (raw strings)
        candidates: List[List[str]],     # List of candidate lists per reference
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512
    ):
        """Initialize the scorer with multiple candidates per reference
        
        Args:
            references: List of reference strings
            candidates: List of candidate lists for each reference
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum token length for truncation
        """
        if _text_scorer_import_error is not None:
            raise ImportError(
                "TextScorer dependencies are missing. Install transformers, nltk, rouge-score."
            ) from _text_scorer_import_error
        # Store original texts for string-based metrics
        self.references = references
        self.candidates = candidates
        
        # Initialize tokenizer and preprocess
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length
        
        # Preprocess for token-based metrics (BLEU, METEOR)
        self.ref_tokens = [self._tokenize(text) for text in references]
        self.cand_tokens = [
            [self._tokenize(cand) for cand in cand_list]
            for cand_list in candidates
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text while removing special tokens"""
        return [
            token for token in self.tokenizer.tokenize(
                text,
                truncation=True,
                max_length=self.max_length
            )
            if token not in {'[PAD]', '[CLS]', '[SEP]'}
        ]

    def compute_bleu(self) -> Dict[str, float]:
        """Compute corpus BLEU-2 and BLEU-4 scores"""
        bleu2_scores, bleu4_scores = [], []
        
        for ref, cand_list in zip(self.ref_tokens, self.cand_tokens):
            # Wrap reference in double list for nltk format
            wrapped_ref = [[ref]]  # format: [[[ref1_tokens], [ref2_tokens], ...]]
            
            for cand in cand_list:
                # Calculate both BLEU scores
                bleu2 = corpus_bleu(wrapped_ref, [cand], weights=(0.5, 0.5))
                bleu4 = corpus_bleu(wrapped_ref, [cand], weights=(0.25, 0.25, 0.25, 0.25))
                
                bleu2_scores.append(bleu2)
                bleu4_scores.append(bleu4)
                
        return {
            'bleu2': np.mean(bleu2_scores) if bleu2_scores else 0.0,
            'bleu4': np.mean(bleu4_scores) if bleu4_scores else 0.0
        }

    def compute_meteor(self) -> float:
        """Compute average METEOR score across all candidates"""
        scores = []
        for ref, cand_list in zip(self.ref_tokens, self.cand_tokens):
            for cand in cand_list:
                # METEOR requires: reference(s) as list of lists, candidate as list
                score = meteor_score([ref], cand)
                scores.append(score)
        return np.mean(scores) if scores else 0.0

    def compute_rouge(self) -> Dict[str, float]:
        """Compute ROUGE scores using original text inputs"""
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=False  # Match MolT5 evaluation setup
        )
        
        rouge_scores = defaultdict(list)
        
        for ref_str, cand_list in zip(self.references, self.candidates):
            for cand_str in cand_list:
                scores = scorer.score(cand_str, ref_str)  # (candidate, reference)
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    rouge_scores[metric].append(scores[metric].fmeasure)
                    
        return {
            'rouge1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
            'rouge2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
            'rougeL': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
        }

    def report(self) -> Dict[str, float]:
        """Aggregate all evaluation metrics
        
        Returns:
            Combined dictionary of scores
        """
        bleu_scores = self.compute_bleu()
        meteor_score = self.compute_meteor()
        rouge_scores = self.compute_rouge()
        return {
            'bleu2': bleu_scores['bleu2'],
            'bleu4': bleu_scores['bleu4'],
            'meteor': meteor_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        }

class MolsScorer:
    """Evaluates molecular properties and quality metrics"""
    
    @staticmethod
    def _safe_parse(smiles: str) -> Optional[Chem.Mol]:
        """Safely parse SMILES with error handling"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol: Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    @staticmethod
    def _get_fingerprint(mol: Chem.Mol) -> Optional[DataStructs.ExplicitBitVect]:
        """Generate Morgan fingerprint"""
        try:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except Exception:
            return None
        
    @staticmethod
    def count_fused_rings(mol):
        # The method is generated by GPT-4. 
        ri = mol.GetRingInfo()  
        rings = list(ri.AtomRings())  
        ring_connections = [set() for _ in range(len(rings))]  
    
        for i, ring1 in enumerate(rings):  
            for j in range(i+1, len(rings)):  
                ring2 = rings[j]  
                if len(set(ring1).intersection(set(ring2))) >= 2:  
                    ring_connections[i].add(j)  
                    ring_connections[j].add(i)  
    
        visited = [False]*len(rings)  
        def dfs(v):  
            visited[v] = True  
            size = 1  
            for w in ring_connections[v]:  
                if not visited[w]:  
                    size += dfs(w)  
            return size  
    
        max_fused = 0  
        for i in range(len(rings)):  
            if not visited[i]:  
                max_fused = max(max_fused, dfs(i))  
    
        return max_fused

    def __init__(
        self,
        hyp_smiles_list: List[str],
        ref_smiles: str,
        docking: bool = False,
        # Compatibility/pairwise arguments (kept for compatibility if needed, or we can deprecate)
        pocket_path: Optional[str] = None,
        p1_path: Optional[str] = None,
        p2_path: Optional[str] = None,
        # New Listwise arguments
        active_pocket_paths: Optional[List[str]] = None,
        inactive_pocket_paths: Optional[List[str]] = None,
        
        docking_cache_dir: Optional[str] = None,
        output_complex_dir: Optional[str] = None
    ):
        """
        Initialize molecular scorer
        
        Args:
            hyp_smiles_list: Candidate SMILES strings
            ref_smiles: Reference SMILES string
            docking: Enable docking calculations
            
            # New Interface
            active_pocket_paths: List of paths to active protein pockets
            inactive_pocket_paths: List of paths to inactive protein pockets
            
            # Compatibility interface (Deprecated)
            pocket_path: Path to protein pocket file (compatibility mode, for single pocket)
            p1_path: Path to P1 pocket file (for dual-pocket docking)
            p2_path: Path to P2 pocket file (for dual-pocket docking)
            
            docking_cache_dir: Docking cache directory
            output_complex_dir: Output directory for complexes
        """
        # Parse molecules
        self.ref_mol = self._safe_parse(ref_smiles)
        if not self.ref_mol:
            raise ValueError(f"Invalid reference SMILES: {ref_smiles}")
            
        self.mols = []
        self.validity = []
        for smi in hyp_smiles_list:
            mol = self._safe_parse(smi)
            if mol:
                self.mols.append(mol)
                self.validity.append(True)
            else:
                self.mols.append(None)
                self.validity.append(False)

        # Docking setup
        self.docking_enabled = docking
        if docking:
            self.active_paths = []
            self.inactive_paths = []
            
            # Handle New Listwise Interface
            if active_pocket_paths or inactive_pocket_paths:
                if active_pocket_paths:
                    self.active_paths = active_pocket_paths
                if inactive_pocket_paths:
                    self.inactive_paths = inactive_pocket_paths
                    
            # Handle compatibility pairwise/single interface
            elif p1_path or p2_path:
                if p1_path: self.active_paths.append(p1_path)
                if p2_path: self.active_paths.append(p2_path)
            elif pocket_path:
                self.active_paths.append(pocket_path)
            else:
                raise ValueError("Docking enabled but no pocket paths provided (active_pocket_paths, p1_path, etc.)")

            # Validate paths
            for p in self.active_paths + self.inactive_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Pocket file missing: {p}")
            
            self.docking_cache_dir = docking_cache_dir or os.path.join(os.getcwd(), "docking_cache")
            self.output_complex_dir = output_complex_dir
            os.makedirs(self.docking_cache_dir, exist_ok=True)
            if output_complex_dir:
                os.makedirs(output_complex_dir, exist_ok=True)

        # Calculate metrics
        self._calculate_metrics()

    def _dock_single_pocket(self, pocket_path: str, pocket_name: str = "") -> Tuple[List[float], List[float]]:
        """
        Dock all molecules to a single pocket.
        
        Args:
            pocket_path: Path to the pocket PDB file
            pocket_name: Name for logging (e.g., "P1", "P2", or empty for compatibility mode)
        
        Returns:
            Tuple of (dock_scores, ligand_efficiency) lists
        """
        dock_scores = []
        ligand_efficiency = []
        prefix = f"{pocket_name} " if pocket_name else ""
        
        for idx, mol in enumerate(self.mols):
            if mol is None:
                dock_scores.append(np.nan)
                ligand_efficiency.append(np.nan)
                continue
            
            try:
                mol_smiles = Chem.MolToSmiles(mol)
                output_filename = f"{pocket_name.lower()}_complex_{idx}.pdbqt" if pocket_name else f"complex_{idx}.pdbqt"
                output_path = os.path.join(self.output_complex_dir, output_filename) \
                    if self.output_complex_dir else None
                
                dock_score = dock(
                    pocket_path,
                    mol_smiles,
                    output_complex_path=output_path,
                    cache_dir=self.docking_cache_dir
                )
                
                # Normalize abnormal scores: docking affinity通常为负值
                if (
                    dock_score is None
                    or not isinstance(dock_score, (int, float))
                    or math.isnan(dock_score)
                    or math.isinf(dock_score)
                    or dock_score >= 200  # compatibility sentinel 250/500
                    or dock_score >= 0    # positive affinities are suspicious
                ):
                    logger.warning(f"{prefix}docking abnormal score for molecule {idx}: {dock_score}")
                    dock_scores.append(np.nan)
                    ligand_efficiency.append(np.nan)
                else:
                    dock_scores.append(dock_score)
                    ligand_efficiency.append(dock_score / self.num_atoms[idx])
            except Exception as e:
                logger.error(f"{prefix}docking failed for molecule {idx}: {str(e)}")
                dock_scores.append(np.nan)
                ligand_efficiency.append(np.nan)
        
        return dock_scores, ligand_efficiency

    def _calculate_metrics(self):
        """Calculate all molecular metrics"""
        # Basic properties
        self.sas = [sascorer.calculateScore(mol) if mol is not None else np.nan for mol in self.mols]
        
        # Normalized SA (0 to 1, higher is better)
        # Original SA is 1 (easy) to 10 (hard)
        # Formula: 1 - (SA - 1) / 9
        self.sa_norm = [1 - (s - 1) / 9 if not np.isnan(s) else np.nan for s in self.sas]
        
        self.qed = [QED.qed(mol) if mol is not None else np.nan for mol in self.mols]
        self.logp = [Crippen.MolLogP(mol) if mol is not None else np.nan for mol in self.mols]
        self.good_logp = [0 <= logp <= 5 for logp in self.logp]
        self.molwt = [Descriptors.MolWt(mol) if mol is not None else np.nan for mol in self.mols]
        self.num_atoms = [mol.GetNumAtoms() if mol is not None else np.nan for mol in self.mols]
        self.lipinski = [self._check_lipinski(mol) if mol is not None else np.nan for mol in self.mols]
        self.fused_rings = [self.count_fused_rings(mol) if mol is not None else np.nan for mol in self.mols]
        
        # Similarity calculation
        ref_fp = self._get_fingerprint(self.ref_mol)
        self.similarity = [
            DataStructs.TanimotoSimilarity(self._get_fingerprint(mol), ref_fp)
            if (mol and ref_fp) else np.nan
            for mol in self.mols
        ]
        
        # Diversity calculation
        self.diversity = self._compute_diversity()
        
        # Docking scores
        self.dock_results_active = {}   # {path: (scores, efficiency)}
        self.dock_results_inactive = {} # {path: (scores, efficiency)}
        
        if self.docking_enabled:
            nan_list = [np.nan] * len(self.mols)
            
            # Active Proteins
            for path in self.active_paths:
                pocket_name = os.path.splitext(os.path.basename(path))[0]
                scores, eff = self._dock_single_pocket(path, f"Active_{pocket_name}")
                self.dock_results_active[path] = (scores, eff)
                
            # Inactive Proteins
            for path in self.inactive_paths:
                pocket_name = os.path.splitext(os.path.basename(path))[0]
                scores, eff = self._dock_single_pocket(path, f"Inactive_{pocket_name}")
                self.dock_results_inactive[path] = (scores, eff)

    @staticmethod
    def _check_lipinski(mol: Chem.Mol) -> bool:
        """Check Lipinski's rule of five"""
        return sum([
            Descriptors.ExactMolWt(mol) < 500,
            Lipinski.NumHDonors(mol) <= 5,
            Lipinski.NumHAcceptors(mol) <= 10,
            -2 <= Crippen.MolLogP(mol) <= 5
        ]) >= 3

    def _compute_diversity(self) -> float:
        """Calculate molecular diversity using parallel processing"""
        if len(self.mols) < 2:
            return 0.0
            
        fps = [self._get_fingerprint(mol) for mol in self.mols]
        fps = [fp for fp in fps if fp is not None]
        
        if len(fps) < 2:
            return 0.0

        total_sim = 0.0
        count = 0
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                total_sim += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                count += 1
                
        return 1 - (total_sim / count) if count > 0 else 0.0

    def report(self, mean: bool = True) -> Dict:
        """Generate molecular evaluation report"""
        metrics = {
            'sas': self.sas,
            'sa_norm': self.sa_norm,
            'qed': self.qed,
            'logp': self.logp,
            'good_logp': self.good_logp,
            'molwt': self.molwt,
            'num_atoms': self.num_atoms,
            'lipinski': [float(v) for v in self.lipinski],
            'fused_rings': self.fused_rings,
            'similarity': self.similarity,
            'validity': self.validity,
            'diversity': [self.diversity]
        }
        
        if self.docking_enabled:
            if not mean:
                metrics['dock_scores_active_by_pocket'] = {
                    path: scores for path, (scores, _) in self.dock_results_active.items()
                }
                metrics['dock_scores_inactive_by_pocket'] = {
                    path: scores for path, (scores, _) in self.dock_results_inactive.items()
                }
                metrics['ligand_efficiency_active_by_pocket'] = {
                    path: eff for path, (_, eff) in self.dock_results_active.items()
                }
                metrics['ligand_efficiency_inactive_by_pocket'] = {
                    path: eff for path, (_, eff) in self.dock_results_inactive.items()
                }

            # Aggregate scores from all active pockets
            all_active_scores = []
            all_active_eff = []
            
            for path, (scores, eff) in self.dock_results_active.items():
                # We can report per-pocket or average
                # For summary, we average across all active pockets for each molecule?
                # Or just flatten?
                # Let's average per molecule first if we want "Average Affinity to Actives"
                all_active_scores.append(scores)
                all_active_eff.append(eff)
            
            if all_active_scores:
                # scores is [N_mols], all_active_scores is [N_pockets, N_mols]
                # avg_active_scores: [N_mols]
                avg_active_scores = np.nanmean(all_active_scores, axis=0)
                avg_active_eff = np.nanmean(all_active_eff, axis=0)
                
                metrics['dock_scores_active_avg'] = avg_active_scores
                metrics['ligand_efficiency_active_avg'] = avg_active_eff
                
            # Aggregate scores from all inactive pockets
            all_inactive_scores = []
            all_inactive_eff = []
            
            for path, (scores, eff) in self.dock_results_inactive.items():
                all_inactive_scores.append(scores)
                all_inactive_eff.append(eff)
                
            if all_inactive_scores:
                avg_inactive_scores = np.nanmean(all_inactive_scores, axis=0)
                avg_inactive_eff = np.nanmean(all_inactive_eff, axis=0)
                
                metrics['dock_scores_inactive_avg'] = avg_inactive_scores
                metrics['ligand_efficiency_inactive_avg'] = avg_inactive_eff

        return {k: np.nanmean(v) for k, v in metrics.items() if v is not None} if mean else metrics
class MultiMolsScorer:
    def __init__(self,
                 hyp_smiles_list: List[List[str]],
                 ref_smiles_list: List[str],
                 docking: bool = False,
                 # Compatibility/pairwise arguments
                 pocket_paths: Optional[List[str]] = None,
                 p1_paths: Optional[List[str]] = None,
                 p2_paths: Optional[List[str]] = None,
                 # New Listwise arguments
                 active_pocket_paths_list: Optional[List[List[str]]] = None,
                 inactive_pocket_paths_list: Optional[List[List[str]]] = None,
                 
                 docking_cache_dir: Optional[str] = None,
                 output_complex_base_dir: Optional[str] = None,
                 num_workers: int = 4):
        """
        Multi-group molecular scorer with parallel processing
        
        Args:
            hyp_smiles_list: List of candidate SMILES lists for each group
            ref_smiles_list: List of reference SMILES for each group
            docking: Enable docking calculations
            
            # New Interface
            active_pocket_paths_list: List of lists of active pocket paths
            inactive_pocket_paths_list: List of lists of inactive pocket paths
            
            # Compatibility interface
            pocket_paths: Protein pocket paths for each group (compatibility mode, for single pocket)
            p1_paths: P1 pocket paths for each group (for dual-pocket docking)
            p2_paths: P2 pocket paths for each group (for dual-pocket docking)
            
            docking_cache_dir: Directory for docking cache files
            output_complex_base_dir: Base directory for docking output complexes
            num_workers: Number of parallel workers
        """
        # Parameter validation
        assert len(hyp_smiles_list) == len(ref_smiles_list), "Mismatched group count"
        
        # Prepare params for each group
        self.group_params = []
        for idx, (hyp_smiles, ref_smi) in enumerate(zip(hyp_smiles_list, ref_smiles_list)):
            params = {
                "hyp_smiles_list": hyp_smiles,
                "ref_smiles": ref_smi,
                "docking": docking,
                "docking_cache_dir": docking_cache_dir,
                "output_complex_dir": os.path.join(output_complex_base_dir, f"group_{idx}") 
                                    if output_complex_base_dir else None
            }
            
            if docking:
                # Map Listwise arguments
                if active_pocket_paths_list:
                    params["active_pocket_paths"] = active_pocket_paths_list[idx] if idx < len(active_pocket_paths_list) else []
                if inactive_pocket_paths_list:
                    params["inactive_pocket_paths"] = inactive_pocket_paths_list[idx] if idx < len(inactive_pocket_paths_list) else []
                
                # Map compatibility arguments
                if p1_paths or p2_paths:
                    # If compatibility paths are provided, we can also add them to active paths for backward compatibility
                    # But MolsScorer handles compatibility args too, so we can pass them directly if we want.
                    # However, MolsScorer Logic: if active_paths is set, it ignores p1/p2.
                    # Let's pass compatibility args if they exist so MolsScorer can decide.
                    if p1_paths and idx < len(p1_paths): params["p1_path"] = p1_paths[idx]
                    if p2_paths and idx < len(p2_paths): params["p2_path"] = p2_paths[idx]
                
                if pocket_paths and idx < len(pocket_paths):
                    params["pocket_path"] = pocket_paths[idx]

            self.group_params.append((idx, params))

        # Process groups
        if num_workers is None or num_workers <= 1:
            # Sequential processing (no extra workers)
            results = []
            for idx, params in self.group_params:
                results.append(self._process_group(idx, params))
        else:
            if docking:
                # When docking is enabled, avoid nested process pools:
                # inner dock() already uses multiprocessing.Process.
                # Here we parallelize groups with threads instead.
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(self._process_group, idx, params)
                        for idx, params in self.group_params
                    ]
                    results = [f.result() for f in futures]
            else:
                # Non-docking case: safe to use process pool
                with Pool(processes=num_workers) as pool:
                    results = pool.starmap(self._process_group, self.group_params)
        
        # Store results as {group_idx: scorer}
        self.scorers = {idx: scorer for idx, scorer in results if scorer is not None}

    @staticmethod
    def _process_group(idx: int, params: dict) -> Tuple[int, Optional[MolsScorer]]:
        """Process a single group with error handling"""
        try:
            return idx, MolsScorer(**params)
        except Exception as e:
            logger.error(f"Group {idx} processing failed: {str(e)}")
            return idx, None

    def report(self, mean: bool = False) -> Dict:
        """
        Generate unified report across all groups
        
        Args:
            mean: 
                True - Calculate mean of group means (two-level averaging)
                False - Return distribution reports per group
        
        Returns:
            results (Dict):
            - Dict of average metrics across group averages(when mean is True)
            - Dict of raw distributions per group (when mean is False)
        """
        if mean:
            return self._calculate_mean_of_means()
        return self._get_group_distributions()

    def _calculate_mean_of_means(self) -> Dict[str, float]:
        """Two-level averaging: group mean -> mean of group means"""
        valid_groups = [s for s in self.scorers.values() if s is not None]
        if not valid_groups:
            return {}
            
        # First level: Calculate group means
        group_means = []
        for scorer in valid_groups:
            try:
                group_report = scorer.report(mean=True)
                group_means.append(group_report)
            except Exception as e:
                logger.warning(f"Skipping invalid group: {str(e)}")
                continue

        # Second level: Aggregate across groups
        metric_aggregator = defaultdict(list)
        for report in group_means:
            for metric, value in report.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_aggregator[metric].append(value)

        results = {}
        results['group_count'] = len(group_means)
        for k, v in metric_aggregator.items():
            results[k] = np.nanmean(v) 
        return results

    def _get_group_distributions(self) -> Dict[int, Dict[str, List]]:
        """Return raw distributions per group with original indices"""
        return {
            idx: scorer.report(mean=False)
            for idx, scorer in self.scorers.items()
            if scorer is not None
        }
