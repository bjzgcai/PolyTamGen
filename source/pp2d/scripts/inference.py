#!/usr/bin/env python3
"""
Optimized inference script for PP2D model (Listwise Architecture).
Loads test data (listwise profiles), model, performs inference, and saves results.
"""

import argparse
import json
import logging
import pickle as pkl
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to path
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from src.config import Config
    from src.data import LigandProfileDataset, ListwiseCollate
    from src.models.pp2d_model import PP2D
    from src.tokenizer import NatureLM1BTokenizer
    from src.utils.generation_utils import (
        build_bad_words_ids,
        build_generation_allowed_token_ids,
        build_prefix_allowed_tokens_fn,
        build_generation_suppressed_token_ids,
        build_prompt_only_batch,
        extract_smiles_from_decoded_sequence,
        finalize_generation_metrics,
        init_generation_metrics,
        update_generation_metrics,
    )
except ImportError as e:
    print(f"Error: Could not import project modules. Details: {e}")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _get_inference_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _save_generation_metrics(metrics: Dict[str, int], output_dir: Path, filename_prefix: str) -> None:
    summary = finalize_generation_metrics(metrics)
    metrics_path = output_dir / f"{filename_prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Generation metrics saved to %s | valid_smiles_rate=%.4f | forbidden_domain_token_rate=%.4f | mol_closure_rate=%.4f",
        metrics_path,
        summary["valid_smiles_rate"],
        summary["forbidden_domain_token_rate"],
        summary["mol_closure_rate"],
    )


def load_config(config_path: Path) -> Config:
    """Load configuration from a YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    return Config.from_yaml(config_path)


def load_model(config: Config, checkpoint_path: Path, device: torch.device) -> PP2D:
    """Load trained PP2D model from a checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Drop reference model weights (DPO checkpoints) to avoid unexpected keys.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('reference_model.')}

    # Detect LoRA-style checkpoints and enable LoRA if needed.
    has_lora_keys = any('lora_A.' in k or 'lora_B.' in k for k in state_dict.keys())
    has_peft_base = any(k.startswith('llm.base_model.') for k in state_dict.keys())
    if (has_lora_keys or has_peft_base) and not config.pp2d.use_lora:
        logger.info("Detected LoRA checkpoint; enabling config.pp2d.use_lora for inference.")
        config.pp2d.use_lora = True

    model = PP2D(config=config)

    strict = not (has_lora_keys or has_peft_base)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}...")

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully and set to evaluation mode.")
    return model


def load_test_data(
    config: Config,
    tokenizer: NatureLM1BTokenizer,
    batch_size: int,
    test_data_path: Path,
    split_path: Path,
    lmdb_path: Path,
    split_key: Optional[str] = None,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Tuple[str, DataLoader]]:
    """
    Load test datasets using Master file + Split index.
    """
    data_loaders = []
    dataset_name = split_key if split_key else "test_split"
    
    logger.info(f"Loading test data from master file: {test_data_path} using split: {split_path}")
    if split_key:
        logger.info(f"Using split key: {split_key}")
    
    # Temporarily override config to force dataset to use these paths
    config.data.data_path = str(test_data_path)
    config.data.split_path = str(split_path)
    config.data.lmdb_path = str(lmdb_path)
    
    try:
        # Initialize dataset in 'test' mode.
        # It will read data_path/split_path from config if present.
        test_dataset = LigandProfileDataset(
            config=config,
            tokenizer=tokenizer,
            split=split_key if split_key else 'test'
        )
        collate_fn = ListwiseCollate(tokenizer)
        
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive when set")
            original_len = len(test_dataset)
            test_dataset = Subset(test_dataset, list(range(min(limit, original_len))))
            logger.info(f"Applied inference sample limit: {len(test_dataset)} / {original_len} samples.")

        if len(test_dataset) == 0:
            logger.warning(f"Test dataset from split is empty.")
        else:
            if num_workers is None:
                num_workers = config.training.num_workers

            use_persistent_workers = persistent_workers and (num_workers > 0)

            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=use_persistent_workers,
            )
            if num_workers > 0 and prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

            test_loader = DataLoader(test_dataset, **loader_kwargs)
            logger.info(f"Test dataset loaded: {len(test_dataset)} samples.")
            data_loaders.append((dataset_name, test_loader))
            
    except Exception as e:
        logger.error(f"Failed to load test split: {e}")
        sys.exit(1)
    
    return data_loaders


def generate_target_id(active_ids: List[str], inactive_ids: List[str], smiles: str) -> str:
    """
    Generate a unique target ID for a listwise sample.
    Format: hash(sorted_actives + sorted_inactives + smiles_prefix)
    """
    # Sort to ensure deterministic ID regardless of order
    sorted_actives = sorted(active_ids)
    sorted_inactives = sorted(inactive_ids)
    
    sig = f"ACT:{','.join(sorted_actives)}_INACT:{','.join(sorted_inactives)}"
    # Add first 20 chars of SMILES to distinguish different reference ligands for same target set
    if smiles:
        sig += f"_SMI:{smiles[:20]}"
        
    return hashlib.md5(sig.encode()).hexdigest()


def generate_molecules(
    model: PP2D,
    batch: Dict[str, Any],
    tokenizer: NatureLM1BTokenizer,
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str]]:
    """Generate molecules for a single batch of data."""
    model_device = next(model.parameters()).device
    prompt_batch = build_prompt_only_batch(batch, tokenizer)

    # Move tensors to device
    batch_on_device = {}
    for k, v in prompt_batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(model_device, non_blocking=True)
        else:
            batch_on_device[k] = v

    use_amp = bool(getattr(args, "amp", False)) and model_device.type == "cuda"
    amp_dtype = _get_inference_autocast_dtype(model_device)
    bad_words_ids = None if getattr(args, "disable_domain_token_filter", False) else build_bad_words_ids(tokenizer)
    prefix_allowed_tokens_fn = None
    suppress_tokens = None
    if not getattr(args, "disable_molecule_vocab_constraint", False):
        generation_allowed_token_ids = build_generation_allowed_token_ids(tokenizer)
        prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(generation_allowed_token_ids)
        suppress_tokens = build_generation_suppressed_token_ids(tokenizer)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
        generated_outputs = model.generate(
            batch=batch_on_device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=not args.disable_sampling,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature if not args.disable_sampling else 1.0,
            top_p=args.top_p if not args.disable_sampling else 1.0,
            top_k=args.top_k if not args.disable_sampling else 50,
            tokenizer=tokenizer,
            bad_words_ids=bad_words_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            suppress_tokens=suppress_tokens,
        )

    if 'decoded_sequences' in generated_outputs:
        decoded_sequences = generated_outputs['decoded_sequences']
    else:
        generated_ids = generated_outputs['generated_ids']
        decoded_sequences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Clean sequences
    cleaned_sequences = []
    prompt_smiles_list = [] # Placeholder if we want to return prompts later
    
    for seq in decoded_sequences:
        cleaned = extract_smiles_from_decoded_sequence(seq)
        
        # Simple check for empty
        if not cleaned:
            cleaned = ""

        cleaned_sequences.append(cleaned)
        prompt_smiles_list.append("")
    
    return cleaned_sequences, prompt_smiles_list, decoded_sequences


def save_results(results: Dict[str, List], output_dir: Path, filename_prefix: str, num_return_sequences: int):
    """
    Save results in Pickle and JSON formats.
    """
    # --- Save as Pickle ---
    pickle_path = output_dir / f"{filename_prefix}_results.pkl"
    with open(pickle_path, 'wb') as f:
        pkl.dump(results, f)
    logger.info(f"Results dictionary saved to: {pickle_path}")

    # --- Save as JSON ---
    naturelm_format = {}
    
    target_ids = results['target_ids']
    generated_smiles = results['generated_smiles']
    reference_smiles = results['reference_smiles']
    active_pockets = results.get('active_pocket_ids', [])
    inactive_pockets = results.get('inactive_pocket_ids', [])
    
    total_gen = len(generated_smiles)
    
    for i in range(0, total_gen, num_return_sequences):
        t_id = target_ids[i]
        
        # Collect the generated sequences for this sample
        batch_gens = generated_smiles[i : i + num_return_sequences]
        ref = reference_smiles[i] if i < len(reference_smiles) else ""
        
        # Metadata
        actives = active_pockets[i] if i < len(active_pockets) else []
        inactives = inactive_pockets[i] if i < len(inactive_pockets) else []
        
        entry = {
            'generated_smiles': batch_gens,
            'reference_smiles': ref,
            'active_pockets': actives,
            'inactive_pockets': inactives
        }
        
        naturelm_format[t_id] = entry

    json_path = output_dir / f"{filename_prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump(naturelm_format, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")


def merge_shard_results(shard_dirs: List[Path], output_dir: Path, num_return_sequences: int) -> None:
    """Merge combined_inference_results.pkl from shard directories and save to output_dir."""
    merged: Dict[str, List] = {
        'target_ids': [],
        'reference_smiles': [],
        'generated_smiles': [],
        'dataset_names': [],
        'active_pocket_ids': [],
        'inactive_pocket_ids': []
    }

    for shard_dir in shard_dirs:
        pkl_path = shard_dir / "combined_inference_results.pkl"
        if not pkl_path.exists():
            logger.warning(f"Missing shard results: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            shard = pkl.load(f)
        for key in merged:
            merged[key].extend(shard.get(key, []))

    save_results(merged, output_dir, "combined_inference", num_return_sequences)

    merged_metrics = init_generation_metrics()
    for shard_dir in shard_dirs:
        metrics_path = shard_dir / "combined_inference_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            shard_metrics = json.load(f)
        for key in ("total_sequences", "valid_smiles_count", "forbidden_domain_token_count", "mol_closure_count"):
            merged_metrics[key] += int(shard_metrics.get(key, 0))
    _save_generation_metrics(merged_metrics, output_dir, "combined_inference")


def run_inference(args: argparse.Namespace):
    """Main inference pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if getattr(args, "tf32", False) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul/cudnn.")

    if getattr(args, "cudnn_benchmark", False) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled.")
    
    config = load_config(Path(args.config))
    
    # Tokenizer
    tokenizer = NatureLM1BTokenizer.from_pretrained(
        config.pp2d.llm_model_name_or_path,
        trust_remote_code=True
    )
    # Ensure PAD token
    if tokenizer.pad_token is None:
        if getattr(tokenizer, 'eos_token', None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
        
    # Model
    model = load_model(config, Path(args.checkpoint), device)
    
    # Data
    test_data_path = Path(args.test_data_path)
    split_path = Path(args.split_path)
    lmdb_path = Path(args.lmdb_path)
    data_loaders = load_test_data(
        config, 
        tokenizer, 
        args.batch_size,
        test_data_path=test_data_path,
        split_path=split_path,
        lmdb_path=lmdb_path,
        split_key=args.split_key,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        limit=args.limit,
    )
    
    combined_results: Dict[str, List] = {
        'target_ids': [],
        'reference_smiles': [],
        'generated_smiles': [],
        'dataset_names': [],
        'active_pocket_ids': [], 
        'inactive_pocket_ids': []
    }
    combined_metrics = init_generation_metrics()
    
    for dataset_name, test_loader in data_loaders:
        logger.info(f"\n--- Starting inference for dataset: {dataset_name} ---")
        dataset_results: Dict[str, List] = {k: [] for k in combined_results}
        dataset_metrics = init_generation_metrics()
        
        for batch in tqdm(test_loader, desc=f"Inference - {dataset_name}", unit="batch"):
            if batch is None:
                continue
            
            # Generate
            generated_smiles, _, decoded_sequences = generate_molecules(model, batch, tokenizer, args)
            update_generation_metrics(dataset_metrics, decoded_sequences, generated_smiles)
            
            batch_size = len(batch.get('active_ids', []))
            expected_len = batch_size * args.num_return_sequences
            
            if len(generated_smiles) != expected_len:
                logger.warning(f"Mismatch in generated SMILES count! Expected {expected_len}, got {len(generated_smiles)}. Skipping batch.")
                continue

            dataset_results['generated_smiles'].extend(generated_smiles)
            
            # Metadata expansion
            active_ids_batch = batch['active_ids']     # List of lists
            inactive_ids_batch = batch['inactive_ids'] # List of lists
            ref_smiles_batch = batch['reference_smiles']
            
            num_ret = args.num_return_sequences
            for i in range(batch_size):
                # Generate Target ID
                t_id = generate_target_id(active_ids_batch[i], inactive_ids_batch[i], ref_smiles_batch[i])

                # Expand for num_return_sequences (vectorized list extension)
                dataset_results['target_ids'].extend([t_id] * num_ret)
                dataset_results['reference_smiles'].extend([ref_smiles_batch[i]] * num_ret)
                dataset_results['active_pocket_ids'].extend([active_ids_batch[i]] * num_ret)
                dataset_results['inactive_pocket_ids'].extend([inactive_ids_batch[i]] * num_ret)
                dataset_results['dataset_names'].extend([dataset_name] * num_ret)
        
        num_generated = len(dataset_results['generated_smiles'])
        logger.info(f"Completed inference for '{dataset_name}': {num_generated} molecules generated.")
        
        # Save individual
        save_results(dataset_results, output_dir, f"{dataset_name}_inference", args.num_return_sequences)
        _save_generation_metrics(dataset_metrics, output_dir, f"{dataset_name}_inference")
        
        # Merge
        for key in combined_results:
            combined_results[key].extend(dataset_results[key])
        for metric_key in combined_metrics:
            combined_metrics[metric_key] += dataset_metrics[metric_key]

    logger.info("\n--- Combined Inference Complete ---")
    save_results(combined_results, output_dir, "combined_inference", args.num_return_sequences)
    _save_generation_metrics(combined_metrics, output_dir, "combined_inference")
    
    logger.info("\nSummary by dataset:")
    dataset_counts = {name: combined_results['dataset_names'].count(name) for name in set(combined_results['dataset_names'])}
    for name, count in dataset_counts.items():
        logger.info(f"  - {name}: {count} molecules")


def _parse_devices(devices_str: str) -> List[int]:
    devices = []
    for part in devices_str.split(','):
        part = part.strip()
        if not part:
            continue
        devices.append(int(part))
    if not devices:
        raise ValueError("--devices is empty")
    return devices


def _split_indices(indices: List[int], num_shards: int) -> List[List[int]]:
    if num_shards <= 1:
        return [indices]
    # round-robin split to balance sizes
    shards = [[] for _ in range(num_shards)]
    for i, idx in enumerate(indices):
        shards[i % num_shards].append(idx)
    return shards


def _write_split_file(base_split: Dict[str, List[int]], split_key: str, indices: List[int], out_path: Path) -> None:
    split = {"train": [], "val": [], "test": []}
    if split_key not in split:
        raise ValueError("split_key must be one of: train, val, test")
    split[split_key] = indices
    with open(out_path, 'wb') as f:
        pkl.dump(split, f)


def _run_single_shard(args: argparse.Namespace, device: int, split_path: Path, output_dir: Path) -> None:
    shard_args = argparse.Namespace(**vars(args))
    shard_args.device = device
    shard_args.split_path = str(split_path)
    shard_args.output_dir = str(output_dir)
    run_inference(shard_args)


def run_multi_gpu_inference(args: argparse.Namespace) -> None:
    devices = _parse_devices(args.devices)
    num_shards = args.num_shards or len(devices)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --devices was provided.")

    visible_count = torch.cuda.device_count()
    max_device = max(devices)
    if max_device >= visible_count:
        raise RuntimeError(
            f"Invalid device id in --devices: {devices}. "
            f"Visible CUDA devices: 0..{visible_count - 1}. "
            "Check CUDA_VISIBLE_DEVICES or choose valid ids."
        )

    # Load base split indices
    with open(args.split_path, 'rb') as f:
        split_dict = pkl.load(f)

    split_key = args.split_key if args.split_key else 'test'
    indices = split_dict.get(split_key, [])
    if not indices:
        raise RuntimeError(f"Split '{split_key}' is empty in {args.split_path}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when set")
        indices = indices[: args.limit]

    shards = _split_indices(indices, num_shards)
    if len(shards) < len(devices):
        devices = devices[:len(shards)]

    base_output = Path(args.output_dir)
    split_root = base_output / "shards"
    split_root.mkdir(parents=True, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    procs = []
    for i, device in enumerate(devices):
        shard_dir = base_output / f"shard_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_split_path = split_root / f"split_shard_{i}.pkl"
        _write_split_file(split_dict, split_key, shards[i], shard_split_path)

        p = ctx.Process(
            target=_run_single_shard,
            args=(args, device, shard_split_path, shard_dir),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Shard process failed with exit code {p.exitcode}")

    # Merge shard outputs
    shard_dirs = [base_output / f"shard_{i}" for i in range(len(devices))]
    logger.info("Merging shard outputs into combined_inference results...")
    merge_shard_results(shard_dirs, base_output, args.num_return_sequences)


def main():
    parser = argparse.ArgumentParser(description="Run inference with PP2D Listwise model.")
    
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    
    # Data Arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument("--test_data_path", type=str, required=True, help="Path to master data file (.pkl).")
    data_group.add_argument("--split_path", type=str, required=True, help="Path to split indices file (.pkl).")
    data_group.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB file containing protein features.")
    data_group.add_argument("--split_key", type=str, default=None, help="Key in split.pkl to use (e.g., test_simple, test_hard).")
    data_group.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Sampling
    parser.add_argument("--disable_sampling", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--disable_domain_token_filter", action='store_true',
                        help="Allow non-SMILES domain tokens during generation.")
    parser.add_argument("--disable_molecule_vocab_constraint", action='store_true',
                        help="Disable molecule-only vocabulary constraint during generation.")

    # Performance
    parser.add_argument("--amp", action='store_true', help="Enable AMP autocast (float16) for faster inference on CUDA.")
    parser.add_argument("--tf32", action='store_true', help="Enable TF32 for matmul/cudnn on CUDA.")
    parser.add_argument("--cudnn_benchmark", action='store_true', help="Enable cuDNN benchmark for fixed-size inputs.")

    # DataLoader performance
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader num_workers.")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch_factor (per worker).")
    parser.add_argument("--persistent_workers", action='store_true', help="Enable persistent DataLoader workers.")
    parser.add_argument("--no_pin_memory", action='store_true', help="Disable DataLoader pin_memory.")
    
    # System
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated GPU ids for multi-process inference (e.g., 0,1,2,3).")
    parser.add_argument("--num_shards", type=int, default=None, help="Override number of shards (default: len(devices)).")

    args = parser.parse_args()
    args.pin_memory = not args.no_pin_memory

    if args.devices:
        logger.info(f"Using multi-GPU inference on devices: {args.devices}")
        run_multi_gpu_inference(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
    
