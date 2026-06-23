import yaml
import os
import argparse
import sys
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import List, Dict, Optional, Union, Any, Type
from pathlib import Path

# ==========================================
# 1. Helper Functions
# ==========================================

def str2bool(v):
    """
    Converts command line string arguments to boolean values.
    Handles 'yes', 'true', 't', 'y', '1' as True and their opposites as False.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==========================================
# 2. Configuration Dataclasses
# ==========================================

@dataclass
class PocketEncoderConfig:
    """Configuration for the Enhanced Pocket Encoder."""
    # Basic Feature Dimensions
    num_atom_features: int = 29
    dist_feat_dim: int = 16
    dir_feat_dim: int = 3
    seq_feat_dim: int = 8
    embedding_dim: int = 256  # Final output dimension of the global pocket embedding
    
    radius_cutoff: float = 10.0

    # Core Network: EGNN
    egnn_dim: int = 128                # Internal feature dimension for each atom/residue
    egnn_depth: int = 5                # Number of message passing layers
    
    # Diffusion Model Parameters
    diffusion_num_timesteps: int = 1000
    diffusion_coord_loss_weight: float = 1.0
    
    # Advanced Pretraining Parameters
    contrastive_temperature: float = 0.1
    contrastive_projection_dim: int = 128
    
    # Masked Node Prediction
    mask_ratio: float = 0.15
    
    # Graph Property Prediction
    num_graph_properties: int = 5
    
    # Multi-task Loss Weights
    diffusion_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    mask_loss_weight: float = 0.1
    property_loss_weight: float = 0.05

@dataclass
class PP2DConfig:
    """Configuration for the PP2D (PALM) model architecture."""
    llm_model_name_or_path: str = "models/base-llm/NLM-1b"
    pocket_encoder_model: Optional[str] = None  # Path to pretrained pocket encoder
    max_length: int = 512
    
    # Feature Mode Flags
    use_precomputed_features: bool = False
    precomputed_feature_dim: int = 1024 
    
    # Interactor Flags
    protein_adapter_type: str = "qformer" # Options: "qformer", "linear"
    adapter_layers: int = 2  
    
    # Listwise Architecture parameters
    num_queries: int = 32
    use_set_transformer: bool = True
    set_transformer_layers: int = 2
    set_transformer_heads: int = 8
    rope_theta: float = 10000.0 
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    lora_dropout: float = 0.1
    
    # Load prompt templates dynamically
    @property
    def prompt_templates(self):
        from .prompt_templates import get_templates
        # Return listwise templates by default
        return get_templates("listwise")
    
    @property
    def prompt_templates_active_only(self):
        from .prompt_templates import get_templates
        return get_templates("listwise_active_only")
    
    pretrained_model_path: Optional[str] = None
    
@dataclass
class LossConfig:
    """Configuration for the loss functions."""
    pad_token_id: Optional[int] = None
    # SMILES Loss Configuration
    smiles_loss_type: str = "cross_entropy" # Options: "cross_entropy", "label_smoothing"
    label_smoothing: float = 0.1

    
@dataclass
class LRSchedulerConfig:
    """Configuration for the Learning Rate Scheduler."""
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    scheduler_params: Optional[dict] = None
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.01
    step_size: int = 30
    gamma: float = 0.1
    exponential_gamma: float = 0.95
    patience: int = 10
    factor: float = 0.5
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    seed: int = 42
    resume: bool = False
    epochs: int = 50
    batch_size: int = 8
    update_freq: int = 1
    learning_rate: float = 1e-4
    num_workers: int = 4
    persistent_workers: bool = False
    pin_memory: bool = True
    dataloader_timeout_sec: int = 0
    log_interval: int = 100
    keep_top_k: int = 3
    log_level: str = "INFO"
    log_dir: str = "logs"
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    
    # Checkpoint saving
    save_interval_steps: Optional[int] = None 
    
    # Specific learning rates
    base_learning_rate: Optional[float] = None
    encoder_learning_rate: Optional[float] = None
    llm_learning_rate: Optional[float] = None
    
    # DataLoader sampling
    max_active: int = 5
    max_inactive: int = 8
    min_active: int = 1
    
    prob_no_inactives: float = 0.5 
    prob_single_active: float = 0.2 
    
    use_mixed_precision: bool = False
    mixed_precision_dtype: str = "fp16"

    # Checkpoint saving options
    save_weights_only: bool = True
    save_weights_dtype: str = "bf16"
    save_optimizer_last: bool = True
    generation_validation_enabled: bool = False
    generation_validation_num_samples: int = 128
    generation_validation_max_new_tokens: int = 256
    generation_validation_num_beams: int = 1
    generation_validation_disable_sampling: bool = True
    generation_validation_temperature: float = 1.0
    generation_validation_top_p: float = 0.95
    generation_validation_top_k: int = 100
    generation_best_checkpoint_enabled: bool = True
    generation_best_keep_top_k: int = 1
    generation_validation_distributed: bool = False
    strict_protein_alignment_check: bool = True

@dataclass
class DataConfig:
    """Configuration for data paths and splitting."""
    train_data: List[str] = field(default_factory=list)
    valid_data: List[str] = field(default_factory=list)
    test_data: List[str] = field(default_factory=list)

    preference_path: Optional[str] = None
    preference_train_path: Optional[str] = None
    preference_valid_path: Optional[str] = None
    preference_test_path: Optional[str] = None
    
    data_path: Optional[str] = None   
    split_path: Optional[str] = None  
    
    lmdb_path: str = ""
    pdb_dir: Optional[str] = None
    encoder_tsv_files: List[str] = field(default_factory=list)

    # Debugging / Subsetting
    train_subset: bool = False
    valid_subset: bool = False
    test_subset: bool = False
    train_subset_size: int = 100000
    valid_subset_size: int = 1000
    test_subset_size: int = 100

@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    beta: float = 0.1
    reference_model_path: Optional[str] = None
    reference_free: bool = False
    length_normalize_logps: bool = False

@dataclass
class DistributedConfig:
    """Configuration for distributed training (DDP)."""
    distributed: bool = False
    world_size: Optional[int] = None
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = False

@dataclass
class Config:
    """
    Main Configuration Class.
    Combines all sub-configurations into a single structure.
    """
    pocket_encoder: PocketEncoderConfig = field(default_factory=PocketEncoderConfig)
    pp2d: PP2DConfig = field(default_factory=PP2DConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    
    # Note: 'encoder_training' has been removed as per requirements.
    
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    training_mode: str = "pp2d"
    output_path: str = "models/model.pth"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        config = cls()
        config._update_from_dict(yaml_data)
        return config

    def _coerce_value(self, field_type: Type, value: Any) -> Any:
        """Coerce YAML values to the dataclass field type."""
        if value is None:
            return None

        # Unwrap Optional[T]
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            args = [t for t in field_type.__args__ if t is not type(None)]
            if len(args) == 1:
                field_type = args[0]

        # Handle List[T]
        if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            elem_type = field_type.__args__[0] if field_type.__args__ else Any
            if isinstance(value, list):
                return [self._coerce_value(elem_type, v) for v in value]
            return value

        # Primitive types
        if field_type is bool:
            return str2bool(value) if isinstance(value, str) else bool(value)
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
        if field_type is str:
            return str(value)

        return value
    
    def _update_from_dict(self, data: Dict):
        """Recursively update config attributes from a dictionary."""
        field_types = {f.name: f.type for f in fields(self)}
        for k, v in data.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if is_dataclass(attr) and isinstance(v, dict):
                    self._update_dataclass(attr, v)
                else:
                    field_type = field_types.get(k, Any)
                    setattr(self, k, self._coerce_value(field_type, v))

    def _update_dataclass(self, obj, data: Dict):
        """Helper to update nested dataclasses."""
        field_types = {f.name: f.type for f in fields(obj)}
        for k, v in data.items():
            if hasattr(obj, k):
                attr = getattr(obj, k)
                if is_dataclass(attr) and isinstance(v, dict):
                    self._update_dataclass(attr, v)
                else:
                    field_type = field_types.get(k, Any)
                    setattr(obj, k, self._coerce_value(field_type, v))

    def to_yaml(self, save_path: str):
        """Save the current configuration to a YAML file."""
        with open(save_path, 'w') as f:
            yaml.dump(asdict(self), f, sort_keys=False)

# ==========================================
# 3. Smart Argument Parsing Logic
# ==========================================

def _get_all_leaf_fields(dataclass_type: Type, prefix: str = "") -> List[tuple]:
    """
    Recursively collect all leaf fields from the dataclass structure.
    
    Returns:
        List of tuples: (nested_path, field_name, field_info)
        Example: [('distributed.world_size', 'world_size', Field(...)), ...]
    """
    leaves = []
    for field_info in fields(dataclass_type):
        field_name = field_info.name
        # Build the dot-separated path (e.g., "training.lr_scheduler.warmup_steps")
        current_path = f"{prefix}{field_name}" if prefix else field_name
        
        field_type = field_info.type
        # Handle Optional[T] by extracting T
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            args = field_type.__args__
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                field_type = non_none_types[0]

        if is_dataclass(field_type):
            # Recursively go deeper
            leaves.extend(_get_all_leaf_fields(field_type, prefix=f"{current_path}."))
        else:
            # It's a leaf node
            leaves.append((current_path, field_name, field_info))
    return leaves

def get_parser_from_config(config_class: Type[Config] = Config) -> argparse.ArgumentParser:
    """
    Creates an argument parser with "Smart Flattening".
    
    Logic:
    1. If a field name is unique across the whole config (e.g., 'world_size'), 
       it creates a flat argument: --world_size.
    2. If a field name appears multiple times (name collision), 
       it creates a nested argument: --section.param.
    """
    parser = argparse.ArgumentParser(description="Auto-generated training script.")
    
    # Required argument: The base config file
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")

    # 1. Collect all fields first
    all_leaves = _get_all_leaf_fields(config_class)
    
    # 2. Count occurrences of each field name to detect collisions
    name_counts = {}
    for _, name, _ in all_leaves:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    # 3. Add arguments to parser
    for path, name, field_info in all_leaves:
        # Determine the CLI flag name
        if name_counts[name] == 1:
            # Unique? Flatten it! (e.g., --world_size)
            arg_name = f"--{name}"
            dest_name = name 
        else:
            # Duplicate? Keep nested path! (e.g., --training.seed)
            arg_name = f"--{path}"
            dest_name = path 

        # Determine how to handle the type in argparse
        field_type = field_info.type
        
        # Unwrap Optional
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            args = field_type.__args__
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                field_type = non_none_types[0]
        
        # Handle Lists (nargs='+')
        if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            parser.add_argument(arg_name, dest=dest_name, nargs='+', help=f"List override for {path}")
            continue

        # Handle Bools (custom converter)
        type_fn = field_type
        if field_type is bool:
            type_fn = str2bool
        
        # Add the argument
        # required=False because we assume defaults come from the YAML/Dataclass
        parser.add_argument(arg_name, dest=dest_name, type=type_fn, help=f"Override {path}", required=False)

    return parser

def update_config_from_args(config: Config, args: argparse.Namespace):
    """
    Updates the config object using the arguments parsed from CLI.
    Reconstructs the mapping from flat names to nested paths.
    """
    arg_dict = vars(args)
    
    # Rebuild the mapping logic to know where to put the values
    all_leaves = _get_all_leaf_fields(type(config))
    name_counts = {}
    path_map = {} # Mapping: argument_name -> nested_path
    
    # Count names
    for _, name, _ in all_leaves:
        name_counts[name] = name_counts.get(name, 0) + 1
        
    # Build map
    for path, name, _ in all_leaves:
        if name_counts[name] == 1:
            # If unique, the arg name was the simple name
            path_map[name] = path 
        else:
            # If duplicate, the arg name was the full path
            path_map[path] = path 

    # Apply updates
    for arg_key, value in arg_dict.items():
        if arg_key == 'config' or value is None:
            continue
        
        # Find where this argument belongs in the nested config
        target_path = path_map.get(arg_key)
        
        if not target_path:
            continue

        # Navigate to the object and set the value
        parts = target_path.split('.')
        obj = config
        try:
            # Go down to the parent object
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            # Set attribute on the leaf
            leaf_attr = parts[-1]
            if hasattr(obj, leaf_attr):
                setattr(obj, leaf_attr, value)
                # print(f"  [Config Update] Overriding {target_path} = {value}") 
        except AttributeError:
            print(f"  [Warning] Could not set {target_path}")
            pass

    return config

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    # 1. Build Parser
    parser = get_parser_from_config(Config)
    
    # 2. Parse Arguments
    # Note: If running in Jupyter/Colab, use args=[] or similar
    args = parser.parse_args()
    
    # 3. Load Base Config from YAML
    try:
        config = Config.from_yaml(args.config)
        print(f"Successfully loaded base config from: {args.config}")
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
        
    # 4. Override with CLI Arguments
    config = update_config_from_args(config, args)
    
    # --- Debugging Output: Verify Values ---
    print("-" * 40)
    print("Final Configuration (Selected Values):")
    print(f"  World Size (Distributed): {config.distributed.world_size}")
    print(f"  Learning Rate (Training): {config.training.learning_rate}")
    print(f"  Batch Size (Training):    {config.training.batch_size}")
    print(f"  Data Path:                {config.data.data_path}")
    print("-" * 40)
    
    # --- Start Training Logic Here ---
    # train(config)

if __name__ == "__main__":
    main()
    
