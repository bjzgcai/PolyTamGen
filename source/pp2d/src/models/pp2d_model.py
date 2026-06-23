import logging
import os
import torch
import torch.nn as nn
from contextlib import nullcontext
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Optional, Any

try:
    from .pocket_encoder import PocketEncoder
except ImportError:
    PocketEncoder = None 

from .set_transformer import ProteinSetTransformer
from .adapter import ProteinAdapter
from ..config import Config
from ..data import efficient_pad_sequences
from ..utils.loss_utils import masked_label_smoothing_nll

# Use unified logger
logger = logging.getLogger(__name__)

class PP2D(nn.Module):
    """
    The definitive end-to-end Pocket-Pair Drug Design (PP2D) model.
    Listwise Architecture with ProteinAdapter and ProteinSetTransformer.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._warned_missing_allowed_target = False
        self.register_buffer("label_smoothing_allowed_mask", torch.zeros(0, dtype=torch.bool), persistent=False)
        
        # Mixed precision support
        self.use_mixed_precision = getattr(config.training, 'use_mixed_precision', False)
        self.mixed_precision_dtype = getattr(config.training, 'mixed_precision_dtype', 'fp16')
        
        # --- 1. Instantiate Pocket Encoder ---
        
        self.use_precomputed_features = getattr(config.pp2d, 'use_precomputed_features', False)
        
        if self.use_precomputed_features:
            self.pocket_encoder = None
            input_dim = config.pp2d.precomputed_feature_dim
            logger.info(f"Using precomputed pocket features with dim: {input_dim}")
        else:
            if PocketEncoder is None:
                raise ImportError("PocketEncoder could not be imported. Cannot proceed without it in graph mode.")
            self.pocket_encoder = PocketEncoder(config.pocket_encoder)
            input_dim = config.pocket_encoder.embedding_dim
            logger.info(f"Using Pocket encoder. Features will be node sequences (dim: {input_dim})")
            
            if config.pp2d.pocket_encoder_model:
                self._load_pretrained_encoder(config.pp2d.pocket_encoder_model)
        
        # --- 2. Load and Configure the Language Model ---
        llm_config = AutoConfig.from_pretrained(config.pp2d.llm_model_name_or_path)
        
        base_llm = AutoModelForCausalLM.from_pretrained(
            config.pp2d.llm_model_name_or_path,
            config=llm_config
        )

        self.vocab_size = llm_config.vocab_size
        self.llm_hidden_size = llm_config.hidden_size
        logger.info(f"Using LLM vocab size: {self.vocab_size}, hidden size: {self.llm_hidden_size}")

        if config.pp2d.use_lora:
            lora_config = LoraConfig(
                r=config.pp2d.lora_r,
                lora_alpha=config.pp2d.lora_alpha,
                target_modules=config.pp2d.lora_target_modules,
                lora_dropout=config.pp2d.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(base_llm, lora_config)
            logger.info("LoRA enabled for LLM.")
            self.llm.print_trainable_parameters()
        else:
            self.llm = base_llm
            logger.info("LoRA disabled. Base LLM loaded; trainability is controlled by training.llm_learning_rate.")

        # --- 3. Listwise Interaction Components ---
        
        # Protein Adapter: Maps protein features to LLM space
        self.adapter = ProteinAdapter(config, input_dim, self.llm_hidden_size)
        logger.info(f"ProteinAdapter initialized (mode: {config.pp2d.protein_adapter_type})")

        # Set Transformer: Models interactions between proteins
        if config.pp2d.use_set_transformer:
            self.set_transformer = ProteinSetTransformer(
                config=config,
                embed_dim=self.llm_hidden_size
            )
            logger.info("ProteinSetTransformer initialized.")
        else:
            self.set_transformer = None
            logger.info("ProteinSetTransformer disabled.")

        logger.info(f"End-to-end PP2D initialized.")

    def set_label_smoothing_allowed_token_ids(self, token_ids: List[int]):
        device = self.label_smoothing_allowed_mask.device
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        if token_ids:
            valid = [idx for idx in token_ids if 0 <= int(idx) < self.vocab_size]
            if valid:
                mask[torch.tensor(valid, dtype=torch.long, device=device)] = True
        self.label_smoothing_allowed_mask = mask

    def _compute_label_smoothing_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        epsilon = float(getattr(self.config.loss, "label_smoothing", 0.1))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            return shift_logits.sum() * 0.0

        flat_logits = shift_logits[valid_mask]
        flat_targets = shift_labels[valid_mask]
        log_probs = F.log_softmax(flat_logits, dim=-1)

        if self.label_smoothing_allowed_mask.numel() != log_probs.size(-1):
            raise ValueError(
                "label_smoothing_allowed_mask size mismatch: "
                f"mask={self.label_smoothing_allowed_mask.numel()}, vocab={log_probs.size(-1)}"
            )

        allowed_mask = self.label_smoothing_allowed_mask.to(log_probs.device)
        missing_targets = (~allowed_mask[flat_targets]).any().item()
        if missing_targets and not self._warned_missing_allowed_target:
            logger.warning(
                "Found training labels outside allowed token set when using label smoothing; "
                "those labels keep their NLL term but smoothing mass is still restricted to allowed tokens."
            )
            self._warned_missing_allowed_target = True

        return masked_label_smoothing_nll(
            log_probs=log_probs,
            targets=flat_targets,
            allowed_mask=allowed_mask,
            epsilon=epsilon,
        )

    def _load_pretrained_encoder(self, encoder_path: str):
        """Load pretrained pocket encoder weights."""
        import os
        
        if not os.path.exists(encoder_path):
            logger.warning(f"Pretrained encoder not found at {encoder_path}. Using random initialization.")
            return
        
        if self.pocket_encoder is None:
            return

        try:
            logger.info(f"Loading pretrained pocket encoder from: {encoder_path}")
            checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            filtered_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace('module.', '')
                if not clean_key.startswith('contrastive_learning.'):
                    filtered_state_dict[clean_key] = value
            
            missing_keys, unexpected_keys = self.pocket_encoder.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in encoder checkpoint: {missing_keys[:10]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in encoder checkpoint: {unexpected_keys[:10]}...")
                
            logger.info("Pretrained pocket encoder loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained encoder: {e}")
            logger.warning("Using random initialization for pocket encoder.")
    
    def _prepare_llm_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encapsulates the process from pocket features to final LLM input embeddings.
        Supports injecting full feature sequences (Q-Former queries or Linear sequences).
        """
        device = batch['input_ids'].device
        batch_size = batch['input_ids'].size(0)

        # --- Step A: Adapter (Flat Processing) ---
        flat_feats = batch['flat_protein_feats']
        flat_mask = batch.get('flat_protein_mask', None)
        
        # Pass through adapter
        # Output: [N_Total, S, D_llm]
        # S is num_queries (qformer) or Max_L (linear)
        prot_tokens = self.adapter(flat_feats, flat_mask)
        
        # --- Step B: Reconstruct Batch ---
        gather_indices = batch['gather_indices'] # [Batch, Max_P]
        max_prots = gather_indices.shape[1]
        n_total, S, d_llm = prot_tokens.shape
        
        # Flatten to [N_Total, S * D_llm] for gathering
        prot_tokens_flat = prot_tokens.reshape(n_total, -1)
        
        # Debugging for potential index out of bounds
        if gather_indices.numel() > 0:
            max_idx = gather_indices.max().item()
            if max_idx >= n_total:
                logger.error(f"CRITICAL: Index out of bounds in _prepare_llm_inputs")
                logger.error(f"n_total (proteins): {n_total}")
                logger.error(f"gather_indices max: {max_idx}")
                logger.error(f"gather_indices shape: {gather_indices.shape}")
                # We can try to clamp it to avoid crash, but this indicates a logic error in Collate
                gather_indices = gather_indices.clamp(max=n_total-1)
        
        # Handle padding indices (-1 -> 0)
        gather_indices_clamped = gather_indices.clamp(min=0)
        valid_protein_mask = (gather_indices != -1)
        
        # Gather: [Batch, Max_P, S * D_llm]
        gathered_flat = F.embedding(gather_indices_clamped, prot_tokens_flat)
        
        # Reshape: [Batch, Max_P, S, D_llm]
        batch_prot_tokens = gathered_flat.reshape(batch_size, max_prots, S, d_llm)
        
        # --- Step C: Interaction (Set Transformer) ---
        if self.set_transformer is not None:
            role_ids = batch['role_ids']
            
            # Prepare residue-level mask if needed
            batch_prot_mask = None
            if flat_mask is not None and self.config.pp2d.protein_adapter_type == "linear":
                if flat_mask.dtype != torch.float:
                    flat_mask = flat_mask.float()
                
                batch_prot_mask = F.embedding(gather_indices_clamped, flat_mask)
            else:
                num_queries = batch_prot_tokens.shape[2]
                batch_prot_mask = valid_protein_mask.unsqueeze(-1).expand(-1, -1, num_queries).float()
            
            batch_prot_tokens = self.set_transformer(batch_prot_tokens, role_ids, protein_masks=batch_prot_mask)

        # --- Step D: Injection ---
        smiles_embedding_layer = self.llm.get_input_embeddings()
        
        # Replace -1 (placeholder) with a valid token ID (e.g. 0) to avoid embedding errors
        # We will overwrite these embeddings anyway
        safe_input_ids = batch['input_ids'].clone()
        safe_input_ids[safe_input_ids == -1] = 0 
        
        if isinstance(smiles_embedding_layer, nn.Module):
            base_embeds = smiles_embedding_layer(safe_input_ids)
        else:
            base_embeds = F.embedding(safe_input_ids, smiles_embedding_layer)
            
        original_labels = batch['labels']
        
        final_compacted_sequences = []
        final_compacted_labels = []
        
        strict_alignment_check = bool(getattr(self.config.training, 'strict_protein_alignment_check', True))

        for i in range(batch_size):
            current_text_embeds = base_embeds[i]
            current_labels = original_labels[i]
            
            # Find <protein> tokens
            protein_token_mask = batch['protein_token_mask'][i]
            injection_indices = torch.nonzero(protein_token_mask, as_tuple=False).squeeze(-1)
            
            num_injections = len(injection_indices)
            
            injections = []
            if num_injections > 0:
                # Get features for this sample
                sample_feats = batch_prot_tokens[i] # [Max_P, S, D]
                sample_indices = gather_indices[i]  # [Max_P]
                valid_protein_count = int((sample_indices != -1).sum().item())

                if strict_alignment_check and num_injections != valid_protein_count:
                    raise ValueError(
                        "Protein placeholder-feature alignment mismatch: "
                        f"sample={i}, placeholders={num_injections}, proteins={valid_protein_count}, "
                        f"active_ids={batch.get('active_ids', [None] * batch_size)[i]}, "
                        f"inactive_ids={batch.get('inactive_ids', [None] * batch_size)[i]}"
                    )
                
                for k, idx in enumerate(injection_indices):
                    # 1. Get the features [S, D]
                    # Note: This assumes the order in gather_indices matches <protein> tokens
                    if strict_alignment_check and k >= valid_protein_count:
                        raise ValueError(
                            "Protein feature index out of range during injection: "
                            f"sample={i}, injection_index={k}, valid_protein_count={valid_protein_count}"
                        )

                    feat_vec = sample_feats[k] 
                    
                    # 2. Retrieve Valid Length (Crucial for 'linear' mode)
                    # Look up the original mask using gather_indices
                    if flat_mask is not None:
                        original_flat_idx = sample_indices[k].item()
                        # If it was a padding slot (which shouldn't happen for valid injections), len=0
                        if original_flat_idx >= flat_mask.size(0): 
                            valid_len = int(S)
                        else:
                            # Count non-zero elements in the mask for this protein
                            valid_len = int(flat_mask[original_flat_idx].sum().item())
                        
                        # If adapter changed length (e.g. qformer), S is fixed, valid_len might not apply directly
                        # For qformer, S is num_queries, we keep all.
                        # For linear, S is Max_L, we cut to valid_len.
                        if self.config.pp2d.protein_adapter_type == "linear":
                            valid_len = max(0, min(valid_len, int(S)))
                            feat_vec = feat_vec[:valid_len]
                        # For qformer, feat_vec is [Num_Q, D], keep all
                    
                    # 3. Create Labels (-100 for protein tokens)
                    feat_len = feat_vec.shape[0]
                    feat_label = torch.full((feat_len,), -100, dtype=torch.long, device=device)
                    
                    injections.append({
                        'idx': idx.item(),
                        'feat': feat_vec,
                        'label': feat_label
                    })
            
            # Reconstruct sequence
            if not injections:
                final_compacted_sequences.append(current_text_embeds)
                final_compacted_labels.append(current_labels)
                continue
            
            seq_parts, label_parts = [], []
            last_idx = 0
            for injection in injections:
                injection_point = injection['idx']
                # Append text BEFORE <protein>
                if last_idx < injection_point:
                    seq_parts.append(current_text_embeds[last_idx:injection_point])
                    label_parts.append(current_labels[last_idx:injection_point])
                
                # Append Protein Features (S tokens) REPLACING <protein>
                seq_parts.append(injection['feat'])
                label_parts.append(injection['label'])
                
                # Skip the original <protein> token
                last_idx = injection_point + 1
            
            # Append remaining text
            if last_idx < len(current_text_embeds):
                seq_parts.append(current_text_embeds[last_idx:])
                label_parts.append(current_labels[last_idx:])
            
            # Concatenate
            final_compacted_sequences.append(torch.cat([p for p in seq_parts if p.numel() > 0], dim=0))
            final_compacted_labels.append(torch.cat([p for p in label_parts if p.numel() > 0], dim=0))

        # Pad batches (New lengths will vary greatly)
        final_embeds = efficient_pad_sequences(final_compacted_sequences, padding_value=0.0, padding_side='left')
        final_labels = efficient_pad_sequences(final_compacted_labels, padding_value=-100, padding_side='left')
        
        # New Attention Mask
        new_max_len = final_embeds.shape[1]
        real_lengths = torch.tensor([s.shape[0] for s in final_compacted_sequences], device=device)
        new_attention_mask = torch.arange(new_max_len, device=device)[None, :] >= (new_max_len - real_lengths)[:, None]
        
        batch['attention_mask'] = new_attention_mask.long()
        batch['labels'] = final_labels

        return {
            "final_embeds": final_embeds
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        """
        The complete forward pass.
        """
        prepared_inputs = self._prepare_llm_inputs(batch)
        final_embeds = prepared_inputs['final_embeds']
        
        if final_embeds.dtype != self.llm.dtype:
            final_embeds = final_embeds.to(self.llm.dtype)
        
        smiles_loss_type = str(getattr(self.config.loss, "smiles_loss_type", "cross_entropy")).lower()
        compute_loss = bool(batch.get("compute_loss", True))

        llm_kwargs = {
            "inputs_embeds": final_embeds,
            "attention_mask": batch['attention_mask'],
            "output_hidden_states": False,
            "return_dict": True,
        }
        if compute_loss and smiles_loss_type == "cross_entropy":
            llm_kwargs["labels"] = batch['labels']

        llm_output = self.llm(**llm_kwargs)

        if not compute_loss:
            loss = None
        elif smiles_loss_type == "label_smoothing":
            loss = self._compute_label_smoothing_loss(llm_output.logits, batch['labels'])
        else:
            loss = llm_output.loss
        
        return {
            "logits": llm_output.logits,
            "loss": loss,
            "labels": batch['labels']
        }
        
    def get_parameter_groups(self):
        """
        Returns a list of parameter groups for the optimizer.
        """
        module_configs = []
        
        if self.pocket_encoder is not None:
            module_configs.append((self.pocket_encoder, 'encoder_learning_rate', 1e-6))
        
        # New modules
        module_configs.extend([
            (self.adapter, 'base_learning_rate', None),
            (self.set_transformer, 'base_learning_rate', None),
            (self.llm, 'llm_learning_rate', None),
        ])
        
        def _coerce_lr(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    logger.warning(f"Invalid learning rate value: {value}")
                    return None
            return None

        def get_lr_with_fallback(*lr_keys, default=1e-5):
            for key in lr_keys:
                lr = getattr(self.config.training, key, None)
                lr = _coerce_lr(lr)
                if lr is not None:
                    return lr
            return _coerce_lr(default)
        
        base_lr = get_lr_with_fallback('base_learning_rate', 'learning_rate', default=1e-5)
        
        llm_lr = get_lr_with_fallback('llm_learning_rate', 'base_learning_rate', 'learning_rate', default=base_lr)
        
        param_groups = []
        
        for module, lr_key, fallback_lr in module_configs:
            if module is None: continue
                
            try:
                trainable_params = list(filter(lambda p: p.requires_grad, module.parameters()))
            except Exception as e:
                logger.warning(f"Error accessing parameters: {e}")
                continue
            
            if not trainable_params:
                continue
                
            if lr_key == 'llm_learning_rate': lr = llm_lr
            elif lr_key == 'base_learning_rate': lr = base_lr
            else: lr = get_lr_with_fallback(lr_key, default=fallback_lr if fallback_lr is not None else base_lr)
            
            if lr is None or lr < 0:
                continue
            elif lr == 0:
                # If learning rate is 0, freeze these parameters
                for param in trainable_params:
                    param.requires_grad = False
                logger.info(f"Freezing {module.__class__.__name__} parameters (lr=0)")
                continue
            
            group_name = f"{module.__class__.__name__}:{lr_key}"
            param_groups.append({"params": trainable_params, "lr": lr, "name": group_name})
        
        if not param_groups:
            dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            param_groups.append({"params": [dummy_param], "lr": 1e-8, "name": "dummy_param"})
        
        return param_groups

    @torch.inference_mode()
    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        tokenizer: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate SMILES strings.
        """
        self.eval()
        
        # Determine EOS/PAD
        pad_token_id = getattr(self.llm.config, 'pad_token_id', None)
        eos_token_id = getattr(self.llm.config, 'eos_token_id', None)
        
        if tokenizer:
            if pad_token_id is None: pad_token_id = tokenizer.pad_token_id
            if eos_token_id is None: eos_token_id = tokenizer.eos_token_id

        prepared_inputs = self._prepare_llm_inputs(batch)
        final_embeds = prepared_inputs['final_embeds']

        amp_enabled = self.use_mixed_precision and final_embeds.device.type == "cuda"
        amp_dtype = torch.float16
        if str(self.mixed_precision_dtype).lower() in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with autocast_ctx:
            gen_ids = self.llm.generate(
                inputs_embeds=final_embeds,
                attention_mask=batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                # Sampling can occasionally surface invalid logits on long runs;
                # ask HF generate() to scrub/renormalize before multinomial draw.
                remove_invalid_values=True,
                renormalize_logits=True,
                **kwargs
            )
        
        if tokenizer is not None:
            decoded_sequences = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        else:
            decoded_sequences = []
            
        return {
            "generated_ids": gen_ids,
            "decoded_sequences": decoded_sequences,
        }

    @staticmethod
    def from_pretrained(pretrained_path: str, config: Config, map_location='cpu'):
        model = PP2D(config)
        checkpoint = torch.load(pretrained_path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('reference_model.')}
        model.load_state_dict(filtered_state_dict, strict=False)
        return model
