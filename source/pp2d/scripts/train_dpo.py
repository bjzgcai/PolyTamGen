#!/usr/bin/env python3
"""Train PP2D with Direct Preference Optimization."""

import os
import sys
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config, get_parser_from_config, update_config_from_args
from src.dpo import DPOTrainer
from src.utils.distributed_utils import cleanup_ddp, find_free_port, setup_ddp
from src.utils.training_recorder import TrainingRecorder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(args):
    raw_config_path = Path(args.config).expanduser()
    if raw_config_path.is_absolute():
        config_path = raw_config_path
    else:
        config_path = (Path.cwd() / raw_config_path).resolve()
        if not config_path.exists():
            config_path = (PROJECT_ROOT / raw_config_path).resolve()
    os.chdir(PROJECT_ROOT)
    config = Config.from_yaml(config_path)
    config = update_config_from_args(config, args)
    config.training_mode = "dpo"
    return config


def _run_dry_run(trainer):
    """Run one DPO train and validation micro-batch without saving checkpoints."""
    if trainer.train_loader is None or trainer.valid_loader is None:
        raise RuntimeError("Dry run requires initialized train and valid loaders.")

    train_batch = next(iter(trainer.train_loader), None)
    if train_batch is None or (isinstance(train_batch, dict) and not train_batch):
        raise RuntimeError("Dry run could not fetch a non-empty DPO training batch.")

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    train_batch = trainer._move_to_device(train_batch)
    with trainer._autocast_context():
        train_outputs = trainer.train_step(train_batch)
        train_loss = train_outputs["loss"]
    if not torch.isfinite(train_loss).all():
        raise RuntimeError("Dry run DPO training loss is not finite.")
    if train_loss.requires_grad:
        if trainer.scaler.is_enabled():
            trainer.scaler.scale(train_loss).backward()
        else:
            train_loss.backward()
    trainer.optimizer.zero_grad(set_to_none=True)

    valid_batch = next(iter(trainer.valid_loader), None)
    if valid_batch is None or (isinstance(valid_batch, dict) and not valid_batch):
        raise RuntimeError("Dry run could not fetch a non-empty DPO validation batch.")

    trainer.model.eval()
    valid_batch = trainer._move_to_device(valid_batch)
    with torch.no_grad():
        with trainer._autocast_context():
            valid_outputs = trainer.validation_step(valid_batch)
            valid_loss = valid_outputs["loss"]
    if not torch.isfinite(valid_loss).all():
        raise RuntimeError("Dry run DPO validation loss is not finite.")

    trainer.recorder.info(
        "DPO dry run succeeded | "
        f"train_loss={float(train_loss.detach().cpu()):.4f} | "
        f"valid_loss={float(valid_loss.detach().cpu()):.4f}"
    )


def main_worker(rank, world_size, config, dry_run=False):
    is_distributed = config.distributed.distributed
    try:
        if is_distributed:
            setup_ddp(rank, world_size, master_port=config.distributed.master_port)

        recorder = TrainingRecorder(
            exp_name="pp2d_dpo",
            log_dir=config.training.log_dir,
            tb_dir=config.tensorboard_dir,
            config=config,
        )

        if rank == 0:
            recorder.info(f"--- Starting DPO Training (Rank {rank}/{world_size}) ---")
            recorder.info(f"Mode: {'Distributed' if is_distributed else 'Single GPU'}")
            recorder.info(f"Device: {torch.cuda.get_device_name(rank) if torch.cuda.is_available() else 'CPU'}")

        trainer = DPOTrainer(config=config, recorder=recorder)
        trainer.setup()

        if dry_run:
            _run_dry_run(trainer)
            if rank == 0:
                recorder.info("DPO dry run completed successfully.")
            return

        trainer.train()

        if rank == 0:
            recorder.info("DPO training completed successfully.")

    except KeyboardInterrupt:
        if rank == 0:
            print("\n>>> Interrupted by user. Exiting worker...")
        sys.exit(130)
    except Exception as exc:
        if rank == 0:
            traceback.print_exc()
            print(f"!!! DPO training failed with error: {exc}")
        raise
    finally:
        if is_distributed:
            try:
                cleanup_ddp()
            except Exception:
                pass
        if "recorder" in locals():
            recorder.close()


if __name__ == "__main__":
    parser = get_parser_from_config(Config)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Initialize DPO and run one train/validation micro-batch without saving checkpoints.",
    )
    args = parser.parse_args()

    config = _load_config(args)

    if config.distributed.distributed:
        if config.distributed.world_size is None:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                print("Warning: Distributed requested but no GPUs found. Falling back to single process.")
                config.distributed.distributed = False
                config.distributed.world_size = 1
            else:
                config.distributed.world_size = gpu_count

        world_size = int(config.distributed.world_size or 1)
        if world_size > 1:
            free_port = find_free_port(int(config.distributed.master_port))
            config.distributed.master_port = str(free_port)
            print(f"Launching DPO DDP on {world_size} GPUs (Port: {free_port})...")
            mp.spawn(main_worker, args=(world_size, config, args.dry_run), nprocs=world_size, join=True)
        else:
            print("Distributed requested but world_size=1. Running single process.")
            config.distributed.distributed = False
            config.distributed.world_size = 1
            main_worker(0, 1, config, dry_run=args.dry_run)
    else:
        print("Launching single-process DPO training...")
        config.distributed.world_size = 1
        main_worker(0, 1, config, dry_run=args.dry_run)
