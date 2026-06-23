import os
import torch
import torch.distributed as dist
import socket
import logging
from contextlib import contextmanager
from typing import Optional, Generator, Tuple

# Configure a module-level logger
logger = logging.getLogger(__name__)

def get_local_rank(default_rank: int = 0) -> int:
    """Returns the process-local CUDA rank when available."""
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        return int(local_rank_env)

    num_devices = max(torch.cuda.device_count(), 1)
    return default_rank % num_devices

def _get_barrier_kwargs() -> dict:
    if torch.cuda.is_available():
        return {"device_ids": [torch.cuda.current_device()]}
    return {}

def ddp_barrier() -> None:
    """Runs a distributed barrier with explicit CUDA device binding when available."""
    if not dist.is_available() or not dist.is_initialized():
        return

    barrier_kwargs = _get_barrier_kwargs()
    if barrier_kwargs:
        dist.barrier(**barrier_kwargs)
    else:
        dist.barrier()

def find_free_port(start_port: int = 12355) -> int:
    """Finds a free port on localhost."""
    for port in range(start_port, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No free ports found in the specified range.")

def setup_ddp(rank: int, world_size: int, master_port: Optional[str] = None, master_addr: str = "localhost", backend: str = "nccl") -> None:
    """
    Initializes the Distributed Data Parallel (DDP) environment.
    
    Args:
        rank: Global rank of the process.
        world_size: Total number of processes.
        master_port: Port to use for communication. If None, expects MASTER_PORT env var.
        master_addr: Address of the master node.
        backend: DDP backend (nccl for GPU, gloo for CPU).

    Notes:
        CUDA device selection prefers LOCAL_RANK when available.
    """
    try:
        # Set environment variables if explicitly provided
        os.environ["MASTER_ADDR"] = master_addr
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
        elif "MASTER_PORT" not in os.environ:
             # Fallback if neither arg nor env var is set
            os.environ["MASTER_PORT"] = str(find_free_port())

        # Set device
        if torch.cuda.is_available():
            local_rank = get_local_rank(rank)
            torch.cuda.set_device(local_rank)

        # Initialize process group
        init_kwargs = dict(backend=backend, rank=rank, world_size=world_size)
        dist.init_process_group(**init_kwargs)

        # Barrier to ensure all processes have started
        ddp_barrier()
        
        if rank == 0:
            if torch.cuda.is_available():
                logger.info(f"DDP device mapping: rank0 uses cuda:{torch.cuda.current_device()}")
            logger.info(f"DDP Initialized: World Size={world_size}, Master={master_addr}:{os.environ['MASTER_PORT']}")
            
    except Exception as e:
        logger.error(f"DDP Setup failed on rank {rank}: {e}")
        cleanup_ddp()
        raise e

def cleanup_ddp() -> None:
    """Cleans up the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduces a tensor across all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    
    with torch.no_grad():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= world_size
    return tensor

@contextmanager
def ddp_context(rank: int, world_size: int, master_port: Optional[str] = None) -> Generator[None, None, None]:
    try:
        setup_ddp(rank, world_size, master_port=master_port)
        yield
    finally:
        cleanup_ddp()
        
