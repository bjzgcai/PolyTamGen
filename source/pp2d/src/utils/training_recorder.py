import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any, Union
import torch
from torch.utils.tensorboard import SummaryWriter

# Import distributed helpers to determine rank without circular imports
from .distributed_utils import is_main_process, get_rank

class ColoredFormatter(logging.Formatter):
    """
    Custom Logging Formatter to add ANSI colors to console output based on log levels.
    This makes reading logs in the terminal significantly easier.
    """
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Purple
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Apply colors only to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class TrainingRecorder:
    """
    A unified recorder class that handles both Python logging and TensorBoard logging.
    
    Key Features:
    1. DDP Safe: Rank 0 logs to console and TensorBoard. All ranks log to files.
    2. Hyperparameter Logging: Can flatten and log complex config objects.
    3. Metric Logging: Unified interface for tracking loss, accuracy, etc.
    """
    
    def __init__(self, 
                 exp_name: str, 
                 log_dir: str = "logs", 
                 tb_dir: str = "runs", 
                 config: Any = None):
        """
        Initialize the recorder.
        
        Args:
            exp_name (str): Unique name for the experiment.
            log_dir (str): Directory to save text log files.
            tb_dir (str): Directory to save TensorBoard runs.
            config (Any, optional): Configuration object/dict to log immediately.
        """
        self.rank = get_rank()
        self.is_main = is_main_process()
        self.exp_name = exp_name
        
        # ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        if self.is_main:
            os.makedirs(tb_dir, exist_ok=True)
            
        # 1. Setup Python Logger
        self.logger = self._setup_logger(log_dir)
        
        # 2. Setup TensorBoard (Only on main process)
        self.writer: Optional[SummaryWriter] = None
        if self.is_main:
            # Use timestamp to avoid overwriting previous runs of the same experiment name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{exp_name}_{timestamp}"
            tb_log_path = os.path.join(tb_dir, run_name)
            
            self.writer = SummaryWriter(log_dir=tb_log_path)
            self.info(f"TensorBoard initialized at: {tb_log_path}")
            
            # Log hyperparameters if provided
            if config:
                self.log_hparams(config)

    def _setup_logger(self, log_dir: str) -> logging.Logger:
        """
        Configures the logging instance.
        
        Strategy:
        - Console: Only Rank 0 prints INFO+. Other ranks only print ERROR+.
        - File: All ranks write to their own log file (debug level).
        """
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Prevent double logging if root logger is configured
        
        # Clear existing handlers to avoid duplication during re-initialization
        if logger.hasHandlers():
            logger.handlers.clear()

        # Define formatters
        # File includes timestamp, rank, file location
        file_fmt = logging.Formatter(
            fmt=f'[%(asctime)s][Rank {self.rank}][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Console is cleaner
        console_fmt = ColoredFormatter(
            fmt=f'[%(asctime)s][Rank {self.rank}][%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # --- Handler 1: Rotating File Handler (All Ranks) ---
        # Each rank gets its own file: exp_name_rank0.log, exp_name_rank1.log...
        log_filename = f"{self.exp_name}_rank{self.rank}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_filename),
            maxBytes=50*1024*1024, # 50 MB
            backupCount=5
        )
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # --- Handler 2: Console Handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_fmt)
        
        if self.is_main:
            # Main process logs everything interesting
            console_handler.setLevel(logging.INFO)
        else:
            # Workers stay quiet unless something breaks
            console_handler.setLevel(logging.ERROR)
            
        logger.addHandler(console_handler)
        
        return logger

    # --- Logging Interface ---

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)
        
    def critical(self, msg: str) -> None:
        self.logger.critical(msg)

    # --- TensorBoard Interface ---
    @staticmethod
    def _to_val(v):
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            return v.item()
        elif isinstance(v, (float, int)):
            return v
        return v

    def log_metric(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar metric to TensorBoard. Only performs action on Rank 0.
        """
        if self.writer and self.is_main:
            self.writer.add_scalar(tag, self._to_val(value), step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log a dictionary of metrics.
        """
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.log_metric(tag, v, step)

    def log_hparams(self, config: Any) -> None:
        """
        Log hyperparameters to TensorBoard. Handles nested Dicts or Objects.
        """
        if not self.writer or not self.is_main:
            return

        # 1. Convert config object to flat dictionary
        params_dict = self._flatten_config(config)
        
        # 2. Log as Text (easier to read sometimes)
        formatted_text = "  \n".join([f"**{k}**: {v}" for k, v in params_dict.items()])
        self.writer.add_text("Hyperparameters", formatted_text, 0)
        
        # 3. Log as HParams (for parallel coordinates view)
        # Note: We pass empty metrics dict because we don't know the final metrics yet
        self.writer.add_hparams(hparam_dict=params_dict, metric_dict={})
        
        self.info("Hyperparameters logged to TensorBoard.")

    def _flatten_config(self, config: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Recursively flattens a configuration object/dictionary.
        """
        items = []
        # Determine if it's a dict or object
        config_dict = config if isinstance(config, dict) else vars(config)
        
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            elif hasattr(v, '__dict__'):
                items.extend(self._flatten_config(vars(v), new_key, sep=sep).items())
            else:
                # Convert list/tuples to string for TensorBoard compatibility
                if isinstance(v, (list, tuple)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)

    def close(self) -> None:
        """
        Close all writers and handlers.
        """
        if self.writer:
            self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            