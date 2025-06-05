"""
Logging utilities for AlphaZero Reversi.
"""
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import Config

class Logger:
    """Logger for training and evaluation metrics."""
    
    def __init__(self, config: Config, log_dir: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            config: Configuration object
            log_dir: Directory to save logs (default: config.logging.log_dir)
        """
        self.config = config
        self.log_dir = log_dir or config.logging.log_dir
        self.run_name = f"{config.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(self.log_dir, self.run_name)
        
        # Create log directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set up console logging
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console.setFormatter(formatter)
        
        # Set up file logging
        log_file = os.path.join(self.run_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.console)
        self.logger.addHandler(file_handler)
        
        # Initialize TensorBoard
        self.writer = None
        if config.logging.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))
        
        # Save config
        self.save_config()
    
    def save_config(self):
        """Save the configuration to a JSON file."""
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            import json
            from dataclasses import asdict
            json.dump(asdict(self.config), f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ''):
        """
        Log metrics to console and TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Log to console
        log_str = f"Step {step}:"
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                log_str += f" {name}={value:.4f}"
            else:
                log_str += f" {name}={value}"
        self.logger.info(log_str)
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}{name}", value, step)
                elif isinstance(value, torch.Tensor):
                    self.writer.add_scalar(f"{prefix}{name}", value.item(), step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """Log a histogram of values to TensorBoard."""
        if self.writer is not None:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self.writer.add_histogram(tag, values, step)
    
    def log_embedding(self, tag: str, embedding: torch.Tensor, metadata: Optional[List[Any]] = None, 
                     label_img: Optional[torch.Tensor] = None, step: int = 0):
        """
        Log embeddings to TensorBoard.
        
        Args:
            tag: Tag for the embedding
            embedding: Tensor of shape (N, D) where N is the number of embeddings and D is the dimension
            metadata: Optional list of metadata for each embedding
            label_img: Optional tensor of shape (N, C, H, W) containing images for each embedding
            step: Current step
        """
        if self.writer is not None:
            self.writer.add_embedding(
                embedding,
                metadata=metadata,
                label_img=label_img,
                tag=tag,
                global_step=step
            )
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """Log text to TensorBoard."""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """Log model graph to TensorBoard."""
        if self.writer is not None:
            self.writer.add_graph(model, input_to_model)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log learning rate to TensorBoard."""
        if self.writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                self.writer.add_scalar(f'lr/group_{i}', param_group['lr'], step)
    
    def close(self):
        """Close the logger and flush all pending logs."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        
        # Remove handlers to prevent duplicate logging
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
    
    def __del__(self):
        """Ensure resources are properly released."""
        self.close()


def setup_logger(config: Config) -> Logger:
    """
    Set up and return a logger instance.
    
    Args:
        config: Configuration object
        
    Returns:
        Logger instance
    """
    return Logger(config)
