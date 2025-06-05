"""
Main script to run the AlphaZero Reversi training pipeline with default configuration.
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import Config, get_default_config
from src.trainer.pipeline import AlphaZeroPipeline
from src.logger import Logger

def load_config(config_path=None):
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = Config.load(config_path)
    else:
        print("Using default configuration")
        config = get_default_config()
    
    # Set device
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.training.device}")
    
    # Create necessary directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.self_play.save_dir, exist_ok=True)
    os.makedirs(config.tournament.output_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    
    return config

def main():
    """Run the complete AlphaZero training pipeline."""
    parser = argparse.ArgumentParser(description='Run AlphaZero Reversi training pipeline')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Base directory for all outputs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update output directories to be under the run directory
    config.training.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    config.self_play.save_dir = os.path.join(run_dir, 'self_play')
    config.tournament.output_dir = os.path.join(run_dir, 'tournament')
    config.logging.log_dir = os.path.join(run_dir, 'logs')
    
    # Save the final config
    os.makedirs(run_dir, exist_ok=True)
    config.save(os.path.join(run_dir, 'config.json'))
    
    print(f"Starting training run in {run_dir}")
    print(f"Configuration: {json.dumps(config.to_dict(), indent=2, default=str)}")
    
    # Initialize and run the pipeline
    pipeline = AlphaZeroPipeline(config)
    
    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        pipeline.load_checkpoint(args.resume)
    
    # Run the training
    try:
        pipeline.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        # Save final checkpoint
        pipeline._save_checkpoint(pipeline.current_iteration, {'status': 'interrupted'})
    
    print(f"Training completed. Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
