"""
Main script to run the AlphaZero Reversi training pipeline.
"""
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.absolute()))

from src.config import Config
from src.trainer.pipeline import AlphaZeroPipeline

def setup_directories(config):
    """Create necessary directories for the run."""
    # Create a timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f"runs/run_{timestamp}")
    
    # Update paths in config
    config.training.checkpoint_dir = str(run_dir / "checkpoints")
    config.self_play.save_dir = str(run_dir / "self_play")
    config.tournament.output_dir = str(run_dir / "tournament")
    config.logging.log_dir = str(run_dir / "logs")
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.self_play.save_dir, exist_ok=True)
    os.makedirs(config.tournament.output_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    
    return config, run_dir

def main():
    """Run the training pipeline with the specified configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AlphaZero Reversi training')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        config = Config.load(args.config)
    else:
        print(f"Config file {args.config} not found, using default configuration")
        from src.config import get_default_config
        config = get_default_config()
    
    # Setup directories and update config paths
    config, run_dir = setup_directories(config)
    
    # Save the final config
    config.save(run_dir / "config.json")
    print(f"Configuration saved to {run_dir}/config.json")
    
    # Initialize and run the pipeline
    pipeline = AlphaZeroPipeline(config)
    
    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        pipeline.load_checkpoint(args.resume)
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Project: {config.project_name}")
    print(f"Device: {config.training.device}")
    print(f"Model: {config.model.num_res_blocks} res blocks, {config.model.num_filters} filters")
    print(f"Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
    print(f"Checkpoints: {config.training.checkpoint_dir}")
    print(f"Logs: {config.logging.log_dir}")
    print("===================\n")
    
    # Run the training
    try:
        pipeline.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        # Save final checkpoint
        pipeline._save_checkpoint(pipeline.current_iteration, {'status': 'interrupted'})
    
    print(f"\nTraining completed. Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
