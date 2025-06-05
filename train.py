"""
AlphaZero training pipeline for Reversi.
"""
import os
import argparse
import json
from datetime import datetime

from src.trainer.pipeline import train_from_config

def create_default_config():
    """Create and save a default configuration file."""
    from src.config import get_default_config
    
    config = get_default_config()
    config_file = "config.json"
    
    # Set some reasonable defaults
    config.training.num_epochs = 1000
    config.training.batch_size = 64
    config.training.learning_rate = 0.001
    config.training.checkpoint_dir = "checkpoints"
    
    config.self_play.num_games = 100
    config.self_play.num_parallel_games = 4
    config.self_play.save_dir = "self_play_data"
    
    config.mcts.num_simulations = 800
    config.mcts.c_puct = 1.0
    config.mcts.temperature = 1.0
    
    config.tournament.rounds = 10
    config.tournament.output_dir = "tournament_results"
    config.tournament.elo_file = "elo_ratings.json"
    
    config.logging.log_dir = "logs"
    config.logging.use_tensorboard = True
    
    # Save the config
    config.save(config_file)
    print(f"Created default configuration file: {config_file}")
    return config_file

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train AlphaZero for Reversi')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create default config
    create_parser = subparsers.add_parser('create-config', help='Create a default configuration file')
    create_parser.add_argument('--output', type=str, default='config.json',
                             help='Output config file path')
    
    # Train model
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', type=str, default='config.json',
                             help='Path to config file')
    
    # Continue training
    continue_parser = subparsers.add_parser('continue', help='Continue training from a checkpoint')
    continue_parser.add_argument('checkpoint', type=str,
                                help='Path to checkpoint file')
    continue_parser.add_argument('--config', type=str, default=None,
                                help='Path to config file (if different from checkpoint)')
    
    args = parser.parse_args()
    
    if args.command == 'create-config':
        create_default_config()
    elif args.command == 'train':
        train_from_config(args.config)
    elif args.command == 'continue':
        # Load config from checkpoint if not specified
        if args.config is None:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'config' in checkpoint:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(checkpoint['config'], f, indent=2)
                    args.config = f.name
            else:
                raise ValueError("No config found in checkpoint. Please specify --config")
        
        # Initialize and load checkpoint
        from src.trainer.pipeline import AlphaZeroPipeline
        pipeline = AlphaZeroPipeline()
        pipeline.load_checkpoint(args.checkpoint)
        
        # Continue training
        pipeline.train()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
