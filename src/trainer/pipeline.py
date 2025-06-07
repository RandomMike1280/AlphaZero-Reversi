"""
Training pipeline for AlphaZero Reversi.
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json

from ..model import AlphaZeroNetwork
from ..mcts import MCTS
from ..self_play.self_play import SelfPlay
from ..arena import Arena, ELOPlayer, ELORatingSystem
from ..game import ReversiGame
from ..logger import Logger, setup_logger
from ..config import Config, get_default_config

class AlphaZeroPipeline:
    """End-to-end training pipeline for AlphaZero."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config if config is not None else get_default_config()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.device = self._get_device()
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = self._init_criterion()
        self.logger = setup_logger(self.config)
        
        # Log device info
        self.logger.logger.info(f"Using device: {self.device}")
        if str(self.device) == 'cuda':
            self.logger.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Training state
        self.current_iteration = 0
        self.best_elo = -float('inf')
        
        # Create directories
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.self_play.save_dir, exist_ok=True)
        os.makedirs(self.config.tournament.output_dir, exist_ok=True)
        
    def _get_device(self) -> torch.device:
        """Get the available device (CUDA if available, else CPU)."""
        try:
            # Try to use CUDA if it's requested and available
            if self.config.training.device == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            # Fall back to CPU
            return torch.device('cpu')
        except Exception as e:
            print(f"Warning: Could not use CUDA: {e}. Falling back to CPU.")
            return torch.device('cpu')
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _init_model(self) -> AlphaZeroNetwork:
        """Initialize the neural network model."""
        model = AlphaZeroNetwork(
            board_size=self.config.model.board_size,
            num_res_blocks=self.config.model.num_res_blocks,
            num_filters=self.config.model.num_filters
        )
        return model.to(self.device)
    
    def _init_optimizer(self) -> optim.Optimizer:
        """Initialize the optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _init_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Initialize the learning rate scheduler."""
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.training.lr_milestones,
            gamma=self.config.training.lr_gamma
        )
    
    def _init_criterion(self) -> Dict[str, callable]:
        """Initialize loss functions."""
        return {
            'policy': nn.CrossEntropyLoss(),
            'value': nn.MSELoss()
        }
    
    def train(self):
        """Run the full training pipeline."""
        self.logger.logger.info("Starting AlphaZero training pipeline")
        
        try:
            for iteration in range(self.current_iteration, self.config.training.num_epochs):
                self.current_iteration = iteration
                self.logger.logger.info(f"\n=== Iteration {iteration + 1}/{self.config.training.num_epochs} ===")
                
                # Generate self-play data
                self.logger.logger.info("Generating self-play data...")
                train_data = self._generate_self_play_data()
                
                # Train the model
                self.logger.logger.info("Training model...")
                train_metrics = self._train_epoch(train_data)
                
                # Update learning rate
                self.scheduler.step()
                
                # Evaluate the model
                self.logger.logger.info("Evaluating model...")
                eval_metrics = self._evaluate_model()
                
                # Save checkpoint
                if (iteration + 1) % self.config.training.save_interval == 0:
                    self._save_checkpoint(iteration, eval_metrics)
                
                # Log metrics
                metrics = {**train_metrics, **eval_metrics}
                self.logger.log_metrics(metrics, iteration)
                
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted. Saving current state...")
        
        self.logger.logger.info("Training completed!")
        self.logger.close()
    
    def _generate_self_play_data(self) -> Dict[str, np.ndarray]:
        """Generate self-play data using the current model."""
        self_play_args = {
            'num_simulations': self.config.mcts.num_simulations,
            'c_puct': self.config.mcts.c_puct,
            'temperature': self.config.mcts.temperature,
            'dirichlet_alpha': self.config.mcts.dirichlet_alpha,
            'dirichlet_epsilon': self.config.mcts.dirichlet_epsilon,
            'num_parallel_games': self.config.self_play.num_parallel_games,
            'save_dir': self.config.self_play.save_dir
        }
        self_play = SelfPlay(
            model=self.model,
            args=self_play_args
        )
        
        # Generate games
        games = self_play.generate_games(self.config.self_play.num_games)
        
        # Convert to training data
        states = []
        policy_targets = []
        value_targets = []
        
        board_size = self.config.model.board_size
        expected_policy_length = board_size * board_size + 1  # +1 for pass move
        
        for game_idx, game in enumerate(games):
            game_states = game.get('states', [])
            game_action_probs = game.get('action_probs', [])
            game_values = game.get('values', [])
            
            # Debug info
            print(f"Game {game_idx}: {len(game_states)} states, {len(game_action_probs)} action_probs, {len(game_values)} values")
            
            # Check for any mismatched lengths within the same game
            min_length = min(len(game_states), len(game_action_probs), len(game_values))
            if min_length == 0:
                print(f"Warning: Game {game_idx} has no valid data")
                continue
                
            # Only use the data that has matching lengths
            for i in range(min_length):
                # Convert action probs to fixed size if needed
                if isinstance(game_action_probs[i], dict):
                    # Convert dict to fixed-size vector
                    fixed_probs = np.zeros(expected_policy_length, dtype=np.float32)
                    for (row, col), prob in game_action_probs[i].items():
                        if (row, col) == (-1, -1):  # Pass move
                            fixed_probs[-1] = prob
                        else:
                            idx = row * board_size + col
                            fixed_probs[idx] = prob
                    policy_targets.append(fixed_probs)
                else:
                    # Already a vector, ensure it's the right size
                    if len(game_action_probs[i]) != expected_policy_length:
                        print(f"Warning: Unexpected policy target length {len(game_action_probs[i])}, expected {expected_policy_length}")
                        fixed_probs = np.zeros(expected_policy_length, dtype=np.float32)
                        min_len = min(len(game_action_probs[i]), expected_policy_length)
                        fixed_probs[:min_len] = game_action_probs[i][:min_len]
                        policy_targets.append(fixed_probs)
                    else:
                        policy_targets.append(game_action_probs[i])
                
                states.append(game_states[i])
                value_targets.append(game_values[i])
        
        # Convert to numpy arrays
        try:
            # Check for empty data
            if not states or not policy_targets or not value_targets:
                raise ValueError("No valid training data was generated. Check the self-play output.")
            
            states = np.array(states, dtype=np.float32)
            policy_targets = np.array(policy_targets, dtype=np.float32)
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)
            
            print(f"Generated training data - States: {states.shape}, Policy targets: {policy_targets.shape}, Value targets: {value_targets.shape}")
            
            # Verify shapes
            if len(states) == 0 or len(policy_targets) == 0 or len(value_targets) == 0:
                raise ValueError(f"Empty data in one or more arrays. Shapes - states: {states.shape}, policy: {policy_targets.shape}, values: {value_targets.shape}")
                
            if len(states) != len(policy_targets) or len(states) != len(value_targets):
                raise ValueError(f"Mismatched array lengths - states: {len(states)}, policy: {len(policy_targets)}, values: {len(value_targets)}")
                
            # Verify policy targets sum to ~1 (within tolerance)
            policy_sums = policy_targets.sum(axis=1)
            valid_sums = np.isclose(policy_sums, 1.0, atol=1e-3)
            if not np.all(valid_sums):
                invalid_count = np.sum(~valid_sums)
                print(f"Warning: {invalid_count}/{len(valid_sums)} policy targets do not sum to ~1")
                # Normalize the policy targets
                policy_targets = policy_targets / (policy_sums.reshape(-1, 1) + 1e-10)
            
            return {
                'states': states,
                'policy_targets': policy_targets,
                'value_targets': value_targets
            }
            
        except Exception as e:
            print(f"Error converting to numpy arrays: {e}")
            print(f"States length: {len(states)}")
            if states:
                print(f"First state shape: {np.array(states[0]).shape if hasattr(states[0], '__len__') else 'scalar'}")
            print(f"Policy targets length: {len(policy_targets)}")
            if policy_targets:
                print(f"First policy target shape: {np.array(policy_targets[0]).shape if hasattr(policy_targets[0], '__len__') else 'scalar'}")
            print(f"Value targets length: {len(value_targets)}")
            if value_targets:
                print(f"First value target: {value_targets[0]}")
            print("\nSample of policy targets (first 5):")
            for i, pt in enumerate(policy_targets[:5]):
                print(f"{i}: Length={len(pt) if hasattr(pt, '__len__') else 'scalar'}, Type={type(pt)}")
                if hasattr(pt, '__len__'):
                    print(f"   Sum: {sum(pt) if hasattr(pt, 'sum') else 'N/A'}")
            raise
    
    def _train_epoch(self, train_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Create dataset and data loader
        dataset = TensorDataset(
            torch.FloatTensor(train_data['states']),
            torch.FloatTensor(train_data['policy_targets']),
            torch.FloatTensor(train_data['value_targets'])
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for batch_idx, (states, policy_targets, value_targets) in enumerate(data_loader):
            # Move data to device
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_preds = self.model.predict(states)
            
            # Calculate policy loss (cross entropy)
            policy_loss = self.criterion['policy'](
                policy_logits.view(-1, policy_logits.size(-1)),
                policy_targets.argmax(dim=1)
            )
            
            # Ensure value predictions and targets have matching shapes
            value_preds = value_preds.squeeze(-1)  # Remove last dimension if it's 1
            if value_targets.dim() > 1:
                value_targets = value_targets.squeeze(-1)  # Also remove from targets if needed
                
            value_loss = self.criterion['value'](
                value_preds,
                value_targets
            )
            
            # Weighted loss
            loss = (
                self.config.training.policy_loss_weight * policy_loss +
                self.config.training.value_loss_weight * value_loss
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Log batch progress
            if (batch_idx + 1) % 10 == 0:
                self.logger.logger.info(
                    f"Batch {batch_idx + 1}/{len(data_loader)}: "
                    f"Loss: {loss.item():.4f} "
                    f"(Policy: {policy_loss.item():.4f}, "
                    f"Value: {value_loss.item():.4f})"
                )
        
        # Calculate average losses
        avg_loss = total_loss / len(data_loader)
        avg_policy_loss = total_policy_loss / len(data_loader)
        avg_value_loss = total_value_loss / len(data_loader)
        
        return {
            'train/loss': avg_loss,
            'train/policy_loss': avg_policy_loss,
            'train/value_loss': avg_value_loss,
            'train/lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model using tournaments."""
        self.model.eval()
        
        # Create arena and add current model
        arena = Arena()
        
        # Add current model
        current_player = ELOPlayer(
            player_id=f"iter_{self.current_iteration}",
            model=self.model,
            mcts_params={
                'num_simulations': self.config.tournament.num_simulations,
                'c_puct': self.config.tournament.c_puct,
                'temperature': 0.1  # Use low temperature for evaluation
            },
            device=self.device
        )
        arena.add_player(current_player)
        
        # Add previous best model if available
        best_model_path = os.path.join(self.config.training.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            best_model = AlphaZeroNetwork(
                board_size=self.config.model.board_size,
                num_res_blocks=self.config.model.num_res_blocks,
                num_filters=self.config.model.num_filters
            ).to(self.device)
            
            # Load the state dict
            state_dict = torch.load(best_model_path, map_location=self.device)
            
            # Handle both scripted and non-scripted models
            if any(k.startswith('_script_module.') for k in state_dict.keys()):
                # Create a new state dict without the _script_module prefix
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('_script_module.'):
                        new_k = k.replace('_script_module.', '')
                        new_state_dict[new_k] = v
                state_dict = new_state_dict
            
            # Load the state dict
            best_model.load_state_dict(state_dict)
            best_model.eval()
            
            best_player = ELOPlayer(
                player_id="best_model",
                model=best_model,
                mcts_params={
                    'num_simulations': self.config.tournament.num_simulations,
                    'c_puct': self.config.tournament.c_puct,
                    'temperature': 0.1
                },
                device=self.device
            )
            arena.add_player(best_player)
        
        # Add random player as baseline
        random_player = ELOPlayer("random", model=None)
        arena.add_player(random_player)
        
        # Run tournament
        results = arena.run_tournament(
            rounds=self.config.tournament.rounds,
            verbose=self.config.logging.verbose
        )
        
        # Get current model's ELO rating
        current_elo = None
        for player in results['leaderboard']:
            if player['player_id'] == f"iter_{self.current_iteration}":
                current_elo = player['rating']
                break
        
        # Save results
        if current_elo is not None and current_elo > self.best_elo:
            self.best_elo = current_elo
            self._save_checkpoint(self.current_iteration, {'elo': current_elo}, is_best=True)
        
        return {
            'eval/elo': current_elo if current_elo is not None else 0.0,
            'eval/best_elo': self.best_elo
        }
    
    def _save_checkpoint(self, iteration: int, metrics: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_elo': self.best_elo,
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f'checkpoint_{iteration:04d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
            self.logger.logger.info(f"New best model saved with ELO: {self.best_elo:.2f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint, handling both JIT-scripted and regular models."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load the checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if this is a JIT-scripted model checkpoint by looking at the keys
        is_jit_checkpoint = any(k.startswith('_script_module.') for k in checkpoint.keys())
        
        if is_jit_checkpoint:
            # This is a JIT-scripted model checkpoint
            # Save the JIT model to a temporary file and load it properly
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                torch.save(checkpoint, tmp.name)
                try:
                    # Load the JIT model
                    self.model = torch.jit.load(tmp.name, map_location=self.device)
                    self.logger.logger.info(f"Loaded JIT-scripted model from {checkpoint_path}")
                    # Load training state if available
                    if 'iteration' in checkpoint:
                        self.current_iteration = checkpoint['iteration']
                    if 'best_elo' in checkpoint:
                        self.best_elo = checkpoint['best_elo']
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    return
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
        
        # Handle regular checkpoint loading
        if 'model_state_dict' in checkpoint:
            # This is a regular checkpoint with separate model state
            model_state = checkpoint['model_state_dict']
            
            # Check if the model state is JIT-scripted
            if any(k.startswith('_script_module.') for k in model_state.keys()):
                # Create a new state dict without the '_script_module.' prefix
                new_state_dict = {}
                for k, v in model_state.items():
                    if k.startswith('_script_module.'):
                        new_k = k.replace('_script_module.', '')
                        new_state_dict[new_k] = v
                    else:
                        new_state_dict[k] = v
                model_state = new_state_dict
            
            self.model.load_state_dict(model_state)
        
        # Load training state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'iteration' in checkpoint:
            self.current_iteration = checkpoint['iteration']
        if 'best_elo' in checkpoint:
            self.best_elo = checkpoint['best_elo']
        
        self.logger.logger.info(f"Loaded checkpoint from {checkpoint_path} (iteration {self.current_iteration})")
        
        # If the model supports JIT compilation, compile it
        if hasattr(self.model, 'compile') and callable(getattr(self.model, 'compile')):
            self.model.compile()
            self.logger.logger.info("Model compiled with JIT")

def train_from_config(config_path: Optional[str] = None):
    """
    Train a model using the specified config file.
    
    Args:
        config_path: Path to config file. If None, uses default config.
    """
    # Load config
    if config_path is not None:
        config = Config.load(config_path)
    else:
        config = get_default_config()
    
    # Initialize and run pipeline
    pipeline = AlphaZeroPipeline(config)
    pipeline.train()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AlphaZero for Reversi')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    args = parser.parse_args()
    train_from_config(args.config)
