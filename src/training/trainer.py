"""
Training loop for the AlphaZero model.
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

from ..model import AlphaZeroNetwork
from ..self_play import SelfPlay
from ..mcts import MCTS

class Trainer:
    """Trainer for the AlphaZero model."""
    
    def __init__(self, model: AlphaZeroNetwork, args: dict):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            args: Dictionary of training arguments:
                - lr: Learning rate
                - batch_size: Batch size for training
                - num_epochs: Number of training epochs
                - weight_decay: L2 regularization strength
                - checkpoint_dir: Directory to save model checkpoints
                - device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.args = args
        self.device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.get('lr', 0.001),
            weight_decay=args.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.get('lr_milestones', [100, 200]),
            gamma=args.get('lr_gamma', 0.1)
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = args.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Loss function
        self.mse_loss = torch.nn.MSELoss()
        
        # Best validation loss
        self.best_loss = float('inf')
    
    def train(self, train_data: Dict[str, np.ndarray], val_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Train the model.
        
        Args:
            train_data: Dictionary containing training data:
                - states: Array of game states (n, 3, board_size, board_size)
                - action_probs: Array of action probabilities (n, board_size * board_size + 1)
                - values: Array of game outcomes (n,)
            val_data: Optional validation data with the same structure as train_data
        """
        # Convert data to PyTorch tensors
        train_states = torch.FloatTensor(train_data['states']).to(self.device)
        train_action_probs = torch.FloatTensor(train_data['action_probs']).to(self.device)
        train_values = torch.FloatTensor(train_data['values']).to(self.device)
        
        # Create dataset and data loader
        train_dataset = TensorDataset(train_states, train_action_probs, train_values)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.get('batch_size', 64),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Convert validation data if provided
        val_loader = None
        if val_data is not None:
            val_states = torch.FloatTensor(val_data['states']).to(self.device)
            val_action_probs = torch.FloatTensor(val_data['action_probs']).to(self.device)
            val_values = torch.FloatTensor(val_data['values']).to(self.device)
            
            val_dataset = TensorDataset(val_states, val_action_probs, val_values)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.get('batch_size', 64),
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        num_epochs = self.args.get('num_epochs', 100)
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Print progress
            log_msg = f"Epoch {epoch}/{num_epochs} - " \
                     f"Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            print(log_msg)
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.args.get('checkpoint_freq', 10) == 0:
                self._save_checkpoint(epoch)
    
    def _train_epoch(self, data_loader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (states, action_probs, values) in enumerate(data_loader):
            # Move data to device
            states = states.to(self.device, non_blocking=True)
            action_probs = action_probs.to(self.device, non_blocking=True)
            values = values.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - use predict for consistency with inference
            policy_logits, value_preds = self.model.predict(states)
            
            # Calculate losses
            # Policy loss: cross-entropy between predicted and target policy
            policy_loss = -torch.mean(torch.sum(action_probs * F.log_softmax(policy_logits, dim=1), dim=1))
            
            # Value loss: MSE between predicted and actual values
            value_loss = self.mse_loss(value_preds, values)
            
            # Total loss (weighted sum)
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} [{batch_idx * len(states)}/{len(data_loader.dataset)} "
                      f"({100. * batch_idx / len(data_loader):.0f}%)]\t"
                      f"Loss: {loss.item():.6f}")
        
        return total_loss / len(data_loader)
    
    def _validate(self, data_loader) -> float:
        """
        Validate the model.
        
        Args:
            data_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, action_probs, values in data_loader:
                # Move data to device
                states = states.to(self.device, non_blocking=True)
                action_probs = action_probs.to(self.device, non_blocking=True)
                values = values.to(self.device, non_blocking=True)
                
                # Forward pass - use predict for consistency with training
                policy_logits, value_preds = self.model.predict(states)
                
                # Calculate losses
                policy_loss = -torch.mean(torch.sum(action_probs * F.log_softmax(policy_logits, dim=1), dim=1))
                value_loss = self.mse_loss(value_preds, values)
                loss = policy_loss + value_loss
                
                # Update statistics
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'args': self.args
        }
        
        # Save checkpoint
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        
        # If this is the best model, save a separate copy
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, model: AlphaZeroNetwork, args: dict):
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load the weights into
            args: Training arguments
            
        Returns:
            Trainer instance with loaded weights
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create trainer
        trainer = cls(model, args)
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state if available
        if 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state if available
        if 'scheduler' in checkpoint and hasattr(trainer, 'scheduler'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load best loss if available
        if 'best_loss' in checkpoint:
            trainer.best_loss = checkpoint['best_loss']
        
        return trainer
