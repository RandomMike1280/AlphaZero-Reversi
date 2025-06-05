"""
Neural network architecture for AlphaZero Reversi.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and batch normalization."""
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)

class AlphaZeroNetwork(nn.Module):
    """AlphaZero neural network for Reversi."""
    
    def __init__(self, board_size: int = 8, num_res_blocks: int = 5, num_filters: int = 128):
        """
        Initialize the AlphaZero network.
        
        Args:
            board_size: Size of the board (default: 8 for standard Reversi)
            num_res_blocks: Number of residual blocks
            num_filters: Number of filters in convolutional layers
        """
        super().__init__()
        self.board_size = board_size
        
        # Initial convolution
        self.conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)  # +1 for pass move
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
               Channels: [player_pieces, opponent_pieces, valid_moves]
               
        Returns:
            Tuple of (policy_logits, value)
        """
        # Initial convolution
        x = F.relu(self.bn(self.conv(x)))
        
        # Residual tower
        x = self.res_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy, value.squeeze(1)  # Remove extra dimension from value
    
    def predict(self, board_state: 'np.ndarray', valid_moves: 'np.ndarray' = None) -> Tuple['np.ndarray', float]:
        """
        Make a prediction for a single board state.
        
        Args:
            board_state: Board state as a numpy array of shape (board_size, board_size)
                        with values: 0=empty, 1=player, 2=opponent
            valid_moves: Optional mask of valid moves (1 for valid, 0 for invalid)
            
        Returns:
            Tuple of (policy_probs, value)
        """
        import numpy as np
        
        # Convert to tensor and add batch dimension
        player_pieces = (board_state == 1).astype(np.float32)
        opponent_pieces = (board_state == 2).astype(np.float32)
        
        if valid_moves is None:
            valid_moves = np.ones_like(board_state, dtype=np.float32)
        else:
            valid_moves = valid_moves.astype(np.float32)
        
        # Stack into input tensor (3, board_size, board_size)
        x = np.stack([player_pieces, opponent_pieces, valid_moves], axis=0)
        x = torch.from_numpy(x).unsqueeze(0).to(next(self.parameters()).device)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(x)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        
        return policy_probs, value
