"""
Neural network architecture for AlphaZero Reversi with JIT compilation support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings

# Suppress JIT warnings for cleaner output
warnings.filterwarnings('ignore', message='.*JIT-compiled functions cannot take variable number of arguments.*')

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
    """AlphaZero neural network for Reversi with JIT compilation support."""
    
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
        self.num_filters = num_filters
        
        # Initial convolution
        self.conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
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
        
        # JIT-related attributes
        self._jit_compiled = False
        self._script_module = None
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @torch.jit.export
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
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = F.relu(self.policy_bn(policy))
        batch_size = x.size(0)
        policy = policy.contiguous().view(batch_size, -1)  # Flatten with contiguity
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(x)
        value = F.relu(self.value_bn(value))
        value = value.contiguous().view(batch_size, -1)  # Flatten with contiguity
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy, value.squeeze(1)  # Remove extra dimension from value
    
    def compile(self) -> None:
        """
        Compile the model using TorchScript for improved performance.
        Uses torch.jit.script for better compatibility with Python control flow.
        """
        if self._jit_compiled:
            return
            
        try:
            # Create a scripted version of the model
            self._script_module = torch.jit.script(self)
            self._jit_compiled = True
        except Exception as e:
            warnings.warn(f"JIT compilation failed: {str(e)}. Falling back to eager mode.")
            self._script_module = None
            self._jit_compiled = False
    
    @torch.jit.export
    def predict(self, board_state: torch.Tensor, valid_moves: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value for the given board state.
        Compatible with both JIT and eager modes.
        
        Args:
            board_state: Input tensor of shape (batch_size, 3, board_size, board_size)
            valid_moves: Optional tensor of valid moves (not used, kept for compatibility)
            
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: Tensor of shape (batch_size, board_size * board_size + 1)
            - value: Tensor of shape (batch_size,)
        """
        # Add batch dimension if needed
        if board_state.dim() == 3:
            board_state = board_state.unsqueeze(0)
            
        # Forward pass through the model
        if self._jit_compiled and self._script_module is not None:
            return self._script_module.forward(board_state)
        return self.forward(board_state)
        
    def predict_from_numpy(self, board_state: np.ndarray, valid_moves: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value from numpy arrays.
        This is a convenience method that handles numpy array conversion.
        
        Args:
            board_state: Numpy array of shape (batch_size, 3, board_size, board_size)
            valid_moves: Optional numpy array of valid moves (not used, kept for compatibility)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        device = next(self.parameters()).device
        board_tensor = torch.from_numpy(board_state).float().to(device)
        return self.predict(board_tensor)
    
    def to(self, *args, **kwargs):
        """Override to handle device changes with JIT compilation."""
        self._jit_compiled = False
        self._script_module = None
        return super().to(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """Override train to disable JIT when in training mode."""
        if mode and self._jit_compiled:
            self._jit_compiled = False
            self._script_module = None
        return super().train(mode)
    
    def eval(self):
        """Override eval to enable JIT compilation if not already compiled."""
        result = super().eval()
        if not self._jit_compiled and not self.training:
            self.compile()
        return result

    def make_prediction(self, board_state: np.ndarray, valid_moves: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Make a prediction for a single board state.
        
        Args:
            board_state: Numpy array of shape (3, board_size, board_size)
                        representing [player_pieces, opponent_pieces, valid_moves]
            valid_moves: Optional mask of valid moves (not used in this implementation)
            
        Returns:
            Tuple of (action_probs, value)
            - action_probs: Numpy array of shape (board_size * board_size + 1,)
            - value: float in [-1, 1]
        """
        # Ensure model is in eval mode and compiled if possible
        self.eval()
        
        # Add batch dimension if needed
        if board_state.ndim == 3:
            board_state = np.expand_dims(board_state, axis=0)
        
        with torch.no_grad():
            # Use predict_from_numpy which handles the numpy to tensor conversion
            policy_logits, value = self.predict_from_numpy(board_state)
            
            # Convert to probabilities
            action_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            value = value.item()
            
            # Convert to numpy
            action_probs = action_probs.cpu().numpy()
            
        return action_probs, value
