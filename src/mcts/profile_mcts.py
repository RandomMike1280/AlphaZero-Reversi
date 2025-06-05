"""
Profile the MCTS implementation to identify performance bottlenecks.

This script uses cProfile to analyze the performance of the MCTS algorithm,
breaking down the time spent in different functions and methods.
"""

import cProfile
import pstats
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import game and model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from game.game import ReversiGame
from model.network import AlphaZeroNetwork
from mcts.mcts import MCTS, MCTSNode

class ProfilingWrapper:
    """Wrapper class to profile MCTS operations."""
    
    def __init__(self, board_size: int = 8, num_simulations: int = 100, batch_size: int = 16):
        """Initialize the profiler with game and model."""
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        
        # Initialize game and model
        self.game = ReversiGame(size=board_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_dummy_model().to(self.device)
        self.model.eval()
        
        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            num_simulations=num_simulations,
            batch_size=batch_size
        )
    
    def _create_dummy_model(self):
        """Create a dummy model for profiling."""
        class DummyModel(torch.nn.Module):
            def __init__(self, board_size=8):
                super().__init__()
                self.board_size = board_size
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.policy_conv = torch.nn.Conv2d(64, 2, kernel_size=1)
                self.policy_fc = torch.nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
                self.value_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
                self.value_fc1 = torch.nn.Linear(board_size * board_size, 64)
                self.value_fc2 = torch.nn.Linear(64, 1)
            
            def forward(self, x):
                batch_size = x.size(0)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                
                # Policy head
                policy = self.policy_conv(x)
                policy = policy.view(batch_size, -1)
                policy = self.policy_fc(policy)
                
                # Value head
                value = torch.relu(self.value_conv(x))
                value = value.view(batch_size, -1)
                value = torch.relu(self.value_fc1(value))
                value = torch.tanh(self.value_fc2(value))
                
                return policy, value
        
        return DummyModel(board_size=self.board_size)
    
    def profile_search(self, num_searches: int = 10):
        """Profile the MCTS search operation."""
        print(f"Profiling MCTS search with {num_searches} searches...")
        
        def run_searches():
            for _ in range(num_searches):
                self.game = ReversiGame(size=self.board_size)  # Reset game
                self.mcts.search(self.game)
        
        # Run with cProfile
        pr = cProfile.Profile()
        pr.enable()
        run_searches()
        pr.disable()
        
        # Print results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Show top 20 functions by cumulative time
        print(s.getvalue())
    
    def profile_node_operations(self, num_nodes: int = 1000):
        """Profile MCTS node operations."""
        print(f"Profiling MCTS node operations with {num_nodes} nodes...")
        
        def create_nodes():
            root = MCTSNode(1.0, 1)  # Player 1's turn
            for i in range(num_nodes):
                node = MCTSNode(1.0, 1 + (i % 2), parent=root)
                root.children[(i % 8, i % 8)] = node
            return root
        
        def traverse_nodes(node, depth=0, max_depth=10):
            if depth >= max_depth or not node.children:
                return
            for child in node.children.values():
                traverse_nodes(child, depth + 1, max_depth)
        
        # Profile node creation
        pr = cProfile.Profile()
        pr.enable()
        root = create_nodes()
        traverse_nodes(root)
        pr.disable()
        
        # Print results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())
    
    def profile_batch_processing(self, batch_size: int = 16, num_batches: int = 10):
        """Profile batch processing in MCTS."""
        print(f"Profiling batch processing with batch_size={batch_size}, num_batches={num_batches}...")
        
        # Create a batch of random inputs
        def create_batch():
            return torch.randn(batch_size, 3, self.board_size, self.board_size, 
                            device=self.device)
        
        # Profile forward pass
        pr = cProfile.Profile()
        pr.enable()
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = create_batch()
                policy, value = self.model(batch)
                _ = policy.cpu().numpy()
                _ = value.cpu().numpy()
        
        pr.disable()
        
        # Print results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())


def main():
    """Run the profiler with default settings."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile MCTS implementation')
    parser.add_argument('--board-size', type=int, default=8, help='Board size')
    parser.add_argument('--simulations', type=int, default=100, help='Number of MCTS simulations')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--profile', choices=['search', 'nodes', 'batch', 'all'], 
                       default='all', help='Which part to profile')
    
    args = parser.parse_args()
    
    profiler = ProfilingWrapper(
        board_size=args.board_size,
        num_simulations=args.simulations,
        batch_size=args.batch_size
    )
    
    if args.profile in ['search', 'all']:
        print("\n" + "="*50)
        print("PROFILING MCTS SEARCH")
        print("="*50)
        profiler.profile_search(num_searches=min(100, args.simulations))
    
    if args.profile in ['nodes', 'all']:
        print("\n" + "="*50)
        print("PROFILING NODE OPERATIONS")
        print("="*50)
        profiler.profile_node_operations(num_nodes=1000)
    
    if args.profile in ['batch', 'all']:
        print("\n" + "="*50)
        print("PROFILING BATCH PROCESSING")
        print("="*50)
        profiler.profile_batch_processing(
            batch_size=args.batch_size,
            num_batches=64
        )


if __name__ == "__main__":
    main()
