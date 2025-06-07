"""
Benchmarking script for MCTS performance evaluation.

Measures:
- Time taken per search
- Nodes evaluated per second
- Memory usage
- GPU utilization (if available)
"""

import time
import os
import sys
import tracemalloc
import psutil
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path to import game and model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.game import ReversiGame
from model.network import AlphaZeroNetwork
from src.mcts.mcts import MCTS  # Absolute import

class DummyModel(nn.Module):
    """A dummy model for benchmarking MCTS without training."""
    
    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)  # +1 for pass
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def predict(self, x):
        # Input shape: (batch, 3, board_size, board_size)
        batch_size = x.size(0)
        
        # Shared trunk
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)  # (batch, board_size*board_size + 1)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # (-1, 1)
        
        return policy, value


class MCTSBenchmark:
    """Benchmark MCTS performance with various configurations."""
    
    def __init__(self, board_size: int = 8, num_simulations: int = 800, 
                 batch_sizes: List[int] = [1, 8, 16, 32], use_cuda: bool = True):
        """
        Initialize the benchmark.
        
        Args:
            board_size: Size of the game board (will be used as 'size' for ReversiGame)
            num_simulations: Number of MCTS simulations to run
            batch_sizes: List of batch sizes to test
            use_cuda: Whether to use CUDA if available
        """
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.batch_sizes = batch_sizes
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Initialize game and model
        self.game = ReversiGame(size=board_size)  # Use 'size' parameter instead of 'board_size'
        self.model = DummyModel(board_size=board_size).to(self.device)
        self.model.eval()
        
        # Benchmark results
        self.results = []
    
    def reset_game(self):
        """Reset the game to its initial state."""
        self.game = ReversiGame(size=self.board_size)  # Use 'size' parameter
    
    def run_benchmark(self, num_runs: int = 5) -> Dict:
        """
        Run the benchmark with different batch sizes.
        
        Args:
            num_runs: Number of runs per configuration
            
        Returns:
            Dictionary containing benchmark results
        """
        print(f"Running MCTS benchmark with {num_runs} runs per configuration")
        print(f"Device: {self.device}")
        print(f"Board size: {self.board_size}")
        print(f"Simulations per run: {self.num_simulations}")
        
        for batch_size in self.batch_sizes:
            print(f"\n=== Batch size: {batch_size} ===")
            
            # Initialize MCTS with current batch size
            mcts = MCTS(
                model=self.model,
                num_simulations=self.num_simulations,
                batch_size=batch_size
            )
            
            # Warmup
            print("  Warming up...")
            for _ in range(2):
                mcts.search(self.game)
            
            # Run benchmark
            run_times = []
            nodes_processed = []
            memory_usages = []
            
            for run in range(num_runs):
                # Reset game state
                self.reset_game()
                
                # Start memory tracking
                tracemalloc.start()
                process = psutil.Process()
                start_mem = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run MCTS
                start_time = time.time()
                visit_counts = mcts.search(self.game)
                elapsed = time.time() - start_time
                
                # Measure memory after search
                current_mem = process.memory_info().rss / 1024 / 1024  # MB
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate metrics
                total_nodes = sum(visit_counts.values())
                nodes_per_sec = total_nodes / elapsed if elapsed > 0 else 0
                
                run_times.append(elapsed)
                nodes_processed.append(total_nodes)
                memory_usages.append(current_mem - start_mem)
                
                print(f"  Run {run + 1}/{num_runs}:")
                print(f"    Time: {elapsed:.3f}s")
                print(f"    Nodes: {total_nodes}")
                print(f"    Nodes/s: {nodes_per_sec:,.0f}")
                print(f"    Memory usage: {current_mem - start_mem:.2f} MB")
                print(f"    Peak memory: {peak / 1024 / 1024:.2f} MB")
            
            # Calculate statistics
            avg_time = np.mean(run_times)
            avg_nodes_per_sec = np.mean([n/t for n, t in zip(nodes_processed, run_times) if t > 0])
            avg_memory = np.mean(memory_usages)
            
            result = {
                'batch_size': batch_size,
                'avg_time': avg_time,
                'avg_nodes_per_sec': avg_nodes_per_sec,
                'avg_memory_mb': avg_memory,
                'num_simulations': self.num_simulations,
                'runs': num_runs,
                'device': str(self.device)
            }
            
            self.results.append(result)
            print("\n  Average:")
            print(f"    Time: {avg_time:.3f}s")
            print(f"    Nodes/s: {avg_nodes_per_sec:,.0f}")
            print(f"    Memory: {avg_memory:.2f} MB")
        
        return self.results
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        if not self.results:
            print("No benchmark results available. Run run_benchmark() first.")
            return
        
        print("\n=== Benchmark Summary ===")
        print(f"Device: {self.device}")
        print(f"Board size: {self.board_size}")
        print(f"Simulations per run: {self.num_simulations}")
        print("-" * 50)
        print("Batch Size | Avg Time (s) | Nodes/s    | Memory (MB)")
        print("-" * 50)
        
        for result in self.results:
            print(f"{result['batch_size']:^9} | {result['avg_time']:^11.3f} | "
                  f"{result['avg_nodes_per_sec']:^10,.0f} | {result['avg_memory_mb']:^10.2f}")
        
        print("-" * 50)


def main():
    """Run the benchmark with default settings."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark MCTS performance')
    parser.add_argument('--board-size', type=int, default=8, help='Board size')
    parser.add_argument('--simulations', type=int, default=800, help='Number of MCTS simulations')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 16, 32, 64],
                       help='Batch sizes to test')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    benchmark = MCTSBenchmark(
        board_size=args.board_size,
        num_simulations=args.simulations,
        batch_sizes=args.batch_sizes,
        use_cuda=not args.no_cuda
    )
    
    benchmark.run_benchmark(num_runs=args.runs)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
