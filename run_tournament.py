"""
Script for running tournaments between different AlphaZero Reversi models.
"""
import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime

from src.arena import Arena, ELOPlayer, ELORatingSystem
from src.model import AlphaZeroNetwork
from src.game import ReversiGame

def load_model(model_path: str, device: str) -> AlphaZeroNetwork:
    """Load a model from a checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create model with default parameters (will be overridden by the checkpoint)
    model = AlphaZeroNetwork(board_size=8)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='Run a tournament between Reversi models')
    
    # Model parameters
    parser.add_argument('--model-dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--elo-file', type=str, default='elo_ratings.json',
                       help='File to save/load ELO ratings')
    
    # Tournament parameters
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of rounds to play')
    parser.add_argument('--num-simulations', type=int, default=400,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--c-puct', type=float, default=1.0,
                       help='Exploration constant for MCTS')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu). Auto-detected if not specified.')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                       help='Directory to save tournament results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed game information')
    parser.add_argument('--print-games', action='store_true',
                       help='Print game moves during the tournament')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize ELO rating system
    elo_file = os.path.join(args.output_dir, args.elo_file)
    if os.path.exists(elo_file):
        print(f"Loading ELO ratings from {elo_file}")
        elo = ELORatingSystem.load_ratings(elo_file)
    else:
        print("Starting new ELO rating system")
        elo = ELORatingSystem(k=32, initial_rating=1500.0)
    
    # Initialize arena
    arena = Arena(elo_system=elo)
    
    # Add a random player as baseline
    arena.add_player(ELOPlayer("random", model=None))
    
    # Find all model checkpoints
    model_files = []
    if os.path.isdir(args.model_dir):
        for root, _, files in os.walk(args.model_dir):
            for file in files:
                if file.endswith('.pth') or file.endswith('.pt') or file.endswith('.pth.tar'):
                    model_files.append(os.path.join(root, file))
    
    if not model_files:
        print(f"No model files found in {args.model_dir}")
    
    # Add models to arena
    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Skip if player already exists
        if model_name in [p.player_id for p in arena.players.values()]:
            continue
        
        try:
            print(f"Loading model: {model_name}")
            model = load_model(model_path, args.device)
            
            # Create player with MCTS
            mcts_params = {
                'num_simulations': args.num_simulations,
                'c_puct': args.c_puct,
                'temperature': 1.0
            }
            
            player = ELOPlayer(
                player_id=model_name,
                model=model,
                mcts_params=mcts_params,
                device=args.device
            )
            
            arena.add_player(player)
            print(f"Added player: {model_name}")
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    if len(arena.players) < 2:
        print("Need at least 2 players to start a tournament")
        return
    
    # Print participants
    print("\nTournament Participants:")
    for i, player_id in enumerate(arena.players.keys(), 1):
        print(f"{i}. {player_id}")
    
    # Run tournament
    print(f"\nStarting tournament with {args.rounds} rounds...")
    results = arena.run_tournament(rounds=args.rounds, 
                                 verbose=args.verbose,
                                 print_games=args.print_games)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'tournament_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'rounds': args.rounds,
            'participants': list(arena.players.keys()),
            'leaderboard': [{'player': p['player_id'], 'rating': p['rating']} 
                          for p in results['leaderboard']]
        }, f, indent=2)
    
    # Save updated ELO ratings
    arena.elo.save_ratings(elo_file)
    
    print(f"\nTournament completed! Results saved to {results_file}")
    print("\nFinal Leaderboard:")
    arena.print_leaderboard()

if __name__ == '__main__':
    main()
