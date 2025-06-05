"""
Self-play implementation for generating training data.
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import os
import time
from datetime import datetime

from ..game import ReversiGame
from ..mcts import MCTS
from ..model import AlphaZeroNetwork

class SelfPlay:
    """Self-play for generating training data."""
    
    def __init__(self, model: AlphaZeroNetwork, args: dict):
        """
        Initialize the self-play generator.
        
        Args:
            model: The neural network model
            args: Dictionary of arguments containing:
                - num_simulations: Number of MCTS simulations per move
                - c_puct: Exploration constant for MCTS
                - temperature: Temperature for action selection
                - dirichlet_alpha: Alpha parameter for Dirichlet noise
                - dirichlet_epsilon: Epsilon for mixing in Dirichlet noise
                - num_parallel_games: Number of games to play in parallel
                - save_dir: Directory to save training data
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
        self.args = args
        self.mcts = MCTS(
            model=model,
            c_puct=args.get('c_puct', 1.0),
            num_simulations=args.get('num_simulations', 800)
        )
        
        # Create save directory if it doesn't exist
        self.save_dir = args.get('save_dir', 'self_play_data')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"SelfPlay initialized on device: {self.device}")
    
    def generate_games(self, num_games: int) -> List[Dict]:
        """
        Generate self-play games.
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            List of game data dictionaries, each containing:
                - states: List of game states
                - action_probs: List of action probabilities from MCTS
                - winners: List of winners (1 for player 1, -1 for player 2, 0 for draw)
        """
        all_games = []
        
        for game_idx in range(num_games):
            start_time = time.time()
            print(f"\nGenerating game {game_idx + 1}/{num_games}")
            
            # Initialize game
            game = ReversiGame()
            game_data = {
                'states': [],
                'action_probs': [],
                'winners': []
            }
            
            # Play the game
            while not game.is_game_over():
                # Get action probabilities from MCTS
                action, action_probs = self.mcts.get_action_probs(
                    game,
                    temperature=self.args.get('temperature', 1.0)
                )
                
                # Add the current state (before the move)
                self._add_state_to_game_data(game, game_data)
                
                # Add the action probabilities for the current state
                game_data['action_probs'].append(action_probs)
                
                # Make the move
                row, col = action
                if (row, col) == (-1, -1):  # Pass
                    game.pass_turn()
                else:
                    game.make_move(row, col)
                
                # Update MCTS tree
                self.mcts.update_with_move(action)
            
            # Determine the winner
            winner = game.get_winner()
            if winner == 2:  # Convert to -1 for player 2
                winner = -1
            elif winner == 0:  # Draw
                winner = 0
            
            # Ensure we have the same number of states and action probabilities
            min_length = min(len(game_data['states']), len(game_data['action_probs']))
            
            # Truncate both lists to the minimum length
            game_data['states'] = game_data['states'][:min_length]
            game_data['action_probs'] = game_data['action_probs'][:min_length]
            
            # Add the winner to each state in the game
            game_data['winners'] = [winner] * min_length
            
            # Save the game data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"game_{timestamp}_{game_idx}.pt")
            torch.save(game_data, save_path)
            
            all_games.append(game_data)
            
            # Print game stats
            duration = time.time() - start_time
            print(f"Game {game_idx + 1} completed in {duration:.1f}s. "
                  f"Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == -1 else 'Draw'}")
        
        return all_games
    
    def _add_state_to_game_data(self, game: ReversiGame, game_data: Dict):
        """
        Add the current game state to the game data.
        
        Args:
            game: The current game state
            game_data: Dictionary to store game data
        """
        # Convert game state to input format for the neural network
        board_state = game.get_board_state()
        player_pieces = (board_state == 1).astype(np.float32)
        opponent_pieces = (board_state == 2).astype(np.float32)
        
        # Get valid moves
        valid_moves = np.zeros_like(board_state, dtype=np.float32)
        for row, col in game.get_valid_moves():
            if (row, col) != (-1, -1):  # Skip pass move for the mask
                valid_moves[row, col] = 1.0
        
        # Stack into input tensor (3, board_size, board_size)
        state = np.stack([player_pieces, opponent_pieces, valid_moves], axis=0)
        
        # Add to game data
        game_data['states'].append(state)
    
    def generate_training_data(self, num_games: int) -> Dict[str, np.ndarray]:
        """
        Generate training data from self-play games.
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            Dictionary containing:
                - states: Array of game states (n, 3, board_size, board_size)
                - action_probs: Array of action probabilities (n, board_size * board_size + 1)
                - values: Array of game outcomes (n,)
        """
        games = self.generate_games(num_games)
        
        # Collect all states, action_probs, and values
        all_states = []
        all_action_probs = []
        all_values = []
        
        for game in games:
            all_states.extend(game['states'])
            all_action_probs.extend(game['action_probs'])
            all_values.extend(game['winners'])
        
        # Convert to numpy arrays
        states = np.array(all_states, dtype=np.float32)
        action_probs = np.array(all_action_probs, dtype=np.float32)
        values = np.array(all_values, dtype=np.float32)
        
        return {
            'states': states,
            'action_probs': action_probs,
            'values': values
        }
