"""
Arena for running tournaments between different models with ELO rating.
"""
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import time
from datetime import datetime

from ..game import ReversiGame
from ..mcts import MCTS
from ..model import AlphaZeroNetwork

class ELORatingSystem:
    """ELO rating system for tracking model performance."""
    
    def __init__(self, k: float = 32, initial_rating: float = 1500.0):
        """
        Initialize the ELO rating system.
        
        Args:
            k: K-factor, controls how much ratings change after each game
            initial_rating: Initial rating for new players
        """
        self.k = k
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.games_played: Dict[str, int] = {}
        self.history: List[Dict] = []
    
    def add_player(self, player_id: str, rating: Optional[float] = None):
        """Add a new player to the rating system."""
        if player_id not in self.ratings:
            self.ratings[player_id] = rating if rating is not None else self.initial_rating
            self.games_played[player_id] = 0
    
    def get_rating(self, player_id: str) -> float:
        """Get the current rating of a player."""
        return self.ratings.get(player_id, self.initial_rating)
    
    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate the expected score of player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, player_a: str, player_b: str, score_a: float):
        """
        Update ratings after a game.
        
        Args:
            player_a: ID of player A
            player_b: ID of player B
            score_a: Score for player A (1.0 for win, 0.5 for draw, 0.0 for loss)
        """
        # Ensure both players exist
        self.add_player(player_a)
        self.add_player(player_b)
        
        # Get current ratings
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        
        # Calculate expected scores
        expected_a = self.get_expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        # Update ratings
        new_rating_a = rating_a + self.k * (score_a - expected_a)
        new_rating_b = rating_b + self.k * ((1 - score_a) - expected_b)
        
        # Update player stats
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        self.games_played[player_a] += 1
        self.games_played[player_b] += 1
        
        # Record the game
        game_record = {
            'timestamp': time.time(),
            'player_a': player_a,
            'player_b': player_b,
            'score_a': score_a,
            'score_b': 1.0 - score_a,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b
        }
        self.history.append(game_record)
        
        return game_record
    
    def get_leaderboard(self) -> List[Dict]:
        """Get the current leaderboard sorted by rating."""
        leaderboard = []
        for player_id, rating in self.ratings.items():
            leaderboard.append({
                'player_id': player_id,
                'rating': rating,
                'games_played': self.games_played[player_id]
            })
        
        # Sort by rating (descending)
        leaderboard.sort(key=lambda x: x['rating'], reverse=True)
        return leaderboard
    
    def save_ratings(self, filepath: str):
        """Save the current ratings to a JSON file."""
        data = {
            'k': self.k,
            'initial_rating': self.initial_rating,
            'ratings': self.ratings,
            'games_played': self.games_played,
            'history': self.history,
            'last_updated': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_ratings(cls, filepath: str) -> 'ELORatingSystem':
        """Load ratings from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        elo = cls(k=data['k'], initial_rating=data['initial_rating'])
        elo.ratings = {k: float(v) for k, v in data['ratings'].items()}
        elo.games_played = {k: int(v) for k, v in data['games_played'].items()}
        elo.history = data.get('history', [])
        
        return elo


class ELOPlayer:
    """A player with ELO rating that can play games."""
    
    def __init__(self, player_id: str, model: Optional[AlphaZeroNetwork] = None, 
                 mcts_params: Optional[Dict] = None, device: str = 'cuda'):
        """
        Initialize an ELO player.
        
        Args:
            player_id: Unique identifier for the player
            model: The neural network model (None for random player)
            mcts_params: Parameters for MCTS (if using a model)
            device: Device to run the model on
        """
        self.player_id = player_id
        self.model = model
        self.device = device
        self.mcts = None
        
        if model is not None:
            self.model.eval()
            self.model.to(device)
            
            # Set up MCTS
            if mcts_params is None:
                mcts_params = {
                    'num_simulations': 800,
                    'c_puct': 1.0,
                    'temperature': 1.0
                }
            
            self.mcts = MCTS(
                model=model,
                c_puct=mcts_params.get('c_puct', 1.0),
                num_simulations=mcts_params.get('num_simulations', 800)
            )
    
    def get_move(self, game: ReversiGame) -> Tuple[int, int]:
        """Get the next move for the current game state."""
        if self.model is None:
            # Random player
            valid_moves = game.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else (-1, -1)
        
        # Use MCTS to get the best move
        action, _ = self.mcts.get_action_probs(
            game,
            temperature=1.0  # Use temperature=1.0 for exploration during training
        )
        
        return action
    
    def reset(self):
        """Reset the player's internal state (e.g., MCTS tree)."""
        if self.mcts is not None:
            self.mcts = MCTS(
                model=self.model,
                c_puct=self.mcts.c_puct,
                num_simulations=self.mcts.num_simulations
            )


class Arena:
    """Arena for running tournaments between different players."""
    
    def __init__(self, elo_system: Optional[ELORatingSystem] = None):
        """
        Initialize the arena.
        
        Args:
            elo_system: Optional ELO rating system to use
        """
        self.elo = elo_system if elo_system is not None else ELORatingSystem()
        self.players: Dict[str, ELOPlayer] = {}
    
    def add_player(self, player: ELOPlayer):
        """Add a player to the arena."""
        self.players[player.player_id] = player
        self.elo.add_player(player.player_id)
    
    def play_game(self, player1_id: str, player2_id: str, verbose: bool = False) -> int:
        """
        Play a single game between two players.
        
        Args:
            player1_id: ID of the first player (plays first)
            player2_id: ID of the second player
            verbose: Whether to print game progress
            
        Returns:
            1 if player1 wins, 0.5 for a draw, 0 if player2 wins
        """
        if player1_id not in self.players or player2_id not in self.players:
            raise ValueError(f"One or both players not found: {player1_id}, {player2_id}")
        
        player1 = self.players[player1_id]
        player2 = self.players[player2_id]
        
        # Reset player states
        player1.reset()
        player2.reset()
        
        # Initialize game
        game = ReversiGame()
        
        if verbose:
            print(f"Starting game: {player1_id} (Black) vs {player2_id} (White)")
            print(game)
        
        # Game loop
        while not game.is_game_over():
            current_player = player1 if game.current_player == 1 else player2
            
            # Get move
            move = current_player.get_move(game)
            
            # Make move
            if move == (-1, -1):  # Pass
                game.pass_turn()
                if verbose:
                    print(f"{current_player.player_id} passes")
            else:
                row, col = move
                game.make_move(row, col)
                if verbose:
                    print(f"{current_player.player_id} plays at ({row}, {col})")
                    print(game)
        
        # Determine winner
        black_count, white_count = game.get_piece_count()
        
        if verbose:
            print(f"Game over. Black: {black_count}, White: {white_count}")
            if black_count > white_count:
                print(f"{player1_id} (Black) wins!")
            elif white_count > black_count:
                print(f"{player2_id} (White) wins!")
            else:
                print("It's a draw!")
        
        # Return result from player1's perspective
        if black_count > white_count:
            return 1.0  # player1 wins
        elif white_count > black_count:
            return 0.0  # player2 wins
        else:
            return 0.5  # draw
    
    def run_tournament(self, rounds: int = 100, verbose: bool = False) -> Dict:
        """
        Run a round-robin tournament between all players.
        
        Args:
            rounds: Number of rounds to play (each player plays each other player this many times)
            verbose: Whether to print game results
            
        Returns:
            Dictionary with tournament results
        """
        player_ids = list(self.players.keys())
        num_players = len(player_ids)
        
        if num_players < 2:
            raise ValueError("Need at least 2 players for a tournament")
        
        results = {
            'games_played': 0,
            'matchups': {},
            'start_time': time.time(),
            'end_time': None,
            'rounds': []
        }
        
        # Initialize matchups
        for i in range(num_players):
            for j in range(i + 1, num_players):
                p1, p2 = player_ids[i], player_ids[j]
                match_key = f"{p1}_vs_{p2}"
                results['matchups'][match_key] = {
                    'player1': p1,
                    'player2': p2,
                    'games_played': 0,
                    'wins1': 0,
                    'wins2': 0,
                    'draws': 0
                }
        
        # Play rounds
        for round_num in range(rounds):
            round_results = {
                'round': round_num + 1,
                'games': []
            }
            
            # Play each matchup
            for i in range(num_players):
                for j in range(i + 1, num_players):
                    p1, p2 = player_ids[i], player_ids[j]
                    
                    # Alternate who goes first
                    if (i + j + round_num) % 2 == 0:
                        p1, p2 = p2, p1
                    
                    # Play the game
                    result = self.play_game(p1, p2, verbose=verbose)
                    
                    # Update ELO ratings
                    self.elo.update_ratings(p1, p2, result)
                    
                    # Track results
                    match_key = f"{p1}_vs_{p2}" if f"{p1}_vs_{p2}" in results['matchups'] else f"{p2}_vs_{p1}"
                    results['matchups'][match_key]['games_played'] += 1
                    results['games_played'] += 1
                    
                    if result == 1.0:  # p1 wins
                        results['matchups'][match_key]['wins1'] += 1
                    elif result == 0.0:  # p2 wins
                        results['matchups'][match_key]['wins2'] += 1
                    else:  # draw
                        results['matchups'][match_key]['draws'] += 1
                    
                    # Add to round results
                    round_results['games'].append({
                        'player1': p1,
                        'player2': p2,
                        'result': result,
                        'elo1_before': self.elo.get_rating(p1) - (self.elo.k * (result - self.elo.get_expected_score(self.elo.get_rating(p1), self.elo.get_rating(p2)))),
                        'elo2_before': self.elo.get_rating(p2) - (self.elo.k * ((1 - result) - self.elo.get_expected_score(self.elo.get_rating(p2), self.elo.get_rating(p1)))),
                        'elo1_after': self.elo.get_rating(p1),
                        'elo2_after': self.elo.get_rating(p2)
                    })
            
            # Add round results
            results['rounds'].append(round_results)
            
            if verbose:
                print(f"\n--- After Round {round_num + 1} ---")
                self.print_leaderboard()
        
        # Finalize results
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        results['leaderboard'] = self.elo.get_leaderboard()
        
        return results
    
    def print_leaderboard(self):
        """Print the current leaderboard."""
        leaderboard = self.elo.get_leaderboard()
        print("\nCurrent Leaderboard:")
        print("Rank  Player ID               Rating  Games Played")
        print("----  ---------------------  -------  ------------")
        
        for i, player in enumerate(leaderboard, 1):
            print(f"{i:4d}  {player['player_id']:22s}  {player['rating']:7.1f}  {player['games_played']:12d}")
    
    def save_results(self, filepath: str):
        """Save tournament results to a JSON file."""
        # Save ELO ratings
        elo_file = os.path.splitext(filepath)[0] + '_elo.json'
        self.elo.save_ratings(elo_file)
        
        # Save tournament results
        with open(filepath, 'w') as f:
            json.dump(self.elo.get_leaderboard(), f, indent=2)
