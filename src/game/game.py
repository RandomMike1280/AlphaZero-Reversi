"""
Reversi game module.
Handles game flow and state management.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .board import Board

class ReversiGame:
    """
    Main game class for Reversi that manages the game state and flow.
    """
    
    def __init__(self, size: int = 8):
        """
        Initialize a new Reversi game.
        
        Args:
            size: Size of the board (default: 8 for standard Reversi)
        """
        self.board = Board(size)
        self.size = size
        self.current_player = Board.BLACK  # Black moves first
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.board = Board(self.size)
        self.current_player = Board.BLACK
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move on the board.
        
        Args:
            row: Row of the move (0-based)
            col: Column of the move (0-based)
            
        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if self.game_over:
            return False
        
        # Make a copy of the current board state for the move history
        board_before = self.board.copy()
        
        # Try to make the move
        move_made = self.board.make_move(row, col, self.current_player)
        
        if move_made:
            # Record the move
            self.move_history.append({
                'player': self.current_player,
                'move': (row, col),
                'board_before': board_before,
                'board_after': self.board.copy()
            })
            
            # Update game state
            self.game_over = self.board.game_over
            self.winner = self.board.winner
            
            # Switch player if game isn't over
            if not self.game_over:
                self.current_player = self.board.current_player
                
                # If next player has no valid moves, switch back
                if not self.get_valid_moves() or self.get_valid_moves() == [(-1, -1)]:
                    self.current_player = Board.WHITE if self.current_player == Board.BLACK else Board.BLACK
        
        return move_made
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the current player.
        
        Returns:
            List of (row, col) tuples representing valid moves
        """
        return self.board.get_valid_moves(self.current_player)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.game_over
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game.
        
        Returns:
            int: Board.BLACK, Board.WHITE, or 0 for draw, None if game not over
        """
        return self.board.winner if self.game_over else None
    
    def get_score(self) -> Tuple[int, int]:
        """
        Get the current score (black, white).
        
        Returns:
            Tuple of (black_score, white_score)
        """
        return self.board.get_score()
    
    def get_board_state(self) -> np.ndarray:
        """
        Get the current board state as a numpy array.
        
        Returns:
            2D numpy array representing the board state
        """
        self.board._ensure_board_updated()
        return self.board._board.copy()
    
    def get_current_player(self) -> int:
        """
        Get the current player.
        
        Returns:
            int: Board.BLACK or Board.WHITE
        """
        return self.current_player
    
    def get_move_history(self) -> List[Dict[str, Any]]:
        """
        Get the move history.
        
        Returns:
            List of move dictionaries containing player, move, and board states
        """
        return self.move_history.copy()
        
    def get_canonical_state(self) -> np.ndarray:
        """
        Returns the canonical board state for the neural network.
        The state is always from the perspective of the current player.
        
        Returns:
            A numpy array of shape (3, board_size, board_size)
            - Channel 0: Current player's pieces (1s)
            - Channel 1: Opponent's pieces (1s)
            - Channel 2: Valid moves mask (1s)
        """
        # Get the board state and current player
        board_state = self.get_board_state()
        current_player = self.get_current_player()
        
        # Create canonical board representation using vectorized operations
        current_player_mask = (board_state == current_player)
        opponent_mask = (board_state != 0) & ~current_player_mask  # Non-zero and not current player
        
        # Convert to float32 for neural network input
        current_player_pieces = current_player_mask.astype(np.float32)
        opponent_pieces = opponent_mask.astype(np.float32)
        
        # Create valid moves mask using vectorized operations
        valid_moves = self.get_valid_moves()
        valid_moves_mask = np.zeros_like(board_state, dtype=np.float32)
        if valid_moves:
            valid_indices = np.array([(r, c) for r, c in valid_moves if (r, c) != (-1, -1)], dtype=np.int32)
            if len(valid_indices) > 0:
                valid_moves_mask[valid_indices[:, 0], valid_indices[:, 1]] = 1.0
        
        return np.stack([current_player_pieces, opponent_pieces, valid_moves_mask], axis=0)
    
    def copy(self) -> 'ReversiGame':
        """Create a deep copy of the game."""
        new_game = ReversiGame(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        return new_game
        
    def __str__(self) -> str:
        """String representation of the game state."""
        # Delegate to the Board's __str__ method
        result = str(self.board)
        
        # Add current player and score if not already included
        if "Current player:" not in result:
            result += f"\nCurrent player: {'Black' if self.current_player == Board.BLACK else 'White'}"
            black, white = self.get_score()
            result += f"\nScore - Black: {black}, White: {white}"
            
            if self.game_over:
                if self.winner == 0:
                    result += "\nGame over! It's a draw!"
                else:
                    winner = 'Black' if self.winner == Board.BLACK else 'White'
                    result += f"\nGame over! {winner} wins!"
        
        return result
