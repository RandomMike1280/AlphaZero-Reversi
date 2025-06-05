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
        return self.board.board.copy()
    
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
    
    def __str__(self) -> str:
        """String representation of the game state."""
        symbols = {Board.EMPTY: '.', Board.BLACK: 'B', Board.WHITE: 'W'}
        rows = []
        
        # Add column headers
        rows.append('  ' + ' '.join(str(i) for i in range(self.size)))
        
        for i in range(self.size):
            row = [str(i)]
            for j in range(self.size):
                row.append(symbols[self.board.board[i][j]])
            rows.append(' '.join(row))
        
        # Add current player and score
        rows.append(f"\nCurrent player: {'Black' if self.current_player == Board.BLACK else 'White'}")
        black, white = self.get_score()
        rows.append(f"Score - Black: {black}, White: {white}")
        
        if self.game_over:
            if self.winner == 0:
                rows.append("Game over! It's a draw!")
            else:
                winner = 'Black' if self.winner == Board.BLACK else 'White'
                rows.append(f"Game over! {winner} wins!")
        
        return '\n'.join(rows)
