"""
Board module for Reversi.
Handles the game board state and move validation.
"""
from typing import List, Tuple, Optional, Set
import numpy as np

class Board:
    """
    Represents the Reversi game board and its state.
    """
    
    # Directions: (row, col) deltas for all 8 possible directions
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Board values
    EMPTY = 0
    BLACK = 1  # Player 1
    WHITE = 2  # Player 2
    
    def __init__(self, size: int = 8):
        """
        Initialize a new game board.
        
        Args:
            size: Size of the board (default: 8 for standard Reversi)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self._setup_initial_board()
        self.current_player = self.BLACK
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def _setup_initial_board(self) -> None:
        """Set up the initial board configuration."""
        mid = self.size // 2
        self.board[mid-1][mid-1] = self.WHITE
        self.board[mid][mid] = self.WHITE
        self.board[mid-1][mid] = self.BLACK
        self.board[mid][mid-1] = self.BLACK
    
    def get_valid_moves(self, player: int = None) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the specified player.
        
        Args:
            player: The player to get moves for (BLACK or WHITE)
            
        Returns:
            List of (row, col) tuples representing valid moves
        """
        if player is None:
            player = self.current_player
            
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        valid_moves = set()
        
        # Find all empty cells that are adjacent to opponent's pieces
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != self.EMPTY:
                    continue
                    
                # Check all directions from this empty cell
                for dr, dc in self.DIRECTIONS:
                    r, c = row + dr, col + dc
                    found_opponent = False
                    
                    # Move in direction until we find player's piece or an empty cell/edge
                    while 0 <= r < self.size and 0 <= c < self.size:
                        if self.board[r][c] == opponent:
                            found_opponent = True
                        elif self.board[r][c] == player and found_opponent:
                            valid_moves.add((row, col))
                            break
                        else:  # Empty or same player without opponent in between
                            break
                        r += dr
                        c += dc
        
        # If no valid moves, check if opponent has any valid moves
        if not valid_moves:
            # Check if opponent has any valid moves
            opponent_has_moves = False
            for row in range(self.size):
                for col in range(self.size):
                    if self.board[row][col] == opponent:
                        for dr, dc in self.DIRECTIONS:
                            r, c = row + dr, col + dc
                            if (0 <= r < self.size and 0 <= c < self.size and 
                                self.board[r][c] == self.EMPTY):
                                # Check if this is a valid move for opponent
                                r2, c2 = row - dr, col - dc
                                found_player = False
                                while 0 <= r2 < self.size and 0 <= c2 < self.size:
                                    if self.board[r2][c2] == opponent:
                                        found_player = True
                                        break
                                    elif self.board[r2][c2] == player:
                                        if found_player:
                                            opponent_has_moves = True
                                            break
                                        else:
                                            break
                                    else:  # Empty
                                        break
                                    r2 -= dr
                                    c2 -= dc
                                if opponent_has_moves:
                                    break
                    if opponent_has_moves:
                        break
                if opponent_has_moves:
                    break
            
            if opponent_has_moves:
                return [(-1, -1)]  # Pass move
        
        return list(valid_moves)
    
    def has_any_valid_move(self, player: int = None) -> bool:
        """Check if the player has any valid moves."""
        if player is None:
            player = self.current_player
            
        # Get valid moves without checking opponent's moves to prevent recursion
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        
        # Check for any valid move for the player
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.EMPTY:
                    # Check all directions from this empty cell
                    for dr, dc in self.DIRECTIONS:
                        r, c = row + dr, col + dc
                        found_opponent = False
                        
                        # Move in direction until we find player's piece or an empty cell/edge
                        while 0 <= r < self.size and 0 <= c < self.size:
                            if self.board[r][c] == opponent:
                                found_opponent = True
                            elif self.board[r][c] == player and found_opponent:
                                return True  # Found at least one valid move
                            else:  # Empty or same player without opponent in between
                                break
                            r += dr
                            c += dc
        
        return False  # No valid moves found
    
    def make_move(self, row: int, col: int, player: int = None) -> bool:
        """
        Make a move on the board.
        
        Args:
            row: Row of the move (0-based)
            col: Column of the move (0-based)
            player: The player making the move (BLACK or WHITE)
            
        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if player is None:
            player = self.current_player
        
        # Handle pass move
        if (row, col) == (-1, -1):
            if not self.get_valid_moves(player):
                self.current_player = self.WHITE if player == self.BLACK else self.BLACK
                self.move_history.append((-1, -1))
                return True
            return False
        
        # Check if move is valid
        if (row, col) not in self.get_valid_moves(player):
            return False
        
        # Place the piece
        self.board[row][col] = player
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        
        # Flip opponent's pieces
        flipped = set()
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
                
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    flipped.update(to_flip)
                    break
        
        # Flip all captured pieces
        for r, c in flipped:
            self.board[r][c] = player
        
        # Update game state
        self.move_history.append((row, col))
        self.current_player = opponent
        
        # Check if game is over
        self._check_game_over()
        
        return True
    
    def _check_game_over(self) -> None:
        """Check if the game is over and update game state."""
        # Check if the board is full
        if np.all(self.board != self.EMPTY):
            self.game_over = True
            self._determine_winner()
            return
            
        # Check if current player has any valid moves
        current_moves = self.get_valid_moves(self.current_player)
        
        # If current player has no valid moves, check opponent
        if not current_moves or current_moves == [(-1, -1)]:
            opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK
            opponent_moves = self.get_valid_moves(opponent)
            
            # If opponent has valid moves, just switch turns
            if opponent_moves and opponent_moves != [(-1, -1)]:
                self.current_player = opponent
            # If neither player has valid moves, game is over
            else:
                self.game_over = True
                self._determine_winner()
    
    def _determine_winner(self) -> None:
        """Determine the winner based on piece count."""
        black_count = np.sum(self.board == self.BLACK)
        white_count = np.sum(self.board == self.WHITE)
        
        if black_count > white_count:
            self.winner = self.BLACK
        elif white_count > black_count:
            self.winner = self.WHITE
        else:
            self.winner = 0  # Draw
    
    def get_score(self) -> Tuple[int, int]:
        """
        Get the current score (black, white).
        
        Returns:
            Tuple of (black_score, white_score)
        """
        black = np.sum(self.board == self.BLACK)
        white = np.sum(self.board == self.WHITE)
        return black, white
    
    def is_valid_move(self, row: int, col: int, player: int = None) -> bool:
        """Check if a move is valid."""
        if player is None:
            player = self.current_player
        return (row, col) in self.get_valid_moves(player)
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.move_history = self.move_history.copy()
        return new_board
        
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {self.EMPTY: '.', self.BLACK: 'B', self.WHITE: 'W'}
        rows = []
        
        # Add column headers
        rows.append('  ' + ' '.join(str(i) for i in range(self.size)))
        
        for i in range(self.size):
            row = [str(i)]
            for j in range(self.size):
                row.append(symbols[self.board[i][j]])
            rows.append(' '.join(row))
        
        # Add current player and score
        rows.append(f"\nCurrent player: {'Black' if self.current_player == self.BLACK else 'White'}")
        black, white = self.get_score()
        rows.append(f"Score - Black: {black}, White: {white}")
        
        if self.game_over:
            if self.winner == 0:
                rows.append("Game over! It's a draw!")
            else:
                winner = 'Black' if self.winner == self.BLACK else 'White'
                rows.append(f"Game over! {winner} wins!")
        
        return '\n'.join(rows)
