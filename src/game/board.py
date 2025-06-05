"""
Board module for Reversi.
Handles the game board state, move validation, and Zobrist hashing.
Uses bitboard representation for optimal performance.
"""
from typing import List, Tuple, Optional
import numpy as np
import random

class Board:
    """
    Represents the Reversi game board using bitboard representation.
    Each player's pieces are stored in a 64-bit integer (for 8x8 board).
    """
    
    # Board dimensions
    SIZE = 8
    BOARD_SIZE = SIZE * SIZE
    
    # Player constants
    EMPTY = 0
    BLACK = 1  # Player 1
    WHITE = 2  # Player 2
    
    def __init__(self, size: int = 8):
        """Initialize a new Reversi board."""
        if size != 8:
            raise ValueError("Only 8x8 board is supported")
            
        self.size = size
        self.black = 0x0000000810000000  # Initial black pieces
        self.white = 0x0000001008000000  # Initial white pieces
        self.current_player = self.BLACK
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.passed_moves_in_a_row = 0
        self._board = np.zeros((size, size), dtype=int)
        self._update_board_state()
    
    def _ensure_board_updated(self) -> None:
        """Ensure the numpy array representation is up to date."""
        self._update_board_state()
        
    def _update_board_state(self) -> None:
        """Update the numpy array representation of the board from bitboards."""
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                pos = 1 << (i * 8 + j)
                if self.black & pos:
                    self._board[i, j] = self.BLACK
                elif self.white & pos:
                    self._board[i, j] = self.WHITE
                else:
                    self._board[i, j] = self.EMPTY
                    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(self.size)
        new_board.black = self.black
        new_board.white = self.white
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.move_history = self.move_history.copy()
        new_board.passed_moves_in_a_row = self.passed_moves_in_a_row
        new_board._update_board_state()
        return new_board
        
    def get_valid_moves(self, player: int = None) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the given player.
        
        Args:
            player: The player to get valid moves for. If None, uses current player.
            
        Returns:
            List of (row, col) tuples representing valid moves
        """
        if player is None:
            player = self.current_player
            
        valid_moves = []
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, row: int, col: int, player: int = None) -> bool:
        """
        Make a move on the board.
        
        Args:
            row: Row of the move (0-based), or -1 for pass
            col: Column of the move (0-based), or -1 for pass
            player: The player making the move (BLACK or WHITE). If None, uses current_player
                
        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if self.game_over:
            return False
                
        # Handle pass move
        if row == -1 and col == -1:
            # Check if passing is allowed (player has no valid moves)
            if self.has_any_valid_move(player or self.current_player):
                return False  # Player must make a move if possible
                    
            # Record the pass
            self.passed_moves_in_a_row += 1
            self.move_history.append((row, col, player or self.current_player))
            
            # Check if game is over (two passes in a row)
            if self.passed_moves_in_a_row >= 2:
                self.game_over = True
                self._determine_winner()
            else:
                # Switch player
                self.current_player = self.WHITE if (player or self.current_player) == self.BLACK else self.BLACK
            
            return True
            
        # Handle normal move
        if player is None:
            player = self.current_player
            
        # Check if the move is valid
        if not self.is_valid_move(row, col, player):
            return False
            
        # Reset passed moves counter
        self.passed_moves_in_a_row = 0
        
        # Make the move
        square = row * 8 + col
        move_bit = 1 << square
        
        # Get the pieces that would be flipped
        flipped = self._get_flipped_pieces((row, col), player)
        flip_bits = 0
        for r, c in flipped:
            flip_bits |= 1 << (r * 8 + c)
            
        # Update the bitboards
        if player == self.BLACK:
            self.black ^= move_bit | flip_bits
            self.white ^= flip_bits
        else:
            self.white ^= move_bit | flip_bits
            self.black ^= flip_bits
        
        # Switch player
        self.current_player = self.WHITE if player == self.BLACK else self.BLACK
        
        # Update board state
        self._update_board_state()
        
        # Check if the next player has any valid moves
        if not self._check_game_over() and not self.has_any_valid_move(self.current_player):
            self.passed_moves_in_a_row += 1
            self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
            
            if self.passed_moves_in_a_row >= 2:
                self.game_over = True
                self._determine_winner()
        
        return True
    
    def is_valid_move(self, row: int, col: int, player: int = None) -> bool:
        """Check if a move is valid."""
        if player is None:
            player = self.current_player
            
        # Check if the position is empty and within bounds
        if (row < 0 or row >= self.SIZE or col < 0 or col >= self.SIZE or 
            self._board[row, col] != self.EMPTY):
            return False
            
        # Check if the move would flip any opponent's pieces
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if (r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE or 
                self._board[r, c] != opponent):
                continue
                
            r += dr
            c += dc
            while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
                if self._board[r, c] == player:
                    return True
                if self._board[r, c] == self.EMPTY:
                    break
                r += dr
                c += dc
                
        return False
    
    def has_any_valid_move(self, player: int = None) -> bool:
        """Check if the player has any valid moves."""
        if player is None:
            player = self.current_player
            
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.is_valid_move(i, j, player):
                    return True
        return False
    
    def _get_flipped_pieces(self, move: Tuple[int, int], player: int) -> List[Tuple[int, int]]:
        """Get the list of pieces that would be flipped by a move."""
        row, col = move
        if self._board[row, col] != self.EMPTY:
            return []
            
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        flipped = []
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
                if self._board[r, c] == opponent:
                    to_flip.append((r, c))
                elif self._board[r, c] == player:
                    flipped.extend(to_flip)
                    break
                else:  # Empty
                    break
                    
                r += dr
                c += dc
                
        return flipped
    
    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        if self.game_over:
            return True
            
        # Game is over if no valid moves for either player
        if not self.has_any_valid_move(self.BLACK) and not self.has_any_valid_move(self.WHITE):
            self.game_over = True
            self._determine_winner()
            return True
            
        return False
    
    def _determine_winner(self) -> None:
        """Determine the winner based on piece counts."""
        black_count = bin(self.black).count('1')
        white_count = bin(self.white).count('1')
        
        if black_count > white_count:
            self.winner = self.BLACK
        elif white_count > black_count:
            self.winner = self.WHITE
        else:
            self.winner = 0  # Draw
    
    def get_board_state(self) -> np.ndarray:
        """Return a copy of the current board state."""
        return self._board.copy()
    
    def __str__(self) -> str:
        """Return a string representation of the board."""
        symbols = {self.EMPTY: '.', self.BLACK: 'B', self.WHITE: 'W'}
        rows = []
        for i in range(self.SIZE):
            row = [symbols[self._board[i, j]] for j in range(self.SIZE)]
            rows.append(' '.join(row))
        
        status = ["\n".join(rows)]
        status.append(f"Current player: {'Black' if self.current_player == self.BLACK else 'White'}")
        
        black_count = bin(self.black).count('1')
        white_count = bin(self.white).count('1')
        status.append(f"Score - Black: {black_count}, White: {white_count}")
        
        if self.game_over:
            if self.winner == 0:
                status.append("Game over! It's a draw!")
            else:
                winner = 'Black' if self.winner == self.BLACK else 'White'
                status.append(f"Game over! {winner} wins!")
        
        return "\n".join(status)

    # For backward compatibility
    def __call__(self, row: int, col: int, player: int = None) -> bool:
        return self.make_move(row, col, player)
        
    def get_board_state(self) -> np.ndarray:
        """
        Get the current board state as a numpy array.
        
        Returns:
            2D numpy array representing the board state
        """
        self._ensure_board_updated()
        return self._board.copy()
        
    def get_score(self) -> Tuple[int, int]:
        """
        Get the current score (black, white).
        
        Returns:
            Tuple of (black_score, white_score)
        """
        black_count = self.bit_count(self.black)
        white_count = self.bit_count(self.white)
        return (black_count, white_count)
        
    @staticmethod
    def bit_count(x: int) -> int:
        """Count the number of set bits in a 64-bit integer."""
        # This is a fast implementation of bit counting (Hamming weight)
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
        return ((x * 0x0101010101010101) & 0xffffffffffffffff) >> 56
