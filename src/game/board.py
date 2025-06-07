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
        Get all valid moves for the given player using bitboard operations.
        
        Args:
            player: The player to get valid moves for. If None, uses current player.
            
        Returns:
            List of (row, col) tuples representing valid moves
        """
        if player is None:
            player = self.current_player
            
        # Get the bitboard for the current player and opponent
        player_board = self.black if player == self.BLACK else self.white
        opponent_board = self.white if player == self.BLACK else self.black
        empty_squares = ~(self.black | self.white) & 0xFFFFFFFFFFFFFFFF  # Mask to 64 bits
        
        # Directions: E, W, S, N, SE, NW, SW, NE
        directions = [
            (1, 0),    # East
            (-1, 0),   # West
            (0, 1),    # South
            (0, -1),   # North
            (1, 1),    # South-East
            (-1, -1),  # North-West
            (-1, 1),   # South-West
            (1, -1)    # North-East
        ]
        
        valid_moves_bb = 0
        
        for dx, dy in directions:
            # Calculate the shift amount
            shift = dx + dy * 8
            
            # Get the player's pieces that have opponent's pieces in the given direction
            candidates = 0
            if shift > 0:
                candidates = (player_board << shift) & opponent_board
            else:
                candidates = (player_board >> -shift) & opponent_board
            
            # Propagate in the direction until we hit an empty square or the edge of the board
            for _ in range(5):  # Maximum of 5 steps needed on an 8x8 board
                if shift > 0:
                    candidates |= ((candidates << shift) & opponent_board)
                else:
                    candidates |= ((candidates >> -shift) & opponent_board)
            
            # The valid moves are the empty squares adjacent to the end of the line of opponent's pieces
            if shift > 0:
                valid_moves_bb |= (candidates << shift) & empty_squares
            else:
                valid_moves_bb |= (candidates >> -shift) & empty_squares
        
        # Convert the bitboard to a list of (row, col) tuples
        valid_moves = []
        for i in range(64):
            if valid_moves_bb & (1 << i):
                row, col = divmod(i, 8)
                valid_moves.append((row, col))
                
        return valid_moves
    
    def make_move(self, row: int, col: int, player: int = None) -> bool:
        """
        Make a move on the board using bitboard operations.
        
        Args:
            row: Row of the move (0-based), or -1 for pass
            col: Column of the move (0-based), or -1 for pass
            player: The player making the move (BLACK or WHITE). If None, uses current_player
                
        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if player is None:
            player = self.current_player
            
        # Handle pass move
        if row == -1 and col == -1:
            # Check if passing is allowed (player has no valid moves)
            if self.get_valid_moves(player):  # Using our optimized get_valid_moves
                return False
                
            # Record the pass
            self.passed_moves_in_a_row += 1
            self.move_history.append((row, col, player))
            
            # Switch player
            self.current_player = 3 - player  # Toggle between BLACK (1) and WHITE (2)
            
            # Check if game is over (two passes in a row)
            if self.passed_moves_in_a_row >= 2:
                self.game_over = True
                self._determine_winner()
            return True
            
        # Handle normal move
        move_bit = 1 << (row * 8 + col)
        
        # Check if the move is valid using the fast bitboard method
        valid_moves = self.get_valid_moves(player)
        valid_moves_bb = 0
        for r, c in valid_moves:
            valid_moves_bb |= 1 << (r * 8 + c)
            
        if not (move_bit & valid_moves_bb):
            return False
            
        # Reset passed moves counter
        self.passed_moves_in_a_row = 0
        
        # Get the player's and opponent's bitboards
        if player == self.BLACK:
            player_board = self.black
            opponent_board = self.white
        else:
            player_board = self.white
            opponent_board = self.black
            
        # Calculate all pieces to flip
        flip_mask = 0
        
        # Directions: E, W, S, N, SE, NW, SW, NE
        directions = [1, -1, 8, -8, 7, -7, 9, -9]
        
        # Edge masks to prevent wrapping around the board edges
        edge_masks = {
            1: 0xFEFEFEFEFEFEFEFE,  # East (no wrap from right to left)
            -1: 0x7F7F7F7F7F7F7F7F,  # West (no wrap from left to right)
            7: 0xFEFEFEFEFEFEFEFE,   # SW (no wrap from right to left)
            -7: 0x7F7F7F7F7F7F7F7F,  # NE (no wrap from left to right)
            9: 0x7F7F7F7F7F7F7F7F,   # SE (no wrap from right to left)
            -9: 0xFEFEFEFEFEFEFEFE   # NW (no wrap from left to right)
        }

        for d in directions:
            line_mask = 0
            curr = move_bit
            edge_mask = edge_masks.get(abs(d), 0xFFFFFFFFFFFFFFFF)
            
            # Shift and check for opponent pieces
            for _ in range(self.size - 1):
                curr = (curr << d) if d > 0 else (curr >> -d)
                if not (curr & opponent_board & edge_mask):
                    break  # Stop if we hit an empty square or our own piece
                line_mask |= curr
                
            # If the line ends with one of our pieces, add it to the flip_mask
            if (curr & player_board & edge_mask):
                flip_mask |= line_mask

        # Update bitboards with XOR
        if player == self.BLACK:
            self.black ^= move_bit | flip_mask
            self.white ^= flip_mask
        else:
            self.white ^= move_bit | flip_mask
            self.black ^= flip_mask
            
        # Record the move
        self.move_history.append((row, col, player))
            
        # Switch player and check for game over conditions
        self.current_player = 3 - player
        
        # Update the numpy board representation
        self._update_board_state()
        
        # If the new player has no moves, pass the turn
        if not self.get_valid_moves(self.current_player):
            self.current_player = 3 - self.current_player
            self.passed_moves_in_a_row += 1
            
            # If this player also has no moves, game is over
            if not self.get_valid_moves(self.current_player):
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
            
        # Use the optimized get_valid_moves method
        return len(self.get_valid_moves(player)) > 0
    
    def _get_flipped_pieces(self, move: Tuple[int, int], player: int):
        """
        Get the list of pieces that would be flipped by a move.
        This is kept for backward compatibility but is not used by the optimized make_move.
        """
        row, col = move
        move_bit = 1 << (row * 8 + col)
        
        if player == self.BLACK:
            player_board = self.black
            opponent_board = self.white
        else:
            player_board = self.white
            opponent_board = self.black
            
        flip_mask = 0
        
        # Directions: E, W, S, N, SE, NW, SW, NE
        directions = [1, -1, 8, -8, 7, -7, 9, -9]
        
        # Edge masks to prevent wrapping around the board edges
        edge_masks = {
            1: 0xFEFEFEFEFEFEFEFE,  # East (no wrap from right to left)
            -1: 0x7F7F7F7F7F7F7F7F,  # West (no wrap from left to right)
            7: 0xFEFEFEFEFEFEFEFE,   # SW (no wrap from right to left)
            -7: 0x7F7F7F7F7F7F7F7F,  # NE (no wrap from left to right)
            9: 0x7F7F7F7F7F7F7F7F,   # SE (no wrap from right to left)
            -9: 0xFEFEFEFEFEFEFEFE   # NW (no wrap from left to right)
        }

        for d in directions:
            line_mask = 0
            curr = move_bit
            edge_mask = edge_masks.get(abs(d), 0xFFFFFFFFFFFFFFFF)
            
            # Shift and check for opponent pieces
            for _ in range(self.size - 1):
                curr = (curr << d) if d > 0 else (curr >> -d)
                if not (curr & opponent_board & edge_mask):
                    break  # Stop if we hit an empty square or our own piece
                line_mask |= curr
                
            # If the line ends with one of our pieces, add it to the flip_mask
            if (curr & player_board & edge_mask):
                flip_mask |= line_mask
        
        # Convert flip_mask to list of (row, col) tuples
        flipped = []
        for i in range(64):
            if flip_mask & (1 << i):
                row, col = divmod(i, 8)
                flipped.append((row, col))
                
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
