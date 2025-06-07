# distutils: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t, int8_t, int64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Constants
cdef int BOARD_SIZE = 8
cdef int NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
cdef int BLACK = 1
cdef int WHITE = 2
cdef int EMPTY = 0

# Direction offsets for move generation
cdef int[8] DIRECTIONS = [
    -9, -8, -7,  # NW, N, NE
    -1,       1,  # W,  E
     7,  8,  9   # SW, S, SE
]

# Bitboard masks for edge detection
cdef uint64_t NOT_A_FILE = 0xFEFEFEFEFEFEFEFE  # ~0x0101010101010101
cdef uint64_t NOT_H_FILE = 0x7F7F7F7F7F7F7F7F  # ~0x8080808080808080

cdef class Board:
    cdef uint64_t black
    cdef uint64_t white
    cdef int current_player
    cdef int game_over
    cdef int winner
    cdef int passed_moves_in_a_row
    
    def __cinit__(self, size=8):
        if size != 8:
            raise ValueError("Only 8x8 board is supported")
        self.black = 0x0000000810000000  # Initial black pieces
        self.white = 0x0000001008000000  # Initial white pieces
        self.current_player = BLACK
        self.game_over = 0
        self.winner = 0
        self.passed_moves_in_a_row = 0
    
    cpdef bint make_move(self, int row, int col, int player=-1):
        """Make a move on the board."""
        if player == -1:
            player = self.current_player
            
        # Handle pass move
        if row == -1 and col == -1:
            if len(self.get_valid_moves()) > 0:
                return False
                
            self.passed_moves_in_a_row += 1
            self.current_player = 3 - self.current_player
            
            if self.passed_moves_in_a_row >= 2:
                self.game_over = 1
                self._determine_winner()
            return True
            
        # Convert row,col to bit position
        cdef int pos = row * 8 + col
        cdef uint64_t move_bit = <uint64_t>1 << (63 - pos)
        
        # Check if the move is valid
        cdef uint64_t valid_moves = self._get_valid_moves_bitboard()
        if not (move_bit & valid_moves):
            return False
            
        # Reset passed moves counter
        self.passed_moves_in_a_row = 0
        
        # Get the player's and opponent's bitboards
        cdef uint64_t player_bb = self.black if player == BLACK else self.white
        cdef uint64_t opponent_bb = self.white if player == BLACK else self.black
        
        # Calculate all pieces to flip
        cdef uint64_t flip_mask = self._get_flip_mask(move_bit, player_bb, opponent_bb)
        
        # Update bitboards
        if player == BLACK:
            self.black ^= move_bit | flip_mask
            self.white ^= flip_mask
        else:
            self.white ^= move_bit | flip_mask
            self.black ^= flip_mask
            
        # Switch player
        self.current_player = 3 - self.current_player
        
        # Check if the next player has any valid moves
        if not self._check_game_over() and self._get_valid_moves_bitboard() == 0:
            self.passed_moves_in_a_row += 1
            self.current_player = 3 - self.current_player
            
            if self.passed_moves_in_a_row >= 2:
                self.game_over = 1
                self._determine_winner()
                
        return True
    
    cpdef list get_valid_moves(self):
        """Get all valid moves for the current player."""
        cdef uint64_t valid_moves_bb = self._get_valid_moves_bitboard()
        cdef list moves = []
        cdef int i
        
        for i in range(64):
            if valid_moves_bb & (<uint64_t>1 << (63 - i)):
                moves.append((i // 8, i % 8))
                
        return moves
    
    cdef uint64_t _get_valid_moves_bitboard(self):
        """Get valid moves as a bitboard."""
        cdef uint64_t player_bb = self.black if self.current_player == BLACK else self.white
        cdef uint64_t opponent_bb = self.white if self.current_player == BLACK else self.black
        cdef uint64_t empty = ~(self.black | self.white)
        cdef uint64_t valid_moves = 0
        cdef int direction
        
        for direction in DIRECTIONS:
            valid_moves |= self._find_valid_direction(player_bb, opponent_bb, empty, direction)
            
        return valid_moves
    
    cdef uint64_t _find_valid_direction(self, uint64_t player_bb, uint64_t opponent_bb, 
                                      uint64_t empty, int direction):
        """Find valid moves in a specific direction."""
        cdef uint64_t candidates, flip, valid = 0
        
        if direction in {1, -7, 9}:  # E, SW, SE
            candidates = opponent_bb & ((player_bb >> -direction) & NOT_H_FILE)
            flip = (candidates >> -direction) & NOT_H_FILE
            valid |= empty & flip
            flip = (flip >> -direction) & opponent_bb
            
            while flip != 0:
                valid |= empty & flip
                flip = (flip >> -direction) & opponent_bb & NOT_H_FILE
                
        elif direction in {-1, 7, -9}:  # W, NE, NW
            candidates = opponent_bb & ((player_bb << -direction) & NOT_A_FILE)
            flip = (candidates << -direction) & NOT_A_FILE
            valid |= empty & flip
            flip = (flip << -direction) & opponent_bb
            
            while flip != 0:
                valid |= empty & flip
                flip = (flip << -direction) & opponent_bb & NOT_A_FILE
                
        else:  # N, S
            candidates = opponent_bb & (player_bb >> -direction)
            flip = candidates >> -direction
            valid |= empty & flip
            flip = (flip >> -direction) & opponent_bb
            
            while flip != 0:
                valid |= empty & flip
                flip = (flip >> -direction) & opponent_bb
                
        return valid
    
    cdef uint64_t _get_flip_mask(self, uint64_t move_bit, uint64_t player_bb, uint64_t opponent_bb):
        """Get the mask of pieces to flip for a move."""
        cdef uint64_t flip_mask = 0
        cdef int direction
        
        for direction in DIRECTIONS:
            flip_mask |= self._find_flips_in_direction(move_bit, player_bb, opponent_bb, direction)
            
        return flip_mask
    
    cdef uint64_t _find_flips_in_direction(self, uint64_t move_bit, uint64_t player_bb, 
                                          uint64_t opponent_bb, int direction):
        """Find flips in a specific direction."""
        cdef uint64_t flip = 0
        cdef uint64_t temp = 0
        cdef uint64_t mask = 0
        
        if direction in {1, -7, 9}:  # E, SW, SE
            mask = NOT_H_FILE
            temp = (move_bit >> -direction) & mask & opponent_bb
            
            while temp != 0:
                flip |= temp
                temp = (temp >> -direction) & mask & opponent_bb
                
            if (temp & player_bb) == 0:
                return 0
                
        elif direction in {-1, 7, -9}:  # W, NE, NW
            mask = NOT_A_FILE
            temp = (move_bit << -direction) & mask & opponent_bb
            
            while temp != 0:
                flip |= temp
                temp = (temp << -direction) & mask & opponent_bb
                
            if (temp & player_bb) == 0:
                return 0
                
        else:  # N, S
            temp = (move_bit >> -direction) & opponent_bb
            
            while temp != 0:
                flip |= temp
                temp = (temp >> -direction) & opponent_bb
                
            if (temp & player_bb) == 0:
                return 0
                
        return flip
    
    cdef bint _check_game_over(self):
        """Check if the game is over."""
        if self.game_over:
            return True
            
        # Game is over if no valid moves for either player
        cdef int current = self.current_player
        
        # Check if current player has moves
        if self._get_valid_moves_bitboard() != 0:
            return False
            
        # Switch player and check
        self.current_player = 3 - current
        cdef bint opponent_has_moves = (self._get_valid_moves_bitboard() != 0)
        self.current_player = current
        
        if not opponent_has_moves:
            self.game_over = 1
            self._determine_winner()
            return True
            
        return False
    
    cdef void _determine_winner(self):
        """Determine the winner based on piece counts."""
        cdef int black_count = self._count_bits(self.black)
        cdef int white_count = self._count_bits(self.white)
        
        if black_count > white_count:
            self.winner = BLACK
        elif white_count > black_count:
            self.winner = WHITE
        else:
            self.winner = 0  # Draw
    
    cdef int _count_bits(self, uint64_t x):
        """Count the number of set bits in a 64-bit integer."""
        cdef int count = 0
        while x:
            count += 1
            x &= x - 1
        return count
    
    def get_board_state(self):
        """Return the board state as a numpy array."""
        cdef np.ndarray[np.int8_t, ndim=2] board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        cdef int i, row, col
        
        for i in range(64):
            row, col = divmod(i, 8)
            if (self.black >> (63 - i)) & 1:
                board[row, col] = BLACK
            elif (self.white >> (63 - i)) & 1:
                board[row, col] = WHITE
                
        return board
    
    def copy(self):
        """Create a deep copy of the board."""
        cdef Board new_board = Board()
        new_board.black = self.black
        new_board.white = self.white
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.passed_moves_in_a_row = self.passed_moves_in_a_row
        return new_board
    
    def __str__(self):
        """String representation of the board."""
        cdef str s = ""
        cdef int i, row, col
        cdef uint64_t pos
        
        s += "  " + " ".join(str(i) for i in range(BOARD_SIZE)) + "\n"
        
        for row in range(BOARD_SIZE):
            s += str(row) + " "
            for col in range(BOARD_SIZE):
                pos = <uint64_t>1 << (63 - (row * 8 + col))
                if self.black & pos:
                    s += "B "
                elif self.white & pos:
                    s += "W "
                else:
                    s += ". "
            s += "\n"
            
        return s
