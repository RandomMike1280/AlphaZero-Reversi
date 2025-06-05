"""
Test script for the Reversi game implementation.
"""
from src.game.game import ReversiGame
import numpy as np

def test_initial_board():
    """Test the initial board setup."""
    game = ReversiGame()
    board = game.get_board_state()
    
    # Check board size
    assert board.shape == (8, 8), "Board should be 8x8"
    
    # Check initial pieces
    mid = 3  # 8//2 - 1
    assert board[mid][mid] == 2      # White
    assert board[mid+1][mid+1] == 2  # White
    assert board[mid][mid+1] == 1    # Black
    assert board[mid+1][mid] == 1    # Black
    
    # Check empty squares
    empty_count = np.sum(board == 0)
    assert empty_count == 60, "Should have 60 empty squares initially"
    
    print("Initial board test passed!")


def test_valid_moves():
    """Test valid move generation."""
    game = ReversiGame()
    valid_moves = game.get_valid_moves()
    
    # Black's valid moves in the initial position
    expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
    assert set(valid_moves) == set(expected_moves), \
        f"Expected valid moves {expected_moves}, got {valid_moves}"
    
    print("Valid moves test passed!")


def test_make_move():
    """Test making moves and capturing pieces."""
    game = ReversiGame()
    
    # Make a valid move
    assert game.make_move(2, 3), "Should be a valid move"
    board = game.get_board_state()
    
    # Check if pieces were captured
    assert board[2][3] == 1, "Move should place black piece"
    assert board[3][3] == 1, "Should capture white piece"
    
    # Check current player changed to white
    assert game.get_current_player() == 2, "Should be white's turn"
    
    print("Make move test passed!")


def test_game_over():
    """Test game over condition by filling the board completely."""
    # Create a standard 8x8 board for testing
    game = ReversiGame(8)
    
    # Set up a board where there's only one empty square and one valid move
    # that will end the game
    
    # First, reset the board to a known state
    game.board.black = 0x0000000000000000
    game.board.white = 0x0000000000000000
    game.current_player = game.board.BLACK
    game.game_over = False
    
    # Set up a position where there's only one empty square at (0,0)
    # and one black piece at (0,1) that can be captured by white
    game.board.black = 0x0000000000000002  # Black piece at (0,1)
    game.board.white = 0x0000000000000000  # No white pieces initially
    
    # Fill the rest of the board with alternating colors
    for i in range(8):
        for j in range(8):
            if i > 0 or j > 1:  # Skip the first two positions we just set
                pos = i * 8 + j
                if (i + j) % 2 == 0:
                    game.board.white |= (1 << pos)
                else:
                    game.board.black |= (1 << pos)
    
    # Make sure (0,0) is empty and (0,1) is black
    game.board.black &= ~0x0000000000000001  # Clear (0,0)
    game.board.white &= ~0x0000000000000002  # Clear (0,1)
    game.board.black |= 0x0000000000000002   # Set (0,1) to black
    
    # It's white's turn to move
    game.current_player = game.board.WHITE
    
    # Update the board state
    game.board._update_board_state()
    
    print("\nInitial board:")
    print(game)
    
    # Make the final move to fill the board - this should capture the black piece
    assert game.make_move(0, 0), "Final move should be valid"
    
    print("\nFinal board:")
    print(game)
    print(f"Game over: {game.is_game_over()}")
    print(f"Winner: {game.get_winner()}")
    print(f"Score: {game.get_score()}")
    
    # The game should be over after filling the board
    assert game.is_game_over(), "Game should be over after filling the board"
    
    # The winner should be White since the board is all white
    black_count = game.board.bit_count(game.board.black)
    white_count = game.board.bit_count(game.board.white)
    expected_winner = game.board.WHITE  # Should be all white pieces
    
    print(f"\nFinal score - Black: {black_count}, White: {white_count}")
    print(f"Expected winner: {'Black' if expected_winner == game.board.BLACK else 'White' if expected_winner == game.board.WHITE else 'Draw'}")
    
    assert game.get_winner() == expected_winner, \
        f"Winner should be {expected_winner}, got {game.get_winner()}"
    
    print("Game over test passed!")


if __name__ == "__main__":
    print("Running Reversi game tests...\n")
    
    test_initial_board()
    test_valid_moves()
    test_make_move()
    test_game_over()
    
    print("\nAll tests passed successfully!")
