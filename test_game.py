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
    # Create a small board for testing
    game = ReversiGame(4)
    
    # Set up a board that's one move away from being full
    # B = Black, W = White, . = Empty
    # Current board:
    # W W W W
    # W W W W
    # W W W W
    # W W B .  <- One empty space left, White's turn
    
    # Fill the board with white pieces except for one spot and one black piece
    game.board.board = np.full((4, 4), game.board.WHITE, dtype=np.int8)
    game.board.board[3, 2] = game.board.BLACK  # One black piece that can be captured
    game.board.board[3, 3] = game.board.EMPTY  # One empty space
    game.current_player = game.board.WHITE  # White's turn
    game.game_over = False
    
    print("\nInitial board:")
    print(game)
    
    # Make the final move to fill the board - this should capture the black piece
    print("\nMaking final move to fill the board...")
    # White places a piece in the empty space, capturing the black piece
    assert game.make_move(3, 3), "Final move should be valid"
    
    print("\nFinal board:")
    print(game)
    print(f"Game over: {game.is_game_over()}")
    print(f"Winner: {game.get_winner()}")
    print(f"Score: {game.get_score()}")
    
    # The game should be over after filling the board
    assert game.is_game_over(), "Game should be over after filling the board"
    
    # The winner should be White since the board is all white
    black_count = np.sum(game.board.board == game.board.BLACK)
    white_count = np.sum(game.board.board == game.board.WHITE)
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
