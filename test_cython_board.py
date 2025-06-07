import sys
import time
import numpy as np
from src.game.board_optimized import Board

def test_performance():
    # Test the performance of the board implementation
    print(f"Using {Board.get_implementation()} implementation")
    
    # Create a board
    board = Board()
    
    # Test move generation
    start_time = time.time()
    num_moves = 100_000
    
    for _ in range(num_moves):
        # Get valid moves
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            # No valid moves, pass
            board.make_move(-1, -1)
        else:
            # Make the first valid move
            row, col = valid_moves[0]
            board.make_move(row, col)
            
        # Check if game is over
        if board.game_over:
            board = Board()
    
    elapsed = time.time() - start_time
    moves_per_sec = num_moves / elapsed if elapsed > 0 else float('inf')
    
    print(f"Processed {num_moves} moves in {elapsed:.2f} seconds")
    print(f"{moves_per_sec:.2f} moves per second")
    print("Final board:")
    print(board)

if __name__ == "__main__":
    test_performance()
