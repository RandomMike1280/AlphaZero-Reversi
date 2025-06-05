from src.game.board_new import Board

def test_make_move():
    b = Board()
    print(f"make_move in dir(Board): {'make_move' in dir(Board)}")
    print(f"make_move in dir(b): {'make_move' in dir(b)}")
    print(f"make_move in Board.__dict__: {'make_move' in Board.__dict__}")
    
    # Test a valid move
    print("\nTesting valid move (2, 3):", b.make_move(2, 3, Board.BLACK))
    print(b)
    
    # Test invalid move
    print("\nTesting invalid move (0, 0):", b.make_move(0, 0, Board.BLACK))
    
    # Test pass move when no valid moves
    print("\nTesting pass move:", b.make_move(-1, -1, Board.WHITE))

if __name__ == "__main__":
    test_make_move()
