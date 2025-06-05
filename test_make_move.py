from src.game.board import Board

def test_make_move():
    b = Board()
    print(f"make_move in dir(Board): {'make_move' in dir(Board)}")
    print(f"make_move in dir(b): {'make_move' in dir(b)}")
    print(f"make_move in Board.__dict__: 'make_move' in Board.__dict__")
    print(f"Attributes of Board: {[attr for attr in dir(Board) if not attr.startswith('__')]}")

if __name__ == "__main__":
    test_make_move()
