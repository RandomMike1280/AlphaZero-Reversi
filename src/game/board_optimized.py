from .cython_implementation import CythonBoard, BOARD_IMPL


class Board(CythonBoard):
    """
    Optimized Board class that uses Cython implementation when available.
    This class provides the same interface as the original Board class.
    """
    def __init__(self, size=8):
        super().__init__(size)
        
    def __getattr__(self, name):
        """Delegate attribute access to the implementation."""
        return getattr(self, name)
    
    def __setattr__(self, name, value):
        """Delegate attribute assignment to the implementation."""
        super().__setattr__(name, value)
    
    def __str__(self):
        """String representation of the board."""
        return super().__str__()
    
    def copy(self):
        """Create a deep copy of the board."""
        return self.__class__(self.size)
    
    @classmethod
    def get_implementation(cls):
        """Get the implementation being used (cython or python)."""
        return BOARD_IMPL
