"""
Monte Carlo Tree Search (MCTS) implementation for AlphaZero.
"""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

class MCTSNode:
    """A node in the Monte Carlo search tree."""
    
    def __init__(self, prior: float, turn: int, move: Optional[Tuple[int, int]] = None, parent: 'MCTSNode' = None):
        """
        Initialize a new node.
        
        Args:
            prior: The prior probability of selecting this node's action
            turn: The player whose turn it is (1 for player 1, 2 for player 2)
            move: The move that led to this node (None for root)
            parent: The parent node (None for root)
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.turn = turn  # The player who will make the next move
        self.move = move  # The move that led to this node
        self.parent = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.state = None  # Cached game state
    
    def expanded(self) -> bool:
        """Check if the node has been expanded (has children)."""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Get the average value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """
        Calculate the UCB score for this node.
        
        Args:
            parent_visit_count: Visit count of the parent node
            c_puct: Exploration constant
            
        Returns:
            UCB score
        """
        if self.visit_count == 0:
            return float('inf')
        
        # UCB formula: Q + U
        # Q = node value (average)
        # U = c_puct * P(s, a) * sqrt(parent_visit) / (1 + N(s, a))
        # where P is the prior probability, N is the visit count
        q = self.value()
        u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        # For opponent's turn, we want to minimize the value
        if self.turn != 1:  # Assuming player 1 is the maximizing player
            q = -q
            
        return q + u
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select a child node with the highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            The selected child node
        """
        # Get the child with the highest UCB score
        best_score = -float('inf')
        best_child = None
        
        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self, action_probs: Dict[Tuple[int, int], float], turn: int):
        """
        Expand the node by adding new child nodes.
        
        Args:
            action_probs: Dictionary mapping actions to their prior probabilities
            turn: The player whose turn it is in the child nodes
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                # The turn flips for the child node
                child_turn = 3 - self.turn  # 1 -> 2, 2 -> 1
                self.children[action] = MCTSNode(prob, child_turn, action, self)
    
    def backpropagate(self, value: float):
        """
        Backpropagate the value up the tree.
        
        Args:
            value: The value to backpropagate
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # For the parent, the value is inverted since it's the opponent's turn
            self.parent.backpropagate(-value)
    
    def get_visit_counts(self) -> Dict[Tuple[int, int], int]:
        """
        Get the visit counts for all child nodes.
        
        Returns:
            Dictionary mapping actions to their visit counts
        """
        return {action: child.visit_count for action, child in self.children.items()}


class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero."""
    
    def __init__(self, model, c_puct: float = 1.0, num_simulations: int = 800):
        """
        Initialize the MCTS.
        
        Args:
            model: The neural network model
            c_puct: Exploration constant
            num_simulations: Number of simulations to run per move
        """
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root = None
    
    def search(self, game) -> Dict[Tuple[int, int], int]:
        """
        Run MCTS from the current game state.
        
        Args:
            game: The current game state
            
        Returns:
            Dictionary mapping actions to their visit counts
        """
        # Create root node if it doesn't exist
        if self.root is None:
            self.root = MCTSNode(1.0, game.current_player)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Save the current game state
            game_copy = game.copy()
            self._simulate(game_copy)
        
        # Return visit counts
        return self.root.get_visit_counts()
    
    def _simulate(self, game):
        """
        Run a single simulation from the current node.
        
        Args:
            game: The current game state
            
        Returns:
            The value of the simulation
        """
        node = self.root
        
        # Selection: traverse the tree until we reach a leaf node
        while node.expanded():
            # If it's not the root node, make the move
            if node.move is not None:
                row, col = node.move
                if (row, col) == (-1, -1):  # Pass move
                    game.pass_turn()
                else:
                    game.make_move(row, col)
            
            # Select the best child
            node = node.select_child(self.c_puct)
        
        # If the game is over, backpropagate the result
        if game.is_game_over():
            # Value is from the perspective of the current player
            if game.get_winner() == 1:  # Player 1 wins
                value = 1.0
            elif game.get_winner() == 2:  # Player 2 wins
                value = -1.0
            else:  # Draw
                value = 0.0
                
            node.backpropagate(value)
            return value
        
        # Expansion: expand the node if not terminal
        # Get valid moves
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:  # No valid moves, pass
            # Create a pass move node
            pass_node = MCTSNode(1.0, 3 - node.turn, (-1, -1), node)
            node.children[(-1, -1)] = pass_node
            node = pass_node
            game.pass_turn()
            # Recursively simulate from the new node
            value = self._simulate(game)
            node.backpropagate(value)
            return value
        
        # Get action probabilities and value from the neural network
        board_state = game.get_board_state()
        valid_moves_mask = np.zeros((game.size, game.size), dtype=np.float32)
        for row, col in valid_moves:
            valid_moves_mask[row, col] = 1.0
        
        # Convert to tensor and add batch dimension
        policy_probs, value = self.model.predict(board_state, valid_moves_mask)
        
        # Convert policy to dictionary
        action_probs = {}
        for row in range(game.size):
            for col in range(game.size):
                if valid_moves_mask[row, col] > 0.5:  # Valid move
                    action_probs[(row, col)] = policy_probs[row * game.size + col]
        
        # Also include pass move if it's valid
        if (-1, -1) in valid_moves:
            action_probs[(-1, -1)] = policy_probs[-1]  # Last element is for pass
        
        # Expand the node
        node.expand(action_probs, 3 - node.turn)  # Turn flips for the next player
        
        # Backpropagate the value
        value = value if node.turn == 1 else -value  # Invert value for opponent
        node.backpropagate(value)
        
        return value
    
    def get_action_probs(self, game, temperature: float = 1.0) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Get action probabilities from the current state.
        
        Args:
            game: The current game state
            temperature: Temperature parameter for exploration
                        (1.0 = proportional to visit counts, 0.0 = always choose best)
            
        Returns:
            Tuple of (best_action, action_probs)
        """
        # Get visit counts
        visit_counts = self.search(game)
        actions = list(visit_counts.keys())
        counts = np.array([visit_counts[action] for action in actions])
        
        # Apply temperature
        if temperature == 0.0:
            # Choose the action with the highest visit count
            best_action_idx = np.argmax(counts)
            action_probs = np.zeros(len(actions))
            action_probs[best_action_idx] = 1.0
        else:
            # Apply temperature to the visit counts
            counts = counts ** (1.0 / temperature)
            action_probs = counts / np.sum(counts)
        
        # Choose an action according to the probabilities
        best_action_idx = np.random.choice(len(actions), p=action_probs)
        best_action = actions[best_action_idx]
        
        return best_action, action_probs
    
    def update_with_move(self, move: Optional[Tuple[int, int]] = None):
        """
        Update the tree after making a move.
        
        Args:
            move: The move that was made (None to reset to a new game)
        """
        if move is None:
            self.root = None
        elif move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None  # Remove parent reference to allow garbage collection
        else:
            self.root = None
