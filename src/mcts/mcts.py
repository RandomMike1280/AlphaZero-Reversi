"""
Monte Carlo Tree Search (MCTS) implementation for AlphaZero.
"""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

class MCTSNode:
    """A node in the Monte Carlo search tree with optimizations."""
    
    __slots__ = [
        'visit_count', 'value_sum', 'prior', 'turn', 'move', 'parent', 
        'children', 'state', 'virtual_loss', 'valid_moves', 'is_terminal',
        'terminal_value', 'cached_ucb'
    ]
    
    def __init__(self, prior: float, turn: int, move: Optional[Tuple[int, int]] = None, 
                 parent: 'MCTSNode' = None, valid_moves: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize a new node.
        
        Args:
            prior: The prior probability of selecting this node's action
            turn: The player whose turn it is (1 for player 1, 2 for player 2)
            move: The move that led to this node (None for root)
            parent: The parent node (None for root)
            valid_moves: List of valid moves from this node
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.turn = turn
        self.move = move
        self.parent = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.state = None  # Minimal game state representation
        self.virtual_loss = 0  # For parallel simulations
        self.valid_moves = valid_moves  # Cached valid moves
        self.is_terminal = False
        self.terminal_value = None
        self.cached_ucb = -float('inf')
    
    def expanded(self) -> bool:
        """Check if the node has been expanded (has children)."""
        return len(self.children) > 0 or self.is_terminal
    
    def value(self) -> float:
        """Get the average value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.0, force_recalculate: bool = False) -> float:
        """
        Calculate the UCB score for this node with caching.
        
        Args:
            parent_visit_count: Visit count of the parent node
            c_puct: Exploration constant
            force_recalculate: If True, recalculate even if cached
            
        Returns:
            UCB score
        """
        if self.visit_count == 0:
            return float('inf')
        
        if not force_recalculate and hasattr(self, 'cached_ucb'):
            return self.cached_ucb
        
        # UCB formula: Q + U
        # Q = node value (average)
        # U = c_puct * P(s, a) * sqrt(parent_visit) / (1 + N(s, a) + virtual_loss)
        visits = self.visit_count + self.virtual_loss
        q = self.value_sum / max(1, self.visit_count)
        u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + visits)
        
        # For opponent's turn, we want to minimize the value
        if self.turn != 1:  # Assuming player 1 is the maximizing player
            q = -q
        
        self.cached_ucb = q + u
        return self.cached_ucb
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select a child node with the highest UCB score.
        Optimized to avoid dictionary lookups in the hot loop.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            The selected child node
        """
        best_score = -float('inf')
        best_child = None
        
        # Convert to list once to avoid repeated dict iteration
        children_items = list(self.children.items())
        
        for action, child in children_items:
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, action_probs: Dict[Tuple[int, int], float], turn: int, 
               valid_moves: Optional[List[Tuple[int, int]]] = None):
        """
        Expand the node by creating child nodes for all possible actions.
        
        Args:
            action_probs: Dictionary mapping actions to their prior probabilities
            turn: The player whose turn it is in the child nodes
            valid_moves: List of valid moves from this node (cached for efficiency)
        """
        self.valid_moves = valid_moves if valid_moves is not None else list(action_probs.keys())
        
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    prior=prob,
                    turn=turn,
                    move=action,
                    parent=self,
                    valid_moves=None  # Will be set when expanded
                )
    
    def backpropagate(self, value: float):
        """
        Backpropagate the value up the tree.
        
        Args:
            value: The value to backpropagate (from the perspective of the current player)
        """
        self.visit_count += 1
        self.value_sum += value
        
        # Invalidate cached UCB score
        if hasattr(self, 'cached_ucb'):
            del self.cached_ucb
        
        if self.parent is not None:
            # For the parent, the value is inverted since it's the opponent's turn
            self.parent.backpropagate(-value)
    
    def get_visit_counts(self) -> Dict[Tuple[int, int], int]:
        """
        Get a dictionary mapping actions to their visit counts.
        
        Returns:
            Dictionary mapping actions to their visit counts
        """
        return {action: child.visit_count for action, child in self.children.items()}


class MCTS:
    """
    Optimized Monte Carlo Tree Search for AlphaZero with batched inference and path backpropagation.
    """
    
    def __init__(self, model, c_puct: float = 1.0, num_simulations: int = 800, 
                 batch_size: int = 16, num_threads: int = 1):
        """
        Initialize the MCTS with optimizations.
        
        Args:
            model: The neural network model
            c_puct: Exploration constant
            num_simulations: Number of simulations to run per move
            batch_size: Number of leaf nodes to process in parallel
            num_threads: Number of threads for parallel simulations (not yet implemented)
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.root = None
        self.lock = None  # Will be initialized if using threads
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def search(self, game) -> Dict[Tuple[int, int], int]:
        """
        Run MCTS from the current game state with batched simulations.
        
        Args:
            game: The current game state
            
        Returns:
            Dictionary mapping actions to their visit counts
        """
        if self.root is None:
            self.root = MCTSNode(1.0, game.current_player, valid_moves=game.get_valid_moves())
        
        # Run batched simulations
        leaf_batch = {}
        for _ in range(self.num_simulations):
            game_copy = game.copy()
            path = []
            
            # Selection and expansion
            node, game_copy = self._traverse(game_copy, path)
            
            # Add to batch if not terminal
            if not node.is_terminal and node not in leaf_batch:
                leaf_batch[node] = (game_copy, path)
                
                # Process batch if full
                if len(leaf_batch) >= self.batch_size:
                    self._process_batch(leaf_batch)
                    leaf_batch = {}
        
        # Process remaining leaves
        if leaf_batch:
            self._process_batch(leaf_batch)
        
        return self.root.get_visit_counts()
    
    def _traverse(self, game, path: List[MCTSNode]) -> Tuple[MCTSNode, Any]:
        """Traverse the tree until a leaf node is found."""
        node = self.root
        path.append(node)
        
        while node.expanded() and not node.is_terminal:
            # Apply virtual loss
            node.virtual_loss += 1
            
            # Select best child
            next_move, next_node = None, None
            best_score = -float('inf')
            
            # Check children
            for move, child in node.children.items():
                score = child.ucb_score(node.visit_count, self.c_puct)
                if score > best_score:
                    best_score = score
                    next_move = move
                    next_node = child
            
            # If no valid moves, check if we need to pass
            if next_node is None and (-1, -1) in node.children:  # Pass move
                next_move = (-1, -1)
                next_node = node.children[next_move]
            
            # Make the move
            if next_move == (-1, -1):  # Pass
                game.pass_turn()
            else:
                game.make_move(*next_move)
            
            node = next_node
            path.append(node)
        
        return node, game
    
    def _process_batch(self, leaf_batch: Dict[MCTSNode, Tuple[Any, List[MCTSNode]]]):
        """Process a batch of leaf nodes in parallel."""
        if not leaf_batch:
            return
        
        # Prepare batch inputs
        states = []
        paths = []
        nodes = []
        
        for node, (game_copy, path) in leaf_batch.items():
            # Get valid moves if not already cached
            if node.valid_moves is None:
                node.valid_moves = game_copy.get_valid_moves()
            
            # Handle terminal nodes
            if not node.valid_moves:  # No valid moves
                node.is_terminal = True
                winner = game_copy.get_winner()
                if winner == 1:  # Player 1 wins
                    node.terminal_value = 1.0
                elif winner == 2:  # Player 2 wins
                    node.terminal_value = -1.0
                else:  # Draw
                    node.terminal_value = 0.0
                
                # Backpropagate terminal value
                self._backpropagate_path(path, node.terminal_value)
                continue
            
            # Prepare input tensor
            board_state = game_copy.get_board_state()
            valid_moves_mask = np.zeros((game_copy.size, game_copy.size), dtype=np.float32)
            for row, col in node.valid_moves:
                valid_moves_mask[row, col] = 1.0
            
            player_pieces = (board_state == 1).astype(np.float32)
            opponent_pieces = (board_state == 2).astype(np.float32)
            
            x = np.stack([player_pieces, opponent_pieces, valid_moves_mask], axis=0)
            states.append(x)
            paths.append(path)
            nodes.append(node)
        
        if not states:  # All nodes were terminal
            return
        
        # Convert to tensor and move to device
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        
        # Get batch predictions
        with torch.no_grad():
            policy_logits, values = self.model(states_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze(1).cpu().numpy()
        
        # Process each node in the batch
        for i, (node, path) in enumerate(zip(nodes, paths)):
            if node.is_terminal:
                continue
                
            # Convert policy to action probabilities
            action_probs = {}
            valid_moves_mask = states[i][2]  # Third channel is valid moves
            
            for row in range(valid_moves_mask.shape[0]):
                for col in range(valid_moves_mask.shape[1]):
                    if valid_moves_mask[row, col] > 0.5:  # Valid move
                        action_probs[(row, col)] = policy_probs[i, row * valid_moves_mask.shape[0] + col]
            
            # Include pass move if valid
            if (-1, -1) in node.valid_moves:
                action_probs[(-1, -1)] = policy_probs[i, -1]
            
            # Expand the node
            node.expand(action_probs, 3 - node.turn, node.valid_moves)
            
            # Backpropagate the value
            value = values[i] if node.turn == 1 else -values[i]
            self._backpropagate_path(path + [node], value)
    
    def _backpropagate_path(self, path: List[MCTSNode], value: float):
        """Backpropagate value up the tree using the given path."""
        sign = 1
        for node in reversed(path):
            if node.virtual_loss > 0:
                node.virtual_loss -= 1
            
            node.visit_count += 1
            node.value_sum += sign * value
            
            # Invert value for parent (opponent's perspective)
            sign = -sign
            
            # Invalidate cached UCB score
            if hasattr(node, 'cached_ucb'):
                del node.cached_ucb
    
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
        Update the tree after making a move, reusing relevant subtrees.
        
        Args:
            move: The move that was made (None to reset to a new game)
        """
        if move is None:
            self.root = None
            return
            
        if self.root is None:
            return
            
        if move in self.root.children:
            # Reuse the subtree
            self.root = self.root.children[move]
            self.root.parent = None
            
            # Prune other children to save memory
            self.root.children = {}
        else:
            # No matching child, start fresh
            self.root = None
