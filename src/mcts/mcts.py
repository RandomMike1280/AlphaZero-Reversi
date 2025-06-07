"""
Monte Carlo Tree Search (MCTS) implementation for AlphaZero with transposition tables.
"""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch.distributed as dist

# Define a transposition table entry
class TranspositionTableEntry:
    """Entry in the transposition table."""
    __slots__ = ['value', 'depth', 'node_type', 'best_move']
    
    def __init__(self, value: float, depth: int, node_type: str, best_move: Optional[Tuple[int, int]] = None):
        """
        Initialize a transposition table entry.
        
        Args:
            value: The value of the node
            depth: The depth of the search that produced this value
            node_type: 'EXACT', 'LOWER_BOUND', or 'UPPER_BOUND'
            best_move: The best move found at this node
        """
        self.value = value
        self.depth = depth
        self.node_type = node_type
        self.best_move = best_move

class MCTSNode:
    """A node in the Monte Carlo search tree with optimizations and transposition table support."""
    
    __slots__ = [
        'visit_count', 'value_sum', 'prior', 'turn', 'move', 'parent', 
        'children', 'state', 'virtual_loss', 'valid_moves', 'is_terminal',
        'terminal_value', 'cached_ucb', 'zobrist_hash', 'transposition_key',
        'best_child_move'
    ]
    
    def __init__(self, prior: float, turn: int, move: Optional[Tuple[int, int]] = None, 
                 parent: 'MCTSNode' = None, valid_moves: Optional[List[Tuple[int, int]]] = None,
                 zobrist_hash: Optional[int] = None):
        """
        Initialize a new node.
        
        Args:
            prior: The prior probability of selecting this node's action
            turn: The player whose turn it is (1 for player 1, 2 for player 2)
            move: The move that led to this node (None for root)
            parent: The parent node (None for root)
            valid_moves: List of valid moves from this node
            zobrist_hash: Zobrist hash of the board state at this node
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
        self.zobrist_hash = zobrist_hash  # Zobrist hash of the board state
        self.transposition_key = None  # Key used in transposition table
        self.best_child_move = None  # Best move found during search
    
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
    Optimized Monte Carlo Tree Search for AlphaZero with transposition tables,
    batched inference, and path backpropagation.
    """
    
    def __init__(self, model, c_puct: float = 1.0, num_simulations: int = 800, 
                 batch_size: int = 64, num_threads: int = 1, use_transposition: bool = True):
        """
        Initialize the MCTS with optimizations.
        
        Args:
            model: The neural network model
            c_puct: Exploration constant
            num_simulations: Number of simulations to run per move
            batch_size: Number of leaf nodes to process in parallel
            num_threads: Number of threads for parallel simulations (not yet implemented)
            use_transposition: Whether to use transposition tables
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.root = None
        self.lock = None  # Will be initialized if using threads
        self.use_transposition = use_transposition
        
        # Initialize multi-GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.use_cuda else 0
        self.models = {}
        
        # If using multiple GPUs, create model replicas
        if self.num_gpus > 1:
            self._init_multi_gpu()
        
        # Transposition table: maps zobrist_hash -> TranspositionTableEntry
        self.transposition_table = {}
        
        # Cache for board states we've seen before
        self.state_cache = {}
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def _get_transposition_key(self, node: MCTSNode, depth: int) -> int:
        """
        Generate a transposition table key for the given node and search depth.
        
        Args:
            node: The node to generate a key for
            depth: Current search depth
            
        Returns:
            A unique key for the transposition table
        """
        if node.zobrist_hash is None:
            return None
            
        # Combine hash with depth for more precise matching
        return (node.zobrist_hash, depth, node.turn)
    
    def _lookup_transposition(self, node: MCTSNode, depth: int, alpha: float, beta: float) -> Optional[float]:
        """
        Look up a node in the transposition table.
        
        Args:
            node: The node to look up
            depth: Current search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            
        Returns:
            The value from the transposition table if found, else None
        """
        if not self.use_transposition or node.zobrist_hash is None:
            return None
            
        key = self._get_transposition_key(node, depth)
        if key is None:
            return None
            
        entry = self.transposition_table.get(key)
        if entry is None:
            return None
            
        # Only use the entry if it's from a search at least as deep as ours
        if entry.depth >= depth:
            if entry.node_type == 'EXACT':
                return entry.value
            elif entry.node_type == 'LOWER_BOUND' and entry.value >= beta:
                return entry.value
            elif entry.node_type == 'UPPER_BOUND' and entry.value <= alpha:
                return entry.value
                
        # If we have a best move from the transposition table, use it
        if entry.best_move is not None and entry.best_move in node.children:
            node.best_child_move = entry.best_move
            
        return None
    
    def _store_transposition(self, node: MCTSNode, depth: int, value: float, 
                           node_type: str, best_move: Optional[Tuple[int, int]] = None) -> None:
        """
        Store a node in the transposition table.
        
        Args:
            node: The node to store
            depth: The depth of the search that produced this value
            value: The value to store
            node_type: 'EXACT', 'LOWER_BOUND', or 'UPPER_BOUND'
            best_move: The best move found at this node
        """
        if not self.use_transposition or node.zobrist_hash is None:
            return
            
        key = self._get_transposition_key(node, depth)
        if key is None:
            return
            
        # Only store if we don't have a deeper search already
        existing = self.transposition_table.get(key)
        if existing is None or existing.depth <= depth:
            self.transposition_table[key] = TranspositionTableEntry(
                value=value,
                depth=depth,
                node_type=node_type,
                best_move=best_move
            )
    
    def search(self, game) -> Dict[Tuple[int, int], int]:
        """
        Run MCTS from the current game state with batched simulations and transposition tables.
        
        Args:
            game: The current game state
            
        Returns:
            Dictionary mapping actions to their visit counts
        """
        # Reset the root node with Zobrist hash if available
        zobrist_hash = getattr(game, 'get_zobrist_hash', lambda: None)()
        self.root = MCTSNode(
            1.0, 
            game.current_player, 
            None, 
            None, 
            game.get_valid_moves(),
            zobrist_hash
        )
        
        # Clear the transposition table if it's too large
        if len(self.transposition_table) > 1000000:  # Arbitrary limit
            self.transposition_table.clear()
        
        # Run simulations in batches
        for _ in range(0, self.num_simulations, self.batch_size):
            batch_size = min(self.batch_size, self.num_simulations - _)
            
            # Collect a batch of leaf nodes to evaluate
            leaf_nodes = []
            paths = []
            
            for _ in range(batch_size):
                # Make a copy of the game for simulation
                sim_game = game.copy()
                
                # Traverse the tree to find a leaf node
                path = []
                node, sim_game = self._traverse(sim_game, path)
                
                # If we've reached a terminal state, backpropagate immediately
                if node.is_terminal:
                    self._backpropagate_path(path, node.terminal_value)
                    continue
                
                # Check transposition table for this node
                if self.use_transposition and hasattr(sim_game, 'get_symmetry_hashes'):
                    # Try to find a transposition for any symmetric position
                    for sym_hash in sim_game.get_symmetry_hashes():
                        # Create a temporary node with the symmetric hash
                        temp_node = MCTSNode(1.0, sim_game.current_player, None, None, None, sym_hash)
                        temp_key = self._get_transposition_key(temp_node, 0)  # Depth 0 for now
                        
                        if temp_key in self.transposition_table:
                            entry = self.transposition_table[temp_key]
                            # Use the value from the transposition table
                            self._backpropagate_path(path, entry.value)
                            break
                    else:
                        # No transposition found, add to batch for evaluation
                        leaf_nodes.append((node, sim_game, path))
                else:
                    # No transposition table support, just add to batch
                    leaf_nodes.append((node, sim_game, path))
            
            # Process the batch of leaf nodes
            if leaf_nodes:
                # Convert list of tuples to the format expected by _process_batch
                batch_data = [(node, game, path) for node, game, path in leaf_nodes]
                self._process_batch(batch_data)
        
        # Store best move in the transposition table
        if self.use_transposition and self.root.zobrist_hash is not None and self.root.children:
            best_move = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
            self._store_transposition(
                self.root, 
                depth=0,  # Root level
                value=self.root.value(),
                node_type='EXACT',
                best_move=best_move
            )
        
        # Return the visit counts of the root's children
        return {move: child.visit_count 
                for move, child in self.root.children.items()}
    
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
    
    def _init_multi_gpu(self):
        """Initialize models on multiple GPUs if available."""
        if self.num_gpus <= 1:
            return
            
        # Get model configuration from the existing model
        board_size = self.model.board_size
        num_filters = self.model.num_filters
        # Get number of residual blocks from the model's res_blocks ModuleList
        num_res_blocks = len(self.model.res_blocks)
            
        # Create model replicas for each GPU
        for i in range(self.num_gpus):
            device = torch.device(f'cuda:{i}')
            # Create a new model instance with the same configuration
            model_copy = type(self.model)(
                board_size=board_size,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters
            )
            # Load the state dict to copy weights
            model_copy.load_state_dict(self.model.state_dict())
            # Move to the target device
            self.models[i] = model_copy.to(device)
            self.models[i].eval()
    
    def _predict_batch(self, states_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of states using available GPUs.
        
        Args:
            states_tensor: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (policy_logits, values)
        """
        if self.num_gpus <= 1 or not self.use_cuda:
            # Single GPU or CPU case
            with torch.no_grad():
                return self.model.predict(states_tensor.to(self.device))
        
        # Multi-GPU case
        batch_size = states_tensor.size(0)
        if batch_size < self.num_gpus:
            # If batch size is smaller than number of GPUs, just use one GPU
            with torch.no_grad():
                return self.model.predict(states_tensor.to(self.device))
        
        # Split batch across GPUs
        chunk_size = (batch_size + self.num_gpus - 1) // self.num_gpus
        chunks = [states_tensor[i:i + chunk_size] for i in range(0, batch_size, chunk_size)]
        
        policy_chunks = []
        value_chunks = []
        
        # Process each chunk on its designated GPU
        for i, chunk in enumerate(chunks):
            gpu_idx = i % self.num_gpus
            device = torch.device(f'cuda:{gpu_idx}')
            
            # Move chunk to the target device
            chunk = chunk.to(device)
            
            # Get model for this GPU
            model = self.models[gpu_idx]
            
            # Make sure the model is on the correct device
            if next(model.parameters()).device != device:
                model = model.to(device)
            
            with torch.no_grad():
                policy, value = model.predict(chunk)
                # Move results back to CPU before storing
                policy_chunks.append(policy.cpu())
                value_chunks.append(value.cpu())
        
        # Concatenate results on CPU
        policy_logits = torch.cat(policy_chunks, dim=0)
        values = torch.cat(value_chunks, dim=0)
        
        return policy_logits, values
    
    def _process_batch(self, leaf_batch: List[Tuple[MCTSNode, Any, List[MCTSNode]]]):
        """Process a batch of leaf nodes in parallel."""
        if not leaf_batch:
            return
        
        # Prepare batch inputs
        states = []
        paths = []
        nodes = []
        
        for node, game_copy, path in leaf_batch:
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
        
        # Convert to tensor
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        
        # Get batch predictions using the model's predict method
        policy_logits, values = self._predict_batch(states_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).numpy()
        values = values.numpy()
        
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
            Tuple of (best_action, action_probs) where action_probs is a fixed-size vector
            with length board_size * board_size + 1 (for pass move)
        """
        # Get visit counts
        visit_counts = self.search(game)
        board_size = game.size
        
        # Initialize fixed-size probability vector
        action_probs = np.zeros(board_size * board_size + 1)  # +1 for pass move
        
        # Convert visit counts to probabilities
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            for (row, col), count in visit_counts.items():
                if (row, col) == (-1, -1):  # Pass move
                    action_probs[-1] = count / total_visits
                else:
                    idx = row * board_size + col
                    action_probs[idx] = count / total_visits
        
        # Apply temperature
        if temperature > 0 and not np.all(action_probs == 0):
            # Apply temperature to the visit counts
            temp_probs = action_probs ** (1.0 / temperature)
            action_probs = temp_probs / np.sum(temp_probs)
        
        # Choose an action according to the probabilities
        if temperature == 0.0 or np.all(action_probs == 0):
            # Choose the action with the highest probability
            best_action_idx = np.argmax(action_probs)
        else:
            # Sample according to the probabilities
            best_action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # Convert index back to action
        if best_action_idx == len(action_probs) - 1:  # Pass move
            best_action = (-1, -1)
        else:
            row = best_action_idx // board_size
            col = best_action_idx % board_size
            best_action = (row, col)
        
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
