"""
Configuration parameters for AlphaZero Reversi.
"""
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
import json
import torch

@dataclass
class ModelConfig:
    """Configuration for the neural network model."""
    board_size: int = 8
    num_res_blocks: int = 5
    num_filters: int = 128
    value_head_hidden_size: int = 64
    policy_head_hidden_size: int = 64
    dropout: float = 0.3

@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30  # Switch to deterministic after this many moves
    num_parallel: int = 4  # Number of parallel MCTS simulations

@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation."""
    num_games: int = 100
    num_parallel_games: int = 4
    save_dir: str = "self_play_data"
    save_every: int = 10  # Save games every N iterations
    max_moves: int = 200  # Maximum moves per game before declaring a draw
    temp_threshold: int = 15  # Switch temperature after this many moves
    temperature_threshold: int = 10  # Same as temp_threshold, for backward compatibility
    temp_init: float = 1.0
    temp_final: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # Momentum for SGD optimizer
    lr_milestones: List[int] = field(default_factory=list)  # type: ignore
    lr_gamma: float = 0.1
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1  # Save checkpoint every N epochs
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    gradient_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

@dataclass
class TournamentConfig:
    """Configuration for model tournaments."""
    rounds: int = 10
    num_simulations: int = 400
    c_puct: float = 1.0
    output_dir: str = "tournament_results"
    elo_file: str = "elo_ratings.json"

@dataclass
class LoggingConfig:
    """Configuration for logging and visualization."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    use_tensorboard: bool = True
    save_checkpoints: bool = True
    save_best_only: bool = True
    verbose: bool = True

@dataclass
class Config:
    """Main configuration class."""
    project_name: str = "AlphaZero-Reversi"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            project_name=config_dict.get('project_name', 'AlphaZero-Reversi'),
            seed=config_dict.get('seed', 42),
            model=ModelConfig(**config_dict.get('model', {})),
            mcts=MCTSConfig(**config_dict.get('mcts', {})),
            self_play=SelfPlayConfig(**config_dict.get('self_play', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            tournament=TournamentConfig(**config_dict.get('tournament', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

def get_default_config() -> Config:
    """Get default configuration."""
    config = Config()
    
    # Set default learning rate milestones
    config.training.lr_milestones = [
        config.training.num_epochs // 2,
        3 * config.training.num_epochs // 4
    ]
    
    return config
