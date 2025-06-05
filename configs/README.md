# Configuration Files

This directory contains configuration files for the AlphaZero Reversi project.

## Default Configuration

- `default_config.json`: The default configuration file with recommended settings for training.

## Configuration Options

### Model Configuration
- `board_size`: Size of the game board (default: 8 for standard Reversi)
- `num_res_blocks`: Number of residual blocks in the network
- `num_filters`: Number of filters in convolutional layers
- `value_head_hidden_size`: Size of hidden layers in value head
- `policy_head_hidden_size`: Size of hidden layers in policy head
- `dropout`: Dropout rate for regularization

### MCTS Configuration
- `num_simulations`: Number of MCTS simulations per move
- `c_puct`: Exploration constant for UCB1
- `dirichlet_alpha`: Alpha parameter for Dirichlet noise
- `dirichlet_epsilon`: Epsilon for mixing in Dirichlet noise
- `temperature`: Temperature for move selection
- `num_parallel`: Number of parallel MCTS simulations

### Self-Play Configuration
- `num_games`: Number of games to generate per iteration
- `num_parallel_games`: Number of games to run in parallel
- `save_dir`: Directory to save self-play data
- `save_every`: Save games every N iterations
- `max_moves`: Maximum moves per game before draw
- `temp_threshold`: Switch to deterministic play after this many moves
- `temp_init`: Initial temperature for move selection
- `temp_final`: Final temperature for move selection

### Training Configuration
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs per iteration
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization weight decay
- `momentum`: Momentum for SGD optimizer
- `lr_milestones`: Epochs at which to decay learning rate
- `lr_gamma`: Multiplicative factor of learning rate decay
- `gradient_clip`: Maximum gradient norm for clipping
- `policy_loss_weight`: Weight for policy loss
- `value_loss_weight`: Weight for value loss
- `device`: Device to use for training ('cuda' or 'cpu')

### Tournament Configuration
- `rounds`: Number of tournament rounds to play
- `num_simulations`: Number of MCTS simulations per move in tournament
- `c_puct`: Exploration constant for tournament games
- `output_dir`: Directory to save tournament results
- `elo_file`: File to save ELO ratings

### Logging Configuration
- `log_dir`: Directory to save logs
- `use_tensorboard`: Whether to use TensorBoard
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Using a Custom Configuration

1. Copy `default_config.json` to a new file (e.g., `my_config.json`)
2. Modify the parameters as needed
3. Run the training script with your config:
   ```bash
   python run.py --config configs/my_config.json
   ```

## Resuming Training

To resume training from a checkpoint:

```bash
python run.py --resume path/to/checkpoint.pth
```

## Best Practices

- Start with the default configuration for initial testing
- For faster training, reduce `num_simulations` and `num_games`
- Increase `num_parallel_games` to utilize more CPU cores
- Use a smaller `batch_size` if you run into memory issues
- Monitor training progress with TensorBoard:
  ```bash
  tensorboard --logdir=runs
  ```
