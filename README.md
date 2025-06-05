# AlphaZero Reversi

An implementation of the AlphaZero algorithm for the game of Reversi (Othello) using PyTorch.

## Features

- **Neural Network**: Residual network with policy and value heads
- **Monte Carlo Tree Search (MCTS)**: Efficient search with UCB1 selection
- **Self-Play**: Parallel game generation for training data
- **Training Pipeline**: End-to-end training with checkpointing
- **Tournament System**: ELO-based evaluation of model performance
- **Logging**: TensorBoard integration for monitoring training

## Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/alphazero-reversi.git
   cd alphazero-reversi
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Create a configuration file

```bash
python train.py create-config --output config.json
```

Edit the generated `config.json` to adjust training parameters as needed.

### 2. Start training

```bash
python train.py train --config config.json
```

### 3. Monitor training

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir=logs
```

### 4. Continue training from a checkpoint

```bash
python train.py continue path/to/checkpoint.pth
```

## Project Structure

```
src/
├── arena/                # Tournament and ELO rating system
├── config.py             # Configuration management
├── game/                 # Reversi game implementation
├── logger.py             # Logging utilities
├── mcts/                 # Monte Carlo Tree Search
├── model.py              # Neural network architecture
├── self_play/            # Self-play data generation
└── trainer/              # Training pipeline
```

## Configuration

The configuration system uses dataclasses to manage all parameters. Key configuration sections:

- **model**: Network architecture parameters
- **training**: Training hyperparameters
- **mcts**: MCTS search parameters
- **self_play**: Self-play configuration
- **tournament**: Evaluation settings
- **logging**: Logging and visualization

## Training Process

1. **Self-Play**: The current model plays against itself to generate training data
2. **Training**: The model is trained on the generated data
3. **Evaluation**: The model is evaluated in tournaments against previous versions
4. **Iteration**: The process repeats with the improved model

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DeepMind's AlphaZero paper
- The Reversi/Othello community
- Open source contributorsZero Reversi

An implementation of the AlphaZero algorithm for playing Reversi (Othello). This project includes:

- Reversi game logic with a clean Python API
- Monte Carlo Tree Search (MCTS) with neural network guidance
- Self-play training pipeline
- Model tournament system with ELO rating

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Reversi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `src/`: Source code
  - `game/`: Reversi game logic
  - `mcts/`: Monte Carlo Tree Search implementation
  - `model/`: Neural network architecture
  - `self_play/`: Self-play data generation
  - `training/`: Model training code
- `models/`: Saved model checkpoints
  - `training/`: Models saved during training
  - `tournament/`: Best model from tournament

## Running Tests

To run the unit tests:

```bash
pytest test_game.py -v
```

## Usage

### Playing the Game

```python
from src.game.game import ReversiGame

game = ReversiGame()
print(game)  # Print the initial board

# Make moves: (row, col)
game.make_move(2, 3)  # Black's move
game.make_move(4, 2)  # White's move

print(game)  # Print the updated board
```

### Training

```bash
python src/main.py --mode train --num_games 1000
```

### Tournament

```bash
python src/main.py --mode tournament --num_games 100
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
