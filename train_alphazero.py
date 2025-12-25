"""
Train Catan AI using AlphaZero approach

Auto-detects best device (CUDA > MPS > CPU)

Usage:
    python train_alphazero.py                    # Quick test (20 games)
    python train_alphazero.py --games 100        # Short training
    python train_alphazero.py --games 500        # Medium training
    python train_alphazero.py --games 2000       # Full training (overnight)
"""

import argparse
from alphazero_trainer import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Catan AI with AlphaZero')

    parser.add_argument('--games', type=int, default=20,
                        help='Total games to play (default: 20)')
    parser.add_argument('--sims', type=int, default=30,
                        help='MCTS simulations per move (default: 30)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size (auto-detected if not set)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save-path', type=str, default='models/alphazero',
                        help='Path to save models (default: models/alphazero)')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to load existing model (optional)')

    args = parser.parse_args()

    # Create trainer (auto-detects device)
    trainer = AlphaZeroTrainer(
        model_path=args.load,
        num_simulations=args.sims,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    # Adjust training params based on game count
    if args.games <= 50:
        games_per_train = 5
        train_steps = 10
        save_freq = 10
    elif args.games <= 200:
        games_per_train = 10
        train_steps = 15
        save_freq = 25
    else:
        games_per_train = 10
        train_steps = 20
        save_freq = 50

    # Train!
    trainer.train(
        num_games=args.games,
        games_per_training=games_per_train,
        training_steps_per_batch=train_steps,
        save_frequency=save_freq,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()