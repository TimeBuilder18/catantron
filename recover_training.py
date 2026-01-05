"""
Recovery Script for Collapsed Training

Use this to recover from entropy collapse by:
1. Loading the model from before collapse
2. Resetting optimizer state
3. Applying entropy healing
4. Resuming with stable parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

from network_wrapper import NetworkWrapper
from curriculum_trainer_v3_stable import CurriculumTrainerV3


def heal_policy_entropy(network, device, temperature=2.0, iterations=100):
    """
    Apply entropy healing to a collapsed policy.

    This adds noise and temperature scaling to prevent deterministic collapse.
    """
    print("\n" + "=" * 50)
    print("ENTROPY HEALING")
    print("=" * 50)

    network.eval()

    # Sample random observations
    batch_size = 64
    obs = torch.randn(batch_size, 121).to(device)  # Random obs

    # Get current policy entropy
    with torch.no_grad():
        action_probs, vertex_probs, edge_probs, _, _, _ = network.forward(obs)

        log_probs = torch.log(action_probs + 1e-8)
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        print(f"Pre-healing entropy: {entropy.item():.4f}")

        # Check if healing needed
        if entropy.item() > 1.5:
            print("Entropy looks healthy, no healing needed")
            return

    # Apply healing via gradient descent on entropy maximization
    print(f"Applying entropy healing with temperature={temperature}...")

    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    for i in range(iterations):
        obs = torch.randn(batch_size, 121).to(device)

        # Forward with temperature
        action_probs, vertex_probs, edge_probs, _, _, _ = network.forward(obs)

        # Apply temperature to soften distribution
        action_logits = torch.log(action_probs + 1e-8) / temperature
        action_probs_soft = F.softmax(action_logits, dim=-1)

        # Maximize entropy
        log_probs = torch.log(action_probs_soft + 1e-8)
        entropy = -(action_probs_soft * log_probs).sum(dim=1).mean()

        # Entropy maximization loss
        loss = -entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}/{iterations}: entropy = {entropy.item():.4f}")

    # Check final entropy
    network.eval()
    with torch.no_grad():
        obs = torch.randn(batch_size, 121).to(device)
        action_probs, _, _, _, _, _ = network.forward(obs)
        log_probs = torch.log(action_probs + 1e-8)
        final_entropy = -(action_probs * log_probs).sum(dim=1).mean()
        print(f"\nPost-healing entropy: {final_entropy.item():.4f}")

    print("=" * 50 + "\n")


def find_best_checkpoint(model_dir='models'):
    """Find the best checkpoint based on filename patterns"""
    checkpoints = []

    for f in os.listdir(model_dir):
        if f.endswith('.pt'):
            path = os.path.join(model_dir, f)
            mtime = os.path.getmtime(path)
            checkpoints.append((path, mtime, f))

    if not checkpoints:
        return None

    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    print("\nAvailable checkpoints:")
    for i, (path, mtime, name) in enumerate(checkpoints[:10]):
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"  {i+1}. {name} ({dt.strftime('%Y-%m-%d %H:%M')})")

    return checkpoints


def recover_from_collapse(model_path, output_path=None, heal=True, continue_training=False,
                          total_games=5000, start_phase=None):
    """
    Recover training from a collapsed checkpoint.

    Args:
        model_path: Path to checkpoint to recover from
        output_path: Path to save recovered model
        heal: Whether to apply entropy healing
        continue_training: Whether to continue training after recovery
        total_games: Number of games if continuing training
        start_phase: Starting curriculum phase (0-4)
    """
    print("\n" + "=" * 70)
    print("TRAINING RECOVERY")
    print("=" * 70)
    print(f"Loading model from: {model_path}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create network wrapper
    wrapper = NetworkWrapper(device=str(device))
    wrapper.policy.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model (games_played: {checkpoint.get('games_played', 'unknown')})")

    if heal:
        heal_policy_entropy(wrapper.policy, device)

    # Save recovered model
    if output_path is None:
        base = os.path.splitext(model_path)[0]
        output_path = f"{base}_recovered.pt"

    # Reset optimizer state for fresh training
    optimizer = torch.optim.Adam(wrapper.policy.parameters(), lr=5e-4)

    torch.save({
        'model_state_dict': wrapper.policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'games_played': checkpoint.get('games_played', 0),
        'recovered': True,
        'source_model': model_path,
    }, output_path)
    print(f"Saved recovered model to: {output_path}")

    if continue_training:
        print("\nStarting recovery training with stable parameters...")
        trainer = CurriculumTrainerV3(
            model_path=output_path,
            learning_rate=3e-4,  # Lower LR for recovery
        )

        # Override starting phase if specified
        if start_phase is not None:
            print(f"Starting from phase {start_phase}")

        trainer.train(
            total_games=total_games,
            train_frequency=5,
            train_steps=10,  # Fewer steps for stability
            min_games_per_phase=300,  # Faster phase transitions
        )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover from entropy collapse")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint to recover')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save recovered model')
    parser.add_argument('--no-heal', action='store_true',
                        help='Skip entropy healing step')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training after recovery')
    parser.add_argument('--games', type=int, default=5000,
                        help='Number of games to train if continuing')
    parser.add_argument('--start-phase', type=int, default=None,
                        help='Starting curriculum phase (0=random, 4=full strength)')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')

    args = parser.parse_args()

    if args.list:
        find_best_checkpoint()
    else:
        recover_from_collapse(
            model_path=args.model,
            output_path=args.output,
            heal=not args.no_heal,
            continue_training=args.continue_training,
            total_games=args.games,
            start_phase=args.start_phase,
        )
