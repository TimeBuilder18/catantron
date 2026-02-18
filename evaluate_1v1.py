#!/usr/bin/env python3
"""
1v1 Model Evaluation Script

Benchmarks a trained model against different AI difficulties in 1v1 mode.
Useful for:
- Tracking training progress
- Comparing models
- Identifying which opponent types the model struggles with
"""

import argparse
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from catan_env_pytorch import CatanEnv
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from network_wrapper import NetworkWrapper
from curriculum_trainer_v3_stable import play_opponent_turn


def evaluate_model(model_path, opponent_type='random', num_games=100, num_parallel=8, verbose=True):
    """
    Evaluate a model against a specific opponent type in 1v1.

    Args:
        model_path: Path to the trained model
        opponent_type: 'truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', or 'strong'
        num_games: Number of games to play
        num_parallel: Number of games to run in parallel
        verbose: Print progress

    Returns:
        dict with win_rate, avg_vp, avg_game_length, std_vp
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network_wrapper = NetworkWrapper(model_path=model_path, device=device)
    network = network_wrapper.policy
    network.eval()

    results = {'wins': 0, 'losses': 0, 'vps': [], 'lengths': []}

    def play_single_game():
        """Play a single 1v1 game and return (won, vp, length)"""
        env = PBRSFixedRewardWrapper(player_id=0, victory_points_to_win=10, num_players=2)
        obs, _ = env.reset()

        done = False
        moves = 0
        max_moves = 500

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                # Model's turn
                moves += 1
                with torch.no_grad():
                    observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(device)
                    action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)
                    vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device)
                    edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device)

                    action_probs, vertex_probs, edge_probs, _, _, _ = network.forward(
                        observation, action_mask, vertex_mask, edge_mask
                    )

                    ap = action_probs.cpu().numpy()[0]
                    vp = vertex_probs.cpu().numpy()[0]
                    ep = edge_probs.cpu().numpy()[0]

                # Safe probability normalization
                def safe_probs(probs):
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    if probs.sum() <= 0:
                        probs = np.ones_like(probs) / len(probs)
                    else:
                        probs = probs / probs.sum()
                    return probs

                ap = safe_probs(ap)
                vp_probs = safe_probs(vp)
                ep = safe_probs(ep)

                action_id = np.random.choice(len(ap), p=ap)
                vertex_id = np.random.choice(len(vp_probs), p=vp_probs)
                edge_id = np.random.choice(len(ep), p=ep)

                obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0
                )
                done = terminated or truncated
            else:
                # Opponent's turn
                play_opponent_turn(game, current_id, 1.0, opponent_type, None)
                winner = game.check_victory_conditions()
                if winner is not None:
                    done = True

        my_vp = env.game_env.game.players[0].calculate_victory_points()
        winner = env.game_env.game.check_victory_conditions()
        won = (winner is not None and game.players.index(winner) == 0)

        return won, my_vp, moves

    # Run games in parallel
    start_time = time.time()
    games_completed = 0

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(play_single_game) for _ in range(num_games)]

        for future in as_completed(futures):
            try:
                won, vp, length = future.result()
                if won:
                    results['wins'] += 1
                else:
                    results['losses'] += 1
                results['vps'].append(vp)
                results['lengths'].append(length)
                games_completed += 1

                if verbose and games_completed % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = games_completed / elapsed * 60
                    wr = results['wins'] / games_completed * 100
                    print(f"  Progress: {games_completed}/{num_games} | WR: {wr:.1f}% | Speed: {speed:.1f} g/min")
            except Exception as e:
                print(f"  Warning: Game failed: {e}")

    total_time = time.time() - start_time

    return {
        'opponent': opponent_type,
        'games': num_games,
        'win_rate': results['wins'] / num_games * 100,
        'avg_vp': np.mean(results['vps']),
        'std_vp': np.std(results['vps']),
        'avg_length': np.mean(results['lengths']),
        'time_seconds': total_time,
        'games_per_min': num_games / total_time * 60
    }


def full_benchmark(model_path, num_games=50, num_parallel=8):
    """
    Run a full benchmark against all opponent types.

    Args:
        model_path: Path to the trained model
        num_games: Number of games per opponent type
        num_parallel: Number of parallel games

    Returns:
        dict mapping opponent type to results
    """
    opponents = ['truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', 'strong']
    all_results = {}

    print("\n" + "=" * 70)
    print("1v1 MODEL BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Games per opponent: {num_games}")
    print(f"Parallel games: {num_parallel}")
    print("=" * 70 + "\n")

    for opponent in opponents:
        print(f"\nBenchmarking vs {opponent.upper()}...")
        result = evaluate_model(model_path, opponent, num_games, num_parallel, verbose=True)
        all_results[opponent] = result

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Opponent':<12} {'Win Rate':>10} {'Avg VP':>8} {'Std VP':>8} {'Avg Len':>8}")
    print("-" * 70)
    for opponent, result in all_results.items():
        print(f"{opponent:<12} {result['win_rate']:>9.1f}% {result['avg_vp']:>8.2f} {result['std_vp']:>8.2f} {result['avg_length']:>8.1f}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model in 1v1 mode")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--opponent', type=str, default='all',
                        choices=['truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', 'strong', 'all'],
                        help='Opponent type to evaluate against (default: all)')
    parser.add_argument('--games', type=int, default=50,
                        help='Number of games to play (default: 50)')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel games (default: 8)')
    args = parser.parse_args()

    if args.opponent == 'all':
        full_benchmark(args.model, args.games, args.parallel)
    else:
        print(f"\nEvaluating vs {args.opponent.upper()}...")
        result = evaluate_model(args.model, args.opponent, args.games, args.parallel)
        print(f"\nResults:")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Avg VP: {result['avg_vp']:.2f} +/- {result['std_vp']:.2f}")
        print(f"  Avg Game Length: {result['avg_length']:.1f} moves")
        print(f"  Speed: {result['games_per_min']:.1f} games/min")
