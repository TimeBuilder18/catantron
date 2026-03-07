#!/usr/bin/env python3
"""
1v1 Model Evaluation Script - evaluates a trained model vs AI opponents.
"""

import argparse
import traceback
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from catan_env_pytorch import CatanEnv
from network_wrapper import NetworkWrapper
from curriculum_trainer_v3_stable import play_opponent_turn


def evaluate_model(model_path, opponent_type='random', num_games=100, num_parallel=8, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network_wrapper = NetworkWrapper(model_path=model_path, device=device)
    network = network_wrapper.policy
    network.eval()

    results = {
        'model_wins': 0, 'opp_wins': 0, 'timeouts': 0,
        'vps': [], 'lengths': []
    }
    _lock = __import__('threading').Lock()

    def safe_probs(probs):
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        s = probs.sum()
        if s <= 0:
            return np.ones_like(probs) / len(probs)
        return probs / s

    def play_single_game():
        # Use CatanEnv directly — same as what the trainer wraps
        env = CatanEnv(player_id=0, victory_points_to_win=10, num_players=2)
        obs, _ = env.reset()

        done = False
        moves = 0
        max_moves = 800  # Match trainer's max_moves
        outcome = 'timeout'  # 'model_win', 'opp_win', 'timeout'

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                moves += 1

                with torch.no_grad():
                    ap, vpp, ep, _, _, _ = network.forward(
                        torch.FloatTensor(obs['observation']).unsqueeze(0).to(device),
                        torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device),
                        torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device),
                        torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device),
                    )

                action_id = int(torch.argmax(ap[0]).item())
                vertex_id = int(torch.argmax(vpp[0]).item())
                edge_id   = int(torch.argmax(ep[0]).item())

                obs, _, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0
                )
                if terminated or truncated:
                    winner = game.check_victory_conditions()
                    if winner is not None and game.players.index(winner) == 0:
                        outcome = 'model_win'
                    else:
                        outcome = 'opp_win'
                    done = True

            else:
                success = play_opponent_turn(game, current_id, 1.0, opponent_type, None)
                if not success and game.can_end_turn():
                    game.end_turn()

                # Handle discards after 7 is rolled (opponent doesn't go through env.step)
                if game.waiting_for_discards:
                    env.game_env._handle_automatic_discards()
                    # Safety: if discards still pending, force-clear
                    if game.waiting_for_discards:
                        game.waiting_for_discards = False
                        game.players_must_discard = []
                        game.players_discarded = set()

                # Refresh observation for model
                obs = env._get_obs()

                winner = game.check_victory_conditions()
                if winner is not None:
                    outcome = 'model_win' if game.players.index(winner) == 0 else 'opp_win'
                    done = True

        my_vp = env.game_env.game.players[0].calculate_victory_points()
        return outcome, my_vp, moves

    start_time = time.time()
    games_done = 0

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(play_single_game) for _ in range(num_games)]

        for future in as_completed(futures):
            try:
                outcome, vp, length = future.result()
                with _lock:
                    if outcome == 'model_win':
                        results['model_wins'] += 1
                    elif outcome == 'opp_win':
                        results['opp_wins'] += 1
                    else:
                        results['timeouts'] += 1
                    results['vps'].append(vp)
                    results['lengths'].append(length)
                    games_done += 1

                if verbose and games_done % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = games_done / elapsed * 60
                    wr = results['model_wins'] / games_done * 100
                    print(f"  Progress: {games_done}/{num_games} | "
                          f"WR: {wr:.1f}% | OppWin: {results['opp_wins']} | "
                          f"TO: {results['timeouts']} | Speed: {speed:.1f} g/min")
            except Exception:
                traceback.print_exc()

    total_time = time.time() - start_time

    return {
        'opponent': opponent_type,
        'games': num_games,
        'win_rate': results['model_wins'] / num_games * 100,
        'opp_win_rate': results['opp_wins'] / num_games * 100,
        'timeout_rate': results['timeouts'] / num_games * 100,
        'avg_vp': float(np.mean(results['vps'])) if results['vps'] else 0,
        'std_vp': float(np.std(results['vps'])) if results['vps'] else 0,
        'avg_length': float(np.mean(results['lengths'])) if results['lengths'] else 0,
        'time_seconds': total_time,
        'games_per_min': num_games / total_time * 60,
    }


def full_benchmark(model_path, num_games=50, num_parallel=8):
    opponents = ['truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', 'strong']
    all_results = {}

    print("\n" + "=" * 75)
    print("1v1 MODEL BENCHMARK")
    print("=" * 75)
    print(f"Model: {model_path}")
    print(f"Games per opponent: {num_games} | Parallel: {num_parallel}")
    print("=" * 75 + "\n")

    for opp in opponents:
        print(f"\nEvaluating vs {opp.upper()}...")
        r = evaluate_model(model_path, opp, num_games, num_parallel, verbose=True)
        all_results[opp] = r

    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Opponent':<16} {'WR%':>7} {'OppWR%':>8} {'TO%':>6} {'AvgVP':>7} {'AvgLen':>8}")
    print("-" * 75)
    for opp, r in all_results.items():
        print(f"{opp:<16} {r['win_rate']:>6.1f}% {r['opp_win_rate']:>7.1f}% "
              f"{r['timeout_rate']:>5.1f}% {r['avg_vp']:>7.2f} {r['avg_length']:>8.1f}")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model in 1v1 mode")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--opponent', type=str, default='all',
                        choices=['truly_random', 'weighted_random', 'random',
                                 'very_weak', 'weak', 'medium', 'strong', 'all'])
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--parallel', type=int, default=8)
    args = parser.parse_args()

    if args.opponent == 'all':
        full_benchmark(args.model, args.games, args.parallel)
    else:
        print(f"\nEvaluating vs {args.opponent.upper()}...")
        r = evaluate_model(args.model, args.opponent, args.games, args.parallel)
        print(f"\nResults:")
        print(f"  Model Win Rate:  {r['win_rate']:.1f}%")
        print(f"  Opp Win Rate:    {r['opp_win_rate']:.1f}%")
        print(f"  Timeout Rate:    {r['timeout_rate']:.1f}%")
        print(f"  Avg VP:          {r['avg_vp']:.2f} +/- {r['std_vp']:.2f}")
        print(f"  Avg Game Length: {r['avg_length']:.1f} moves")
        print(f"  Speed:           {r['games_per_min']:.1f} games/min")
