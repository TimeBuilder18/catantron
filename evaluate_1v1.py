#!/usr/bin/env python3
"""
1v1 Model Evaluation Script
Evaluates a trained Catan model against AI opponents and reports detailed stats.

Usage:
    python evaluate_1v1.py --model models/my_model.pt
    python evaluate_1v1.py --model models/my_model.pt --opponent strong --games 100
    python evaluate_1v1.py --model models/my_model.pt --opponent all --games 50
"""

import argparse
import traceback
import threading
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from catan_env_pytorch import CatanEnv
from network_wrapper import NetworkWrapper
from curriculum_trainer_v3_stable import play_opponent_turn


# ──────────────────────────────────────────────────────────────────────────────
# Core game loop (mirrors the trainer exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _play_game(network, device, opponent_type, victory_points=10):
    """
    Play one full game. Returns (outcome, model_vp, opp_vp, num_moves).
    outcome: 'model_win' | 'opp_win' | 'timeout'
    """
    env = CatanEnv(player_id=0, victory_points_to_win=victory_points, num_players=2)
    obs, _ = env.reset()

    done     = False
    moves    = 0          # model-only move counter (matches trainer definition)
    MAX_MOVES = 800       # same as trainer

    while not done and moves < MAX_MOVES:
        game       = env.game_env.game
        current_id = game.players.index(game.get_current_player())

        # ── MODEL TURN ──────────────────────────────────────────────────────
        if current_id == 0:
            moves += 1

            with torch.no_grad():
                ap, vpp, ep, tgive, tget, _ = network.forward(
                    torch.FloatTensor(obs['observation']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device),
                )

            # Sample from network outputs (realistic eval, not always-greedy)
            def _sample(probs):
                p = probs.cpu().numpy()[0]
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                s = p.sum()
                p = p / s if s > 0 else np.ones_like(p) / len(p)
                return int(np.random.choice(len(p), p=p))

            action_id = _sample(ap)
            vertex_id = _sample(vpp)
            edge_id   = _sample(ep)
            give_idx  = _sample(tgive)
            get_idx   = _sample(tget)
            if give_idx == get_idx:            # never trade a resource for itself
                get_idx = (give_idx + 1) % 5

            obs, _, terminated, truncated, _ = env.step(
                action_id, vertex_id, edge_id,
                trade_give_idx=give_idx, trade_get_idx=get_idx,
            )

            if terminated or truncated:
                winner = game.check_victory_conditions()
                outcome = 'model_win' if (winner and game.players.index(winner) == 0) else 'opp_win'
                done = True

        # ── OPPONENT TURN ────────────────────────────────────────────────────
        else:
            success = play_opponent_turn(game, current_id, 1.0, opponent_type, None)
            if not success and game.can_end_turn():
                game.end_turn()

            # Auto-discard after 7 (opponent doesn't go through env.step)
            if game.waiting_for_discards:
                env.game_env._handle_automatic_discards()
                if game.waiting_for_discards:          # safety: force-clear
                    game.waiting_for_discards = False
                    game.players_must_discard  = []
                    game.players_discarded     = set()

            obs = env._get_obs()

            winner = game.check_victory_conditions()
            if winner is not None:
                outcome = 'model_win' if game.players.index(winner) == 0 else 'opp_win'
                done = True

    if not done:
        outcome = 'timeout'

    game    = env.game_env.game
    model_vp = game.players[0].calculate_victory_points()
    opp_vp   = game.players[1].calculate_victory_points()
    return outcome, model_vp, opp_vp, moves


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_path, opponent_type, num_games=100, num_parallel=8, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nw = NetworkWrapper(model_path=model_path, device=device)
    network = nw.policy
    network.eval()

    counters = {'model_win': 0, 'opp_win': 0, 'timeout': 0}
    model_vps, opp_vps, lengths = [], [], []
    lock = threading.Lock()
    done_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_parallel) as ex:
        futures = [ex.submit(_play_game, network, device, opponent_type) for _ in range(num_games)]

        for fut in as_completed(futures):
            try:
                outcome, mvp, ovp, nmoves = fut.result()
            except Exception:
                traceback.print_exc()
                continue

            with lock:
                counters[outcome] += 1
                model_vps.append(mvp)
                opp_vps.append(ovp)
                lengths.append(nmoves)
                done_count += 1

                if verbose and done_count % 10 == 0:
                    elapsed = time.time() - t0
                    wr = counters['model_win'] / done_count * 100
                    speed = done_count / elapsed * 60
                    print(f"  {done_count:4d}/{num_games}  WR {wr:5.1f}%  "
                          f"OppWin {counters['opp_win']:3d}  TO {counters['timeout']:3d}  "
                          f"{speed:.1f} g/min")

    total = done_count or 1
    return {
        'opponent':      opponent_type,
        'games':         total,
        # Win rates
        'win_rate':      counters['model_win'] / total * 100,
        'opp_win_rate':  counters['opp_win']   / total * 100,
        'timeout_rate':  counters['timeout']    / total * 100,
        # VP
        'model_avg_vp':  float(np.mean(model_vps)) if model_vps else 0,
        'model_std_vp':  float(np.std(model_vps))  if model_vps else 0,
        'opp_avg_vp':    float(np.mean(opp_vps))   if opp_vps   else 0,
        # Game length
        'avg_moves':     float(np.mean(lengths))   if lengths   else 0,
        'max_moves':     int(max(lengths))          if lengths   else 0,
        # Speed
        'games_per_min': total / (time.time() - t0) * 60,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

OPPONENTS = ['truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', 'strong']


def _print_result(r):
    print(f"\n  Win Rate  : {r['win_rate']:6.1f}%  (model wins)")
    print(f"  Opp Win   : {r['opp_win_rate']:6.1f}%")
    print(f"  Timeout   : {r['timeout_rate']:6.1f}%")
    print(f"  Model VP  : {r['model_avg_vp']:.2f} ± {r['model_std_vp']:.2f}")
    print(f"  Opp VP    : {r['opp_avg_vp']:.2f}")
    print(f"  Avg moves : {r['avg_moves']:.1f}  (max {r['max_moves']})")
    print(f"  Speed     : {r['games_per_min']:.1f} games/min")


def benchmark(model_path, num_games=50, num_parallel=8):
    print(f"\n{'='*70}")
    print(f"BENCHMARK  {model_path}")
    print(f"{'='*70}")
    print(f"{'Opponent':<16} {'WR%':>7} {'OppW%':>7} {'TO%':>6} "
          f"{'ModelVP':>8} {'OppVP':>7} {'AvgLen':>8}")
    print(f"{'-'*70}")

    all_results = {}
    for opp in OPPONENTS:
        print(f"\n  vs {opp.upper()} …")
        r = evaluate(model_path, opp, num_games, num_parallel, verbose=True)
        all_results[opp] = r
        print(f"  {opp:<14} {r['win_rate']:>6.1f}% {r['opp_win_rate']:>6.1f}% "
              f"{r['timeout_rate']:>5.1f}% {r['model_avg_vp']:>8.2f} "
              f"{r['opp_avg_vp']:>7.2f} {r['avg_moves']:>8.1f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Opponent':<16} {'WR%':>7} {'OppW%':>7} {'TO%':>6} "
          f"{'ModelVP':>8} {'OppVP':>7} {'AvgLen':>8}")
    print(f"{'-'*70}")
    for opp, r in all_results.items():
        print(f"{opp:<16} {r['win_rate']:>6.1f}% {r['opp_win_rate']:>6.1f}% "
              f"{r['timeout_rate']:>5.1f}% {r['model_avg_vp']:>8.2f} "
              f"{r['opp_avg_vp']:>7.2f} {r['avg_moves']:>8.1f}")
    print(f"{'='*70}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Catan model in 1v1 mode")
    parser.add_argument('--model',    type=str, required=True,  help='Path to .pt model file')
    parser.add_argument('--opponent', type=str, default='all',
                        choices=OPPONENTS + ['all'],
                        help='Opponent difficulty (default: all)')
    parser.add_argument('--games',    type=int, default=50,     help='Games per opponent (default: 50)')
    parser.add_argument('--parallel', type=int, default=8,      help='Parallel games (default: 8)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"✅ Loaded model from {args.model}")

    if args.opponent == 'all':
        benchmark(args.model, args.games, args.parallel)
    else:
        print(f"\nEvaluating vs {args.opponent.upper()} ({args.games} games) …")
        r = evaluate(args.model, args.opponent, args.games, args.parallel, verbose=True)
        _print_result(r)


if __name__ == '__main__':
    main()
