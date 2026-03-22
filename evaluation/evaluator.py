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

from environment.catan_env import CatanEnv
from model.model_loader import NetworkWrapper
from training.trainer import play_opponent_turn
from evaluation.quality_score import AgentQualityEvaluator


# ──────────────────────────────────────────────────────────────────────────────
# AQS helper functions
# ──────────────────────────────────────────────────────────────────────────────

DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _calculate_pip_count(player):
    """Sum pip values from all hexes adjacent to player's settlements and cities."""
    pips = 0
    seen_tiles = set()
    for structure in player.settlements + player.cities:
        for tile in structure.position.adjacent_tiles:
            if id(tile) not in seen_tiles and tile.number and tile.resource and tile.resource != 'desert':
                seen_tiles.add(id(tile))
                pips += DICE_PIPS.get(tile.number, 0)
    return pips


def _calculate_resource_diversity(player):
    """Count distinct resource types produced by player's settlements/cities."""
    resources = set()
    for structure in player.settlements + player.cities:
        for tile in structure.position.adjacent_tiles:
            if tile.resource and tile.resource != 'desert':
                rt = tile.get_resource_type()
                if rt:
                    resources.add(rt)
    return len(resources)


def _calculate_port_access(player, game_board):
    """Count how many ports the player has access to."""
    return len(game_board.get_player_ports(player))


def _estimate_resources_spent(player):
    """Estimate total resources spent from structures built."""
    settlements = len(player.settlements)
    cities = len(player.cities)
    roads = len(player.roads)
    dev_cards = sum(player.development_cards.values()) + player.knights_played
    paid_settlements = max(0, settlements + cities - 2)  # cities were settlements first
    paid_roads = max(0, roads - 2)  # 2 initial roads are free
    return (paid_settlements * 4) + (cities * 5) + (paid_roads * 2) + (dev_cards * 3)


# ──────────────────────────────────────────────────────────────────────────────
# Core game loop (mirrors the trainer exactly)
# ──────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0:  'roll_dice',
    1:  'place_sett',
    2:  'place_road',
    3:  'build_sett',
    4:  'build_city',
    5:  'build_road',
    6:  'buy_dev',
    7:  'end_turn',
    8:  'wait',
    9:  'trade_bank',
    10: 'do_nothing',
    11: 'play_knight',
    12: 'play_monopoly',
    13: 'play_year_plenty',
}


def _play_game(network, device, opponent_type, victory_points=10):
    """
    Play one full game. Returns a stats dict with outcome, VP, structures, actions.
    """
    env = CatanEnv(player_id=0, victory_points_to_win=victory_points, num_players=2)
    obs, _ = env.reset()

    done      = False
    moves     = 0          # model-only move counter (matches trainer definition)
    MAX_MOVES = 3000       # 800 was too low: model does many trades/turn → only ~80 real turns
    action_counts = {}     # action_id → count
    bank_trade_count = 0   # AQS: count bank trades (action_id 9)
    robber_placement_count = 0  # AQS: count robber placements (action_id 11)

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

            action_counts[action_id] = action_counts.get(action_id, 0) + 1
            if action_id == 9:
                bank_trade_count += 1
            if action_id == 11:
                robber_placement_count += 1

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

    game  = env.game_env.game
    mp    = game.players[0]
    op    = game.players[1]
    return {
        'outcome':        outcome,
        'model_vp':       mp.calculate_victory_points(),
        'opp_vp':         op.calculate_victory_points(),
        'moves':          moves,
        # structures
        'model_sett':     len(mp.settlements),
        'model_cities':   len(mp.cities),
        'model_roads':    len(mp.roads),
        'opp_sett':       len(op.settlements),
        'opp_cities':     len(op.cities),
        'opp_roads':      len(op.roads),
        # bonuses
        'model_knights':  mp.knights_played,
        'model_largest':  int(mp.has_largest_army),
        'model_longest':  int(mp.has_longest_road),
        'opp_knights':    op.knights_played,
        'opp_largest':    int(op.has_largest_army),
        'opp_longest':    int(op.has_longest_road),
        # dev cards remaining in hand
        'model_dev_cards': sum(mp.development_cards.values()),
        # AQS fields
        'dev_cards_bought':     sum(mp.development_cards.values()) + mp.knights_played,
        'dev_cards_played':     mp.knights_played,
        'total_resources_spent': _estimate_resources_spent(mp),
        'bank_trades_made':     bank_trade_count,
        'robber_placements':    robber_placement_count,
        'pip_count':            _calculate_pip_count(mp),
        'resource_diversity':   _calculate_resource_diversity(mp),
        'port_access_count':    _calculate_port_access(mp, game.game_board),
        # action distribution
        'action_counts':  action_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_path, opponent_type, num_games=100, num_parallel=8, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Auto-detect hidden_dim from checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    hidden_dim = ckpt.get('hidden_dim', None)
    if hidden_dim is None:
        # Infer from fc1 weight shape (fc1 is hidden_dim x hidden_dim)
        sd = ckpt.get('model_state_dict', {})
        if 'fc1.weight' in sd:
            hidden_dim = sd['fc1.weight'].shape[0]
        else:
            hidden_dim = 256
    del ckpt
    nw = NetworkWrapper(model_path=model_path, device=device, hidden_dim=hidden_dim)
    network = nw.policy
    network.eval()

    counters = {'model_win': 0, 'opp_win': 0, 'timeout': 0}
    lists = {k: [] for k in (
        'model_vp', 'opp_vp', 'moves',
        'model_sett', 'model_cities', 'model_roads',
        'opp_sett', 'opp_cities', 'opp_roads',
        'model_knights', 'model_largest', 'model_longest',
        'opp_knights', 'opp_largest', 'opp_longest',
        'model_dev_cards',
        'dev_cards_bought', 'dev_cards_played', 'total_resources_spent',
        'bank_trades_made', 'robber_placements', 'pip_count',
        'resource_diversity', 'port_access_count',
    )}
    combined_actions = {}
    lock = threading.Lock()
    done_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_parallel) as ex:
        futures = [ex.submit(_play_game, network, device, opponent_type) for _ in range(num_games)]

        for fut in as_completed(futures):
            try:
                g = fut.result()
            except Exception:
                traceback.print_exc()
                continue

            with lock:
                counters[g['outcome']] += 1
                for k in lists:
                    lists[k].append(g[k])
                for aid, cnt in g['action_counts'].items():
                    combined_actions[aid] = combined_actions.get(aid, 0) + cnt
                done_count += 1

                if verbose and done_count % 10 == 0:
                    elapsed = time.time() - t0
                    wr = counters['model_win'] / done_count * 100
                    speed = done_count / elapsed * 60
                    print(f"  {done_count:4d}/{num_games}  WR {wr:5.1f}%  "
                          f"OppWin {counters['opp_win']:3d}  TO {counters['timeout']:3d}  "
                          f"{speed:.1f} g/min")

    def avg(k): return float(np.mean(lists[k])) if lists[k] else 0
    def pct(k): return float(np.mean(lists[k])) * 100 if lists[k] else 0

    total = done_count or 1
    total_actions = sum(combined_actions.values()) or 1
    action_pcts = {ACTION_NAMES.get(aid, str(aid)): cnt / total_actions * 100
                   for aid, cnt in sorted(combined_actions.items())}

    # Calculate AQS
    aqs_evaluator = AgentQualityEvaluator(window_size=num_games)
    for i in range(len(lists['model_vp'])):
        aqs_evaluator.record_game({
            'won': lists['model_vp'][i] >= 10,
            'agent_vp': lists['model_vp'][i],
            'opponent_vp': lists['opp_vp'][i],
            'agent_turns': lists['moves'][i],
            'settlements_built': lists['model_sett'][i] + lists['model_cities'][i],
            'cities_built': lists['model_cities'][i],
            'roads_built': lists['model_roads'][i],
            'has_longest_road': bool(lists['model_longest'][i]),
            'has_largest_army': bool(lists['model_largest'][i]),
            'knights_played': lists['model_knights'][i],
            'opponent_type': opponent_type,
            'dev_cards_bought': lists['dev_cards_bought'][i],
            'dev_cards_played': lists['dev_cards_played'][i],
            'total_resources_spent': lists['total_resources_spent'][i],
            'bank_trades_made': lists['bank_trades_made'][i],
            'robber_placements': lists['robber_placements'][i],
            'pip_count': lists['pip_count'][i],
            'resource_diversity': lists['resource_diversity'][i],
            'port_access_count': lists['port_access_count'][i],
        })
    aqs_breakdown = aqs_evaluator.get_breakdown()

    return {
        'opponent':         opponent_type,
        'games':            total,
        # Win rates
        'win_rate':         counters['model_win'] / total * 100,
        'opp_win_rate':     counters['opp_win']   / total * 100,
        'timeout_rate':     counters['timeout']    / total * 100,
        # VP
        'model_avg_vp':     avg('model_vp'),
        'model_std_vp':     float(np.std(lists['model_vp'])) if lists['model_vp'] else 0,
        'opp_avg_vp':       avg('opp_vp'),
        # Game length
        'avg_moves':        avg('moves'),
        'max_moves':        int(max(lists['moves'])) if lists['moves'] else 0,
        # Structures
        'model_avg_sett':   avg('model_sett'),
        'model_avg_cities': avg('model_cities'),
        'model_avg_roads':  avg('model_roads'),
        'opp_avg_sett':     avg('opp_sett'),
        'opp_avg_cities':   avg('opp_cities'),
        'opp_avg_roads':    avg('opp_roads'),
        # Bonuses
        'model_avg_knights': avg('model_knights'),
        'model_largest_pct': pct('model_largest'),
        'model_longest_pct': pct('model_longest'),
        'opp_avg_knights':   avg('opp_knights'),
        'opp_largest_pct':   pct('opp_largest'),
        'opp_longest_pct':   pct('opp_longest'),
        'model_avg_dev':    avg('model_dev_cards'),
        # Actions
        'action_pcts':      action_pcts,
        # Speed
        'games_per_min':    total / (time.time() - t0) * 60,
        # AQS
        'aqs':              aqs_breakdown['aqs'],
        'aqs_tier':         aqs_breakdown['tier'],
        'aqs_breakdown':    aqs_breakdown,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

OPPONENTS = ['truly_random', 'weighted_random', 'random', 'very_weak', 'weak', 'medium', 'strong']


def _print_result(r):
    print(f"\n  ── Outcomes ──────────────────────────────")
    print(f"  Win Rate  : {r['win_rate']:6.1f}%  (model wins)")
    print(f"  Opp Win   : {r['opp_win_rate']:6.1f}%")
    print(f"  Timeout   : {r['timeout_rate']:6.1f}%")
    print(f"\n  ── Victory Points ────────────────────────")
    print(f"  Model VP  : {r['model_avg_vp']:.2f} ± {r['model_std_vp']:.2f}")
    print(f"  Opp VP    : {r['opp_avg_vp']:.2f}")
    print(f"\n  ── Structures (avg at game end) ──────────")
    print(f"  {'':16}  {'Model':>7}  {'Opp':>7}")
    print(f"  {'Settlements':16}  {r['model_avg_sett']:>7.2f}  {r['opp_avg_sett']:>7.2f}")
    print(f"  {'Cities':16}  {r['model_avg_cities']:>7.2f}  {r['opp_avg_cities']:>7.2f}")
    print(f"  {'Roads':16}  {r['model_avg_roads']:>7.2f}  {r['opp_avg_roads']:>7.2f}")
    print(f"\n  ── Bonuses ───────────────────────────────")
    print(f"  {'':16}  {'Model':>7}  {'Opp':>7}")
    print(f"  {'Knights played':16}  {r['model_avg_knights']:>7.2f}  {r['opp_avg_knights']:>7.2f}")
    print(f"  {'Largest Army %':16}  {r['model_largest_pct']:>6.1f}%  {r['opp_largest_pct']:>6.1f}%")
    print(f"  {'Longest Road %':16}  {r['model_longest_pct']:>6.1f}%  {r['opp_longest_pct']:>6.1f}%")
    print(f"  {'Dev cards (hand)':16}  {r['model_avg_dev']:>7.2f}")
    print(f"\n  ── Action Distribution ───────────────────")
    for name, pct in sorted(r['action_pcts'].items(), key=lambda x: -x[1]):
        bar = '█' * int(pct / 2)
        print(f"  {name:<16}  {pct:5.1f}%  {bar}")
    print(f"\n  ── Game Length ───────────────────────────")
    print(f"  Avg moves : {r['avg_moves']:.1f}  (max {r['max_moves']})")
    print(f"  Speed     : {r['games_per_min']:.1f} games/min")

    if 'aqs' in r:
        print(f"\n  ── Agent Quality Score ────────────────────")
        print(f"  AQS         : {r['aqs']:.0f} / 1000  [{r['aqs_tier']}]")
        if 'aqs_breakdown' in r:
            for name, info in r['aqs_breakdown']['components'].items():
                bar_len = int(info['score'] / 2)
                bar = '\u2588' * bar_len + '\u2591' * (50 - bar_len)
                print(f"  {name:15s} {bar} {info['score']:5.1f} "
                      f"(x{info['weight']:.2f} = {info['weighted']:5.1f})")


def benchmark(model_path, num_games=50, num_parallel=8):
    HDR = f"{'Opponent':<16} {'WR%':>7} {'AQS':>5} {'OppW%':>7} {'TO%':>5} {'MVP':>6} {'OVP':>6} {'Sett':>5} {'City':>5} {'Road':>5} {'Knt':>5} {'Len':>6}"
    SEP = '-' * len(HDR)

    print(f"\n{'='*len(HDR)}")
    print(f"BENCHMARK  {model_path}")
    print(f"{'='*len(HDR)}")
    print(HDR)
    print(SEP)

    all_results = {}
    for opp in OPPONENTS:
        print(f"\n  vs {opp.upper()} …")
        r = evaluate(model_path, opp, num_games, num_parallel, verbose=True)
        all_results[opp] = r
        _print_result(r)

    print(f"\n{'='*len(HDR)}")
    print("SUMMARY")
    print(f"{'='*len(HDR)}")
    print(HDR)
    print(SEP)
    for opp, r in all_results.items():
        print(f"{opp:<16} {r['win_rate']:>6.1f}% {r['aqs']:>5.0f} {r['opp_win_rate']:>6.1f}% "
              f"{r['timeout_rate']:>4.1f}% {r['model_avg_vp']:>6.2f} {r['opp_avg_vp']:>6.2f} "
              f"{r['model_avg_sett']:>5.2f} {r['model_avg_cities']:>5.2f} "
              f"{r['model_avg_roads']:>5.2f} {r['model_avg_knights']:>5.2f} "
              f"{r['avg_moves']:>6.1f}")
    print(f"{'='*len(HDR)}")
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
