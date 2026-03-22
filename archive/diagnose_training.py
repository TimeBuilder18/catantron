#!/usr/bin/env python3
"""
Diagnostic script to understand why agent isn't learning to build.

This script plays a few games and logs:
1. What actions are available each turn
2. What the agent chooses
3. Whether builds succeed
4. Resource state
"""

import numpy as np
import torch
from collections import defaultdict

from catan_env_pytorch import CatanEnv
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from curriculum_trainer_v3_stable import play_opponent_turn


def diagnose_game(network=None, opponent_type='passive', verbose=True):
    """Play a single game with detailed logging."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = PBRSFixedRewardWrapper(player_id=0, victory_points_to_win=10, num_players=2)
    obs, _ = env.reset()

    stats = {
        'total_turns': 0,
        'build_settlement_available': 0,
        'build_city_available': 0,
        'build_road_available': 0,
        'build_settlement_attempted': 0,
        'build_city_attempted': 0,
        'build_road_attempted': 0,
        'build_settlement_success': 0,
        'build_city_success': 0,
        'build_road_success': 0,
        'end_turn_chosen': 0,
        'end_turn_with_build_available': 0,
        'resources_when_build_available': [],
        'action_distributions': defaultdict(list),
    }

    action_names = [
        'roll_dice', 'place_settlement', 'place_road',
        'build_settlement', 'build_city', 'build_road',
        'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing'
    ]

    done = False
    moves = 0
    max_moves = 300

    while not done and moves < max_moves:
        game = env.game_env.game
        current = game.get_current_player()
        current_id = game.players.index(current)

        if current_id == 0:
            moves += 1
            stats['total_turns'] += 1

            action_mask = obs['action_mask']
            legal_actions = obs.get('legal_actions', [])
            resources = obs.get('my_resources', {})

            # Track availability
            if action_mask[3] == 1:  # build_settlement
                stats['build_settlement_available'] += 1
                stats['resources_when_build_available'].append(dict(resources))
            if action_mask[4] == 1:  # build_city
                stats['build_city_available'] += 1
            if action_mask[5] == 1:  # build_road
                stats['build_road_available'] += 1

            # Choose action
            if network is None:
                # Random policy
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) == 0:
                    valid_actions = [7]  # end_turn fallback
                action_id = np.random.choice(valid_actions)
                vertex_id = np.random.randint(0, 54)
                edge_id = np.random.randint(0, 72)
            else:
                # Use network
                with torch.no_grad():
                    observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(device)
                    action_mask_t = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)
                    vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device)
                    edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device)

                    action_probs, vertex_probs, edge_probs, _, _, _ = network.forward(
                        observation, action_mask_t, vertex_mask, edge_mask
                    )

                    ap = action_probs.cpu().numpy()[0]
                    vp = vertex_probs.cpu().numpy()[0]
                    ep = edge_probs.cpu().numpy()[0]

                # Normalize
                ap = np.nan_to_num(ap, nan=0.0)
                if ap.sum() > 0:
                    ap = ap / ap.sum()
                else:
                    ap = np.ones(len(ap)) / len(ap)

                vp = np.nan_to_num(vp, nan=0.0)
                if vp.sum() > 0:
                    vp = vp / vp.sum()
                else:
                    vp = np.ones(len(vp)) / len(vp)

                ep = np.nan_to_num(ep, nan=0.0)
                if ep.sum() > 0:
                    ep = ep / ep.sum()
                else:
                    ep = np.ones(len(ep)) / len(ep)

                action_id = np.random.choice(len(ap), p=ap)
                vertex_id = np.random.choice(len(vp), p=vp)
                edge_id = np.random.choice(len(ep), p=ep)

                # Record action distribution for valid actions
                for i, prob in enumerate(ap):
                    if action_mask[i] == 1:
                        stats['action_distributions'][action_names[i]].append(prob)

            action_name = action_names[action_id]

            # Track attempts
            if action_name == 'build_settlement':
                stats['build_settlement_attempted'] += 1
            elif action_name == 'build_city':
                stats['build_city_attempted'] += 1
            elif action_name == 'build_road':
                stats['build_road_attempted'] += 1
            elif action_name == 'end_turn':
                stats['end_turn_chosen'] += 1
                if action_mask[3] == 1 or action_mask[4] == 1 or action_mask[5] == 1:
                    stats['end_turn_with_build_available'] += 1

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(
                action_id, vertex_id, edge_id,
                trade_give_idx=0, trade_get_idx=0
            )

            # Track success
            if action_name == 'build_settlement' and info.get('success', False):
                stats['build_settlement_success'] += 1
                if verbose:
                    print(f"  Turn {moves}: BUILD SETTLEMENT SUCCESS! VP now: {next_obs.get('my_victory_points', 0)}")
            elif action_name == 'build_city' and info.get('success', False):
                stats['build_city_success'] += 1
                if verbose:
                    print(f"  Turn {moves}: BUILD CITY SUCCESS! VP now: {next_obs.get('my_victory_points', 0)}")
            elif action_name == 'build_road' and info.get('success', False):
                stats['build_road_success'] += 1

            obs = next_obs
            done = terminated or truncated
        else:
            # Opponent's turn
            play_opponent_turn(game, current_id, 1.0, opponent_type, None)
            winner = game.check_victory_conditions()
            if winner is not None:
                done = True

    my_vp = env.game_env.game.players[0].calculate_victory_points()
    winner = env.game_env.game.check_victory_conditions()
    won = winner is not None and game.players.index(winner) == 0

    stats['final_vp'] = my_vp
    stats['won'] = won

    return stats


def run_diagnostic(model_path=None, num_games=10, opponent='passive'):
    """Run diagnostic over multiple games."""
    print("=" * 70)
    print("TRAINING DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {model_path if model_path else 'Random Policy'}")
    print(f"Opponent: {opponent}")
    print(f"Games: {num_games}")
    print("=" * 70)

    network = None
    if model_path:
        from network_wrapper import NetworkWrapper
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wrapper = NetworkWrapper(model_path=model_path, device=device)
        network = wrapper.policy
        network.eval()

    all_stats = []
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        stats = diagnose_game(network, opponent, verbose=True)
        all_stats.append(stats)
        print(f"Final VP: {stats['final_vp']}, Won: {stats['won']}")

    # Aggregate stats
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    avg_turns = np.mean([s['total_turns'] for s in all_stats])
    print(f"\nAvg turns per game: {avg_turns:.1f}")

    print(f"\n--- BUILD AVAILABILITY ---")
    total_turns = sum(s['total_turns'] for s in all_stats)
    total_settle_avail = sum(s['build_settlement_available'] for s in all_stats)
    total_city_avail = sum(s['build_city_available'] for s in all_stats)
    total_road_avail = sum(s['build_road_available'] for s in all_stats)

    print(f"Settlement available: {total_settle_avail}/{total_turns} turns ({100*total_settle_avail/total_turns:.1f}%)")
    print(f"City available: {total_city_avail}/{total_turns} turns ({100*total_city_avail/total_turns:.1f}%)")
    print(f"Road available: {total_road_avail}/{total_turns} turns ({100*total_road_avail/total_turns:.1f}%)")

    print(f"\n--- BUILD ATTEMPTS ---")
    total_settle_attempt = sum(s['build_settlement_attempted'] for s in all_stats)
    total_city_attempt = sum(s['build_city_attempted'] for s in all_stats)
    total_road_attempt = sum(s['build_road_attempted'] for s in all_stats)

    print(f"Settlement attempts: {total_settle_attempt} ({100*total_settle_attempt/max(1,total_settle_avail):.1f}% of available)")
    print(f"City attempts: {total_city_attempt} ({100*total_city_attempt/max(1,total_city_avail):.1f}% of available)")
    print(f"Road attempts: {total_road_attempt} ({100*total_road_attempt/max(1,total_road_avail):.1f}% of available)")

    print(f"\n--- BUILD SUCCESS ---")
    total_settle_success = sum(s['build_settlement_success'] for s in all_stats)
    total_city_success = sum(s['build_city_success'] for s in all_stats)
    total_road_success = sum(s['build_road_success'] for s in all_stats)

    print(f"Settlement success: {total_settle_success}/{total_settle_attempt} ({100*total_settle_success/max(1,total_settle_attempt):.1f}%)")
    print(f"City success: {total_city_success}/{total_city_attempt} ({100*total_city_success/max(1,total_city_attempt):.1f}%)")
    print(f"Road success: {total_road_success}/{total_road_attempt} ({100*total_road_success/max(1,total_road_attempt):.1f}%)")

    print(f"\n--- END TURN BEHAVIOR ---")
    total_end_turn = sum(s['end_turn_chosen'] for s in all_stats)
    total_end_with_build = sum(s['end_turn_with_build_available'] for s in all_stats)
    print(f"End turn chosen: {total_end_turn} times")
    print(f"End turn with build available: {total_end_with_build} ({100*total_end_with_build/max(1,total_end_turn):.1f}%)")

    if model_path:
        print(f"\n--- ACTION PROBABILITIES (when action is valid) ---")
        for s in all_stats:
            for action_name, probs in s['action_distributions'].items():
                if probs:
                    avg_prob = np.mean(probs)
                    print(f"  {action_name}: avg prob = {avg_prob:.4f}")

    print(f"\n--- RESULTS ---")
    wins = sum(1 for s in all_stats if s['won'])
    avg_vp = np.mean([s['final_vp'] for s in all_stats])
    print(f"Win rate: {100*wins/num_games:.1f}%")
    print(f"Avg VP: {avg_vp:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to model (None for random)')
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--opponent', type=str, default='passive')
    args = parser.parse_args()

    run_diagnostic(args.model, args.games, args.opponent)
