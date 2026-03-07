#!/usr/bin/env python3
"""
Debug a single game to understand why games always timeout.
Prints VP snapshots and action breakdowns.
"""
import sys
sys.path.insert(0, '.')

import argparse
from collections import Counter

from catan_env_pytorch import CatanEnv
from curriculum_trainer_v3_stable import play_opponent_turn

ACTION_NAMES = [
    'roll_dice', 'place_settlement', 'place_road',
    'build_settlement', 'build_city', 'build_road',
    'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing',
    'play_knight', 'play_monopoly', 'play_year_of_plenty'
]


def debug_game(opponent_type='strong', model_path=None, max_moves=800):
    env = CatanEnv(player_id=0, victory_points_to_win=10, num_players=2)
    obs, _ = env.reset()

    network = None
    if model_path:
        import torch
        from network_wrapper import NetworkWrapper
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nw = NetworkWrapper(model_path=model_path, device=device)
        network = nw.policy
        network.eval()

    done = False
    moves = 0
    total_iters = 0
    p0_actions = Counter()
    p1_actions_count = 0
    p1_end_turns = 0

    print(f"\n{'='*60}")
    print(f"DEBUG GAME: player=model vs opponent={opponent_type}")
    print(f"{'='*60}")

    while not done and moves < max_moves:
        total_iters += 1
        if total_iters > 50000:
            print("SAFETY BREAK: too many iterations without move increment!")
            break

        game = env.game_env.game
        current = game.get_current_player()
        current_id = game.players.index(current)
        p0_vp = game.players[0].calculate_victory_points()
        p1_vp = game.players[1].calculate_victory_points()

        # Print VP every 100 model moves
        if moves % 100 == 0 and moves > 0 and current_id == 0:
            p0_res = sum(game.players[0].resources.values())
            p1_res = sum(game.players[1].resources.values())
            print(f"  Move {moves:4d}: P0_VP={p0_vp} P1_VP={p1_vp} | "
                  f"P0_res={p0_res} P1_res={p1_res} | "
                  f"P1_turns={p1_end_turns}")

        if current_id == 0:
            moves += 1
            action_mask = obs['action_mask']

            if network:
                import torch
                device = next(network.parameters()).device
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
            else:
                # Pick first valid action
                valid = [i for i, m in enumerate(action_mask) if m]
                action_id = valid[0] if valid else 7
                vertex_id, edge_id = 0, 0

            p0_actions[ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else str(action_id)] += 1

            obs, _, terminated, truncated, info = env.step(
                action_id, vertex_id, edge_id, trade_give_idx=0, trade_get_idx=0
            )
            if terminated or truncated:
                winner = game.check_victory_conditions()
                idx = game.players.index(winner) if winner else None
                print(f"\n*** GAME OVER (model step): winner=P{idx}, "
                      f"P0_VP={p0_vp}, P1_VP={p1_vp} ***")
                done = True

        else:
            p1_actions_count += 1
            success = play_opponent_turn(game, current_id, 1.0, opponent_type, None)
            if not success and game.can_end_turn():
                game.end_turn()
                p1_end_turns += 1
            if game.waiting_for_discards:
                env.game_env._handle_automatic_discards()
                if game.waiting_for_discards:
                    game.waiting_for_discards = False
                    game.players_must_discard = []
                    game.players_discarded = set()

            # Count P1 end_turn calls from within play_rule_based_turn
            if success and not game.can_roll_dice() and game.turn_phase == 'ROLL_DICE':
                p1_end_turns += 1  # opponent ended their turn

            obs = env._get_obs()
            winner = game.check_victory_conditions()
            if winner is not None:
                idx = game.players.index(winner)
                print(f"\n*** GAME OVER (opp step): winner=P{idx}, "
                      f"P0_VP={p0_vp}, P1_VP={p1_vp} ***")
                done = True

    # Final report
    game = env.game_env.game
    p0_vp = game.players[0].calculate_victory_points()
    p1_vp = game.players[1].calculate_victory_points()
    p0_set = len(game.players[0].settlements)
    p0_cit = len(game.players[0].cities)
    p1_set = len(game.players[1].settlements)
    p1_cit = len(game.players[1].cities)
    p0_res = sum(game.players[0].resources.values())
    p1_res = sum(game.players[1].resources.values())

    outcome = "TIMEOUT" if not done else "WON"
    print(f"\n{'='*60}")
    print(f"RESULT: {outcome} after {moves} model moves ({total_iters} total iters)")
    print(f"  P0: VP={p0_vp} settlements={p0_set} cities={p0_cit} resources={p0_res}")
    print(f"  P1: VP={p1_vp} settlements={p1_set} cities={p1_cit} resources={p1_res}")
    print(f"  P1 opponent iterations: {p1_actions_count}, end_turns: {p1_end_turns}")
    print(f"\nP0 action distribution:")
    for action, count in sorted(p0_actions.items(), key=lambda x: -x[1]):
        print(f"  {action:<20} {count:5d}  ({count/moves*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--opponent', type=str, default='strong')
    parser.add_argument('--moves', type=int, default=800)
    args = parser.parse_args()
    debug_game(args.opponent, args.model, args.moves)
