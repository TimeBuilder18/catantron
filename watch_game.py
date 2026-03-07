#!/usr/bin/env python3
"""
Watch the model play a single game turn by turn.

Usage:
    python watch_game.py --model models/v3_overnight2_game178016.pt
    python watch_game.py --model models/v3_overnight2_game178016.pt --opponent strong --delay 0.3
    python watch_game.py --model models/v3_overnight2_game178016.pt --delay 0   # no pause
"""

import argparse
import time
import numpy as np
import torch

from catan_env_pytorch import CatanEnv
from game_system import ResourceType
from network_wrapper import NetworkWrapper
from curriculum_trainer_v3_stable import play_opponent_turn

RESOURCE_NAMES = {
    ResourceType.WOOD:  'Wood',
    ResourceType.BRICK: 'Brick',
    ResourceType.WHEAT: 'Wheat',
    ResourceType.SHEEP: 'Sheep',
    ResourceType.ORE:   'Ore',
}
RES_SHORT = {
    ResourceType.WOOD:  'Wo',
    ResourceType.BRICK: 'Br',
    ResourceType.WHEAT: 'Wh',
    ResourceType.SHEEP: 'Sh',
    ResourceType.ORE:   'Or',
}
TRADE_RES = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
             ResourceType.SHEEP, ResourceType.ORE]

ACTION_NAMES = {
    0: 'roll_dice', 1: 'place_sett', 2: 'place_road',
    3: 'build_sett', 4: 'build_city', 5: 'build_road',
    6: 'buy_dev', 7: 'end_turn', 8: 'wait',
    9: 'trade_bank', 10: 'do_nothing',
    11: 'play_knight', 12: 'play_monopoly', 13: 'play_year_plenty',
}


def _greedy(probs):
    """Argmax — zero entropy, always picks the highest-probability action."""
    return int(torch.argmax(probs[0]).item())


def res_str(player):
    parts = [f"{RES_SHORT[r]}:{player.resources[r]}" for r in TRADE_RES]
    return ' '.join(parts)


def state_line(game, label):
    mp = game.players[0]
    op = game.players[1]
    mvp = mp.calculate_victory_points()
    ovp = op.calculate_victory_points()
    return (f"  {label}  VP {mvp}/{ovp}  "
            f"Sett {len(mp.settlements)}/{len(op.settlements)}  "
            f"City {len(mp.cities)}/{len(op.cities)}  "
            f"Road {len(mp.roads)}/{len(op.roads)}  "
            f"Knt {mp.knights_played}/{op.knights_played}  "
            f"[{res_str(mp)}]")


def watch(model_path, opponent_type, victory_points, delay):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nw = NetworkWrapper(model_path=model_path, device=device)
    network = nw.policy
    network.eval()

    env = CatanEnv(player_id=0, victory_points_to_win=victory_points, num_players=2)
    obs, _ = env.reset()

    print(f"\n{'='*70}")
    print(f"WATCHING: model vs {opponent_type.upper()}  ({victory_points}VP)")
    print(f"{'='*70}")
    print(f"  Legend: VP model/opp | Sett/City/Road/Knights | model resources")
    print(f"{'='*70}\n")

    done      = False
    moves     = 0
    turn_num  = 0
    MAX_MOVES = 3000

    while not done and moves < MAX_MOVES:
        game       = env.game_env.game
        current_id = game.players.index(game.get_current_player())

        # ── MODEL TURN ──────────────────────────────────────────────────────
        if current_id == 0:
            moves += 1

            with torch.no_grad():
                ap, vpp, ep, tgive, tget, sv = network.forward(
                    torch.FloatTensor(obs['observation']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device),
                )

            action_id = _greedy(ap)
            vertex_id = _greedy(vpp)
            edge_id   = _greedy(ep)
            give_idx  = _greedy(tgive)
            get_idx   = _greedy(tget)
            if give_idx == get_idx:
                get_idx = (give_idx + 1) % 5

            action_name = ACTION_NAMES.get(action_id, str(action_id))
            value_est   = sv[0, 0].item()

            # Build human-readable detail
            detail = ''
            if action_name == 'trade_bank':
                give_res = TRADE_RES[give_idx]
                get_res  = TRADE_RES[get_idx]
                detail = f" ({RESOURCE_NAMES[give_res]} → {RESOURCE_NAMES[get_res]})"
            elif action_name in ('build_sett', 'place_sett'):
                detail = f" v{vertex_id}"
            elif action_name in ('build_road', 'place_road'):
                detail = f" e{edge_id}"
            elif action_name == 'build_city':
                detail = f" v{vertex_id}"

            # Top action probabilities
            ap_np = ap.cpu().numpy()[0]
            top3  = sorted(enumerate(ap_np), key=lambda x: -x[1])[:3]
            top3_str = '  '.join(f"{ACTION_NAMES.get(i,str(i))}:{p*100:.0f}%" for i, p in top3)

            print(f"[M#{moves:4d}] {action_name:<16}{detail:<22}  val={value_est:+.2f}  top:[{top3_str}]")

            obs, _, terminated, truncated, _ = env.step(
                action_id, vertex_id, edge_id,
                trade_give_idx=give_idx, trade_get_idx=get_idx,
            )

            # Print state after each end_turn or build action
            if action_name in ('end_turn', 'build_sett', 'build_city', 'build_road',
                               'place_sett', 'place_road', 'play_knight'):
                print(state_line(game, "  -->"))

            if terminated or truncated:
                winner = game.check_victory_conditions()
                outcome = 'MODEL WINS' if (winner and game.players.index(winner) == 0) else 'OPP WINS'
                done = True

            if delay > 0:
                time.sleep(delay)

        # ── OPPONENT TURN ────────────────────────────────────────────────────
        else:
            turn_num += 1
            print(f"\n--- Opp turn {turn_num} ---  {state_line(game, '')}")

            success = play_opponent_turn(game, current_id, 1.0, opponent_type, None)
            if not success and game.can_end_turn():
                game.end_turn()

            if game.waiting_for_discards:
                env.game_env._handle_automatic_discards()
                if game.waiting_for_discards:
                    game.waiting_for_discards = False
                    game.players_must_discard  = []
                    game.players_discarded     = set()

            obs = env._get_obs()

            winner = game.check_victory_conditions()
            if winner is not None:
                outcome = 'MODEL WINS' if game.players.index(winner) == 0 else 'OPP WINS'
                done = True

            if delay > 0:
                time.sleep(delay)

    if not done:
        outcome = 'TIMEOUT'

    game = env.game_env.game
    mp   = game.players[0]
    op   = game.players[1]
    print(f"\n{'='*70}")
    print(f"RESULT: {outcome}")
    print(f"  Model  VP={mp.calculate_victory_points()}  "
          f"Sett={len(mp.settlements)}  City={len(mp.cities)}  "
          f"Road={len(mp.roads)}  Knights={mp.knights_played}  "
          f"LargestArmy={mp.has_largest_army}  LongestRoad={mp.has_longest_road}")
    print(f"  Opp    VP={op.calculate_victory_points()}  "
          f"Sett={len(op.settlements)}  City={len(op.cities)}  "
          f"Road={len(op.roads)}  Knights={op.knights_played}  "
          f"LargestArmy={op.has_largest_army}  LongestRoad={op.has_longest_road}")
    print(f"  Moves: {moves}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Watch model play one Catan game")
    parser.add_argument('--model',    type=str, required=True)
    parser.add_argument('--opponent', type=str, default='strong',
                        choices=['truly_random','weighted_random','random',
                                 'very_weak','weak','medium','strong'])
    parser.add_argument('--vp',       type=int, default=10,  help='VP to win (default 10)')
    parser.add_argument('--delay',    type=float, default=0.1, help='Seconds between actions (default 0.1)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    watch(args.model, args.opponent, args.vp, args.delay)


if __name__ == '__main__':
    main()
