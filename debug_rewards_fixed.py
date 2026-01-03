"""
Fixed debug with CORRECT action mapping
"""

import numpy as np
import random
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from simplified_reward_wrapper import SimplifiedRewardWrapper
from game_system import ResourceType


def play_opponent_turn(game, player_id):
    """Random opponent"""
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    game.try_place_initial_road(random.choice(valid), player)
                    return True
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
                return True
        return False

    if game.can_roll_dice():
        game.roll_dice()
        return True

    if game.can_trade_or_build():
        actions = []
        res = player.resources

        can_settlement = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
                         res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1)
        if can_settlement:
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        can_city = (res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3)
        if can_city:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        can_road = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1)
        if can_road:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        if actions and random.random() < 0.9:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                game.try_build_settlement(random.choice(locs), player)
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                game.try_build_road(random.choice(locs), player)
            return True

    if game.can_end_turn():
        game.end_turn()
        return True

    return False


def debug_games(wrapper_type='pbrs_fixed', num_games=10):
    """Debug with CORRECT action mapping"""

    # CORRECT action names from catan_env_pytorch.py:310-314
    action_names = [
        'roll_dice', 'place_settlement', 'place_road',
        'build_settlement', 'build_city', 'build_road',
        'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing'
    ]

    print("=" * 70)
    print(f"DEBUGGING {wrapper_type.upper()} - FIXED")
    print("=" * 70)

    for game_num in range(num_games):
        if wrapper_type == 'pbrs_fixed':
            env = PBRSFixedRewardWrapper(player_id=0)
        else:
            env = SimplifiedRewardWrapper(player_id=0, reward_mode='vp_only')

        obs, _ = env.reset()
        episode_rewards = []
        done = False
        moves = 0
        max_moves = 500

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:  # AI turn
                moves += 1

                action_mask = obs['action_mask']
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]

                if not valid_actions:
                    break

                action_id = np.random.choice(valid_actions)
                vertex_id = None
                edge_id = None

                # FIXED: Check for action_id 1 or 3 for vertices (place_settlement or build_settlement)
                if action_id in [1, 3]:
                    vertex_mask = obs.get('vertex_mask', None)
                    if vertex_mask is not None:
                        valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                        if valid_vertices:
                            vertex_id = np.random.choice(valid_vertices)

                # FIXED: Check for action_id 2 or 5 for edges (place_road or build_road)
                if action_id in [2, 5]:
                    edge_mask = obs.get('edge_mask', None)
                    if edge_mask is not None:
                        valid_edges = [i for i, m in enumerate(edge_mask) if m == 1]
                        if valid_edges:
                            edge_id = np.random.choice(valid_edges)

                next_obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id, trade_give_idx=0, trade_get_idx=0
                )

                episode_rewards.append(reward)
                obs = next_obs
                done = terminated or truncated

            else:  # Opponent turn
                success = play_opponent_turn(game, current_id)
                if not success and game.can_end_turn():
                    game.end_turn()

                winner = game.check_victory_conditions()
                if winner:
                    done = True

        # Summary
        total_reward = sum(episode_rewards)
        final_vp = obs.get('my_victory_points', 0)
        winner_id = None
        if done:
            winner_obj = game.check_victory_conditions()
            if winner_obj:
                winner_id = game.players.index(winner_obj)

        won = (winner_id == 0)

        print(f"Game {game_num+1}: Moves={moves}, VP={final_vp}, Won={won}, Reward={total_reward:.2f}")


if __name__ == "__main__":
    debug_games('pbrs_fixed', num_games=10)
    print()
    debug_games('vp_only', num_games=10)
