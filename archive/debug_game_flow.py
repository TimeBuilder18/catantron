"""
Debug game flow to see why games aren't completing
"""

import numpy as np
import random
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
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
                game.try_upgrade_to_city(random.choice(locs), player)
            elif action_type == 'road':
                game.try_build_road(random.choice(locs), player)
            return True

    if game.can_end_turn():
        game.end_turn()
        return True

    return False


def debug_one_game():
    """Debug a single game with detailed logging"""

    env = PBRSFixedRewardWrapper(player_id=0)
    obs, _ = env.reset()

    episode_rewards = []
    done = False
    moves = 0
    max_moves = 100  # Shorter limit for debugging

    print("Starting game...")
    print(f"Initial phase: {env.game_env.game.is_initial_placement_phase()}")

    while not done and moves < max_moves:
        game = env.game_env.game
        current = game.get_current_player()
        current_id = game.players.index(current)

        if current_id == 0:  # AI turn
            moves += 1

            # Get valid actions
            action_mask = obs['action_mask']
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]

            if not valid_actions:
                print(f"Move {moves}: No valid actions!")
                break

            # Random action
            action_id = np.random.choice(valid_actions)

            action_names = ["NONE", "SETTLEMENT", "CITY", "ROAD", "DEV_CARD",
                          "TRADE_BANK", "TRADE_PLAYER", "PLAY_KNIGHT", "WAIT",
                          "END_TURN", "DISCARD"]

            vertex_id = None
            edge_id = None

            if action_id in [1, 2]:
                vertex_mask = obs.get('vertex_mask', None)
                if vertex_mask is not None:
                    valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                    if valid_vertices:
                        vertex_id = np.random.choice(valid_vertices)

            if action_id == 3:
                edge_mask = obs.get('edge_mask', None)
                if edge_mask is not None:
                    valid_edges = [i for i, m in enumerate(edge_mask) if m == 1]
                    if valid_edges:
                        edge_id = np.random.choice(valid_edges)

            if moves <= 20 or moves % 10 == 0:
                print(f"Move {moves}: AI action={action_names[action_id]}, "
                      f"vertex={vertex_id}, edge={edge_id}, "
                      f"VP={obs.get('my_victory_points', 0)}, "
                      f"initial_phase={game.is_initial_placement_phase()}")

            # Step
            next_obs, reward, terminated, truncated, info = env.step(
                action_id, vertex_id, edge_id, trade_give_idx=0, trade_get_idx=0
            )

            if abs(reward) > 0.5 or reward != 0:
                print(f"  -> Reward: {reward:.2f}")

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

    print(f"\nGame finished:")
    print(f"  Moves: {moves}")
    print(f"  Final VP: {final_vp}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Still in initial phase: {env.game_env.game.is_initial_placement_phase()}")


if __name__ == "__main__":
    debug_one_game()
