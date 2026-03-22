"""
Simple reward debugging - copies exact trainer flow
"""

import numpy as np
import random
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from simplified_reward_wrapper import SimplifiedRewardWrapper
from game_system import ResourceType


def play_opponent_turn(game, player_id):
    """Random opponent - copied from trainer"""
    player = game.players[player_id]

    # Initial placement
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

    # Normal play
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


def debug_game(wrapper_type='pbrs_fixed', num_games=5):
    """Play games using exact trainer flow"""

    print("=" * 70)
    print(f"DEBUGGING {wrapper_type.upper()} - EXACT TRAINER FLOW")
    print("=" * 70)

    for game_num in range(num_games):
        print(f"\nGAME {game_num + 1}:")

        # Create wrapper
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

                # Get valid actions
                action_mask = obs['action_mask']
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]

                if not valid_actions:
                    break

                # Random action
                action_id = np.random.choice(valid_actions)
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

                # Step
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

                # Check if game ended
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

        print(f"  Moves: {moves}, VP: {final_vp}, Won: {won}, Total Reward: {total_reward:.2f}")

        if len(episode_rewards) > 0:
            print(f"  Reward range: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")


if __name__ == "__main__":
    debug_game('pbrs_fixed', num_games=10)
    print("\n")
    debug_game('vp_only', num_games=10)
