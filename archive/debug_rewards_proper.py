"""
Proper reward debugging - simulates actual training game flow
"""

import numpy as np
import random
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from simplified_reward_wrapper import SimplifiedRewardWrapper

def play_random_turn(game, player_id):
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
        from game_system import ResourceType
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


def debug_full_game(wrapper_type='pbrs_fixed', num_games=3):
    """Play complete games with proper game flow"""

    print("=" * 70)
    print(f"DEBUGGING {wrapper_type.upper()} - FULL GAME SIMULATION")
    print("=" * 70)

    for game_num in range(num_games):
        print(f"\n{'='*70}")
        print(f"GAME {game_num + 1}")
        print('='*70)

        # Create wrapper
        if wrapper_type == 'pbrs_fixed':
            env = PBRSFixedRewardWrapper(player_id=0)
        else:
            env = SimplifiedRewardWrapper(player_id=0, reward_mode='vp_only')

        obs, _ = env.reset()

        episode_rewards = []
        vp_changes = []
        last_vp = 0

        steps = 0
        max_steps = 10000

        while steps < max_steps:
            # Check if it's AI's turn
            current_player = env.game_env.game.current_player_index

            if current_player == 0:  # AI turn
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

                obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id
                )

                episode_rewards.append(reward)

                current_vp = obs.get('my_victory_points', 0)
                if current_vp != last_vp:
                    vp_change = current_vp - last_vp
                    vp_changes.append((steps, last_vp, current_vp, vp_change))
                    last_vp = current_vp

                if abs(reward) > 1.0:
                    print(f"  Step {steps}: Reward={reward:.2f}, VP={current_vp}")

                if terminated or truncated:
                    break

            else:  # Opponent turn
                play_random_turn(env.game_env.game, current_player)

            steps += 1

        # Summary
        total_reward = sum(episode_rewards)
        final_vp = obs.get('my_victory_points', 0)
        winner_id = info.get('winner_id', None)
        won = (winner_id == 0)

        print(f"\n{'='*70}")
        print(f"GAME {game_num + 1} SUMMARY:")
        print(f"  Steps: {steps}")
        print(f"  Final VP: {final_vp}")
        print(f"  Won: {won}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  VP Changes: {len(vp_changes)}")

        if vp_changes:
            print(f"  VP progression:")
            for step, old_vp, new_vp, change in vp_changes:
                print(f"    Step {step}: {old_vp} -> {new_vp} (+{change})")

        print(f"  Reward stats:")
        print(f"    Min: {min(episode_rewards):.2f}")
        print(f"    Max: {max(episode_rewards):.2f}")
        print(f"    Mean: {np.mean(episode_rewards):.3f}")
        print('='*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PBRS FIXED WRAPPER")
    print("="*70)
    debug_full_game('pbrs_fixed', num_games=3)

    print("\n\n" + "="*70)
    print("TESTING SIMPLIFIED WRAPPER (VP_ONLY)")
    print("="*70)
    debug_full_game('vp_only', num_games=3)
