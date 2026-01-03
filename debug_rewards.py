"""
Debug script to trace PBRS rewards step-by-step
"""

import numpy as np
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from simplified_reward_wrapper import SimplifiedRewardWrapper

def debug_game(wrapper_type='pbrs_fixed', num_games=5):
    """Play a few games and trace all rewards"""

    print("=" * 70)
    print(f"DEBUGGING {wrapper_type.upper()} REWARDS")
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

        # Track rewards
        step_rewards = []
        vp_rewards = []
        pbrs_rewards = []
        terminal_rewards = []

        step = 0
        done = False

        while not done and step < 500:  # Safety limit
            # Get valid actions
            action_mask = obs['action_mask']
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]

            if not valid_actions:
                print(f"  Step {step}: No valid actions!")
                break

            # Random action
            action_id = np.random.choice(valid_actions)

            # Take step
            vertex_id = None
            edge_id = None

            if action_id in [1, 2]:  # Settlement/City
                vertex_mask = obs.get('vertex_mask', None)
                if vertex_mask is not None:
                    valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                    if valid_vertices:
                        vertex_id = np.random.choice(valid_vertices)

            if action_id == 3:  # Road
                edge_mask = obs.get('edge_mask', None)
                if edge_mask is not None:
                    valid_edges = [i for i, m in enumerate(edge_mask) if m == 1]
                    if valid_edges:
                        edge_id = np.random.choice(valid_edges)

            old_vp = obs.get('my_victory_points', 0)

            obs, reward, terminated, truncated, info = env.step(
                action_id, vertex_id, edge_id
            )

            new_vp = obs.get('my_victory_points', 0)
            vp_change = new_vp - old_vp

            # Try to decompose reward
            if wrapper_type == 'pbrs_fixed':
                # Estimate components
                if terminated:
                    winner_id = info.get('winner_id', None)
                    if winner_id == 0:
                        terminal_r = 100.0
                    else:
                        terminal_r = -10.0
                else:
                    terminal_r = 0.0

                vp_r = vp_change * 10.0 if vp_change > 0 else 0.0
                pbrs_r = reward - vp_r - terminal_r

                if vp_change > 0 or terminated or abs(reward) > 1.0:
                    print(f"  Step {step}: VP {old_vp}->{new_vp} | "
                          f"Reward={reward:.2f} (VP:{vp_r:.1f} + PBRS:{pbrs_r:.2f} + Term:{terminal_r:.1f})")

                vp_rewards.append(vp_r)
                pbrs_rewards.append(pbrs_r)
                terminal_rewards.append(terminal_r)
            else:
                if vp_change > 0 or terminated:
                    print(f"  Step {step}: VP {old_vp}->{new_vp} | Reward={reward:.2f}")

            step_rewards.append(reward)
            step += 1
            done = terminated or truncated

        # Summary
        total_reward = sum(step_rewards)
        final_vp = obs.get('my_victory_points', 0)
        winner_id = info.get('winner_id', None) if done else None
        won = (winner_id == 0)

        print(f"\n{'='*70}")
        print(f"GAME {game_num + 1} SUMMARY:")
        print(f"  Steps: {step}")
        print(f"  Final VP: {final_vp}")
        print(f"  Won: {won}")
        print(f"  Total Reward: {total_reward:.2f}")

        if wrapper_type == 'pbrs_fixed':
            print(f"  Breakdown:")
            print(f"    VP rewards: {sum(vp_rewards):.1f}")
            print(f"    PBRS rewards: {sum(pbrs_rewards):.2f}")
            print(f"    Terminal rewards: {sum(terminal_rewards):.1f}")
            print(f"  Check: {sum(vp_rewards) + sum(pbrs_rewards) + sum(terminal_rewards):.2f}")

        print('='*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PBRS FIXED WRAPPER")
    print("="*70)
    debug_game('pbrs_fixed', num_games=3)

    print("\n\n" + "="*70)
    print("TESTING SIMPLIFIED WRAPPER (VP_ONLY)")
    print("="*70)
    debug_game('vp_only', num_games=3)
