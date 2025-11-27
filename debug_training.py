"""
Debug script to see what the agent is actually doing during training
"""
import sys
sys.path.append('/home/user/catantron')
import numpy as np
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent
import torch

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def debug_single_episode():
    """Play one episode and print what's happening"""
    env = CatanEnv(player_id=0)
    agent = CatanAgent(device='cpu')

    obs, info = env.reset()
    done = False
    step_count = 0
    actions_taken = {i: 0 for i in range(9)}

    action_names = [
        'roll_dice', 'place_settlement', 'place_road',
        'build_settlement', 'build_city', 'build_road',
        'buy_dev_card', 'end_turn', 'wait'
    ]

    resources_by_turn = []
    vp_by_turn = []

    print("\n" + "="*70)
    print("DEBUGGING SINGLE EPISODE")
    print("="*70)

    while not done and step_count < 100:  # Limit to 100 agent steps
        # Check if it's our turn
        if not info.get('is_my_turn', True):
            # Skip opponent turns
            from rule_based_ai import play_rule_based_turn
            current_player = env.game_env.game.current_player_index
            play_rule_based_turn(env, current_player)

            winner = env.game_env.game.check_victory_conditions()
            if winner:
                done = True
                break

            obs = env._get_obs()
            info = env._get_info()
            continue

        # Our turn - record state
        player = env.game_env.game.players[0]
        resources = player.resources
        total_resources = sum(resources.values())
        vp = info['victory_points']

        resources_by_turn.append(total_resources)
        vp_by_turn.append(vp)

        # Get action
        action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value = agent.choose_action(
            obs, obs['action_mask'], obs.get('vertex_mask'), obs.get('edge_mask')
        )

        actions_taken[action] += 1

        # Print every 10 steps
        if step_count % 10 == 0 or action in [1, 2, 3, 4, 5, 6]:  # Always print building actions
            phase = "INITIAL" if env.game_env.game.is_initial_placement_phase() else "NORMAL"
            print(f"Step {step_count:3d} [{phase}] Action: {action_names[action]:15s} | "
                  f"VP: {vp} | Resources: {total_resources:2d} | Value: {value:.2f}")

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
        done = terminated or truncated

        obs = next_obs
        step_count += 1

    print("\n" + "="*70)
    print("EPISODE SUMMARY")
    print("="*70)
    print(f"Total agent steps: {step_count}")
    print(f"Final VP: {vp_by_turn[-1] if vp_by_turn else 0}")
    print(f"Max resources: {max(resources_by_turn) if resources_by_turn else 0}")
    print(f"Avg resources: {np.mean(resources_by_turn) if resources_by_turn else 0:.1f}")
    print(f"\nActions taken:")
    for i, name in enumerate(action_names):
        if actions_taken[i] > 0:
            print(f"  {name:20s}: {actions_taken[i]:3d} times ({100*actions_taken[i]/step_count:.1f}%)")

    # Check if agent ever left initial placement
    max_vp = max(vp_by_turn) if vp_by_turn else 0
    print(f"\nMax VP reached: {max_vp}")
    if max_vp <= 2:
        print("⚠️  PROBLEM: Agent never progressed past initial placement!")

    print("="*70 + "\n")

if __name__ == "__main__":
    for i in range(3):
        print(f"\n{'#'*70}")
        print(f"# DEBUG RUN {i+1}/3")
        print(f"{'#'*70}")
        debug_single_episode()
