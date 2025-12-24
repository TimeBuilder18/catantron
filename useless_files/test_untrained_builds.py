"""
Test if builds work with an UNTRAINED (random) agent
This will tell us if the build mechanism itself is broken
"""
import torch
import numpy as np
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent
from game_system import ResourceType

device = torch.device('cpu')
agent = CatanAgent(device=device)  # Fresh, untrained agent!

print("Testing builds with UNTRAINED agent (random policy)...\n")

for episode in range(20):
    env = CatanEnv()
    obs, _ = env.reset()
    done = False
    step = 0

    action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city',
                    'build_road', 'buy_dev', 'end', 'wait']

    while not done and step < 200:
        step += 1

        # Choose action with untrained (random) policy
        action, vertex, edge, _, _, _, _ = agent.choose_action(
            obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
        )

        # Track builds during normal play
        if action in [3, 4, 5] and not env.game_env.game.is_initial_placement_phase():
            player = env.game_env.game.players[0]
            res_before = {rt: player.resources[rt] for rt in ResourceType}

            # Execute
            next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)

            res_after = {rt: env.game_env.game.players[0].resources[rt] for rt in ResourceType}
            resources_changed = any(res_after[rt] != res_before[rt] for rt in ResourceType)

            if resources_changed:
                print(f"✅ Episode {episode+1}: {action_names[action]} SUCCEEDED!")
                print(f"   Before: W{res_before[ResourceType.WOOD]} B{res_before[ResourceType.BRICK]} "
                      f"Wh{res_before[ResourceType.WHEAT]} S{res_before[ResourceType.SHEEP]} O{res_before[ResourceType.ORE]}")
                print(f"   After:  W{res_after[ResourceType.WOOD]} B{res_after[ResourceType.BRICK]} "
                      f"Wh{res_after[ResourceType.WHEAT]} S{res_after[ResourceType.SHEEP]} O{res_after[ResourceType.ORE]}")
                print(f"   Reward: {reward:.1f}\n")
                # Found a successful build, can stop
                break
            else:
                print(f"❌ Episode {episode+1}: {action_names[action]} FAILED (resources unchanged)")

            obs = next_obs
            done = terminated or truncated
            if done:
                break
        else:
            # Not a build action, just step normally
            next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
            obs = next_obs
            done = terminated or truncated

print("\n" + "="*70)
print("If we see ANY successful builds, the mechanism works.")
print("If ALL builds fail, something is broken in the build execution.")
print("="*70)
