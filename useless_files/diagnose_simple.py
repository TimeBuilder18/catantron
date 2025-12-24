"""
Simple diagnostic - just check if agent ever collects resources AT ALL
"""
import torch
import numpy as np
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent
from game_system import ResourceType

device = torch.device('cpu')
agent = CatanAgent(device=device)
checkpoint = torch.load('models/catan_clean_episode_500.pt', map_location=device)
agent.policy.load_state_dict(checkpoint['model_state_dict'])
agent.policy.eval()

print("Running 5 episodes to check resource collection...\n")

for ep in range(5):
    env = CatanEnv()
    obs, _ = env.reset()
    done = False
    step = 0
    max_resources = {rt: 0 for rt in ResourceType}

    print(f"\n=== Episode {ep+1} ===")

    while not done and step < 200:
        step += 1

        # Choose and execute action
        action, vertex, edge, _, _, _, _ = agent.choose_action(
            obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
        )

        next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
        done = terminated or truncated

        # Track resources
        if not env.game_env.game.is_initial_placement_phase():
            player = env.game_env.game.players[0]
            res = player.resources
            for rt in ResourceType:
                if res[rt] > max_resources[rt]:
                    max_resources[rt] = res[rt]

            # Check if can_trade_or_build returns True
            if step % 20 == 0:
                can_build = env.game_env.game.can_trade_or_build()
                print(f"  Step {step}: can_trade_or_build={can_build} | "
                      f"W{res[ResourceType.WOOD]} B{res[ResourceType.BRICK]} "
                      f"Wh{res[ResourceType.WHEAT]} S{res[ResourceType.SHEEP]} O{res[ResourceType.ORE]}")

        obs = next_obs

    print(f"  Max resources collected: W{max_resources[ResourceType.WOOD]} "
          f"B{max_resources[ResourceType.BRICK]} Wh{max_resources[ResourceType.WHEAT]} "
          f"S{max_resources[ResourceType.SHEEP]} O{max_resources[ResourceType.ORE]}")
    print(f"  Total steps: {step}")

print("\n" + "="*70)
print("If agent never collects resources, initial placement is broken.")
print("If agent collects resources but can_trade_or_build=False, that's the issue.")
print("="*70)
