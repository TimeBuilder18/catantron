"""Quick test to see if auto-play works"""
import torch
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent

device = torch.device('cpu')
agent = CatanAgent(device=device)

# Load model
checkpoint = torch.load('models/catan_clean_episode_500.pt', map_location=device)
agent.policy.load_state_dict(checkpoint['model_state_dict'])
agent.policy.eval()

env = CatanEnv()

# Enable debug
env.game_env._debug_autoplay = True

obs, _ = env.reset()
print("Starting episode with auto-play debug enabled...\n")

for step in range(20):
    action, vertex, edge, _, _, _, _ = agent.choose_action(
        obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
    )

    next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)

    if terminated or truncated:
        break

    obs = next_obs

print("\nDone!")
