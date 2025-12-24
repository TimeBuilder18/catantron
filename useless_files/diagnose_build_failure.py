"""
Check why builds are failing - diagnose edge/vertex selection
"""
import torch
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent
from game_system import ResourceType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

agent = CatanAgent(device=device)
checkpoint = torch.load('models/catan_clean_episode_500.pt', map_location=device)
agent.policy.load_state_dict(checkpoint['model_state_dict'])
agent.policy.eval()

env = CatanEnv()
obs, _ = env.reset()

print("Testing build action execution...\n")

for step in range(100):
    # Choose action
    action, vertex, edge, _, _, _, _ = agent.choose_action(
        obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
    )

    action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city',
                    'build_road', 'buy_dev', 'end', 'wait']

    # Check if it's a build action
    if action == 5:  # build_road
        player = env.game_env.game.players[0]
        res_before = {rt: player.resources[rt] for rt in ResourceType}

        # Get edge mask
        edge_mask = obs['edge_mask']
        valid_edges = [i for i, m in enumerate(edge_mask) if m == 1.0]

        print(f"\n[BUILD_ROAD ATTEMPT]")
        print(f"  Resources before: W{res_before[ResourceType.WOOD]} B{res_before[ResourceType.BRICK]}")
        print(f"  Valid edges: {len(valid_edges)} edges available")
        print(f"  Agent chose edge: {edge}")
        print(f"  Is edge valid? {edge in valid_edges}")

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)

        res_after = {rt: env.game_env.game.players[0].resources[rt] for rt in ResourceType}
        resources_changed = any(res_after[rt] != res_before[rt] for rt in ResourceType)

        print(f"  Resources after: W{res_after[ResourceType.WOOD]} B{res_after[ResourceType.BRICK]}")
        print(f"  Build succeeded? {resources_changed}")
        print(f"  Reward: {reward:.1f}")

        if not resources_changed:
            print(f"  ❌ BUILD FAILED - Resources didn't change!")
            if edge not in valid_edges:
                print(f"     Reason: Agent chose INVALID edge {edge}")
            break
        else:
            print(f"  ✅ BUILD SUCCEEDED!")

        obs = next_obs

        if terminated or truncated:
            break
    else:
        next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
        obs = next_obs

        if terminated or truncated:
            break

print("\nDiagnosis complete!")
