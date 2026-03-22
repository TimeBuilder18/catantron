"""
Debug script to see what happens when agent actually attempts builds
Uses the environment's debug output to trace build failures
"""
import torch
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent
from game_system import ResourceType

device = torch.device('cpu')
agent = CatanAgent(device=device)
checkpoint = torch.load('models/catan_clean_episode_500.pt', map_location=device)
agent.policy.load_state_dict(checkpoint['model_state_dict'])
agent.policy.eval()

print("Running episode to catch build attempt with DEBUG enabled...\n")

env = CatanEnv()
obs, _ = env.reset()
done = False
step = 0

action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city',
                'build_road', 'buy_dev', 'end', 'wait']

build_caught = False

while not done and step < 100:
    step += 1

    # Choose action
    action, vertex, edge, _, _, _, _ = agent.choose_action(
        obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
    )

    # Check if it's a build action
    if action in [3, 4, 5] and not env.game_env.game.is_initial_placement_phase():
        build_caught = True
        player = env.game_env.game.players[0]
        res = player.resources

        print(f"\n{'='*70}")
        print(f"BUILD ACTION CAUGHT at step {step}")
        print(f"Action: {action_names[action]}")
        print(f"Vertex chosen: {vertex}, Edge chosen: {edge}")
        print(f"Resources BEFORE: W{res[ResourceType.WOOD]} B{res[ResourceType.BRICK]} "
              f"Wh{res[ResourceType.WHEAT]} S{res[ResourceType.SHEEP]} O{res[ResourceType.ORE]}")

        # Check masks
        vertex_mask = obs['vertex_mask']
        edge_mask = obs['edge_mask']
        valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1.0]
        valid_edges = [i for i, m in enumerate(edge_mask) if m == 1.0]

        print(f"Valid vertices: {len(valid_vertices)} vertices available")
        print(f"Valid edges: {len(valid_edges)} edges available")
        if action in [3, 4]:  # Needs vertex
            print(f"Vertex {vertex} is valid? {vertex in valid_vertices}")
        if action == 5:  # Needs edge
            print(f"Edge {edge} is valid? {edge in valid_edges}")

        print(f"\nExecuting action (will show [DEBUG] output from environment)...")
        print(f"{'='*70}")

    # Take step (this will print [DEBUG] output if it's a build action)
    next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)

    # If we caught a build, check result
    if build_caught:
        player = env.game_env.game.players[0]
        res = player.resources
        print(f"{'='*70}")
        print(f"Resources AFTER: W{res[ResourceType.WOOD]} B{res[ResourceType.BRICK]} "
              f"Wh{res[ResourceType.WHEAT]} S{res[ResourceType.SHEEP]} O{res[ResourceType.ORE]}")
        print(f"Reward received: {reward:.1f}")
        print(f"Build success: {'✅ YES' if info.get('success', True) else '❌ NO'}")
        print(f"{'='*70}\n")
        break

    obs = next_obs
    done = terminated or truncated

if not build_caught:
    print("\n❌ No build actions attempted in 100 steps!")
    print("   Agent never tried to build, only ended turns.\n")
