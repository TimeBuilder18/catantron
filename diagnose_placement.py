"""
Detailed diagnostic of initial placement and early game
Shows exactly where settlements are placed and what happens
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

print("Running detailed placement diagnostic...\n")

env = CatanEnv()
obs, _ = env.reset()
done = False
step = 0

action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city',
                'build_road', 'buy_dev', 'end', 'wait']

print("="*70)
print("INITIAL PLACEMENT PHASE")
print("="*70)

placement_count = 0

while not done and step < 100:
    step += 1

    # Get current game phase
    is_initial = env.game_env.game.is_initial_placement_phase()

    # Choose action
    action, vertex, edge, _, _, _, _ = agent.choose_action(
        obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
    )

    # Log initial placement actions
    if is_initial and action in [1, 2]:  # place_settlement or place_road
        placement_count += 1
        action_name = action_names[action]
        print(f"\nStep {step}: {action_name}")
        if action == 1:  # Settlement
            print(f"  Vertex index: {vertex}")
            # Try to get vertex coordinates
            all_vertices = env.game_env.game.game_board.vertices
            if 0 <= vertex < len(all_vertices):
                v = all_vertices[vertex]
                print(f"  Position: ({v.x:.1f}, {v.y:.1f})")

                # Check adjacent tiles
                print(f"  Adjacent tiles:")
                for tile in env.game_env.game.game_board.tiles:
                    corners = tile.get_corners()
                    for cx, cy in corners:
                        if abs(cx - v.x) < 0.1 and abs(cy - v.y) < 0.1:
                            res_type = tile.get_resource_type()
                            print(f"    Tile: {res_type.name if res_type else 'DESERT'} | Number: {tile.number}")
                            break
        elif action == 2:  # Road
            print(f"  Edge index: {edge}")

    # Take step
    next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
    done = terminated or truncated

    # Check if we exited initial placement
    was_initial = is_initial
    is_still_initial = env.game_env.game.is_initial_placement_phase()

    if was_initial and not is_still_initial:
        print(f"\n{'='*70}")
        print(f"EXITED INITIAL PLACEMENT at step {step}")
        print(f"Total placements: {placement_count}")

        # Show player's settlements
        player = env.game_env.game.players[0]
        print(f"\nPlayer has {len(player.settlements)} settlements:")
        for i, settlement in enumerate(player.settlements):
            print(f"\n  Settlement {i+1} at ({settlement.position.x:.1f}, {settlement.position.y:.1f})")
            print(f"    Adjacent tiles:")
            for tile in env.game_env.game.game_board.tiles:
                corners = tile.get_corners()
                for cx, cy in corners:
                    if abs(cx - settlement.position.x) < 0.1 and abs(cy - settlement.position.y) < 0.1:
                        res_type = tile.get_resource_type()
                        res_name = res_type.name if res_type else 'DESERT'
                        print(f"      {res_name:8s} | Number: {tile.number if tile.number else 'N/A'}")
                        break

        print(f"\n{'='*70}")
        print(f"NORMAL PLAY - First 20 steps")
        print(f"{'='*70}")
        break

    obs = next_obs

# Now track normal play
normal_play_step = 0
while not done and normal_play_step < 20:
    normal_play_step += 1

    # Get valid actions
    valid_actions = [action_names[i] for i, mask in enumerate(obs['action_mask']) if mask == 1]

    # Choose action
    action, vertex, edge, _, _, _, _ = agent.choose_action(
        obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
    )

    action_name = action_names[action]

    # Get current resources
    player = env.game_env.game.players[0]
    res = player.resources

    print(f"\nStep {step + normal_play_step}: {action_name}")
    print(f"  Valid: {valid_actions}")
    print(f"  Resources: W{res[ResourceType.WOOD]} B{res[ResourceType.BRICK]} "
          f"Wh{res[ResourceType.WHEAT]} S{res[ResourceType.SHEEP]} O{res[ResourceType.ORE]}")

    if action_name == 'roll':
        print(f"  >>> ROLLING DICE")

    # Take step
    next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)

    # Check if resources changed
    new_res = env.game_env.game.players[0].resources
    resource_gained = any(new_res[rt] != res[rt] for rt in ResourceType)
    if resource_gained:
        print(f"  >>> RESOURCES GAINED! New: W{new_res[ResourceType.WOOD]} B{new_res[ResourceType.BRICK]} "
              f"Wh{new_res[ResourceType.WHEAT]} S{new_res[ResourceType.SHEEP]} O{new_res[ResourceType.ORE]}")

    done = terminated or truncated
    obs = next_obs

print(f"\n{'='*70}")
print("DIAGNOSIS COMPLETE")
print("="*70)
