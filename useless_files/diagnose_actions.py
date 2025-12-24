"""
Diagnostic script to understand why agent won't build
Tracks action probabilities and choices to identify the issue
"""
import torch
import numpy as np
from catan_env_pytorch import CatanEnv
from agent_gpu import CatanAgent

# Load trained model
device = torch.device('cpu')
print("Loading trained model from models/catan_clean_episode_500.pt")

agent = CatanAgent(device=device)
checkpoint = torch.load('models/catan_clean_episode_500.pt', map_location=device)
agent.policy.load_state_dict(checkpoint['model_state_dict'])
agent.policy.eval()

print(f"Model loaded! Running 50 episodes to find build opportunities...\n")

action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city',
                'build_road', 'buy_dev', 'end', 'wait']

all_opportunities = []

# Run MULTIPLE episodes to find build opportunities
for episode in range(50):
    env = CatanEnv()
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < 200:
        step += 1

        # Get action probabilities from the policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs['observation']).unsqueeze(0).to(device)
            action_mask_tensor = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)

            # Get probabilities from policy (already masked)
            action_probs_tensor, _, _, _ = agent.policy(state_tensor, action_mask=action_mask_tensor)
            action_probs = action_probs_tensor[0].cpu().numpy()

        # Check if build actions are available
        valid_actions = [action_names[i] for i, mask in enumerate(obs['action_mask']) if mask == 1]
        build_actions_available = any(a in valid_actions for a in ['build_sett', 'build_city', 'build_road'])

        # Log when build actions are available
        if build_actions_available and not env.game_env.game.is_initial_placement_phase():
            player = env.game_env.game.players[0]
            resources = player.resources
            from game_system import ResourceType

            print(f"\n📍 Episode {episode+1}, Step {step}: BUILD ACTIONS AVAILABLE!")
            print(f"   Valid actions: {valid_actions}")
            print(f"   Resources: W{resources[ResourceType.WOOD]} B{resources[ResourceType.BRICK]} "
                  f"Wh{resources[ResourceType.WHEAT]} S{resources[ResourceType.SHEEP]} O{resources[ResourceType.ORE]}")
            print(f"   Action probabilities:")
            for i, (name, prob) in enumerate(zip(action_names, action_probs)):
                if obs['action_mask'][i] == 1:
                    print(f"      {name:12s}: {prob*100:5.1f}%")

            all_opportunities.append({
                'episode': episode+1,
                'step': step,
                'valid': valid_actions,
                'probs': {name: action_probs[i] for i, name in enumerate(action_names) if obs['action_mask'][i] == 1},
                'resources': {
                    'W': resources[ResourceType.WOOD],
                    'B': resources[ResourceType.BRICK],
                    'Wh': resources[ResourceType.WHEAT],
                    'S': resources[ResourceType.SHEEP],
                    'O': resources[ResourceType.ORE]
                }
            })

        # Choose action
        action, vertex, edge, _, _, _, value = agent.choose_action(
            obs, obs['action_mask'], obs['vertex_mask'], obs['edge_mask']
        )

        # Log if a build action was actually chosen
        if action in [3, 4, 5]:  # build_settlement, build_city, build_road
            print(f"   ✅ CHOSE: {action_names[action]} (vertex={vertex}, edge={edge})")

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
        done = terminated or truncated
        obs = next_obs

print(f"\n{'='*70}")
print(f"DIAGNOSTIC RESULTS (50 episodes):")
print(f"{'='*70}")
print(f"Build opportunities found: {len(all_opportunities)}")

if len(all_opportunities) == 0:
    print("\n❌ NO BUILD OPPORTUNITIES IN 50 EPISODES!")
    print("   This is very unusual - action mask may be broken.")
else:
    print(f"\n📊 Build opportunity analysis:")
    build_chosen_count = sum(1 for opp in all_opportunities
                            if max(opp['probs'].get('build_sett', 0),
                                  opp['probs'].get('build_city', 0),
                                  opp['probs'].get('build_road', 0)) > 0.5)

    print(f"   Times build action chosen: {build_chosen_count}/{len(all_opportunities)}")

    # Show details of first few opportunities
    print(f"\n   First 3 build opportunities:")
    for i, opp in enumerate(all_opportunities[:3]):
        print(f"\n   Ep{opp['episode']} Step{opp['step']} | Res: W{opp['resources']['W']} B{opp['resources']['B']} "
              f"Wh{opp['resources']['Wh']} S{opp['resources']['S']} O{opp['resources']['O']}")
        for action, prob in sorted(opp['probs'].items(), key=lambda x: -x[1])[:4]:
            print(f"      {action:12s}: {prob*100:5.1f}%")

print(f"\n{'='*70}")
