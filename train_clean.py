"""Clean training with minimal output"""
import os
import sys
import io
import os
import torch

# M2 optimization - better memory management
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("✅ M2 memory optimization enabled")
import matplotlib
matplotlib.use('Agg')
# Fix Windows encoding issues with emojis - use reconfigure for Python 3.7+
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    # Fallback for older Python versions
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


class NullWriter:
    """A file-like object that discards all output - Windows compatible"""
    def write(self, text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


class SuppressOutput:
    """Context manager to suppress all stdout/stderr output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        # Use custom NullWriter instead of os.devnull for better Windows compatibility
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Import everything WHILE suppressing output
with SuppressOutput():

    from catan_env_pytorch import CatanEnv
    from network_gpu import CatanPolicy
    from agent_gpu import CatanAgent,ExperienceBuffer
    from trainer_gpu import PPOTrainer
    from rule_based_ai import play_rule_based_turn
    from game_system import GameConstants
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import time

# Now run regular training code...
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100000, help="Total episodes to train for an overnight session")
parser.add_argument('--update-freq', type=int, default=50)
parser.add_argument('--save-freq', type=int, default=10000, help="Save a checkpoint every N episodes")
parser.add_argument('--model-name', type=str, default='catan_overnight')
parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning (VP 4→10)')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Training epochs per update')
args = parser.parse_args()

print("=" * 70)
print("CATAN TRAINING - OVERNIGHT OPTIMIZED")
print("=" * 70)
sys.stdout.flush()

# Curriculum learning function for overnight run
def get_vp_target(episode, use_curriculum):
    """Return VP target based on a slow, gradual curriculum for 100k episodes"""
    if not use_curriculum:
        return 10  # Default to full game if curriculum is off

    # New, slower curriculum for overnight training
    if episode < 10000:
        return 4   # Master absolute basics
    elif episode < 25000:
        return 5   # Solidify early game
    elif episode < 45000:
        return 6   # Learn to expand
    elif episode < 65000:
        return 7   # Intermediate strategy
    elif episode < 85000:
        return 8   # Advanced strategy
    else:
        return 10  # Full competitive game

# Auto-detect best device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"🎮 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Batch size: {args.batch_size} (RTX 2080 Super optimized)")
    print(f"   Training epochs: {args.epochs}")

if args.curriculum:
    print(f"\n📚 Curriculum Learning: ENABLED (Overnight Schedule)")
    print(f"   Episodes 0-10000:     VP = 4")
    print(f"   Episodes 10001-25000: VP = 5")
    print(f"   Episodes 25001-45000: VP = 6")
    print(f"   Episodes 45001-65000: VP = 7")
    print(f"   Episodes 65001-85000: VP = 8")
    print(f"   Episodes 85001+:      VP = 10")
else:
    print(f"\n📚 Curriculum Learning: DISABLED")
    GameConstants.VICTORY_POINTS_TO_WIN = 10 # Set to default

print(f"\n🎯 Training:")
print(f"   Episodes: {args.episodes}")
print(f"   Update frequency: {args.update_freq}")
print(f"   Save frequency: {args.save_freq}")
print(f"   Model name: {args.model_name}")
print()
sys.stdout.flush()

env = CatanEnv(player_id=0)
agent = CatanAgent(device=device)

trainer = PPOTrainer(
    policy=agent.policy,
    learning_rate=3e-4,      # Standard PPO learning rate
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.05,       # Increased exploration
    max_grad_norm=0.5,
    n_epochs=args.epochs,
    batch_size=args.batch_size
)
buffer = ExperienceBuffer()

episode_rewards = []
episode_vps = []

print(f"\nStarting training...\n")
sys.stdout.flush()
start_time = time.time()

# At the TOP, BEFORE the episode loop (around line 85):
timeout_count = 0
natural_end_count = 0

# Start episode loop:
for episode in range(args.episodes):
    # Update VP target based on curriculum
    current_vp_target = get_vp_target(episode, args.curriculum)
    GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target

    # Collect debug info for periodic printing
    debug_actions = []
    debug_resources = []

    # Suppress game output during episode
    with SuppressOutput():
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        # Adjust max_steps based on VP target
        if current_vp_target >= 8:
            max_steps = 350 # Allow more steps for higher VP games
        elif current_vp_target >= 6:
            max_steps = 300
        else:
            max_steps = 250

        # Episode game loop
        while not done and step_count < max_steps:
            step_count += 1

            # Rule-based AI turn if not our turn
            if not info.get('is_my_turn', True):
                current_player = env.game_env.game.current_player_index
                play_rule_based_turn(env, current_player)
                obs = env._get_obs()
                info = env._get_info()
                continue

            # Get hierarchical action from agent
            (action, vertex, edge, trade_give, trade_get,
             action_log_prob, vertex_log_prob, edge_log_prob,
             trade_give_log_prob, trade_get_log_prob, value) = agent.choose_action(
                obs,
                obs['action_mask'],
                obs['vertex_mask'],
                obs['edge_mask']
            )

            # Collect debug info for episodes 0, 50, 100, etc.
            # Capture turns 10-20 of normal play (after initial placement)
            if episode % 50 == 0 and not env.game_env.game.is_initial_placement_phase():
                if 10 <= step_count <= 20:
                    action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city', 'build_road', 'buy_dev', 'end', 'wait', 'trade_with_bank', 'do_nothing']
                    valid = [action_names[i] for i, mask in enumerate(obs['action_mask']) if mask == 1]
                    player = env.game_env.game.players[0]
                    resources = player.resources
                    from game_system import ResourceType
                    debug_actions.append((step_count, valid, action))
                    debug_resources.append({
                        'W': resources[ResourceType.WOOD],
                        'B': resources[ResourceType.BRICK],
                        'Wh': resources[ResourceType.WHEAT],
                        'S': resources[ResourceType.SHEEP],
                        'O': resources[ResourceType.ORE]
                    })

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge, trade_give, trade_get)
            done = terminated or truncated

            # Store ALL hierarchical data in buffer
            buffer.store(
                state=obs['observation'],
                action=action,
                vertex=vertex,
                edge=edge,
                trade_give=trade_give,
                trade_get=trade_get,
                reward=reward,
                action_log_prob=action_log_prob,
                vertex_log_prob=vertex_log_prob,
                edge_log_prob=edge_log_prob,
                trade_give_log_prob=trade_give_log_prob,
                trade_get_log_prob=trade_get_log_prob,
                value=value,
                done=done,
                action_mask=obs['action_mask'],
                vertex_mask=obs['vertex_mask'],
                edge_mask=obs['edge_mask']
            )

            obs = next_obs
            episode_reward += reward

        # After while loop ends, check WHY it ended:
        if step_count >= max_steps:
            timeout_count += 1
        else:
            natural_end_count += 1

    # Store episode reward
    episode_rewards.append(episode_reward)
    episode_vps.append(info.get('victory_points', 0))

    # Print debug info for episodes 0, 50, 100, etc.
    if episode > 0 and episode % 500 == 0 and debug_actions:
        print(f"\n  [DEBUG Ep{episode}] Steps 10-20 of normal play:")
        action_names_map = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city', 'build_road', 'buy_dev', 'end', 'wait', 'trade_with_bank', 'do_nothing']
        for (step, valid, chosen_action), res in zip(debug_actions, debug_resources):
            chosen = action_names_map[chosen_action] if chosen_action < len(action_names_map) else f"#{chosen_action}"
            print(f"    Step {step}: Valid={valid} | Chose={chosen} | Res: W{res['W']} B{res['B']} Wh{res['Wh']} S{res['S']} O{res['O']}")

    # Print progress every 100 episodes for a cleaner log
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_vp = np.mean(episode_vps[-100:])
        progress = (episode + 1) / args.episodes * 100
        elapsed = time.time() - start_time
        speed = (episode + 1) / (elapsed / 60)

        progress_str = (
            f"[{progress:5.1f}%] Ep {episode + 1:6d}/{args.episodes} | "
            f"VP: {avg_vp:.1f} | Reward: {avg_reward:7.2f} | {speed:4.0f} eps/min"
        )

        progress_str += f" | Target VP: {current_vp_target}"

        print(progress_str)

        # Print timeout stats every 500 episodes
        if (episode + 1) % 500 == 0:
            timeout_pct = timeout_count / (episode + 1) * 100
            natural_pct = natural_end_count / (episode + 1) * 100
            print(f"         📊 Timeouts: {timeout_pct:.1f}% | Natural endings: {natural_pct:.1f}%")

        sys.stdout.flush()

    # Update policy
    if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
        with SuppressOutput():
            metrics = trainer.update_policy(buffer)
        buffer.clear()
        if (episode + 1) % 500 == 0: # Only print loss periodically
            print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f}")
            sys.stdout.flush()

    # Save model
    if (episode + 1) % args.save_freq == 0:
        save_path = f"models/{args.model_name}_episode_{episode + 1}.pt"
        agent.policy.save(save_path)
        print(f"         Checkpoint saved -> {save_path}")
        sys.stdout.flush()

# AFTER all episodes complete:
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print(f"Final timeout rate: {timeout_count}/{args.episodes} = {timeout_count / args.episodes * 100:.1f}%")
print(f"Natural endings: {natural_end_count}/{args.episodes} = {natural_end_count / args.episodes * 100:.1f}%")
elapsed_time = time.time() - start_time
print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
print(f"Average speed: {args.episodes / (elapsed_time / 60):.0f} episodes/min")
print("=" * 70)
sys.stdout.flush()