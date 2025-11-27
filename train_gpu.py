"""
GPU-Optimized Catan PPO Training Script

This version automatically detects and uses GPU if available.
Perfect for training on your RTX 2080 Super!

Usage:
    # Short test
    python train_gpu.py --episodes 100
    
    # Full training session
    python train_gpu.py --episodes 5000 --update-freq 20 --save-freq 500
    
    # Force CPU (for testing)
    python train_gpu.py --episodes 10 --device cpu
"""

import sys
sys.path.append('/home/claude')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

# Suppress game debug output
class QuietMode:
    def write(self, x): pass
    def flush(self): pass

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Rest of your imports...
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from catan_env_pytorch import CatanEnv
from network_gpu import CatanPolicy
from agent_gpu import CatanAgent, ExperienceBuffer
from trainer_gpu import PPOTrainer
from rule_based_ai import play_rule_based_turn


def train(total_episodes=10, update_frequency=5, save_frequency=5, model_name="catan_ppo", device=None):
    """
    Train Catan AI using PPO
    
    Args:
        total_episodes: Number of games to play
        update_frequency: Update policy every N episodes
        save_frequency: Save model every N episodes
        model_name: Base name for saved models
        device: 'cuda', 'cpu', or None (auto-detect)
    """
    #print("=" * 70)
    #print("CATAN PPO TRAINING - GPU ACCELERATED")
    #print("=" * 70)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    #print(f"\n🎮 Using device: {device}")
    if device.type == 'cuda':
        #print(f"   GPU: {torch.cuda.get_device_name(0)}")
        #print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Clear cache
        torch.cuda.empty_cache()
    
    #print(f"\n📋 Training Configuration:")
    #print(f"   Total episodes: {total_episodes}")
    #print(f"   Update frequency: {update_frequency} episodes")
    #print(f"   Save frequency: {save_frequency} episodes")
    #print("=" * 70 + "\n")
    
    # Create environment
    env = CatanEnv(player_id=0)
    #print("✅ Environment created\n")
    
    # Create agent with GPU support
    agent = CatanAgent(device=device)
    #print("✅ Agent created\n")
    
    # Create trainer with larger batch size for GPU
    # RTX 2080 Super optimization: 8GB VRAM allows for large batches
    if device.type == 'cuda':
        batch_size = 1024  # Doubled from 512 for RTX 2080 Super
    else:
        batch_size = 64

    trainer = PPOTrainer(
        policy=agent.policy,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.05,  # Increased from 0.01 for more exploration
        n_epochs=20,  # Increased from 15 for better learning
        batch_size=batch_size
    )
    #print("✅ Trainer created\n")
    
    buffer = ExperienceBuffer()
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_vps = []
    update_times = []
    
    #print("🚀 Starting training...\n")
    start_time = time.time()
    
    for episode in range(total_episodes):
        episode_start = time.time()
        
        # Reset environment
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Play one episode
        while not done:
            # Handle other players' turns (rule-based AI)
            if not info.get('is_my_turn', True):
                max_wait = 100
                waited = 0
                
                while not info.get('is_my_turn', True) and not done and waited < max_wait:
                    # Turn limit check
                    if env.game_env.game.turn_number > 500:
                        done = True
                        break
                    
                    current_player = env.game_env.game.current_player_index
                    
                    # Use rule-based AI for opponents
                    success = play_rule_based_turn(env, current_player)
                    
                    if not success:
                        # Fallback: force end turn
                        if env.game_env.game.can_end_turn():
                            env.game_env.game.end_turn()
                    
                    # Check victory
                    winner = env.game_env.game.check_victory_conditions()
                    if winner:
                        done = True
                        break
                    
                    obs = env._get_obs()
                    info = env._get_info()
                    waited += 1
                
                if not info.get('is_my_turn', True) or done:
                    break
            
            # Our agent's turn - get ALL 7 return values (hierarchical action)
            action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value = agent.choose_action(
                obs, obs['action_mask'], obs.get('vertex_mask'), obs.get('edge_mask')
            )

            # DEBUG: Print when build actions are available (every 50 episodes)
            if episode % 50 == 0 and not env.game_env.game.is_initial_placement_phase():
                action_names = ['roll', 'place_sett', 'place_road', 'build_sett', 'build_city', 'build_road', 'buy_dev', 'end', 'wait']
                valid = [action_names[i] for i, mask in enumerate(obs['action_mask']) if mask == 1]
                player = env.game_env.game.players[0]
                resources = player.resources
                from game_system import ResourceType
                print(f"  [Ep{episode}] Valid actions: {valid} | Resources: W{resources[ResourceType.WOOD]} B{resources[ResourceType.BRICK]} Wh{resources[ResourceType.WHEAT]} S{resources[ResourceType.SHEEP]} O{resources[ResourceType.ORE]}")

            # Take step in environment - pass vertex and edge indices
            next_obs, reward, terminated, truncated, info = env.step(action, vertex, edge)
            done = terminated or truncated

            # Store experience with ALL hierarchical data
            buffer.store(
                obs['observation'],
                action,
                vertex,
                edge,
                reward,
                action_log_prob,
                vertex_log_prob,
                edge_log_prob,
                value,
                done,
                obs['action_mask'],
                obs.get('vertex_mask', np.ones(54)),
                obs.get('edge_mask', np.ones(72))
            )
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        episode_time = time.time() - episode_start
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_vps.append(info.get('victory_points', 0))
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_vp = np.mean(episode_vps[-10:])
            elapsed = time.time() - start_time
            eps_per_min = (episode + 1) / (elapsed / 60)

            pass  # Progress printing suppressed
            #print(f"Episode {episode + 1}/{total_episodes} | "
            #      f"Reward: {avg_reward:.2f} | VP: {avg_vp:.1f} | "
            #      f"Length: {episode_length} | Time: {episode_time:.1f}s | "
            #      f"Speed: {eps_per_min:.1f} eps/min")

            # GPU memory usage
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                mem_cached = torch.cuda.memory_reserved(0) / 1e9
                pass  # GPU memory printing suppressed
                #print(f"   GPU Memory: {mem_allocated:.2f}GB allocated, {mem_cached:.2f}GB cached")
        
        # Update policy
        if (episode + 1) % update_frequency == 0 and len(buffer) > 0:
            #print(f"\n📊 Updating policy (buffer size: {len(buffer)})...")
            update_start = time.time()
            
            metrics = trainer.update_policy(buffer)
            
            update_time = time.time() - update_start
            update_times.append(update_time)
            
            #print(f"   Policy loss: {metrics['policy_loss']:.4f}")
            #print(f"   Value loss: {metrics['value_loss']:.4f}")
            #print(f"   Entropy: {metrics['entropy']:.4f}")
            #print(f"   Update time: {update_time:.2f}s\n")
            
            buffer.clear()
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save model
        if (episode + 1) % save_frequency == 0:
            save_path = f"models/{model_name}_episode_{episode + 1}.pt"
            agent.policy.save(save_path)
            #print(f"💾 Model saved: {save_path}\n")
    
    total_time = time.time() - start_time
    
    #print("\n" + "=" * 70)
    #print("✅ TRAINING COMPLETE!")
    #print("=" * 70)
    #print(f"Total time: {total_time/60:.1f} minutes")
    #print(f"Average time per episode: {total_time/total_episodes:.2f}s")
    if update_times:
        pass  # Update time printing suppressed
        #print(f"Average update time: {np.mean(update_times):.2f}s")

    # Save final model
    final_path = f"models/{model_name}_final.pt"
    agent.policy.save(final_path)
    #print(f"💾 Final model saved: {final_path}")
    
    # Plot results
    plot_training_progress(episode_rewards, episode_vps, model_name)
    
    return episode_rewards, episode_vps


def plot_training_progress(rewards, vps, model_name):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        ax1.plot(range(9, len(rewards)), moving_avg, label='Moving Avg (10)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot victory points
    ax2.plot(vps, alpha=0.3, label='Episode VP')
    if len(vps) >= 10:
        moving_avg = np.convolve(vps, np.ones(10) / 10, mode='valid')
        ax2.plot(range(9, len(vps)), moving_avg, label='Moving Avg (10)', linewidth=2)
    ax2.axhline(y=10, color='r', linestyle='--', label='Win Threshold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Victory Points')
    ax2.set_title('Victory Points Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png', dpi=150)
    #print(f"📊 Training plot saved: {model_name}_training.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Catan PPO agent with GPU support')
    parser.add_argument('--episodes', type=int, default=1000, help='Total episodes')
    parser.add_argument('--update-freq', type=int, default=20, help='Update every N episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='Save every N episodes')
    parser.add_argument('--model-name', type=str, default='catan_ppo', help='Model name')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda, cpu, or auto (default)')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train
    rewards, vps = train(
        total_episodes=args.episodes,
        update_frequency=args.update_freq,
        save_frequency=args.save_freq,
        model_name=args.model_name,
        device=args.device
    )
    
    #print("\n🎉 Training finished!")
    #print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    #print(f"Final average VP (last 100): {np.mean(vps[-100:]):.1f}")
    
    # GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        #print("🧹 GPU cache cleared")
