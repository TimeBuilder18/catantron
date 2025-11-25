import sys
sys.path.append('/home/claude')

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from catan_env_pytorch import CatanEnv
from network import CatanPolicy
from agent import CatanAgent, ExperienceBuffer
from trainer import PPOTrainer

def train(total_episodes=10, update_frequency=5, save_frequency=5, model_name="catan_ppo"):
    print("=" * 70)
    print("CATAN PPO TRAINING")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Update frequency: {update_frequency} episodes")
    print(f"Save frequency: {save_frequency} episodes")
    print("=" * 70 + "\n")
    env = CatanEnv(player_id=0)
    print("✅ Environment created\n")
    agent = CatanAgent()
    print("✅ Agent created\n")
    trainer = PPOTrainer(policy=agent.policy,learning_rate=3e-4,gamma=0.99,gae_lambda=0.95,clip_epsilon=0.2,n_epochs=10,batch_size=64)
    print("✅ Trainer created\n")
    buffer = ExperienceBuffer()
    episode_rewards = []
    episode_lengths = []
    episode_vps = []
    print("🚀 Starting training...\n")

    for episode in range(total_episodes):
        # Reset environment
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # Play one episode
        while not done:
            # Get action from agent
            action, log_prob, value = agent.choose_action(obs, obs['action_mask'])

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            buffer.store(
                obs['observation'],
                action,
                reward,
                log_prob,
                value,
                done,
                obs['action_mask']
            )
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_length += 1

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_vps.append(info.get('victory_points', 0))

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_vp = np.mean(episode_vps[-10:])
            print(f"Episode {episode + 1}/{total_episodes} | "
                  f"Reward: {avg_reward:.2f} | VP: {avg_vp:.1f} | "
                  f"Length: {episode_length}")

        # Update policy
        if (episode + 1) % update_frequency == 0 and len(buffer) > 0:
            print(f"\n📊 Updating policy (buffer size: {len(buffer)})...")
            metrics = trainer.update_policy(buffer)
            print(f"   Policy loss: {metrics['policy_loss']:.4f}")
            print(f"   Value loss: {metrics['value_loss']:.4f}")
            print(f"   Entropy: {metrics['entropy']:.4f}\n")
            buffer.clear()

        # Save model
        if (episode + 1) % save_frequency == 0:
            save_path = f"models/{model_name}_episode_{episode + 1}.pt"
            torch.save(agent.policy.state_dict(), save_path)
            print(f"💾 Model saved: {save_path}\n")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

    # Save final model
    final_path = f"models/{model_name}_final.pt"
    torch.save(agent.policy.state_dict(), final_path)
    print(f"💾 Final model saved: {final_path}")

    # Plot results
    plot_training_progress(episode_rewards, episode_vps, model_name)

    return episode_rewards, episode_vps


def plot_training_progress(rewards, vps, model_name):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        ax1.plot(range(9, len(rewards)), moving_avg, label='Moving Avg (10)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot victory points
    ax2.plot(vps, alpha=0.3, label='Episode VP')
    if len(vps) >= 10:
        moving_avg = np.convolve(vps, np.ones(10) / 10, mode='valid')
        ax2.plot(range(9, len(vps)), moving_avg, label='Moving Avg (10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Victory Points')
    ax2.set_title('Victory Points Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')
    print(f"📊 Training plot saved: {model_name}_training.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Catan PPO agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Total episodes')
    parser.add_argument('--update-freq', type=int, default=10, help='Update every N episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='Save every N episodes')
    parser.add_argument('--model-name', type=str, default='catan_ppo', help='Model name')

    args = parser.parse_args()

    # Create models directory
    import os

    os.makedirs('models', exist_ok=True)

    # Train
    rewards, vps = train(
        total_episodes=args.episodes,
        update_frequency=args.update_freq,
        save_frequency=args.save_freq,
        model_name=args.model_name
    )

    print("\n🎉 Training finished!")
    print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average VP (last 100): {np.mean(vps[-100:]):.1f}")