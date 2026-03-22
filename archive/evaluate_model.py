"""
Evaluation Script for Catan AI

Diagnoses model behavior by:
1. Loading a trained model
2. Playing test games
3. Showing action probability distributions
4. Measuring entropy, win rate, VP distribution
5. Identifying if model is stuck/deterministic
"""

import torch
import numpy as np
from collections import defaultdict, Counter
import os

from catan_env_pytorch import CatanEnv
from network_wrapper import NetworkWrapper
from curriculum_trainer_v2_fixed import play_opponent_turn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class ModelEvaluator:
    def __init__(self, model_path):
        self.device = get_device()
        print(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy
        self.network.eval()

        print(f"✅ Model loaded successfully")
        print(f"Device: {self.device}\n")

    def get_action_and_stats(self, obs, sample=True):
        """Get action from network and return detailed stats"""
        with torch.no_grad():
            observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
            action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)
            vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(self.device)
            edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(self.device)

            action_probs, vertex_probs, edge_probs, _, _, value = self.network.forward(
                observation, action_mask, vertex_mask, edge_mask
            )

            ap = action_probs.cpu().numpy()[0]
            vp = vertex_probs.cpu().numpy()[0]
            ep = edge_probs.cpu().numpy()[0]
            val = value.cpu().numpy()[0]

        # Calculate entropy
        def calc_entropy(probs):
            # Avoid log(0)
            probs = np.clip(probs, 1e-10, 1.0)
            return -np.sum(probs * np.log(probs))

        action_entropy = calc_entropy(ap)
        vertex_entropy = calc_entropy(vp)
        edge_entropy = calc_entropy(ep)

        # Get action
        if sample:
            action_id = np.random.choice(len(ap), p=ap)
            vertex_id = np.random.choice(len(vp), p=vp)
            edge_id = np.random.choice(len(ep), p=ep)
        else:
            action_id = np.argmax(ap)
            vertex_id = np.argmax(vp)
            edge_id = np.argmax(ep)

        return {
            'action': (action_id, vertex_id, edge_id),
            'action_probs': ap,
            'vertex_probs': vp,
            'edge_probs': ep,
            'action_entropy': action_entropy,
            'vertex_entropy': vertex_entropy,
            'edge_entropy': edge_entropy,
            'value': val,
            'max_action_prob': np.max(ap),
            'max_vertex_prob': np.max(vp),
            'max_edge_prob': np.max(ep),
        }

    def play_game(self, opponent_random_prob=1.0, verbose=False):
        """Play a single game and track statistics"""
        env = CatanEnv(player_id=0)
        obs, _ = env.reset()

        game_stats = {
            'moves': 0,
            'total_reward': 0,
            'action_entropies': [],
            'value_predictions': [],
            'max_action_probs': [],
        }

        done = False
        moves = 0
        max_moves = 500

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                # Our agent's turn
                moves += 1

                stats = self.get_action_and_stats(obs, sample=True)
                action_id, vertex_id, edge_id = stats['action']

                game_stats['action_entropies'].append(stats['action_entropy'])
                game_stats['value_predictions'].append(stats['value'])
                game_stats['max_action_probs'].append(stats['max_action_prob'])

                if verbose and moves <= 5:
                    print(f"\n  Move {moves}:")
                    print(f"    Action entropy: {stats['action_entropy']:.4f}")
                    print(f"    Value prediction: {stats['value']:.2f}")
                    print(f"    Max action prob: {stats['max_action_prob']:.4f}")
                    print(f"    Top 3 actions: {np.argsort(stats['action_probs'])[-3:][::-1]}")

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0
                )

                game_stats['total_reward'] += reward
                obs = next_obs
                done = terminated or truncated
            else:
                # Opponent's turn
                success = play_opponent_turn(game, current_id, opponent_random_prob)
                if not success and game.can_end_turn():
                    game.end_turn()

                winner = game.check_victory_conditions()
                if winner is not None:
                    done = True

        game_stats['moves'] = moves

        # Get final stats
        my_vp = env.game_env.game.players[0].calculate_victory_points()
        winner = env.game_env.game.check_victory_conditions()
        winner_id = game.players.index(winner) if winner else None

        game_stats['vp'] = my_vp
        game_stats['won'] = (winner_id == 0)

        return game_stats

    def evaluate(self, num_games=50, opponent_random_prob=1.0, verbose_first=True):
        """Evaluate model over multiple games"""
        print("=" * 70)
        print(f"EVALUATING MODEL")
        print("=" * 70)
        print(f"Games: {num_games}")
        print(f"Opponent: {opponent_random_prob*100:.0f}% random")
        print("=" * 70 + "\n")

        all_stats = {
            'wins': 0,
            'vps': [],
            'rewards': [],
            'avg_entropy': [],
            'avg_value': [],
            'avg_max_prob': [],
        }

        for i in range(num_games):
            verbose = (verbose_first and i == 0)
            if verbose:
                print(f"🎮 Game 1 (detailed):")

            stats = self.play_game(opponent_random_prob, verbose=verbose)

            if stats['won']:
                all_stats['wins'] += 1

            all_stats['vps'].append(stats['vp'])
            all_stats['rewards'].append(stats['total_reward'])
            all_stats['avg_entropy'].append(np.mean(stats['action_entropies']))
            all_stats['avg_value'].append(np.mean(stats['value_predictions']))
            all_stats['avg_max_prob'].append(np.mean(stats['max_action_probs']))

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_games} games...")

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        win_rate = all_stats['wins'] / num_games * 100
        avg_vp = np.mean(all_stats['vps'])
        std_vp = np.std(all_stats['vps'])
        avg_reward = np.mean(all_stats['rewards'])
        avg_entropy = np.mean(all_stats['avg_entropy'])
        avg_value = np.mean(all_stats['avg_value'])
        avg_max_prob = np.mean(all_stats['avg_max_prob'])

        print(f"\n📊 Performance:")
        print(f"  Win Rate: {win_rate:.1f}% ({all_stats['wins']}/{num_games})")
        print(f"  Avg VP: {avg_vp:.2f} ± {std_vp:.2f}")
        print(f"  VP Distribution: {Counter(all_stats['vps'])}")
        print(f"  Avg Reward: {avg_reward:.1f}")

        print(f"\n🧠 Model Behavior:")
        print(f"  Avg Action Entropy: {avg_entropy:.4f}")
        print(f"  Avg Value Prediction: {avg_value:.2f}")
        print(f"  Avg Max Action Prob: {avg_max_prob:.4f}")

        # Diagnose issues
        print(f"\n🔍 Diagnosis:")
        if avg_entropy < 0.01:
            print("  ❌ CRITICAL: Entropy collapsed (< 0.01)")
            print("     Model is completely deterministic!")
        elif avg_entropy < 0.1:
            print("  ⚠️  WARNING: Very low entropy (< 0.1)")
            print("     Model has limited exploration")
        elif avg_entropy < 0.5:
            print("  ⚠️  Entropy somewhat low (< 0.5)")
        else:
            print("  ✅ Entropy healthy (>= 0.5)")

        if avg_max_prob > 0.95:
            print("  ❌ Model is too confident (max prob > 0.95)")
        elif avg_max_prob > 0.8:
            print("  ⚠️  Model fairly confident (max prob > 0.8)")
        else:
            print("  ✅ Model confidence reasonable")

        if win_rate == 0:
            print("  ❌ 0% win rate - model not learning!")
        elif win_rate < 5:
            print("  ⚠️  Very low win rate (< 5%)")
        elif win_rate < 15:
            print("  ⚙️  Low but learning (5-15%)")
        else:
            print("  ✅ Decent win rate (>= 15%)")

        if avg_vp < 2.5:
            print("  ❌ Very low VP scores (< 2.5)")
        elif avg_vp < 3.5:
            print("  ⚠️  Low VP scores (< 3.5)")
        elif avg_vp < 5.0:
            print("  ⚙️  Moderate VP scores (3.5-5.0)")
        else:
            print("  ✅ Good VP scores (>= 5.0)")

        print("\n" + "=" * 70)

        return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Catan AI model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (e.g., models/curriculum_v2_fixed_phase1.pt)')
    parser.add_argument('--games', type=int, default=50,
                        help='Number of games to evaluate (default: 50)')
    parser.add_argument('--opponent', type=float, default=1.0,
                        help='Opponent random probability 0-1 (default: 1.0 = fully random)')
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model)
    results = evaluator.evaluate(
        num_games=args.games,
        opponent_random_prob=args.opponent
    )


if __name__ == "__main__":
    main()
