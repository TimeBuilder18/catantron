"""
Simplified Reward Wrapper for Catan AI

The existing reward system is too complex for policy gradient methods.
This wrapper provides simpler reward options:

1. SPARSE: Win (+100) / Loss (-10) only
2. VP_ONLY: Reward based on VP changes (+10 per VP)
3. SIMPLIFIED: VP + basic build rewards

For MCTS/AlphaZero/PPO, simpler rewards work better!
"""

import numpy as np
from environment.catan_env import CatanEnv
from environment.reward_utils import PositionalRewardMixin


class SimplifiedRewardWrapper(PositionalRewardMixin):
    """Wraps CatanEnv with simplified rewards for better learning"""

    def __init__(self, player_id=0, reward_mode='vp_only', victory_points_to_win=10, num_players=4):
        """
        Args:
            player_id: Which player is the AI (0-3)
            reward_mode: 'sparse', 'vp_only', or 'simplified'
            victory_points_to_win: VP needed to win (default 10)
            num_players: Number of players (2 for 1v1, 4 for standard)
        """
        self.env = CatanEnv(player_id=player_id, victory_points_to_win=victory_points_to_win, num_players=num_players)
        self.num_players = num_players
        self.reward_mode = reward_mode
        self.player_id = player_id
        self.last_vp = 0
        self.victory_points_to_win = victory_points_to_win
    def reset(self):
        """Reset environment"""
        obs, info = self.env.reset()
        self.last_vp = obs.get('my_victory_points', 0)
        self._shaping_reset()
        return obs, info

    def step(self, action_id, vertex_id=None, edge_id=None, trade_give_idx=None, trade_get_idx=None):
        """Step with simplified rewards"""
        # Take action in underlying env
        obs, orig_reward, terminated, truncated, info = self.env.step(
            action_id, vertex_id, edge_id, trade_give_idx, trade_get_idx
        )

        # Calculate simplified reward
        reward = self._calculate_simple_reward(obs, terminated, info)

        return obs, reward, terminated, truncated, info

    def _calculate_simple_reward(self, obs, terminated, info):
        """Calculate reward based on mode"""
        current_vp = obs.get('my_victory_points', 0)
        vp_gain = current_vp - self.last_vp
        self.last_vp = current_vp

        if self.reward_mode == 'sparse':
            # SPARSE: Only win/loss matters
            if terminated:
                winner_id = info.get('winner_id', None)
                if winner_id == self.player_id:
                    return 100.0  # Win!
                else:
                    return -10.0  # Loss
            return 0.0  # No reward during game

        elif self.reward_mode == 'vp_only':
            # VP_ONLY: Reward VP changes strongly
            reward = 0.0

            # VP gain is the main signal
            if vp_gain > 0:
                reward += vp_gain * 10.0  # +10 per VP

            # City bonus: cities cost more resources but are strategically superior.
            # Without this, vp_only treats city and settlement identically (+10 VP each).
            # City total = 10 (VP) + 20 (bonus) = 30. Road chain = 2+10+20=32 over 3 actions
            # (10.7/action) vs city alone (30/action) — cities still clearly best single action.
            if info.get('built_city'):
                reward += 20.0

            # Settlement quality bonus (pip score + diversity + port + blocking)
            player = self.env.game_env.game.players[self.player_id]
            action_name_tmp = info.get('action_name', '')
            if info.get('built_settlement') or action_name_tmp == 'place_settlement':
                reward += self._settlement_shaping(player)

            # Road bonus: roads are prerequisite for settlement expansion → more cities.
            # Without this, agent never learns to build roads and VP caps at ~4
            # (2 initial settlements upgraded to cities → stuck, no new settlement spots).
            if info.get('action_name') == 'build_road' and info.get('success', False):
                reward += 2.0  # base
                reward += self._road_expansion_shaping(player)  # quality bonus

            # Effective trade: reward trades that create a build opportunity
            if info.get('bank_trade') and info.get('success') and info.get('trade_led_to_build_opportunity'):
                reward += 2.0

            # Dev card play rewards
            action_name = info.get('action_name', '')
            if action_name == 'play_knight' and info.get('success', False):
                reward += 8.0
                if info.get('stolen_resource'):
                    reward += 3.0
                    if info.get('steal_completes_build'):
                        reward += 5.0  # Stolen card unlocked a build
            elif action_name == 'play_monopoly' and info.get('success', False):
                reward += info.get('stolen_count', 0) * 3.0  # +3 per card stolen
            elif action_name == 'play_year_of_plenty' and info.get('success', False):
                reward += 5.0
            elif action_name == 'buy_dev_card' and info.get('got_vp_card'):
                reward += 15.0  # VP card: instant +1 VP (vp_diff gives +10 too)

            # Win bonus (must dominate sum of all VP-step rewards so winning is #1 objective)
            if terminated:
                winner_id = info.get('winner_id', None)
                if winner_id == self.player_id:
                    reward += 100.0  # Win dominates: 8 VP gains (80) + win (100) > 4 VP gains (40)
                else:
                    reward -= 20.0  # Loss stings — opponent reaching 10 VP = failed urgently

            return reward

        elif self.reward_mode == 'simplified':
            # SIMPLIFIED: VP + basic build rewards
            reward = 0.0

            # VP changes (main signal)
            if vp_gain > 0:
                reward += vp_gain * 10.0

            # Building rewards (encourage expansion).
            # FIX: CatanEnv.step() does info.update(step_info) so keys are at info top level,
            # not nested under 'step_info'. Use info.get() directly.
            if info.get('built_city'):
                reward += 5.0  # Cities are good
            if info.get('built_settlement'):
                reward += 3.0  # Settlements are good

            # Road bonus: same fix as vp_only — roads must have non-zero reward
            if info.get('action_name') == 'build_road' and info.get('success', False):
                reward += 2.0  # Enables settlement → city expansion chain

            # Dev card play rewards
            action_name = info.get('action_name', '')
            if action_name == 'play_knight' and info.get('success', False):
                reward += 8.0
                if info.get('stolen_resource'):
                    reward += 3.0
                    if info.get('steal_completes_build'):
                        reward += 5.0
            elif action_name == 'play_monopoly' and info.get('success', False):
                reward += info.get('stolen_count', 0) * 3.0
            elif action_name == 'play_year_of_plenty' and info.get('success', False):
                reward += 5.0
            elif action_name == 'buy_dev_card' and info.get('got_vp_card'):
                reward += 15.0

            # Small inaction penalty (encourage action)
            if info.get('action_name') == 'end_turn':
                legal_actions = info.get('legal_actions', [])
                if 'build_city' in legal_actions:
                    reward -= 2.0  # Should have built city!

            # Win/loss
            if terminated:
                winner_id = info.get('winner_id', None)
                if winner_id == self.player_id:
                    reward += 100.0  # Increased from 50: win must dominate all per-step rewards
                else:
                    reward -= 20.0  # Increased from -5: losing should sting

            return reward

        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def __getattr__(self, name):
        """Delegate attribute access to underlying env"""
        return getattr(self.env, name)


def compare_reward_systems(num_games=10):
    """Compare different reward modes"""
    print("=" * 70)
    print("REWARD SYSTEM COMPARISON")
    print("=" * 70)

    modes = ['sparse', 'vp_only', 'simplified']

    for mode in modes:
        print(f"\nMode: {mode}")
        print("-" * 40)

        env = SimplifiedRewardWrapper(player_id=0, reward_mode=mode)
        total_rewards = []

        for game in range(num_games):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < 500:
                # Random action for comparison
                action_mask = obs['action_mask']
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if not valid_actions:
                    break

                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action, 0, 0, 0, 0)

                episode_reward += reward
                done = terminated or truncated
                steps += 1

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)

        print(f"  Avg reward: {avg_reward:.1f} ± {std_reward:.1f}")
        print(f"  Range: [{min_reward:.1f}, {max_reward:.1f}]")
        print(f"  Games played: {num_games}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print("For policy gradient / PPO:")
    print("  → Use 'vp_only' or 'simplified'")
    print("  → Clear signal: VP gains = good")
    print("  → No PBRS decay causing negative drift")
    print("\nFor sparse reward methods (e.g., pure MCTS):")
    print("  → Use 'sparse'")
    print("  → Let search figure out intermediate steps")
    print("=" * 70)


if __name__ == "__main__":
    # Test the wrapper
    print("Testing SimplifiedRewardWrapper...\n")

    # Quick test
    env = SimplifiedRewardWrapper(player_id=0, reward_mode='vp_only')
    obs, _ = env.reset()
    print(f"Initial VP: {obs['my_victory_points']}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Action mask: {len(obs['action_mask'])} actions")

    # Compare modes
    print("\n")
    compare_reward_systems(num_games=10)
