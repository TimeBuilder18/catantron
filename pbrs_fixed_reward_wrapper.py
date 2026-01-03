"""
Fixed PBRS Reward Wrapper

The original PBRS implementation had the right idea but wrong scale.
This version:
1. Scales potential function to ±5 max (not ±50)
2. Has STRONG terminal rewards (win=+100, loss=-10)
3. PBRS is a small guide, not the main signal

Result: PBRS helps learning but doesn't dominate
"""

import numpy as np
from catan_env_pytorch import CatanEnv


class PBRSFixedRewardWrapper:
    """PBRS with correct scaling - shaping helps but doesn't dominate"""

    def __init__(self, player_id=0):
        self.env = CatanEnv(player_id=player_id)
        self.player_id = player_id
        self.last_potential = 0.0
        self.gamma = 0.99

    def reset(self):
        obs, info = self.env.reset()
        # Calculate initial potential
        player = self.env.game_env.game.players[self.player_id]
        self.last_potential = self._calculate_scaled_potential(player)
        return obs, info

    def _calculate_scaled_potential(self, player):
        """
        Calculate potential but SCALE IT DOWN!
        Original: 0-50 range
        Fixed: 0-5 range (10x smaller)
        """
        # Use the original potential function
        original_potential = self.env._calculate_potential(player)

        # Scale down by 10x to prevent dominating
        scaled_potential = original_potential / 10.0

        return scaled_potential

    def step(self, action_id, vertex_id=None, edge_id=None, trade_give_idx=None, trade_get_idx=None):
        """Step with fixed PBRS"""
        # Take action
        obs, _, terminated, truncated, info = self.env.step(
            action_id, vertex_id, edge_id, trade_give_idx, trade_get_idx
        )

        # Calculate new potential
        player = self.env.game_env.game.players[self.player_id]
        new_potential = self._calculate_scaled_potential(player)

        # PBRS shaping reward (small)
        pbrs_reward = self.gamma * new_potential - self.last_potential
        self.last_potential = new_potential

        # Base rewards (large!)
        base_reward = 0.0

        # VP changes (main signal)
        vp_diff = obs.get('my_victory_points', 0) - obs.get('prev_vp', 0)
        if vp_diff > 0:
            base_reward += vp_diff * 10.0  # +10 per VP

        # Terminal rewards (STRONG)
        if terminated:
            winner_id = info.get('winner_id', None)
            if winner_id == self.player_id:
                base_reward += 100.0  # HUGE win bonus
            else:
                base_reward -= 10.0  # Small loss penalty

        # Total reward = base + PBRS shaping
        total_reward = base_reward + pbrs_reward

        # Track prev VP for next step
        obs['prev_vp'] = obs.get('my_victory_points', 0)

        return obs, total_reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def compare_pbrs_scales():
    """Show the difference between original and fixed PBRS"""
    print("=" * 70)
    print("PBRS SCALING COMPARISON")
    print("=" * 70)

    from game_system import Player, GameBoard

    # Create dummy player with typical mid-game state
    # (This is just for demonstration)
    print("\nTypical mid-game state:")
    print("  - 3 settlements, 1 city")
    print("  - 8 roads")
    print("  - Some resources")
    print("  - ~6 VP")

    print("\n" + "-" * 70)
    print("ORIGINAL PBRS (broken):")
    print("-" * 70)
    print("  Potential value: ~25-30")
    print("  Per-step reward: 0.99*25 - 25 = -0.25")
    print("  Over 200 steps: -50")
    print("  Win bonus: +25")
    print("  TOTAL WIN GAME: -50 + 25 = -25 ❌")
    print("")
    print("  Problem: PBRS decay dominates!")

    print("\n" + "-" * 70)
    print("FIXED PBRS (scaled):")
    print("-" * 70)
    print("  Potential value: ~2.5-3.0 (10x smaller)")
    print("  Per-step reward: 0.99*2.5 - 2.5 = -0.025")
    print("  Over 200 steps: -5")
    print("  Win bonus: +100")
    print("  TOTAL WIN GAME: -5 + 100 = +95 ✅")
    print("")
    print("  Fixed: Base reward dominates, PBRS guides!")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("PBRS should be SMALLER than base rewards!")
    print("  - Base (win/loss): ±100 magnitude")
    print("  - PBRS (shaping): ±5 magnitude")
    print("  - Ratio: 20:1 base-to-shaping")
    print("")
    print("This way:")
    print("  ✅ Winning matters most (base reward)")
    print("  ✅ PBRS helps guide toward good states")
    print("  ✅ AI learns to win, not minimize steps")
    print("=" * 70)


if __name__ == "__main__":
    compare_pbrs_scales()
