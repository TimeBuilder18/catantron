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
from environment.catan_env import CatanEnv
from environment.reward_utils import PositionalRewardMixin


class PBRSFixedRewardWrapper(PositionalRewardMixin):
    """PBRS with correct scaling - shaping helps but doesn't dominate"""

    def __init__(self, player_id=0, victory_points_to_win=10, num_players=4):
        self.env = CatanEnv(player_id=player_id, victory_points_to_win=victory_points_to_win, num_players=num_players)
        self.player_id = player_id
        self.num_players = num_players
        self.last_potential = 0.0
        self.last_vp = 0
        self.gamma = 0.99
        self.victory_points_to_win = victory_points_to_win
        self.last_has_largest_army = False
        self.last_has_longest_road = False

    def reset(self):
        obs, info = self.env.reset()
        player = self.env.game_env.game.players[self.player_id]
        self.last_potential = self._calculate_scaled_potential(player)
        self.last_vp = obs.get('my_victory_points', 0)
        self.last_has_largest_army = player.has_largest_army
        self.last_has_longest_road = player.has_longest_road
        self._shaping_reset()
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
        current_vp = obs.get('my_victory_points', 0)
        vp_diff = current_vp - self.last_vp
        if vp_diff > 0:
            base_reward += vp_diff * 10.0  # +10 per VP

        # Phase-aware bonuses: optimal Catan strategy changes through the game.
        #   Early  (VP < 50% of win): EXPAND — roads + settlements claim territory
        #   Mid    (50–80% of win):   CITIES — upgrade production to fund endgame
        #   Late   (>80% of win):     DEV CARDS — largest army / VP cards seal the win
        #
        # Per-action value summary (bonus + 10 VP where applicable):
        #   Phase     road   settle  city   buy_dev  play_knight  play_monopoly  play_yop
        #   Early      +12    +25    +13      +5         +8            +6           +5
        #   Mid         +6    +15    +22      +8         +8            +6           +5
        #   Late        +3    +10    +20     +10         +8            +6           +5
        win_vp = self.victory_points_to_win
        phase = 'early' if current_vp < 0.5 * win_vp else ('late' if current_vp >= 0.8 * win_vp else 'mid')

        road_bonus      = {'early': 12, 'mid':  6, 'late':  3}[phase]
        settle_bonus    = {'early': 25, 'mid': 15, 'late': 10}[phase]
        city_bonus      = {'early': 12, 'mid': 15, 'late': 13}[phase]
        dev_card_bonus  = {'early':  5, 'mid':  8, 'late': 10}[phase]

        player = self.env.game_env.game.players[self.player_id]

        if info.get('built_city'):
            base_reward += city_bonus

        if info.get('built_settlement'):
            base_reward += settle_bonus
            # Positional quality: pip score, diversity, port access, blocking
            base_reward += self._settlement_shaping(player)

        action_name_step = info.get('action_name', '')
        if action_name_step == 'place_settlement' and info.get('success', True):
            base_reward += self._settlement_shaping(player)

        if info.get('action_name') == 'build_road' and info.get('success', False):
            base_reward += road_bonus
            # Directional quality: BFS toward best reachable settlement spot
            base_reward += self._road_expansion_shaping(player)

        # Effective trade: reward trades that open a build opportunity
        if info.get('bank_trade') and info.get('success') and info.get('trade_led_to_build_opportunity'):
            base_reward += 2.0

        if info.get('action_name') == 'buy_dev_card' and info.get('success', False):
            base_reward += dev_card_bonus

        # Dev card play rewards
        # Reward hierarchy (from most to least valuable per play):
        #   monopoly:  +3 per card stolen (steal 5 = +15, steal 0 = +0, typically > knight)
        #   knight:    +8 base (robber blocks best hex) + 3 steal + 5 if steal completes a build
        #   yop:       +5 (2 free resources of choice — targeted, not luck-based)
        #
        # VP dev card (auto on purchase): vp_diff already gives +10, plus explicit +15 bonus
        # so VP card buy = +10 VP + +15 bonus + dev_card_bonus = strongest single buy action
        action_name = info.get('action_name', '')
        if action_name == 'play_knight' and info.get('success', False):
            base_reward += 8.0  # Robber disruption value (Largest Army bonus tracked separately)
            if info.get('stolen_resource'):
                base_reward += 3.0  # Stole a resource (now always guaranteed if opponent has any)
                if info.get('steal_completes_build'):
                    base_reward += 5.0  # Stolen card was the last one needed for a build
        elif action_name == 'play_monopoly' and info.get('success', False):
            stolen_count = info.get('stolen_count', 0)
            base_reward += stolen_count * 3.0  # +3 per card stolen; beats knight when you steal 4+
        elif action_name == 'play_year_of_plenty' and info.get('success', False):
            base_reward += 5.0  # 2 targeted free resources (completes builds)
        elif action_name == 'buy_dev_card' and info.get('got_vp_card'):
            base_reward += 15.0  # VP card = instant +1 VP. vp_diff gives +10, this seals the deal.

        # 1v1 strategic milestone bonuses (on top of VP reward).
        # In 1v1 these confer permanent advantage beyond just the 2 bonus VP:
        #   Largest Army = ongoing robber control (suppress opponent's best hex every turn)
        #   Longest Road = map denial (cuts expansion routes, forces opponent to spend resources)
        # The VP signal alone undersells these since their denial value doesn't show up in VP.
        has_largest_army = player.has_largest_army
        has_longest_road = player.has_longest_road
        if has_largest_army and not self.last_has_largest_army:
            base_reward += 30.0  # Gained Largest Army: +2 VP already counted + denial bonus
        if has_longest_road and not self.last_has_longest_road:
            base_reward += 20.0  # Gained Longest Road: +2 VP already counted + map control bonus
        self.last_has_largest_army = has_largest_army
        self.last_has_longest_road = has_longest_road

        # Terminal rewards (STRONG)
        if terminated:
            winner_id = info.get('winner_id', None)
            if winner_id == self.player_id:
                base_reward += 100.0  # HUGE win bonus
            else:
                base_reward -= 10.0  # Small loss penalty

        # Total reward = base + PBRS shaping
        total_reward = base_reward + pbrs_reward

        # Track current VP for next step
        self.last_vp = current_vp

        return obs, total_reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def compare_pbrs_scales():
    """Show the difference between original and fixed PBRS"""
    print("=" * 70)
    print("PBRS SCALING COMPARISON")
    print("=" * 70)

    from game.game_system import Player, GameBoard

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
