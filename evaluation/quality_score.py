"""
Agent Quality Score (AQS) for 1v1 Catan AI Evaluation
======================================================
Not all wins are created equal. A win in 45 turns with 3 cities against a strong
opponent is vastly different from a 200-turn win with longest road against a passive bot.

This module provides a composite scoring equation that quantifies agent quality
across multiple dimensions, producing a single 0-1000 AQS rating.

The AQS Formula
===============
    AQS = w1*Outcome + w2*Efficiency + w3*Economy + w4*Tempo + w5*Dominance + w6*Adaptability

Each component is normalized to [0, 100], then weighted and scaled to [0, 1000].

Components
----------
1. Outcome Score     (w=0.30) - Did you win? By how much?
2. Efficiency Score  (w=0.20) - How quickly and cleanly did you win?
3. Economy Score     (w=0.15) - How well did you manage resources?
4. Tempo Score       (w=0.15) - Did you build at the right times?
5. Dominance Score   (w=0.10) - How much did you control the board?
6. Adaptability Score(w=0.10) - Can you beat different opponent types?

Usage
-----
    evaluator = AgentQualityEvaluator()
    evaluator.record_game(game_stats)
    aqs = evaluator.calculate_aqs()
    print(f"Agent Quality Score: {aqs:.0f} / 1000")
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ============================================================================
# Data Structure for a Single Game Result
# ============================================================================

@dataclass
class GameResult:
    """Stats collected from a single game for AQS calculation."""
    # --- Core outcome ---
    won: bool
    agent_vp: int
    opponent_vp: int
    victory_points_to_win: int = 10

    # --- Timing ---
    total_turns: int = 0
    agent_turns: int = 0

    # --- Building milestones ---
    settlements_built: int = 0
    cities_built: int = 0
    roads_built: int = 0
    dev_cards_bought: int = 0
    dev_cards_played: int = 0

    # --- Resource economy ---
    total_resources_gained: int = 0
    total_resources_spent: int = 0
    resources_lost_to_robber: int = 0
    bank_trades_made: int = 0

    # --- Board control ---
    has_longest_road: bool = False
    has_largest_army: bool = False
    knights_played: int = 0
    robber_placements: int = 0

    # --- Opponent info ---
    opponent_type: str = "unknown"
    opponent_difficulty: float = 0.5

    # --- Optional advanced metrics ---
    longest_road_length: int = 0
    port_access_count: int = 0
    pip_count: float = 0.0
    resource_diversity: int = 0


# ============================================================================
# Component Score Calculators
# ============================================================================

def outcome_score(results: List[GameResult]) -> float:
    """
    OUTCOME SCORE (0-100): Did you win, and by how much?

    Formula:
        outcome = 40 * win_rate + 30 * avg_vp_ratio + 30 * margin_bonus
    """
    if not results:
        return 0.0

    wins = sum(1 for r in results if r.won)
    win_rate = wins / len(results)

    avg_vp_ratio = np.mean([
        min(1.0, r.agent_vp / max(1, r.victory_points_to_win))
        for r in results
    ])

    margins = []
    for r in results:
        diff = r.agent_vp - r.opponent_vp
        normalized = diff / max(1, r.victory_points_to_win)
        margins.append(max(0.0, min(1.0, (normalized + 1) / 2)))
    avg_margin = np.mean(margins)

    return 40 * win_rate + 30 * avg_vp_ratio + 30 * avg_margin


def efficiency_score(results: List[GameResult]) -> float:
    """
    EFFICIENCY SCORE (0-100): How quickly and cleanly did you win?

    Formula:
        efficiency = 50 * turn_efficiency + 25 * build_density + 25 * waste_ratio
    """
    if not results:
        return 0.0

    turn_scores = []
    build_scores = []
    waste_scores = []

    for r in results:
        max_turns = 150
        min_turns = 40
        if r.agent_turns > 0:
            t_eff = 1.0 - (r.agent_turns - min_turns) / (max_turns - min_turns)
            t_eff = max(0.0, min(1.0, t_eff))
        else:
            t_eff = 0.0
        turn_scores.append(t_eff)

        total_builds = (r.settlements_built + r.cities_built +
                        r.roads_built + r.dev_cards_bought)
        if r.agent_turns > 0:
            density = total_builds / r.agent_turns
            b_eff = min(1.0, density / 0.3)
        else:
            b_eff = 0.0
        build_scores.append(b_eff)

        if r.total_resources_gained > 0:
            waste = r.resources_lost_to_robber / r.total_resources_gained
            w_eff = max(0.0, 1.0 - waste)
        else:
            w_eff = 0.5
        waste_scores.append(w_eff)

    return (50 * np.mean(turn_scores) +
            25 * np.mean(build_scores) +
            25 * np.mean(waste_scores))


def economy_score(results: List[GameResult]) -> float:
    """
    ECONOMY SCORE (0-100): How well did you manage resources?

    Formula:
        economy = 35 * conversion_rate + 25 * trade_efficiency
                + 20 * resource_diversity_norm + 20 * pip_efficiency
    """
    if not results:
        return 0.0

    conversion_scores = []
    trade_scores = []
    diversity_scores = []
    pip_scores = []

    for r in results:
        if r.total_resources_gained > 0:
            conv = min(1.0, r.total_resources_spent / r.total_resources_gained)
        else:
            conv = 0.0
        conversion_scores.append(conv)

        if r.agent_turns > 0:
            trades_per_turn = r.bank_trades_made / r.agent_turns
            t_eff = max(0.0, 1.0 - max(0, trades_per_turn - 0.05) / 0.15)
        else:
            t_eff = 0.5
        trade_scores.append(t_eff)

        diversity_scores.append(r.resource_diversity / 5.0)

        max_pips = 30.0
        pip_scores.append(min(1.0, r.pip_count / max_pips))

    return (35 * np.mean(conversion_scores) +
            25 * np.mean(trade_scores) +
            20 * np.mean(diversity_scores) +
            20 * np.mean(pip_scores))


def tempo_score(results: List[GameResult]) -> float:
    """
    TEMPO SCORE (0-100): Did you build at the right times?

    Formula:
        tempo = 40 * city_timing + 30 * expansion_rate + 30 * dev_card_usage
    """
    if not results:
        return 0.0

    city_scores = []
    expansion_scores = []
    dev_scores = []

    for r in results:
        if r.settlements_built > 0:
            city_ratio = r.cities_built / r.settlements_built
            c_score = min(1.0, city_ratio / 0.8)
        else:
            c_score = 0.0
        city_scores.append(c_score)

        extra_settlements = max(0, r.settlements_built - 2)
        expected_extra = 2
        e_score = min(1.0, extra_settlements / expected_extra)
        expansion_scores.append(e_score)

        if r.dev_cards_bought > 0:
            usage = r.dev_cards_played / r.dev_cards_bought
            d_score = min(1.0, usage)
        else:
            d_score = 0.4
        dev_scores.append(d_score)

    return (40 * np.mean(city_scores) +
            30 * np.mean(expansion_scores) +
            30 * np.mean(dev_scores))


def dominance_score(results: List[GameResult]) -> float:
    """
    DOMINANCE SCORE (0-100): How much did you control the board?

    Formula:
        dominance = 30 * trophy_rate + 30 * robber_control + 40 * vp_lead
    """
    if not results:
        return 0.0

    trophy_scores = []
    robber_scores = []
    lead_scores = []

    for r in results:
        has_trophy = r.has_longest_road or r.has_largest_army
        trophy_scores.append(1.0 if has_trophy else 0.0)

        if r.agent_turns > 0:
            robber_rate = r.robber_placements / r.agent_turns
            rob_score = min(1.0, robber_rate / 0.08)
        else:
            rob_score = 0.0
        robber_scores.append(rob_score)

        diff = r.agent_vp - r.opponent_vp
        normalized_lead = diff / max(1, r.victory_points_to_win)
        lead_score = max(0.0, min(1.0, (normalized_lead + 0.5) / 1.0))
        lead_scores.append(lead_score)

    return (30 * np.mean(trophy_scores) +
            30 * np.mean(robber_scores) +
            40 * np.mean(lead_scores))


def adaptability_score(results: List[GameResult]) -> float:
    """
    ADAPTABILITY SCORE (0-100): Can you beat different opponent types?

    Each opponent type is weighted by its difficulty.
    """
    if not results:
        return 0.0

    DIFFICULTY_MAP = {
        'passive': 0.1,
        'truly_random': 0.3,
        'random': 0.4,
        'weighted_random': 0.5,
        'very_weak': 0.55,
        'weak': 0.6,
        'medium': 0.8,
        'strong': 1.0,
        'self_play': 1.0,
        'unknown': 0.5,
    }

    by_type: Dict[str, List[GameResult]] = {}
    for r in results:
        key = r.opponent_type.lower()
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(r)

    if not by_type:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    for opp_type, games in by_type.items():
        difficulty = DIFFICULTY_MAP.get(opp_type, games[0].opponent_difficulty)
        wr = sum(1 for g in games if g.won) / len(games)
        avg_vp_ratio = np.mean([g.agent_vp / max(1, g.victory_points_to_win) for g in games])
        type_score = 0.6 * wr + 0.4 * avg_vp_ratio
        weighted_sum += difficulty * type_score * len(games)
        weight_total += difficulty * len(games)

    if weight_total == 0:
        return 0.0

    return 100.0 * (weighted_sum / weight_total)


# ============================================================================
# Elo Rating (for relative comparison between agent versions)
# ============================================================================

class EloTracker:
    """Simple Elo implementation for tracking agent versions."""

    def __init__(self, k_factor: float = 32.0, base_rating: float = 1000.0):
        self.k = k_factor
        self.ratings: Dict[str, float] = {}
        self.base_rating = base_rating

    def get_rating(self, agent_name: str) -> float:
        return self.ratings.get(agent_name, self.base_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner: str, loser: str, draw: bool = False):
        ra = self.get_rating(winner)
        rb = self.get_rating(loser)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        if draw:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 1.0, 0.0

        self.ratings[winner] = ra + self.k * (sa - ea)
        self.ratings[loser] = rb + self.k * (sb - eb)

    def record_game(self, agent_a: str, agent_b: str, a_won: bool, draw: bool = False):
        if draw:
            self.update(agent_a, agent_b, draw=True)
        elif a_won:
            self.update(agent_a, agent_b)
        else:
            self.update(agent_b, agent_a)


# ============================================================================
# Main Evaluator
# ============================================================================

DEFAULT_WEIGHTS = {
    'outcome':      0.30,
    'efficiency':   0.20,
    'economy':      0.15,
    'tempo':        0.15,
    'dominance':    0.10,
    'adaptability': 0.10,
}


class AgentQualityEvaluator:
    """
    Composite Agent Quality Score (AQS) calculator.

    Score interpretation:
        0-200:    Beginner      - Random-level play, no strategy
        200-400:  Novice        - Basic building, poor efficiency
        400-600:  Intermediate  - Wins against weak opponents, some strategy
        600-800:  Advanced      - Consistent wins, good economy and tempo
        800-1000: Expert        - Dominant play across all dimensions
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 window_size: int = 100):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.results: deque = deque(maxlen=window_size)
        self.all_results: List[GameResult] = []
        self.window_size = window_size

    def record_game(self, game_stats: dict):
        """Record a game from a stats dictionary.

        Minimal required keys: 'won', 'agent_vp' (or 'vp'), 'opponent_vp'
        """
        result = GameResult(
            won=game_stats.get('won', False),
            agent_vp=game_stats.get('agent_vp', game_stats.get('vp', 0)),
            opponent_vp=game_stats.get('opponent_vp', 0),
            victory_points_to_win=game_stats.get('victory_points_to_win', 10),
            total_turns=game_stats.get('total_turns', 0),
            agent_turns=game_stats.get('agent_turns', game_stats.get('total_turns', 0) // 2),
            settlements_built=game_stats.get('settlements_built', 0),
            cities_built=game_stats.get('cities_built', 0),
            roads_built=game_stats.get('roads_built', 0),
            dev_cards_bought=game_stats.get('dev_cards_bought', 0),
            dev_cards_played=game_stats.get('dev_cards_played', 0),
            total_resources_gained=game_stats.get('total_resources_gained', 0),
            total_resources_spent=game_stats.get('total_resources_spent', 0),
            resources_lost_to_robber=game_stats.get('resources_lost_to_robber', 0),
            bank_trades_made=game_stats.get('bank_trades_made', 0),
            has_longest_road=game_stats.get('has_longest_road', False),
            has_largest_army=game_stats.get('has_largest_army', False),
            knights_played=game_stats.get('knights_played', 0),
            robber_placements=game_stats.get('robber_placements', 0),
            opponent_type=game_stats.get('opponent_type', 'unknown'),
            opponent_difficulty=game_stats.get('opponent_difficulty', 0.5),
            longest_road_length=game_stats.get('longest_road_length', 0),
            port_access_count=game_stats.get('port_access_count', 0),
            pip_count=game_stats.get('pip_count', 0.0),
            resource_diversity=game_stats.get('resource_diversity', 0),
        )
        self.results.append(result)
        self.all_results.append(result)

    def record_game_result(self, result: GameResult):
        """Record a GameResult directly."""
        self.results.append(result)
        self.all_results.append(result)

    def calculate_aqs(self) -> float:
        """Calculate the Agent Quality Score (0-1000)."""
        results = list(self.results)
        if not results:
            return 0.0

        components = self.get_component_scores(results)
        aqs = sum(
            self.weights[name] * score
            for name, score in components.items()
        ) * 10

        return round(max(0.0, min(1000.0, aqs)), 1)

    def get_component_scores(self, results: Optional[List[GameResult]] = None) -> Dict[str, float]:
        """Get individual component scores (each 0-100)."""
        if results is None:
            results = list(self.results)
        return {
            'outcome':      outcome_score(results),
            'efficiency':   efficiency_score(results),
            'economy':      economy_score(results),
            'tempo':        tempo_score(results),
            'dominance':    dominance_score(results),
            'adaptability': adaptability_score(results),
        }

    def get_breakdown(self) -> dict:
        """Get a full breakdown of the AQS with all components."""
        results = list(self.results)
        components = self.get_component_scores(results)
        aqs = self.calculate_aqs()

        wins = sum(1 for r in results if r.won)
        avg_vp = np.mean([r.agent_vp for r in results]) if results else 0
        avg_opp_vp = np.mean([r.opponent_vp for r in results]) if results else 0
        avg_turns = np.mean([r.agent_turns for r in results]) if results else 0

        return {
            'aqs': aqs,
            'games_evaluated': len(results),
            'components': {
                name: {
                    'score': round(score, 1),
                    'weight': self.weights[name],
                    'weighted': round(score * self.weights[name] * 10, 1),
                }
                for name, score in components.items()
            },
            'summary': {
                'win_rate': wins / max(1, len(results)),
                'avg_agent_vp': round(avg_vp, 2),
                'avg_opponent_vp': round(avg_opp_vp, 2),
                'avg_agent_turns': round(avg_turns, 1),
            },
            'tier': self._get_tier(aqs),
        }

    def get_trend(self, window: int = 20) -> List[float]:
        """Get AQS trend over time using a sliding window."""
        if len(self.all_results) < window:
            return [self.calculate_aqs()]

        trend = []
        for i in range(window, len(self.all_results) + 1, max(1, window // 4)):
            chunk = self.all_results[max(0, i - window):i]
            components = self.get_component_scores(chunk)
            aqs = sum(self.weights[n] * s for n, s in components.items()) * 10
            trend.append(round(aqs, 1))

        return trend

    @staticmethod
    def _get_tier(aqs: float) -> str:
        if aqs >= 800:
            return "Expert"
        elif aqs >= 600:
            return "Advanced"
        elif aqs >= 400:
            return "Intermediate"
        elif aqs >= 200:
            return "Novice"
        else:
            return "Beginner"

    def print_report(self):
        """Print a formatted AQS report to console."""
        breakdown = self.get_breakdown()

        print("\n" + "=" * 70)
        print(f"  AGENT QUALITY SCORE: {breakdown['aqs']:.0f} / 1000  [{breakdown['tier']}]")
        print("=" * 70)
        print(f"  Games evaluated: {breakdown['games_evaluated']}")
        print(f"  Win rate: {breakdown['summary']['win_rate']:.1%}")
        print(f"  Avg VP: {breakdown['summary']['avg_agent_vp']:.1f} vs "
              f"{breakdown['summary']['avg_opponent_vp']:.1f}")
        print(f"  Avg turns: {breakdown['summary']['avg_agent_turns']:.0f}")
        print()

        print("  Component Breakdown:")
        print("  " + "-" * 50)
        for name, info in breakdown['components'].items():
            bar_len = int(info['score'] / 2)
            bar = "\u2588" * bar_len + "\u2591" * (50 - bar_len)
            print(f"  {name:15s} {bar} {info['score']:5.1f} "
                  f"(x{info['weight']:.2f} = {info['weighted']:5.1f})")
        print("  " + "-" * 50)
        print(f"  {'TOTAL':15s} {'':50s} {breakdown['aqs']:>11.0f}")
        print("=" * 70)


# ============================================================================
# Quick Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("Agent Quality Score (AQS) - Demo")
    print("=" * 70)

    evaluator = AgentQualityEvaluator()

    # Simulate a weak agent: wins sometimes but slowly, poor economy
    np.random.seed(42)
    for i in range(50):
        won = np.random.random() < 0.3
        evaluator.record_game({
            'won': won,
            'agent_vp': np.random.randint(4, 8) if not won else 10,
            'opponent_vp': 10 if not won else np.random.randint(4, 8),
            'victory_points_to_win': 10,
            'total_turns': np.random.randint(120, 250),
            'agent_turns': np.random.randint(60, 125),
            'settlements_built': np.random.randint(2, 4),
            'cities_built': np.random.randint(0, 2),
            'roads_built': np.random.randint(3, 8),
            'dev_cards_bought': np.random.randint(0, 3),
            'dev_cards_played': np.random.randint(0, 2),
            'total_resources_gained': np.random.randint(40, 80),
            'total_resources_spent': np.random.randint(20, 50),
            'resources_lost_to_robber': np.random.randint(5, 20),
            'bank_trades_made': np.random.randint(2, 10),
            'has_longest_road': np.random.random() < 0.2,
            'has_largest_army': np.random.random() < 0.1,
            'robber_placements': np.random.randint(0, 3),
            'opponent_type': 'truly_random',
            'pip_count': np.random.uniform(10, 20),
            'resource_diversity': np.random.randint(2, 5),
        })

    evaluator.print_report()

    print("\n\nNow simulating a strong agent...\n")

    strong_evaluator = AgentQualityEvaluator()
    for i in range(50):
        won = np.random.random() < 0.85
        strong_evaluator.record_game({
            'won': won,
            'agent_vp': 10 if won else np.random.randint(6, 9),
            'opponent_vp': np.random.randint(3, 7) if won else 10,
            'victory_points_to_win': 10,
            'total_turns': np.random.randint(60, 100),
            'agent_turns': np.random.randint(30, 50),
            'settlements_built': np.random.randint(3, 5),
            'cities_built': np.random.randint(2, 4),
            'roads_built': np.random.randint(5, 10),
            'dev_cards_bought': np.random.randint(2, 5),
            'dev_cards_played': np.random.randint(2, 4),
            'total_resources_gained': np.random.randint(60, 120),
            'total_resources_spent': np.random.randint(50, 100),
            'resources_lost_to_robber': np.random.randint(2, 8),
            'bank_trades_made': np.random.randint(1, 5),
            'has_longest_road': np.random.random() < 0.5,
            'has_largest_army': np.random.random() < 0.4,
            'robber_placements': np.random.randint(2, 6),
            'opponent_type': 'weighted_random',
            'pip_count': np.random.uniform(18, 28),
            'resource_diversity': np.random.randint(4, 6),
        })

    strong_evaluator.print_report()

    # Show Elo comparison
    print("\n\nElo Comparison (agent versions):")
    print("-" * 40)
    elo = EloTracker()
    for i in range(20):
        elo.record_game("v1_weak", "v2_strong", a_won=(np.random.random() < 0.3))
    for i in range(20):
        elo.record_game("v2_strong", "v3_expert", a_won=(np.random.random() < 0.4))

    for name in ["v1_weak", "v2_strong", "v3_expert"]:
        print(f"  {name:15s}: Elo {elo.get_rating(name):.0f}")
