"""
Rule-based AI opponents with multiple difficulty levels for curriculum training.

The idea is to give the RL agent progressively harder opponents as it
improves. Each difficulty adds another layer of intelligence:

- passive/truly_random: Does literally nothing or picks randomly. For
  the very first phase when the agent is learning basic building.
- weak: Random placements but at least tries to build things.
- medium: Scores settlements by pip count (probability of getting rolled).
- strong: Full heuristic — pip count + resource diversity + port access
  + gap-filling. This is actually pretty good and hard to beat consistently.

The scoring functions (score_vertex, score_edge) are also used by the
specialist AIs and the trainer's opponent selection logic.
"""

from game.game_system import ResourceType
import random


# Pip count for each dice number (probability of being rolled)
DICE_PIPS = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}


def score_vertex(vertex, game_board=None, player=None, consider_diversity=True,
                 consider_expansion=True, existing_settlement=None):
    """
    Score a vertex for settlement placement.

    Args:
        vertex: The vertex to score
        game_board: Optional GameBoard for expansion scoring
        player: Optional player (unused, for future port ownership checks)
        consider_diversity: Include resource diversity bonus
        consider_expansion: Include expansion potential scoring
        existing_settlement: Vertex of player's first settlement (for 2nd placement gap-filling)

    Returns:
        float: Higher = better placement location
    """
    if not vertex.adjacent_tiles:
        return 0

    # === 1. PIP COUNT ===
    pip_score = 0
    resources = {}  # resource_type -> pip_contribution

    for tile in vertex.adjacent_tiles:
        if tile.resource and tile.resource != "desert" and tile.number:
            pips = DICE_PIPS.get(tile.number, 0)
            pip_score += pips
            resource_type = tile.get_resource_type()
            if resource_type:
                resources[resource_type] = resources.get(resource_type, 0) + pips

    # === 2. RESOURCE DIVERSITY ===
    diversity_bonus = 0
    if consider_diversity:
        unique_resources = len(resources)
        diversity_bonus = unique_resources * 3  # 3 pts per unique resource type

        # Penalty for doubling up on the same resource (diminishing returns)
        for res_type, pip_contrib in resources.items():
            # If a resource has >5 pips worth of contribution, it's likely doubled up
            tiles_with_resource = sum(
                1 for tile in vertex.adjacent_tiles
                if tile.resource and tile.get_resource_type() == res_type
            )
            if tiles_with_resource > 1:
                diversity_bonus -= 2  # Penalty for duplicate resource type

    # === 3. HIGH-VALUE RESOURCES BONUS (ore and wheat for cities) ===
    value_bonus = 0
    for tile in vertex.adjacent_tiles:
        if tile.resource == "mountain":  # Ore
            value_bonus += 1
        elif tile.resource == "field":  # Wheat
            value_bonus += 1

    # === 4. GAP-FILLING BONUS (for 2nd settlement placement) ===
    # In 1v1, your second settlement should complement your first:
    # - Cover resources you don't have yet
    # - Avoid doubling up on what you already produce
    gap_bonus = 0
    if existing_settlement is not None and consider_diversity:
        # Get resources already covered by first settlement
        existing_resources = {}
        for tile in existing_settlement.adjacent_tiles:
            if tile.resource and tile.resource != "desert" and tile.number:
                rt = tile.get_resource_type()
                if rt:
                    existing_resources[rt] = existing_resources.get(rt, 0) + DICE_PIPS.get(tile.number, 0)

        # Bonus for each new resource type not covered by existing settlement
        for res_type in resources:
            if res_type not in existing_resources:
                gap_bonus += 4  # Strong bonus for filling a gap
            else:
                gap_bonus -= 1  # Small penalty for overlap

    # === 5. EXPANSION POTENTIAL ===
    # How many good future settlement spots are reachable within 2 roads?
    expansion_bonus = 0
    if consider_expansion and game_board is not None:
        expansion_bonus = _score_expansion_potential(vertex, game_board, depth=2)

    return pip_score + diversity_bonus + value_bonus + gap_bonus + expansion_bonus


def _score_expansion_potential(vertex, game_board, depth=2):
    """
    Score the expansion potential from a given vertex.

    Looks 'depth' roads away and scores reachable vertices that:
    - Are not already occupied
    - Don't violate the distance rule currently
    - Have good pip counts themselves

    In 1v1, we discount future spots heavily because the opponent
    will try to block corridors.

    Returns: expansion score (0-10 range typically)
    """
    FUTURE_DISCOUNT = 0.25  # Future spots are uncertain - don't overvalue

    visited_vertices = {vertex}
    frontier = {vertex}
    expansion_score = 0

    for road_step in range(depth):
        next_frontier = set()
        discount = FUTURE_DISCOUNT ** (road_step + 1)

        for v in frontier:
            # Get edges connected to this vertex
            connected_edges = []
            for edge in game_board.edges:
                if edge.vertex1 == v or edge.vertex2 == v:
                    connected_edges.append(edge)

            for edge in connected_edges:
                # Get the other vertex on this edge
                other = edge.vertex1 if edge.vertex2 == v else edge.vertex2

                if other in visited_vertices:
                    continue
                visited_vertices.add(other)
                next_frontier.add(other)

                # Can this vertex ever become a settlement?
                if other.structure is not None:
                    continue  # Already occupied

                # Check distance rule (no adjacent structures)
                too_close = any(
                    adj.structure is not None
                    for adj in other.adjacent_vertices
                )
                if too_close:
                    continue  # Too close to existing structure

                # Score this future spot
                future_pip = sum(
                    DICE_PIPS.get(tile.number, 0)
                    for tile in other.adjacent_tiles
                    if tile.resource and tile.resource != "desert" and tile.number
                )

                # Only count spots worth settling on (avoid dead spots)
                if future_pip >= 3:
                    expansion_score += future_pip * discount

        frontier = next_frontier

    # Normalize to reasonable range
    return min(expansion_score, 8.0)


def score_edge(edge, player, game):
    """
    Score an edge for road placement.
    Higher score = better expansion potential.
    """
    score = 0

    # Check both endpoints
    for vertex in [edge.vertex1, edge.vertex2]:
        if vertex.structure is None:
            # Can we build here eventually?
            too_close = any(adj.structure is not None for adj in vertex.adjacent_vertices)
            if not too_close:
                # Good expansion spot - score by vertex quality
                score += score_vertex(vertex, consider_diversity=False) / 2

    return score


class RuleBasedAI:
    """
    Rule-based AI with configurable difficulty levels.

    Difficulty levels:
    - 'very_weak': Picks from top 5 spots with randomness (bridge between random and weak)
    - 'weak': Random placement, basic build priorities
    - 'medium': Smart settlement placement (pip scoring), picks from top 3
    - 'strong': Smart placement + resource diversity + strategic roads, picks best
    """

    def __init__(self, difficulty='medium'):
        """
        Initialize the AI with a difficulty level.

        Args:
            difficulty: 'very_weak', 'weak', 'medium', or 'strong'
        """
        self.name = f"Rule-Based AI ({difficulty})"
        self.difficulty = difficulty

    def play_turn(self, game, player_id):
        """
        Execute one turn for the rule-based AI
        
        Args:
            game: The GameSystem object
            player_id: Which player this AI controls (0-3)
        
        Returns:
            bool: True if turn advanced, False if stuck
        """
        player = game.players[player_id]
        
        #print(f"[RULE AI DEBUG] Player {player_id+1} - Phase: {game.game_phase}, Turn phase: {game.turn_phase}")
        #print(f"[RULE AI DEBUG] can_roll_dice: {game.can_roll_dice()}, can_trade_or_build: {game.can_trade_or_build()}, can_end_turn: {game.can_end_turn()}")

        # Handle initial placement phase
        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player)

        # Phase 1: Roll dice if needed
        if game.can_roll_dice():
            result = game.roll_dice()
            #print(f"[RULE AI DEBUG] Rolled dice: {result}")
            if result:
                return True
            return False

        # Phase 2: Try to build (in priority order)
        if game.can_trade_or_build():
            # Get current resources
            resources = player.resources

            # Priority 1: Build city (2 wheat + 3 ore) = 2 VP!
            if self._can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    # Pick a random vertex from available options
                    vertex = random.choice(vertices)
                    success, msg = player.try_build_city(vertex)
                    if success:
                        #print(f"[RULE AI] Player {player_id+1} built a city!")
                        return True

            # Priority 2: Build settlement (1 wood + 1 brick + 1 wheat + 1 sheep) = 1 VP
            if self._can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    vertex = self._choose_best_vertex(vertices, game=game, player=player,
                                                       game_board=game.game_board)
                    success, msg = player.try_build_settlement(vertex, ignore_road_rule=False)
                    if success:
                        #print(f"[RULE AI] Player {player_id+1} built a settlement!")
                        return True

            # Priority 3: Build road (1 wood + 1 brick) = Progress toward longest road
            if self._can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    edge = self._choose_best_edge(edges, player, game)
                    success, msg = player.try_build_road(edge)
                    if success:
                        #print(f"[RULE AI] Player {player_id+1} built a road!")
                        return True

            # Priority 4: Buy development card (1 wheat + 1 sheep + 1 ore)
            if self._can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        #print(f"[RULE AI] Player {player_id+1} bought a dev card!")
                        return True

            # Priority 5: Try bank trading if we're close to affording something
            if self._should_trade(resources):
                success = self._try_beneficial_trade(game, player)
                if success:
                    #print(f"[RULE AI] Player {player_id+1} made a trade!")
                    return True

        # Phase 3: End turn if nothing else to do
        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        # Shouldn't reach here, but safety
        return False

    def _handle_initial_placement(self, game, player_id, player):
        """Handle initial placement phase"""
        #print(f"[RULE AI DEBUG] Handling initial placement for player {player_id+1}")
        #print(f"[RULE AI DEBUG] waiting_for_road: {game.waiting_for_road}")

        # Check if we need to place road
        if game.waiting_for_road:
            #print(f"[RULE AI DEBUG] Need to place road")
            # Find valid road positions (connected to last settlement)
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid_edges = [e for e in edges
                             if e.structure is None and
                             (e.vertex1 == game.last_settlement_vertex or
                              e.vertex2 == game.last_settlement_vertex)]

                if valid_edges:
                    # For medium/strong: pick road leading to best expansion vertex
                    if self.difficulty in ('medium', 'strong'):
                        last_v = game.last_settlement_vertex

                        def score_initial_road(edge):
                            other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                            return score_vertex(other, consider_diversity=True,
                                                consider_expansion=(self.difficulty == 'strong'),
                                                game_board=game.game_board)

                        valid_edges.sort(key=score_initial_road, reverse=True)
                        edge = valid_edges[0] if self.difficulty == 'strong' else random.choice(valid_edges[:min(2, len(valid_edges))])
                    else:
                        edge = random.choice(valid_edges)
                    success, msg = game.try_place_initial_road(edge, player)
                    #print(f"[RULE AI DEBUG] Place road result: {success}, {msg}")
                    if success:
                        #print(f"[RULE AI] Player {player_id+1} placed initial road")
                        return True
                else:
                    pass  # No valid edges found
                    #print(f"[RULE AI DEBUG] No valid edges found!")
            else:
                pass  # No last_settlement_vertex
                #print(f"[RULE AI DEBUG] No last_settlement_vertex!")
        else:
            #print(f"[RULE AI DEBUG] Need to place settlement")
            # Find valid settlement positions
            vertices = game.game_board.vertices
            valid_vertices = []

            for v in vertices:
                if v.structure is None:
                    # Check distance rule
                    too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
                    if not too_close:
                        valid_vertices.append(v)

            #print(f"[RULE AI DEBUG] Found {len(valid_vertices)} valid settlement positions")

            if valid_vertices:
                vertex = self._choose_best_vertex(valid_vertices, game=game, player=player,
                                                   game_board=game.game_board)
                success, msg = game.try_place_initial_settlement(vertex, player)
                #print(f"[RULE AI DEBUG] Place settlement result: {success}, {msg}")
                if success:
                    #print(f"[RULE AI] Player {player_id+1} placed initial settlement")
                    return True
            else:
                pass  # No valid vertices found
                #print(f"[RULE AI DEBUG] No valid vertices found!")

        return False

    def _choose_best_vertex(self, vertices, game=None, player=None, game_board=None):
        """Choose the best vertex based on difficulty level."""
        # Detect if this is 2nd settlement placement
        existing_settlement = None
        if player is not None and hasattr(player, 'settlements') and len(player.settlements) == 1:
            existing_settlement = player.settlements[0].position

        if self.difficulty == 'weak':
            # Weak: Random choice
            return random.choice(vertices)

        elif self.difficulty == 'very_weak':
            # Very Weak: Pip only, no diversity, no expansion, pick from top 5
            scored = [(v, score_vertex(v, consider_diversity=False, consider_expansion=False))
                      for v in vertices]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = min(5, len(scored))
            return random.choice([s[0] for s in scored[:top_n]])

        elif self.difficulty == 'medium':
            # Medium: Pip + diversity, no expansion, pick from top 3
            scored = [(v, score_vertex(
                v,
                consider_diversity=True,
                consider_expansion=False,
                existing_settlement=existing_settlement
            )) for v in vertices]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = min(3, len(scored))
            return random.choice([s[0] for s in scored[:top_n]])

        else:  # strong
            # Full scoring: pip + diversity + gap-filling + expansion
            scored = [(v, score_vertex(
                v,
                game_board=game_board,
                consider_diversity=True,
                consider_expansion=True,
                existing_settlement=existing_settlement
            )) for v in vertices]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

    def _choose_best_edge(self, edges, player, game):
        """Choose the best edge based on difficulty level."""
        if self.difficulty == 'weak':
            # Weak: Random choice
            return random.choice(edges)
        elif self.difficulty == 'very_weak':
            # Very Weak: Basic scoring, pick from top 5
            scored = [(e, score_edge(e, player, game)) for e in edges]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = min(5, len(scored))
            return random.choice([s[0] for s in scored[:top_n]])
        elif self.difficulty == 'medium':
            # Medium: Basic expansion scoring
            scored = [(e, score_edge(e, player, game)) for e in edges]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Add some randomness - pick from top 3
            top_n = min(3, len(scored))
            return random.choice([s[0] for s in scored[:top_n]])
        else:
            # Strong: Best expansion potential
            scored = [(e, score_edge(e, player, game)) for e in edges]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

    def _can_afford_city(self, resources):
        """Check if player can afford a city (2 wheat + 3 ore)"""
        return (resources[ResourceType.WHEAT] >= 2 and
                resources[ResourceType.ORE] >= 3)

    def _can_afford_settlement(self, resources):
        """Check if player can afford a settlement (1 wood + 1 brick + 1 wheat + 1 sheep)"""
        return (resources[ResourceType.WOOD] >= 1 and
                resources[ResourceType.BRICK] >= 1 and
                resources[ResourceType.WHEAT] >= 1 and
                resources[ResourceType.SHEEP] >= 1)

    def _can_afford_road(self, resources):
        """Check if player can afford a road (1 wood + 1 brick)"""
        return (resources[ResourceType.WOOD] >= 1 and
                resources[ResourceType.BRICK] >= 1)

    def _can_afford_dev_card(self, resources):
        """Check if player can afford a dev card (1 wheat + 1 sheep + 1 ore)"""
        return (resources[ResourceType.WHEAT] >= 1 and
                resources[ResourceType.SHEEP] >= 1 and
                resources[ResourceType.ORE] >= 1)

    def _should_trade(self, resources):
        """Decide if we should try trading"""
        # Trade if we have 4+ of one resource (can do 4:1 bank trade)
        for resource_type, amount in resources.items():
            if amount >= 4:
                return True
        return False

    def _try_beneficial_trade(self, game, player):
        """Try to make a beneficial trade to get closer to building something"""
        resources = player.resources

        # Find resources we have excess of (4+)
        excess_resources = []
        for resource_type, amount in resources.items():
            if amount >= 4:
                excess_resources.append(resource_type)

        if not excess_resources:
            return False

        # Decide what we need most
        needed = self._get_most_needed_resource(resources)

        if needed:
            # Try to trade excess for what we need
            give_resource = random.choice(excess_resources)

            # Get best trade ratio (checks for ports)
            ratio = game.game_board.get_best_trade_ratio(player, give_resource)

            # Execute trade if we have enough
            if resources[give_resource] >= ratio:
                success, msg = game.execute_bank_trade(player, give_resource, needed, ratio)
                return success

        return False

    def _get_most_needed_resource(self, resources):
        """Determine which resource we need most to build something"""
        # Priority: Try to complete a city (high value)
        if resources[ResourceType.WHEAT] >= 2 and resources[ResourceType.ORE] < 3:
            return ResourceType.ORE
        if resources[ResourceType.ORE] >= 3 and resources[ResourceType.WHEAT] < 2:
            return ResourceType.WHEAT

        # Try to complete a settlement
        needs = []
        if resources[ResourceType.WOOD] < 1:
            needs.append(ResourceType.WOOD)
        if resources[ResourceType.BRICK] < 1:
            needs.append(ResourceType.BRICK)
        if resources[ResourceType.WHEAT] < 1:
            needs.append(ResourceType.WHEAT)
        if resources[ResourceType.SHEEP] < 1:
            needs.append(ResourceType.SHEEP)

        if needs:
            return random.choice(needs)

        # Default: get wood or brick for roads
        if resources[ResourceType.WOOD] < resources[ResourceType.BRICK]:
            return ResourceType.WOOD
        return ResourceType.BRICK


def play_rule_based_turn(env, player_id, difficulty='medium'):
    """
    Convenience function to play one turn for a rule-based AI

    Args:
        env: CatanEnv instance
        player_id: Which player (0-3)
        difficulty: 'very_weak', 'weak', 'medium', or 'strong'

    Returns:
        bool: True if turn completed successfully
    """
    ai = RuleBasedAI(difficulty=difficulty)
    return ai.play_turn(env.game_env.game, player_id)


# ==================== TEST ====================

if __name__ == "__main__":
    #print("=" * 60)
    #print("TESTING RULE-BASED AI")
    #print("=" * 60)

    # This is just for testing - normally you'd use it in training
    from environment.ai_interface import AIGameEnvironment

    # Create game
    env = AIGameEnvironment()
    env.reset()

    #print("\n✅ Rule-based AI ready!")
    #print("   • Understands resource costs")
    #print("   • Prioritizes high-value actions")
    #print("   • Makes intelligent trades")
    #print("   • Never makes illegal moves")
    #print("\n" + "=" * 60)