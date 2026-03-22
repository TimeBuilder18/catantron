"""
Specialist AIs for Catan 1v1 Training

Each specialist commits to a single whole-game strategy from placement through endgame.
They are designed to expose strategic blindspots in the generalist RL agent by playing
styles it has never encountered — pure city rushing, road blocking, dev card spamming,
aggressive disruption, and port-based trading engines.

Usage:
    from ai.specialist_opponents import play_specialist_turn
    play_specialist_turn(game, player_id, strategy='city_rusher')

Available strategies:
    'city_rusher'         — Cities + dev cards, ore/wheat focus
    'road_blocker'        — Longest Road + corridor denial, wood/brick focus
    'dev_card_spammer'    — Largest Army + VP cards, ore/wheat/sheep focus
    'balanced_aggressor'  — Strong AI clone with aggressive robber/blocking
    'port_specialist'     — Port-based trading engine, adapts to best available port
"""

import random
from game.game_system import ResourceType, DevelopmentCardType, PortType, Settlement, City
from ai.scripted_opponents import score_vertex, score_edge, DICE_PIPS


# ============================================================
# HELPERS
# ============================================================

def _pip_count(vertex):
    """Total pip count for a vertex across all producing tiles."""
    total = 0
    for tile in vertex.adjacent_tiles:
        if tile.resource and tile.resource != "desert" and tile.number:
            total += DICE_PIPS.get(tile.number, 0)
    return total


def _resource_pips(vertex, resource_name):
    """Pip count at a vertex for a specific tile resource string (e.g. 'mountain' for ore)."""
    total = 0
    for tile in vertex.adjacent_tiles:
        if tile.resource == resource_name and tile.number:
            total += DICE_PIPS.get(tile.number, 0)
    return total


def _get_opponent(game, player_id):
    """Get the opponent player in a 1v1 game."""
    for i, p in enumerate(game.players):
        if i != player_id:
            return p
    return None


def _get_opponent_best_tile(game, opponent):
    """Find the highest-pip producing tile adjacent to opponent's structures.

    Used for robber placement — always target the opponent's best hex.
    Returns None if opponent has no structures or all tiles are desert.
    """
    best_tile = None
    best_pips = -1

    for vertex in game.game_board.vertices:
        if vertex.structure and vertex.structure.player == opponent:
            for tile in vertex.adjacent_tiles:
                if tile.resource and tile.resource != "desert" and tile.number:
                    pips = DICE_PIPS.get(tile.number, 0)
                    # Don't place robber on tile that already has it
                    if pips > best_pips and not tile.has_robber:
                        best_pips = pips
                        best_tile = tile

    return best_tile


def _try_play_knight(game, player, opponent):
    """Try to play a knight card, move robber to opponent's best hex, and steal.

    Returns True if knight was successfully played.
    """
    if player.development_cards[DevelopmentCardType.KNIGHT] <= 0:
        return False

    success, msg = game.play_knight_card(player)
    if not success:
        return False

    # Move robber to opponent's highest-pip tile
    best_tile = _get_opponent_best_tile(game, opponent)
    if best_tile:
        game.move_robber_to_tile(best_tile)
        # Steal a resource if opponent is on that tile
        players_on_tile = game.get_players_on_tile(best_tile)
        if opponent in players_on_tile:
            game.steal_random_resource(player, opponent)

    return True


def _can_afford_city(resources):
    return resources[ResourceType.WHEAT] >= 2 and resources[ResourceType.ORE] >= 3


def _can_afford_settlement(resources):
    return (resources[ResourceType.WOOD] >= 1 and resources[ResourceType.BRICK] >= 1 and
            resources[ResourceType.WHEAT] >= 1 and resources[ResourceType.SHEEP] >= 1)


def _can_afford_road(resources):
    return resources[ResourceType.WOOD] >= 1 and resources[ResourceType.BRICK] >= 1


def _can_afford_dev_card(resources):
    return (resources[ResourceType.WHEAT] >= 1 and resources[ResourceType.SHEEP] >= 1 and
            resources[ResourceType.ORE] >= 1)


def _get_valid_initial_vertices(game):
    """Get all vertices valid for initial settlement placement."""
    valid = []
    for v in game.game_board.vertices:
        if v.structure is None:
            too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
            if not too_close:
                valid.append(v)
    return valid


def _get_valid_initial_roads(game):
    """Get edges connected to the last placed settlement."""
    if not game.last_settlement_vertex:
        return []
    return [e for e in game.game_board.edges
            if e.structure is None and
            (e.vertex1 == game.last_settlement_vertex or
             e.vertex2 == game.last_settlement_vertex)]


def _try_conservative_trade(game, player, desired_resources, excess_resources):
    """Trade only when we have 4+ of an excess resource and won't starve our strategy.

    Args:
        game: GameSystem instance
        player: Player doing the trading
        desired_resources: list of ResourceType we want, in priority order
        excess_resources: list of ResourceType we're willing to trade away

    Returns True if a trade was made.
    """
    resources = player.resources

    for give_res in excess_resources:
        ratio = game.game_board.get_best_trade_ratio(player, give_res)
        # Only trade if we have enough surplus (at least ratio + 1 to keep a buffer,
        # or ratio if we have a lot)
        if resources[give_res] < ratio:
            continue
        # After trading, we should still have at least 1 left (conservative)
        if resources[give_res] - ratio < 1 and resources[give_res] < ratio + 2:
            continue

        for want_res in desired_resources:
            if resources[want_res] < 3:  # Only trade for things we actually need
                success, msg = game.execute_bank_trade(player, give_res, want_res, ratio)
                if success:
                    return True

    return False


def _vertex_has_port(vertex, game_board):
    """Check if a vertex is on a port and return the port, or None."""
    for port in game_board.ports:
        if port.vertex1 == vertex or port.vertex2 == vertex:
            return port
    return None


# ============================================================
# SPECIALIST 1: CITY RUSHER
# ============================================================

class CityRusher:
    """Win through cities + dev card VP cards. Focuses on ore/wheat production.

    Strategy: 2 settlements → upgrade both to cities (4 VP) → dev cards for
    VP cards + Largest Army → 10 VP. Never builds roads for Longest Road.
    """

    def play_turn(self, game, player_id):
        player = game.players[player_id]
        opponent = _get_opponent(game, player_id)

        # --- Initial placement ---
        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player)

        # --- Roll dice ---
        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        # --- Build phase ---
        if game.can_trade_or_build():
            resources = player.resources

            # Play knight immediately — path to Largest Army (+2 VP)
            if opponent and _try_play_knight(game, player, opponent):
                return True

            # Priority 1: City — 2 VP each, core of the strategy
            if _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    # Upgrade the settlement with the best ore/wheat production
                    best_v = max(vertices, key=lambda v:
                                 _resource_pips(v, 'mountain') + _resource_pips(v, 'field'))
                    success, msg = player.try_build_city(best_v)
                    if success:
                        return True

            # Priority 2: Dev card — looking for VP cards + knights
            if _can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        return True

            # Priority 3: Settlement — ONLY if all current settlements are already cities
            # (meaning we have room to expand and nothing left to upgrade)
            if len(player.settlements) == 0 and _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    # Pick best ore/wheat vertex
                    best_v = max(vertices, key=lambda v: self._score_vertex(v))
                    success, msg = player.try_build_settlement(best_v)
                    if success:
                        return True

            # Priority 4: Road — ONLY to reach a new ore/wheat settlement spot
            if _can_afford_road(resources) and len(player.settlements) == 0:
                edges = game.get_buildable_edges()
                if edges:
                    # Score edges by whether they lead to ore/wheat vertices
                    best_e = max(edges, key=lambda e: self._score_edge_for_ore_wheat(e))
                    if self._score_edge_for_ore_wheat(best_e) > 0:
                        success, msg = player.try_build_road(best_e)
                        if success:
                            return True

            # Trade: surplus wood/brick for ore/wheat (conservative)
            _try_conservative_trade(
                game, player,
                desired_resources=[ResourceType.ORE, ResourceType.WHEAT, ResourceType.SHEEP],
                excess_resources=[ResourceType.WOOD, ResourceType.BRICK]
            )

        # --- End turn ---
        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False

    def _handle_initial_placement(self, game, player_id, player):
        if game.waiting_for_road:
            valid_edges = _get_valid_initial_roads(game)
            if valid_edges:
                # Point road toward next best ore/wheat vertex
                last_v = game.last_settlement_vertex

                def score_road(edge):
                    other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                    return self._score_vertex(other)

                valid_edges.sort(key=score_road, reverse=True)
                success, msg = game.try_place_initial_road(valid_edges[0], player)
                return success
        else:
            valid_vertices = _get_valid_initial_vertices(game)
            if valid_vertices:
                # Get existing settlement for gap-filling on 2nd placement
                existing = player.settlements[0].position if len(player.settlements) == 1 else None
                scored = [(v, self._score_vertex(v, existing)) for v in valid_vertices]
                scored.sort(key=lambda x: x[1], reverse=True)
                success, msg = game.try_place_initial_settlement(scored[0][0], player)
                return success
        return False

    def _score_vertex(self, vertex, existing_settlement=None):
        """Score vertex heavily weighted toward ore + wheat."""
        ore_pips = _resource_pips(vertex, 'mountain')
        wheat_pips = _resource_pips(vertex, 'field')
        other_pips = _pip_count(vertex) - ore_pips - wheat_pips

        # Ore and wheat are worth 3x; other resources are filler
        score = (ore_pips * 3) + (wheat_pips * 3) + other_pips

        # Gap-fill bonus for 2nd settlement
        if existing_settlement:
            existing_ore = _resource_pips(existing_settlement, 'mountain')
            existing_wheat = _resource_pips(existing_settlement, 'field')
            # Bonus if this fills a gap in ore or wheat
            if existing_ore == 0 and ore_pips > 0:
                score += 5
            if existing_wheat == 0 and wheat_pips > 0:
                score += 5
            # Small bonus for sheep (needed for dev cards)
            existing_sheep = _resource_pips(existing_settlement, 'pasture')
            if existing_sheep == 0 and _resource_pips(vertex, 'pasture') > 0:
                score += 3

        return score

    def _score_edge_for_ore_wheat(self, edge):
        """Score edge by whether it leads toward an ore/wheat settlement spot."""
        score = 0
        for vertex in [edge.vertex1, edge.vertex2]:
            if vertex.structure is None:
                too_close = any(adj.structure is not None for adj in vertex.adjacent_vertices)
                if not too_close:
                    score += _resource_pips(vertex, 'mountain') + _resource_pips(vertex, 'field')
        return score


# ============================================================
# SPECIALIST 2: ROAD BLOCKER
# ============================================================

class RoadBlocker:
    """Win through Longest Road (+2 VP) + settlements + denying opponent expansion.

    Strategy: Spam roads to claim Longest Road, then settle isolated spots.
    Actively builds roads to cut off opponent expansion corridors.
    Never buys dev cards — wastes ore/wheat/sheep that could be traded for wood/brick.
    """

    def play_turn(self, game, player_id):
        player = game.players[player_id]
        opponent = _get_opponent(game, player_id)

        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player, opponent)

        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        if game.can_trade_or_build():
            resources = player.resources

            # Priority 1: Road — spam roads, build 2-3 per turn if affordable
            # Build up to 3 roads per action cycle before considering other builds
            roads_built_this_cycle = 0
            while _can_afford_road(resources) and roads_built_this_cycle < 3:
                edges = game.get_buildable_edges()
                if not edges:
                    break
                best_e = max(edges, key=lambda e: self._score_edge(e, player, opponent, game))
                success, msg = player.try_build_road(best_e)
                if success:
                    roads_built_this_cycle += 1
                    resources = player.resources  # Refresh after spending
                else:
                    break

            if roads_built_this_cycle > 0:
                return True

            # Priority 2: Settlement — when a good isolated spot is reachable
            if _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    best_v = max(vertices, key=lambda v: self._score_settlement(v))
                    success, msg = player.try_build_settlement(best_v)
                    if success:
                        return True

            # Priority 3: City — only if at 8+ VP and need the push to win
            if player.victory_points >= 8 and _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    success, msg = player.try_build_city(random.choice(vertices))
                    if success:
                        return True

            # Priority 4: More roads if we can still afford them
            if _can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    best_e = max(edges, key=lambda e: self._score_edge(e, player, opponent, game))
                    success, msg = player.try_build_road(best_e)
                    if success:
                        return True

            # Never buys dev cards — waste of ore/wheat/sheep

            # Trade: surplus ore/wheat/sheep for wood/brick (conservative)
            _try_conservative_trade(
                game, player,
                desired_resources=[ResourceType.WOOD, ResourceType.BRICK],
                excess_resources=[ResourceType.ORE, ResourceType.WHEAT, ResourceType.SHEEP]
            )

        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False

    def _handle_initial_placement(self, game, player_id, player, opponent):
        if game.waiting_for_road:
            valid_edges = _get_valid_initial_roads(game)
            if valid_edges:
                last_v = game.last_settlement_vertex

                # Point road toward opponent's likely expansion direction
                def score_road(edge):
                    other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                    base = _pip_count(other)
                    # Bonus for edges near board center (blocks more corridors)
                    blocking = self._blocking_bonus(edge, opponent)
                    return base + blocking

                valid_edges.sort(key=score_road, reverse=True)
                success, msg = game.try_place_initial_road(valid_edges[0], player)
                return success
        else:
            valid_vertices = _get_valid_initial_vertices(game)
            if valid_vertices:
                existing = player.settlements[0].position if len(player.settlements) == 1 else None
                scored = [(v, self._score_placement(v, existing, opponent)) for v in valid_vertices]
                scored.sort(key=lambda x: x[1], reverse=True)
                success, msg = game.try_place_initial_settlement(scored[0][0], player)
                return success
        return False

    def _score_placement(self, vertex, existing_settlement, opponent):
        """Score vertex for road blocker: wood/brick focus + positional blocking."""
        wood_pips = _resource_pips(vertex, 'forest')
        brick_pips = _resource_pips(vertex, 'hill')
        other_pips = _pip_count(vertex) - wood_pips - brick_pips

        # Wood and brick are king for roads
        score = (wood_pips * 3) + (brick_pips * 3) + other_pips

        # 2nd settlement: try to cut off opponent expansion
        if existing_settlement and opponent and opponent.settlements:
            opp_vertex = opponent.settlements[0].position
            # Bonus for being on the opposite side of the board from our 1st settlement
            # but near the opponent (to block them)
            for adj in vertex.adjacent_vertices:
                for opp_adj in opp_vertex.adjacent_vertices:
                    if adj == opp_adj:
                        score += 4  # Near opponent = more blocking potential

        if existing_settlement:
            # Gap-fill wood/brick
            existing_wood = _resource_pips(existing_settlement, 'forest')
            existing_brick = _resource_pips(existing_settlement, 'hill')
            if existing_wood == 0 and wood_pips > 0:
                score += 5
            if existing_brick == 0 and brick_pips > 0:
                score += 5

        return score

    def _score_edge(self, edge, player, opponent, game):
        """Score edge for road building: expansion + blocking opponent."""
        # Base expansion score
        base = score_edge(edge, player, game)

        # Blocking bonus: +2 per opponent structure/road adjacent to this edge
        blocking = self._blocking_bonus(edge, opponent)

        return base + blocking

    def _blocking_bonus(self, edge, opponent):
        """Add blocking bonus for edges near opponent's structures/roads."""
        bonus = 0
        if not opponent:
            return 0

        for vertex in [edge.vertex1, edge.vertex2]:
            # Check if opponent has structures nearby
            for adj_v in vertex.adjacent_vertices:
                if adj_v.structure and adj_v.structure.player == opponent:
                    bonus += 3  # Adjacent to opponent settlement/city

            # Check if opponent has roads nearby
            for adj_e in vertex.connected_edges:
                if adj_e.structure and adj_e.structure.player == opponent:
                    bonus += 2  # Adjacent to opponent road

        return bonus

    def _score_settlement(self, vertex):
        """Score settlement spots — prefer high pip count since roads already extend."""
        return _pip_count(vertex)


# ============================================================
# SPECIALIST 3: DEV CARD SPAMMER
# ============================================================

class DevCardSpammer:
    """Win through Largest Army (+2 VP) + VP dev cards.

    Strategy: Buy dev cards every single turn. Cities to double ore/wheat/sheep
    output. Play knights immediately. Minimal road/settlement building.
    """

    def play_turn(self, game, player_id):
        player = game.players[player_id]
        opponent = _get_opponent(game, player_id)

        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player)

        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        if game.can_trade_or_build():
            resources = player.resources

            # Play knight IMMEDIATELY when drawn — always, for Largest Army
            if opponent and _try_play_knight(game, player, opponent):
                return True

            # Priority 1: Dev card — buy one EVERY turn if possible
            if _can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        return True

            # Priority 2: City — doubles ore/wheat output for more dev cards
            if _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    # Upgrade settlement with best ore/wheat/sheep production
                    best_v = max(vertices, key=lambda v:
                                 _resource_pips(v, 'mountain') + _resource_pips(v, 'field') +
                                 _resource_pips(v, 'pasture'))
                    success, msg = player.try_build_city(best_v)
                    if success:
                        return True

            # Priority 3: Settlement — ONLY to reach new ore/wheat/sheep hex
            if _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    # Only build if the vertex has ore/wheat/sheep
                    ows_vertices = [v for v in vertices if
                                    _resource_pips(v, 'mountain') + _resource_pips(v, 'field') +
                                    _resource_pips(v, 'pasture') >= 3]
                    if ows_vertices:
                        best_v = max(ows_vertices, key=lambda v: self._score_vertex(v))
                        success, msg = player.try_build_settlement(best_v)
                        if success:
                            return True

            # Priority 4: Road — ONLY if gating a settlement that unlocks ore/wheat/sheep
            if _can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    best_e = max(edges, key=lambda e: self._score_edge_for_ows(e))
                    if self._score_edge_for_ows(best_e) > 0:
                        success, msg = player.try_build_road(best_e)
                        if success:
                            return True

            # Trade: wood/brick for ore/wheat/sheep (conservative)
            _try_conservative_trade(
                game, player,
                desired_resources=[ResourceType.ORE, ResourceType.WHEAT, ResourceType.SHEEP],
                excess_resources=[ResourceType.WOOD, ResourceType.BRICK]
            )

        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False

    def _handle_initial_placement(self, game, player_id, player):
        if game.waiting_for_road:
            valid_edges = _get_valid_initial_roads(game)
            if valid_edges:
                last_v = game.last_settlement_vertex

                def score_road(edge):
                    other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                    return self._score_vertex(other)

                valid_edges.sort(key=score_road, reverse=True)
                success, msg = game.try_place_initial_road(valid_edges[0], player)
                return success
        else:
            valid_vertices = _get_valid_initial_vertices(game)
            if valid_vertices:
                existing = player.settlements[0].position if len(player.settlements) == 1 else None
                scored = [(v, self._score_vertex(v, existing)) for v in valid_vertices]
                scored.sort(key=lambda x: x[1], reverse=True)
                success, msg = game.try_place_initial_settlement(scored[0][0], player)
                return success
        return False

    def _score_vertex(self, vertex, existing_settlement=None):
        """Score vertex for dev card strategy: ore + wheat + sheep equally weighted."""
        ore_pips = _resource_pips(vertex, 'mountain')
        wheat_pips = _resource_pips(vertex, 'field')
        sheep_pips = _resource_pips(vertex, 'pasture')
        other_pips = _pip_count(vertex) - ore_pips - wheat_pips - sheep_pips

        # All three dev card resources weighted equally at 3x
        score = (ore_pips * 3) + (wheat_pips * 3) + (sheep_pips * 3) + other_pips

        if existing_settlement:
            existing_ore = _resource_pips(existing_settlement, 'mountain')
            existing_wheat = _resource_pips(existing_settlement, 'field')
            existing_sheep = _resource_pips(existing_settlement, 'pasture')
            # Gap-fill bonuses
            if existing_ore == 0 and ore_pips > 0:
                score += 5
            if existing_wheat == 0 and wheat_pips > 0:
                score += 5
            if existing_sheep == 0 and sheep_pips > 0:
                score += 5

        return score

    def _score_edge_for_ows(self, edge):
        """Score edge by whether it leads to ore/wheat/sheep settlement spots."""
        score = 0
        for vertex in [edge.vertex1, edge.vertex2]:
            if vertex.structure is None:
                too_close = any(adj.structure is not None for adj in vertex.adjacent_vertices)
                if not too_close:
                    score += (_resource_pips(vertex, 'mountain') +
                              _resource_pips(vertex, 'field') +
                              _resource_pips(vertex, 'pasture'))
        return score


# ============================================================
# SPECIALIST 4: BALANCED AGGRESSOR
# ============================================================

class BalancedAggressor:
    """Strong AI clone with an aggressive disruption layer.

    Same build priorities as the 'strong' rule-based AI, but:
    - Robber ALWAYS goes to opponent's highest-pip hex
    - Knight cards played immediately for disruption + Largest Army
    - Edge scoring adds +2 blocking bonus near opponent structures
    - Actively targets opponent's production, not just self-optimization
    """

    def play_turn(self, game, player_id):
        player = game.players[player_id]
        opponent = _get_opponent(game, player_id)

        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player)

        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        if game.can_trade_or_build():
            resources = player.resources

            # DISRUPTION: Play knight immediately — robber on opponent's best hex
            if opponent and _try_play_knight(game, player, opponent):
                return True

            # Priority 1: City (same as strong AI)
            if _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    success, msg = player.try_build_city(random.choice(vertices))
                    if success:
                        return True

            # Priority 2: Settlement (same as strong AI, uses full score_vertex)
            if _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    existing = (player.settlements[0].position
                                if len(player.settlements) == 1 else None)
                    scored = [(v, score_vertex(v, game_board=game.game_board,
                                              consider_diversity=True,
                                              consider_expansion=True,
                                              existing_settlement=existing))
                              for v in vertices]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    success, msg = player.try_build_settlement(scored[0][0])
                    if success:
                        return True

            # Priority 3: Road (with blocking bonus)
            if _can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    best_e = max(edges, key=lambda e:
                                 self._score_edge_aggressive(e, player, opponent, game))
                    success, msg = player.try_build_road(best_e)
                    if success:
                        return True

            # Priority 4: Dev card
            if _can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        return True

            # Trade (same as strong AI: generic beneficial trading)
            self._try_trade(game, player)

        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False

    def _handle_initial_placement(self, game, player_id, player):
        """Same as strong AI — full score_vertex with diversity + expansion."""
        if game.waiting_for_road:
            valid_edges = _get_valid_initial_roads(game)
            if valid_edges:
                last_v = game.last_settlement_vertex

                def score_road(edge):
                    other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                    return score_vertex(other, consider_diversity=True,
                                        consider_expansion=True,
                                        game_board=game.game_board)

                valid_edges.sort(key=score_road, reverse=True)
                success, msg = game.try_place_initial_road(valid_edges[0], player)
                return success
        else:
            valid_vertices = _get_valid_initial_vertices(game)
            if valid_vertices:
                existing = player.settlements[0].position if len(player.settlements) == 1 else None
                scored = [(v, score_vertex(v, game_board=game.game_board,
                                           consider_diversity=True,
                                           consider_expansion=True,
                                           existing_settlement=existing))
                          for v in valid_vertices]
                scored.sort(key=lambda x: x[1], reverse=True)
                success, msg = game.try_place_initial_settlement(scored[0][0], player)
                return success
        return False

    def _score_edge_aggressive(self, edge, player, opponent, game):
        """Score edge: base expansion + blocking bonus near opponent."""
        base = score_edge(edge, player, game)

        # DISRUPTION: +2 bonus for each opponent structure/road adjacent
        blocking = 0
        if opponent:
            for vertex in [edge.vertex1, edge.vertex2]:
                for adj_v in vertex.adjacent_vertices:
                    if adj_v.structure and adj_v.structure.player == opponent:
                        blocking += 2
                for adj_e in vertex.connected_edges:
                    if adj_e.structure and adj_e.structure.player == opponent:
                        blocking += 2

        return base + blocking

    def _try_trade(self, game, player):
        """Generic beneficial trading — same logic as strong rule-based AI."""
        resources = player.resources

        # Find excess resources (4+)
        excess = [rt for rt, amt in resources.items() if amt >= 4]
        if not excess:
            return False

        # Determine what we need most
        needed = self._get_most_needed(resources)
        if not needed:
            return False

        give_res = random.choice(excess)
        ratio = game.game_board.get_best_trade_ratio(player, give_res)
        if resources[give_res] >= ratio:
            success, msg = game.execute_bank_trade(player, give_res, needed, ratio)
            return success
        return False

    def _get_most_needed(self, resources):
        """Determine most needed resource (same priority as strong AI)."""
        # City first
        if resources[ResourceType.WHEAT] >= 2 and resources[ResourceType.ORE] < 3:
            return ResourceType.ORE
        if resources[ResourceType.ORE] >= 3 and resources[ResourceType.WHEAT] < 2:
            return ResourceType.WHEAT

        # Settlement
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

        # Default: wood/brick for roads
        if resources[ResourceType.WOOD] < resources[ResourceType.BRICK]:
            return ResourceType.WOOD
        return ResourceType.BRICK


# ============================================================
# SPECIALIST 5: PORT SPECIALIST
# ============================================================

class PortSpecialist:
    """Win through port-based trading engine. Sacrifices pip count for 2:1/3:1 port access.

    Strategy: First settlement on a 2:1 port (ore or wheat preferred). Second settlement
    fills missing resources. Mid-game: accumulate port resource, trade 2:1 for whatever
    is needed. Effectively halves the cost of everything.

    Fallback: If no good port vertex is available (opponent took it or bad board),
    plays like strong AI.
    """

    # Map PortType to the ResourceType it specializes in
    PORT_RESOURCE_MAP = {
        PortType.ORE: ResourceType.ORE,
        PortType.WHEAT: ResourceType.WHEAT,
        PortType.WOOD: ResourceType.WOOD,
        PortType.BRICK: ResourceType.BRICK,
        PortType.SHEEP: ResourceType.SHEEP,
    }

    # Map ResourceType to the tile resource string
    RESOURCE_TILE_MAP = {
        ResourceType.ORE: 'mountain',
        ResourceType.WHEAT: 'field',
        ResourceType.WOOD: 'forest',
        ResourceType.BRICK: 'hill',
        ResourceType.SHEEP: 'pasture',
    }

    # Preferred port types in priority order (ore/wheat best for city engine)
    PORT_PRIORITY = [PortType.ORE, PortType.WHEAT, PortType.SHEEP,
                     PortType.BRICK, PortType.WOOD, PortType.GENERIC]

    def __init__(self):
        self.port_resource = None  # The resource our port specializes in (set during placement)
        self.has_port = False      # Whether we successfully landed on a port
        self._fallback_mode = False

    def play_turn(self, game, player_id):
        player = game.players[player_id]
        opponent = _get_opponent(game, player_id)

        if game.is_initial_placement_phase():
            return self._handle_initial_placement(game, player_id, player)

        # If we failed to get a port, fall back to strong-style play
        if self._fallback_mode:
            return self._play_fallback_turn(game, player_id, player, opponent)

        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        if game.can_trade_or_build():
            resources = player.resources

            # Play knight if we have one
            if opponent and _try_play_knight(game, player, opponent):
                return True

            # TRADING ENGINE: This is the core of the port strategy.
            # Trade port resource 2:1 for whatever we need before building.
            if self.port_resource and resources[self.port_resource] >= 2:
                traded = self._trade_via_port(game, player)
                if traded:
                    return True

            # Priority 1: City — port engine makes ore/wheat effectively cheaper
            if _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    # Prefer upgrading settlement that produces our port resource
                    best_v = max(vertices, key=lambda v: self._score_city_upgrade(v))
                    success, msg = player.try_build_city(best_v)
                    if success:
                        return True

            # Priority 2: Settlement — prefer spots that produce our port resource
            if _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    best_v = max(vertices, key=lambda v: self._score_vertex(v))
                    success, msg = player.try_build_settlement(best_v)
                    if success:
                        return True

            # Priority 3: Dev card
            if _can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        return True

            # Priority 4: Road — to reach new settlement spots producing port resource
            if _can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    best_e = max(edges, key=lambda e: self._score_edge(e, player, game))
                    if self._score_edge(best_e, player, game) > 0:
                        success, msg = player.try_build_road(best_e)
                        if success:
                            return True

            # Conservative trade for non-port resources
            if self.port_resource:
                excess = [rt for rt in ResourceType if rt != self.port_resource]
                _try_conservative_trade(
                    game, player,
                    desired_resources=[self.port_resource],
                    excess_resources=excess
                )

        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False

    def _handle_initial_placement(self, game, player_id, player):
        if game.waiting_for_road:
            valid_edges = _get_valid_initial_roads(game)
            if valid_edges:
                last_v = game.last_settlement_vertex

                def score_road(edge):
                    other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                    return self._score_vertex(other)

                valid_edges.sort(key=score_road, reverse=True)
                success, msg = game.try_place_initial_road(valid_edges[0], player)
                return success
        else:
            valid_vertices = _get_valid_initial_vertices(game)
            if not valid_vertices:
                return False

            is_first = len(player.settlements) == 0

            if is_first:
                # First settlement: find the best port vertex
                best_vertex, port_type = self._find_best_port_vertex(valid_vertices, game)

                if best_vertex and port_type:
                    self.has_port = True
                    if port_type == PortType.GENERIC:
                        self.port_resource = None  # 3:1, no specific resource
                    else:
                        self.port_resource = self.PORT_RESOURCE_MAP.get(port_type)

                    success, msg = game.try_place_initial_settlement(best_vertex, player)
                    return success
                else:
                    # No good port available — enter fallback mode
                    self._fallback_mode = True
                    # Place like strong AI
                    scored = [(v, score_vertex(v, game_board=game.game_board,
                                              consider_diversity=True,
                                              consider_expansion=True))
                              for v in valid_vertices]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    success, msg = game.try_place_initial_settlement(scored[0][0], player)
                    return success
            else:
                # Second settlement: fill missing resources, especially the port resource
                existing = player.settlements[0].position if player.settlements else None
                scored = [(v, self._score_second_settlement(v, existing))
                          for v in valid_vertices]
                scored.sort(key=lambda x: x[1], reverse=True)
                success, msg = game.try_place_initial_settlement(scored[0][0], player)
                return success

        return False

    def _find_best_port_vertex(self, valid_vertices, game):
        """Find the best port vertex to settle on.

        Scores each valid vertex that's on a port by:
        - Port type priority (ore > wheat > sheep > brick > wood > generic)
        - Pip count of the port resource at this vertex (need production of what we'll trade)
        - Total pip count (don't settle on a dead vertex just for the port)

        Returns (vertex, port_type) or (None, None) if no viable port vertex.
        """
        candidates = []

        for vertex in valid_vertices:
            port = _vertex_has_port(vertex, game.game_board)
            if not port:
                continue

            port_type = port.port_type
            total_pips = _pip_count(vertex)

            # Minimum viability: vertex must have at least 4 pips total
            if total_pips < 4:
                continue

            # Score by port priority (lower index = better)
            try:
                priority_score = len(self.PORT_PRIORITY) - self.PORT_PRIORITY.index(port_type)
            except ValueError:
                priority_score = 0

            # Bonus if vertex produces the port's resource (so we can actually trade)
            port_resource_pips = 0
            if port_type in self.PORT_RESOURCE_MAP:
                tile_name = self.RESOURCE_TILE_MAP.get(self.PORT_RESOURCE_MAP[port_type], '')
                port_resource_pips = _resource_pips(vertex, tile_name)

            # Combined score
            score = (priority_score * 5) + (port_resource_pips * 3) + total_pips

            candidates.append((vertex, port_type, score))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]

    def _score_second_settlement(self, vertex, existing_settlement):
        """Score 2nd settlement: fill resource gaps, bonus for port resource production."""
        total = _pip_count(vertex)
        score = total

        if existing_settlement:
            # Gap-fill bonus for each new resource type
            existing_resources = set()
            for tile in existing_settlement.adjacent_tiles:
                rt = tile.get_resource_type() if tile.resource and tile.resource != "desert" else None
                if rt:
                    existing_resources.add(rt)

            for tile in vertex.adjacent_tiles:
                rt = tile.get_resource_type() if tile.resource and tile.resource != "desert" else None
                if rt and rt not in existing_resources:
                    score += 4

        # Bonus for producing our port resource (so we can keep trading)
        if self.port_resource:
            tile_name = self.RESOURCE_TILE_MAP.get(self.port_resource, '')
            score += _resource_pips(vertex, tile_name) * 2

        return score

    def _trade_via_port(self, game, player):
        """Use our 2:1 (or 3:1) port to trade for what we need most.

        Trades immediately whenever port_resource >= ratio AND there's something
        useful to receive. Don't wait or accumulate — trade as soon as possible
        to avoid missing build windows.
        """
        resources = player.resources

        if self.port_resource:
            # 2:1 specialized port — trade immediately when we have enough
            ratio = game.game_board.get_best_trade_ratio(player, self.port_resource)
            if resources[self.port_resource] < ratio:
                return False
            # No buffer requirement — the whole point of the port is to trade aggressively
        else:
            # 3:1 generic port — find any resource we have 3+ of
            return self._trade_via_generic_port(game, player)

        # Determine what we need
        needed = self._get_most_needed(resources)
        if not needed or needed == self.port_resource:
            return False

        success, msg = game.execute_bank_trade(player, self.port_resource, needed)
        return success

    def _trade_via_generic_port(self, game, player):
        """Trade using a 3:1 generic port."""
        resources = player.resources

        needed = self._get_most_needed(resources)
        if not needed:
            return False

        # Find a resource we have 4+ of (3 for trade + 1 buffer)
        for rt in ResourceType:
            if rt == needed:
                continue
            ratio = game.game_board.get_best_trade_ratio(player, rt)
            if resources[rt] >= ratio + 1:
                success, msg = game.execute_bank_trade(player, rt, needed, ratio)
                if success:
                    return True
        return False

    def _get_most_needed(self, resources):
        """Determine most needed resource based on build priorities."""
        # City first
        if resources[ResourceType.WHEAT] >= 2 and resources[ResourceType.ORE] < 3:
            return ResourceType.ORE
        if resources[ResourceType.ORE] >= 3 and resources[ResourceType.WHEAT] < 2:
            return ResourceType.WHEAT

        # Settlement
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

        # Dev card
        if resources[ResourceType.ORE] < 1:
            return ResourceType.ORE
        if resources[ResourceType.WHEAT] < 1:
            return ResourceType.WHEAT
        if resources[ResourceType.SHEEP] < 1:
            return ResourceType.SHEEP

        return None

    def _score_vertex(self, vertex):
        """Score vertex: prefer spots that produce our port resource."""
        total = _pip_count(vertex)
        if self.port_resource:
            tile_name = self.RESOURCE_TILE_MAP.get(self.port_resource, '')
            total += _resource_pips(vertex, tile_name) * 2
        return total

    def _score_city_upgrade(self, vertex):
        """Score city upgrade: prefer settlements producing our port resource."""
        total = _pip_count(vertex)
        if self.port_resource:
            tile_name = self.RESOURCE_TILE_MAP.get(self.port_resource, '')
            total += _resource_pips(vertex, tile_name) * 2
        return total

    def _score_edge(self, edge, player, game):
        """Score edge for road building."""
        base = score_edge(edge, player, game)
        # Bonus for edges leading to vertices that produce port resource
        if self.port_resource:
            tile_name = self.RESOURCE_TILE_MAP.get(self.port_resource, '')
            for vertex in [edge.vertex1, edge.vertex2]:
                if vertex.structure is None:
                    too_close = any(adj.structure is not None for adj in vertex.adjacent_vertices)
                    if not too_close:
                        base += _resource_pips(vertex, tile_name)
        return base

    def _play_fallback_turn(self, game, player_id, player, opponent):
        """Fallback: play like strong AI when no port was available."""
        if game.can_roll_dice():
            result = game.roll_dice()
            return result is not None

        if game.can_trade_or_build():
            resources = player.resources

            if opponent and _try_play_knight(game, player, opponent):
                return True

            if _can_afford_city(resources):
                vertices = game.get_buildable_vertices_for_cities()
                if vertices:
                    success, msg = player.try_build_city(random.choice(vertices))
                    if success:
                        return True

            if _can_afford_settlement(resources):
                vertices = game.get_buildable_vertices_for_settlements()
                if vertices:
                    existing = (player.settlements[0].position
                                if len(player.settlements) == 1 else None)
                    scored = [(v, score_vertex(v, game_board=game.game_board,
                                              consider_diversity=True,
                                              consider_expansion=True,
                                              existing_settlement=existing))
                              for v in vertices]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    success, msg = player.try_build_settlement(scored[0][0])
                    if success:
                        return True

            if _can_afford_road(resources):
                edges = game.get_buildable_edges()
                if edges:
                    best_e = max(edges, key=lambda e: score_edge(e, player, game))
                    success, msg = player.try_build_road(best_e)
                    if success:
                        return True

            if _can_afford_dev_card(resources):
                if not game.dev_deck.is_empty():
                    success, msg = player.try_buy_development_card(game.dev_deck)
                    if success:
                        return True

            _try_conservative_trade(
                game, player,
                desired_resources=[ResourceType.ORE, ResourceType.WHEAT],
                excess_resources=[ResourceType.WOOD, ResourceType.BRICK, ResourceType.SHEEP]
            )

        if game.can_end_turn():
            success, msg = game.end_turn()
            return success

        return False


# ============================================================
# STRATEGY REGISTRY & ENTRY POINT
# ============================================================

# Map strategy names to specialist classes
STRATEGY_CLASSES = {
    'city_rusher': CityRusher,
    'road_blocker': RoadBlocker,
    'dev_card_spammer': DevCardSpammer,
    'balanced_aggressor': BalancedAggressor,
    'port_specialist': PortSpecialist,
}

# Cache specialist instances per strategy to reuse across turns within a game.
# PortSpecialist has state (which port it landed on) so it needs per-game instances.
# For stateless specialists, a single instance is fine.
# WARNING: Stateless specialists (CityRusher, RoadBlocker, DevCardSpammer, BalancedAggressor)
# must NOT have mutable instance variables that persist between play_turn() calls.
# If you add instance state to any of these, move them to per-game caching like PortSpecialist.
_stateless_cache = {}


def play_specialist_turn(game, player_id, strategy='city_rusher'):
    """Play one turn for a specialist AI.

    Drop-in replacement for play_rule_based_turn() — same signature pattern.
    Called from play_opponent_turn() in curriculum_trainer_v3_stable.py.

    Args:
        game: The GameSystem object
        player_id: Which player this AI controls (0 or 1 in 1v1)
        strategy: One of 'city_rusher', 'road_blocker', 'dev_card_spammer',
                  'balanced_aggressor', 'port_specialist'

    Returns:
        bool: True if turn advanced, False if stuck
    """
    if strategy not in STRATEGY_CLASSES:
        raise ValueError(f"Unknown specialist strategy: {strategy}. "
                         f"Available: {list(STRATEGY_CLASSES.keys())}")

    cls = STRATEGY_CLASSES[strategy]

    # PortSpecialist has per-game state — needs fresh instance per game.
    # We store it on the game object to persist across turns within one game.
    if strategy == 'port_specialist':
        cache_key = f'_specialist_{strategy}_{player_id}'
        if not hasattr(game, cache_key):
            setattr(game, cache_key, cls())
        specialist = getattr(game, cache_key)
    else:
        # Stateless specialists can be reused
        if strategy not in _stateless_cache:
            _stateless_cache[strategy] = cls()
        specialist = _stateless_cache[strategy]

    return specialist.play_turn(game, player_id)
