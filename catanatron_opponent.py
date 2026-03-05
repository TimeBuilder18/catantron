"""
catanatron_opponent.py

Implements WeightedRandomPlayer behavior using our game engine.
WeightedRandomPlayer from catanatron weights actions as follows:
  - Build settlement:  weight 5
  - Build city:        weight 4
  - Build road:        weight 3
  - Buy dev card:      weight 2
  - Use dev card:      weight 2
  - Bank trade:        weight 1
  - End turn:          weight 0.1  (only if nothing else available)

This creates a player that strongly prefers building over passing,
which is harder than truly_random (15% build) but easier than our
rule-based opponent (which uses board evaluation).
"""

import random
from game_system import ResourceType, DevelopmentCardType


# Action weights matching catanatron's WeightedRandomPlayer
ACTION_WEIGHTS = {
    'place_settlement': 5,
    'build_settlement': 5,
    'place_city': 4,
    'build_city': 4,
    'place_road': 3,
    'build_road': 3,
    'buy_dev_card': 2,
    'play_knight': 2,
    'play_road_building': 2,
    'play_year_of_plenty': 2,
    'play_monopoly': 2,
    'trade_with_bank': 1,
    'end_turn': 0,
    'roll_dice': 0,
    'wait': 0,
}


def _pip_count(number):
    """Number of dots on dice that produce this number (probability proxy)."""
    if number is None or number == 0:
        return 0
    return 6 - abs(7 - number)


def play_weighted_random_turn(game, player_id):
    """
    Play one turn for a player using WeightedRandomPlayer weighting logic.

    Args:
        game: Our GameSystem instance
        player_id: Index of the player to play

    Returns:
        True when turn is complete
    """
    player = game.players[player_id]
    max_actions = 50  # Safety limit

    for _ in range(max_actions):
        current = game.get_current_player()
        if game.players.index(current) != player_id:
            break  # No longer our turn

        legal_actions = _get_legal_actions(game, player)

        if not legal_actions:
            # Force end turn if stuck
            if game.can_end_turn():
                game.end_turn()
            break

        # Weight actions
        weights = [ACTION_WEIGHTS.get(action_name, 1) for action_name, _ in legal_actions]

        # If all weights are 0 (only end_turn/roll_dice available), give them weight 1
        if sum(weights) == 0:
            weights = [1] * len(weights)

        # Weighted sample
        action_name, action_fn = random.choices(legal_actions, weights=weights, k=1)[0]

        # Execute the action
        try:
            action_fn()
        except Exception:
            # If action fails, try end turn
            if game.can_end_turn():
                game.end_turn()
            break

        # Check if turn ended naturally
        current_after = game.get_current_player()
        if game.players.index(current_after) != player_id:
            break

    # Force end turn if the loop exhausted without ending the turn
    if game.players.index(game.get_current_player()) == player_id and game.can_end_turn():
        game.end_turn()

    return True


def _get_legal_actions(game, player):
    """
    Get list of (action_name, callable) pairs for legal actions.
    Returns actions that can actually be executed right now.
    """
    actions = []
    res = player.resources

    # Initial placement phase
    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            buildable_edges = _get_edges_connected_to_last_settlement(game)
            if buildable_edges:
                edge = random.choice(buildable_edges)
                actions.append(('place_road', lambda e=edge: game.try_place_initial_road(e, player)))
        else:
            valid_vertices = [v for v in game.game_board.vertices
                              if v.structure is None and
                              not any(adj.structure for adj in v.adjacent_vertices)]
            if valid_vertices:
                # Pick by pip count like other opponents do
                valid_vertices.sort(key=lambda v: sum(
                    _pip_count(t.number) for t in v.adjacent_tiles
                    if hasattr(t, 'number') and t.number), reverse=True)
                vertex = valid_vertices[0]
                actions.append(('place_settlement', lambda v=vertex: game.try_place_initial_settlement(v, player)))
        return actions

    # Normal play phase - must roll first
    if not game.dice_rolled:
        actions.append(('roll_dice', lambda: game.roll_dice()))
        return actions

    # Build actions (check resources)
    wood = res.get(ResourceType.WOOD, 0)
    brick = res.get(ResourceType.BRICK, 0)
    wheat = res.get(ResourceType.WHEAT, 0)
    sheep = res.get(ResourceType.SHEEP, 0)
    ore = res.get(ResourceType.ORE, 0)

    # Settlement: wood + brick + wheat + sheep
    if wood >= 1 and brick >= 1 and wheat >= 1 and sheep >= 1:
        buildable = game.get_buildable_vertices_for_settlements()
        if buildable:
            vertex = random.choice(buildable)
            actions.append(('build_settlement', lambda v=vertex: player.try_build_settlement(v)))

    # City: 2 wheat + 3 ore
    if wheat >= 2 and ore >= 3:
        city_candidates = list(player.settlements)
        if city_candidates:
            vertex = random.choice(city_candidates)
            actions.append(('build_city', lambda v=vertex: player.try_build_city(v)))

    # Road: wood + brick
    if wood >= 1 and brick >= 1:
        buildable_edges = game.get_buildable_edges()
        if buildable_edges:
            edge = random.choice(buildable_edges)
            actions.append(('build_road', lambda e=edge: player.try_build_road(e)))

    # Dev card: wheat + sheep + ore
    if wheat >= 1 and sheep >= 1 and ore >= 1 and not game.dev_deck.is_empty():
        actions.append(('buy_dev_card', lambda: game.try_buy_development_card(player)))

    # End turn (always available after rolling, low weight)
    if game.can_end_turn():
        actions.append(('end_turn', lambda: game.end_turn()))

    return actions


def _get_edges_connected_to_last_settlement(game):
    """Get edges connected to the last placed settlement for initial road placement."""
    last_settlement = game.last_settlement_vertex
    if last_settlement is None:
        return []
    all_edges = game.game_board.edges
    return [e for e in all_edges
            if e.structure is None and
            (e.vertex1 == last_settlement or e.vertex2 == last_settlement)]
