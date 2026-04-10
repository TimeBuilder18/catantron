"""
Shared reward utilities used by both PBRS and simplified reward wrappers.

The main thing here is the PositionalRewardMixin — it scores how good
a player's board position is by looking at settlement pip counts,
resource diversity, port access, and road connectivity to future
building spots. Both wrappers mix this in so they don't duplicate
the scoring logic.
"""

from collections import deque
from game.game_system import PortType


class PositionalRewardMixin:
    # ── one-time initialisation ──────────────────────────────────────────────

    def _shaping_reset(self):
        """Call from reset() to clear per-episode shaping state."""
        self._produced_resources = set()   # resource types currently in production
        self._port_access = set()          # ports player currently has access to

    # ── shared pip table ────────────────────────────────────────────────────

    _PIP_MAP = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    def _vertex_pip_and_resources(self, vertex):
        """Return (pip_score, set_of_resource_strings) for a board vertex."""
        pips = 0
        resources = set()
        for tile in vertex.adjacent_tiles:
            if tile.number:
                pips += self._PIP_MAP.get(tile.number, 0)
            if tile.resource and tile.resource != 'desert':
                resources.add(tile.resource)
        return pips, resources

    # ── settlement quality ───────────────────────────────────────────────────

    def _settlement_shaping(self, player):
        """
        One-off bonus when a settlement is placed / built.

        Rewards
        -------
        Pip quality    : up to +4.0   (best ~15-pip vertex → full bonus)
        New resource   : +1.0 each    (first time this resource type enters production)
        Port access    : +5.0 / +2.5  (2:1 / 3:1 — only first time)
        Blocking opp   : +2.0         (opponent road network reaches this vertex)
        """
        if not player.settlements:
            return 0.0

        game = self.env.game_env.game
        new_vertex = player.settlements[-1].position
        pips, res_set = self._vertex_pip_and_resources(new_vertex)

        reward = min(pips / 15.0, 1.0) * 4.0

        # New resource types entering production
        new_types = res_set - self._produced_resources
        reward += len(new_types) * 1.0
        self._produced_resources |= res_set

        # Port access (one-time per port)
        for port in game.game_board.ports:
            if port.vertex1 == new_vertex or port.vertex2 == new_vertex:
                pt_key = (port.port_type, id(port))
                if pt_key not in self._port_access:
                    self._port_access.add(pt_key)
                    reward += 5.0 if port.port_type != PortType.GENERIC else 2.5

        # Blocking: opponent road reaches an adjacent vertex
        for adj in new_vertex.adjacent_vertices:
            for edge in adj.connected_edges:
                if edge.structure and edge.structure.player != player:
                    reward += 2.0
                    break

        return reward

    # ── road expansion quality ───────────────────────────────────────────────

    def _road_expansion_shaping(self, player):
        """
        Quality bonus for a newly built road based on the best settlement
        spot reachable via BFS (depth ≤ 4 edge steps) from its endpoints.

        Score decays with distance so roads *adjacent* to a great spot
        reward more than roads pointing vaguely in that direction.

        Rewards
        -------
        Best reachable pip vertex : up to +4.0  (distance-discounted)
        Toward a port vertex      : +2.0
        New resource type reachable: +1.0
        """
        if not player.roads:
            return 0.0

        game = self.env.game_env.game

        # Vertices blocked by the Catan distance rule
        occupied = set()
        for v in game.game_board.vertices:
            if v.structure is not None:
                occupied.add(id(v))
                for adj in v.adjacent_vertices:
                    occupied.add(id(adj))

        # Port vertex ids for the toward-port bonus
        port_vertices = set()
        for port in game.game_board.ports:
            port_vertices.add(id(port.vertex1))
            port_vertices.add(id(port.vertex2))

        newest_road = player.roads[-1]
        endpoints = [newest_road.position.vertex1, newest_road.position.vertex2]

        MAX_DEPTH = 4
        best_score = 0.0
        best_resources = set()
        toward_port = False

        for start in endpoints:
            queue = deque([(start, 0)])
            visited = {id(start)}
            while queue:
                v, depth = queue.popleft()
                if id(v) not in occupied:
                    pips, res = self._vertex_pip_and_resources(v)
                    # 1.0 at depth 0 → 0.5 at depth MAX_DEPTH
                    discount = 1.0 - (depth / (MAX_DEPTH + 1)) * 0.5
                    score = pips * discount
                    if score > best_score:
                        best_score = score
                        best_resources = res
                if id(v) in port_vertices:
                    toward_port = True
                if depth < MAX_DEPTH:
                    for adj in v.adjacent_vertices:
                        if id(adj) not in visited:
                            visited.add(id(adj))
                            queue.append((adj, depth + 1))

        reward = min(best_score / 15.0, 1.0) * 1.5   # was 3.0 — halved again
        if toward_port:
            reward += 0.3                               # was 0.5
        if best_resources - self._produced_resources:
            reward += 0.2                               # was 0.5

        return min(reward, 2.0)                         # hard cap 2.0 (was 4.0)
