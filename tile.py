import math
# Pygame is only needed for drawing - make it optional for server
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from game_system import ResourceType

DIRECTIONS = [
    (+1, 0), (0, +1), (-1, +1),
    (-1, 0), (0, -1), (+1, -1)
]


class Tile:
    def __init__(self, q, r, size, resource=None, number=None):
        self.q = q
        self.r = r
        self.size = size  # This is the radius from center to vertex
        self.resource = resource
        self.number = number
        self.x, self.y = self.axial_to_pixel()
        self.neighbors = []
        self.has_robber = False  # Track if robber is on this tile

    def get_resource_type(self):
        """Convert string resource to ResourceType enum"""
        resource_map = {
            "forest": ResourceType.WOOD,
            "hill": ResourceType.BRICK,
            "field": ResourceType.WHEAT,
            "mountain": ResourceType.ORE,
            "pasture": ResourceType.SHEEP
        }
        return resource_map.get(self.resource, None)

    def produces_resources(self):
        """Check if this tile produces resources (not desert and no robber)"""
        return self.resource != "desert" and not self.has_robber

    def axial_to_pixel(self):
        # For flat-top hexagons to be perfectly adjacent:
        # - Horizontal distance between centers = size * sqrt(3)
        # - Vertical distance between rows = size * 1.5
        x = self.size * math.sqrt(3) * (self.q + 0.5 * self.r)
        y = self.size * 1.5 * self.r
        return x, y

    def get_corners(self, offset=(0, 0)):
        corners = []
        # Flat-top hexagon: start from top vertex and go clockwise
        for i in range(6):
            angle = math.pi / 3 * i - math.pi / 2  # Start from top, -90 degrees
            cx = self.x + self.size * math.cos(angle) + offset[0]
            cy = self.y + self.size * math.sin(angle) + offset[1]
            corners.append((cx, cy))
        return corners

    def draw(self, surface, fill_color, offset=(0, 0), show_coords=False):
        if not HAS_PYGAME:
            raise ImportError("pygame is required for drawing - install with: pip3 install pygame")

        # Draw filled hexagon
        corners = self.get_corners(offset)
        pygame.draw.polygon(surface, fill_color, corners)

        # Draw border - thin black line
        pygame.draw.polygon(surface, (0, 0, 0), corners, 1)

        # Draw robber if present
        if self.has_robber:
            center_x = self.x + offset[0]
            center_y = self.y + offset[1] - 10  # Offset robber up a bit
            # Draw robber as a dark circle
            pygame.draw.circle(surface, (50, 50, 50), (int(center_x), int(center_y)), 12)
            pygame.draw.circle(surface, (0, 0, 0), (int(center_x), int(center_y)), 12, 2)

        # Draw number token (only if not desert and has number)
        if self.number is not None and self.resource != "desert":
            center_x = self.x + offset[0]
            center_y = self.y + offset[1]

            # Adjust position if robber is present
            if self.has_robber:
                center_y += 15

            # Draw white circle background for number
            pygame.draw.circle(surface, (255, 255, 255), (int(center_x), int(center_y)), 15)
            pygame.draw.circle(surface, (0, 0, 0), (int(center_x), int(center_y)), 15, 2)

            # Draw number text - highlight 6 and 8 in red
            color = (255, 0, 0) if self.number in [6, 8] else (0, 0, 0)
            font = pygame.font.SysFont(None, 24, bold=(self.number in [6, 8]))
            text = font.render(str(self.number), True, color)
            text_rect = text.get_rect(center=(center_x, center_y))
            surface.blit(text, text_rect)

        # Draw coordinates if requested
        if show_coords:
            font = pygame.font.SysFont(None, 16)
            coord_text = f"({self.q},{self.r})"
            text_surface = font.render(coord_text, True, (255, 255, 255))
            surface.blit(text_surface, (self.x + offset[0] - 20, self.y + offset[1] + 25))

    def find_neighbors(self, all_tiles):
        self.neighbors = []
        tile_dict = {(t.q, t.r): t for t in all_tiles}
        for dq, dr in DIRECTIONS:
            neighbor_coord = (self.q + dq, self.r + dr)
            if neighbor_coord in tile_dict:
                self.neighbors.append(tile_dict[neighbor_coord])

    def get_distance_to(self, other_tile):
        """Calculate distance between two tiles"""
        dx = self.x - other_tile.x
        dy = self.y - other_tile.y
        return math.sqrt(dx * dx + dy * dy)