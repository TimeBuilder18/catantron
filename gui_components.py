"""
GUI Components for Catantron
Reusable drawing functions for the polished Catan game interface.
"""

import pygame
import math
from game_system import (ResourceType, PortType, Settlement, City,
                         DevelopmentCardType, GameConstants)

# --- Screen Layout (defaults, updated by set_screen_size) ---
SCREEN_W, SCREEN_H = 1600, 1000
PANEL_X = 1020  # Right panel starts here
PANEL_W = 560   # Panel width
SCALE = 1.0     # Scale factor relative to 1600x1000 reference


def set_screen_size(w, h):
    """Update layout globals to match actual window size."""
    global SCREEN_W, SCREEN_H, PANEL_X, PANEL_W, SCALE
    SCREEN_W = w
    SCREEN_H = h
    SCALE = w / 1600
    PANEL_X = int(w * 1020 / 1600)
    PANEL_W = w - PANEL_X - int(20 * SCALE)

# --- Color Palette ---
BG_COLOR = (25, 20, 18)
PANEL_BG = (35, 30, 28)
PANEL_BORDER = (80, 70, 60)
TEXT_COLOR = (230, 225, 215)
TEXT_DIM = (160, 155, 145)
GOLD = (230, 190, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RESOURCE_DISPLAY_NAMES = {
    "forest": "Wood", "hill": "Brick", "field": "Wheat",
    "mountain": "Ore", "pasture": "Sheep", "desert": "Desert"
}

RESOURCE_TYPE_NAMES = {
    ResourceType.WOOD: "Wood",
    ResourceType.BRICK: "Brick",
    ResourceType.WHEAT: "Wheat",
    ResourceType.ORE: "Ore",
    ResourceType.SHEEP: "Sheep",
}

RESOURCE_TYPE_SHORT = {
    ResourceType.WOOD: "W",
    ResourceType.BRICK: "B",
    ResourceType.WHEAT: "Wh",
    ResourceType.ORE: "O",
    ResourceType.SHEEP: "S",
}

# Rich hex colors
RESOURCE_COLORS = {
    "forest": (34, 120, 34),
    "hill": (180, 80, 40),
    "field": (230, 190, 50),
    "mountain": (120, 130, 150),
    "pasture": (130, 200, 100),
    "desert": (220, 200, 160),
}

RESOURCE_CARD_COLORS = {
    ResourceType.WOOD: (34, 120, 34),
    ResourceType.BRICK: (180, 80, 40),
    ResourceType.WHEAT: (230, 190, 50),
    ResourceType.ORE: (120, 130, 150),
    ResourceType.SHEEP: (130, 200, 100),
}

PORT_TYPE_NAMES = {
    PortType.GENERIC: "3:1",
    PortType.WOOD: "2:1 Wood",
    PortType.BRICK: "2:1 Brick",
    PortType.WHEAT: "2:1 Wheat",
    PortType.SHEEP: "2:1 Sheep",
    PortType.ORE: "2:1 Ore",
}

PORT_COLORS = {
    PortType.GENERIC: (180, 170, 150),
    PortType.WOOD: (34, 120, 34),
    PortType.BRICK: (180, 80, 40),
    PortType.WHEAT: (230, 190, 50),
    PortType.SHEEP: (130, 200, 100),
    PortType.ORE: (120, 130, 150),
}

DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Build costs for button tooltips
BUILD_COSTS = {
    "settlement": {ResourceType.WOOD: 1, ResourceType.BRICK: 1,
                   ResourceType.WHEAT: 1, ResourceType.SHEEP: 1},
    "city": {ResourceType.WHEAT: 2, ResourceType.ORE: 3},
    "road": {ResourceType.WOOD: 1, ResourceType.BRICK: 1},
    "dev_card": {ResourceType.WHEAT: 1, ResourceType.SHEEP: 1, ResourceType.ORE: 1},
}

# --- Font Cache ---
_font_cache = {}

def get_font(size, bold=False, name=None):
    key = (size, bold, name)
    if key not in _font_cache:
        if name:
            _font_cache[key] = pygame.font.SysFont(name, size, bold=bold)
        else:
            for font_name in ["Arial", "Segoe UI", "Helvetica", "DejaVu Sans"]:
                try:
                    _font_cache[key] = pygame.font.SysFont(font_name, size, bold=bold)
                    break
                except:
                    continue
            else:
                _font_cache[key] = pygame.font.SysFont(None, size, bold=bold)
    return _font_cache[key]


# ===========================================================================
# Board Drawing
# ===========================================================================

def draw_hex_tile(screen, tile, offset):
    """Draw a single hex tile with rich visuals."""
    color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
    corners = tile.get_corners(offset)

    # Shadow (offset down-right)
    shadow_offset = 3
    shadow_corners = [(x + shadow_offset, y + shadow_offset) for x, y in corners]
    pygame.draw.polygon(screen, (15, 12, 10), shadow_corners)

    # Fill hex
    pygame.draw.polygon(screen, color, corners)

    # Border
    pygame.draw.polygon(screen, (30, 25, 20), corners, 2)

    # Inner highlight line on top edge for subtle 3D
    if len(corners) >= 2:
        lighter = tuple(min(c + 30, 255) for c in color)
        pygame.draw.line(screen, lighter, corners[0], corners[1], 1)

    cx = tile.x + offset[0]
    cy = tile.y + offset[1]

    # Robber
    if tile.has_robber:
        draw_robber(screen, cx, cy - 12)

    # Number token
    if tile.number is not None and tile.resource != "desert":
        token_y = cy + 12 if tile.has_robber else cy

        # Token circle
        pygame.draw.circle(screen, (250, 245, 230), (int(cx), int(token_y)), 17)
        pygame.draw.circle(screen, (60, 50, 40), (int(cx), int(token_y)), 17, 2)

        # Number
        is_hot = tile.number in (6, 8)
        num_color = (200, 30, 30) if is_hot else (40, 35, 30)
        font = get_font(22, bold=is_hot)
        text = font.render(str(tile.number), True, num_color)
        text_rect = text.get_rect(center=(cx, token_y - 2))
        screen.blit(text, text_rect)

        # Pip dots
        pips = DICE_PIPS.get(tile.number, 0)
        pip_y = token_y + 10
        pip_color = (200, 30, 30) if is_hot else (80, 70, 60)
        total_width = (pips - 1) * 5
        start_x = cx - total_width / 2
        for i in range(pips):
            px = int(start_x + i * 5)
            pygame.draw.circle(screen, pip_color, (px, int(pip_y)), 2)


def draw_robber(screen, x, y):
    """Draw the robber as a dark figure."""
    # Body
    pygame.draw.circle(screen, (40, 35, 30), (int(x), int(y)), 13)
    pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 13, 2)
    # "R" label
    font = get_font(16, bold=True)
    text = font.render("R", True, (200, 50, 50))
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)


def draw_settlement(screen, x, y, color):
    """Draw a settlement as a small house shape."""
    # House: triangle roof on square base
    w, h = 14, 10
    roof_h = 8
    points = [
        (x, y - h // 2 - roof_h),       # roof peak
        (x - w // 2, y - h // 2),        # roof left
        (x - w // 2, y + h // 2),        # base left
        (x + w // 2, y + h // 2),        # base right
        (x + w // 2, y - h // 2),        # roof right
    ]
    pygame.draw.polygon(screen, color, points)
    pygame.draw.polygon(screen, (0, 0, 0), points, 2)


def draw_city(screen, x, y, color):
    """Draw a city as a house with a tower."""
    # Tower (left) + house (right)
    tower_points = [
        (x - 10, y - 14),   # tower top-left
        (x - 10, y + 8),    # tower bottom-left
        (x - 2, y + 8),     # tower bottom-right
        (x - 2, y - 14),    # tower top-right
    ]
    house_points = [
        (x + 1, y - 6),     # roof peak area
        (x - 2, y - 2),     # roof left
        (x - 2, y + 8),     # base left
        (x + 12, y + 8),    # base right
        (x + 12, y - 2),    # roof right
    ]
    pygame.draw.polygon(screen, color, house_points)
    pygame.draw.polygon(screen, (0, 0, 0), house_points, 2)
    pygame.draw.polygon(screen, color, tower_points)
    pygame.draw.polygon(screen, (0, 0, 0), tower_points, 2)


def draw_road(screen, x1, y1, x2, y2, color):
    """Draw a road as a thick line with dark border."""
    pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 7)
    pygame.draw.line(screen, color, (x1, y1), (x2, y2), 4)


def draw_port(screen, port, offset, board_center):
    """Draw a port label outside the board with bridge lines."""
    v1x = port.vertex1.x + offset[0]
    v1y = port.vertex1.y + offset[1]
    v2x = port.vertex2.x + offset[0]
    v2y = port.vertex2.y + offset[1]
    mid_x = (v1x + v2x) / 2
    mid_y = (v1y + v2y) / 2

    # Direction outward from board center
    dx = mid_x - board_center[0]
    dy = mid_y - board_center[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    nx, ny = dx / dist, dy / dist

    # Label position (35px outward)
    label_x = mid_x + nx * 38
    label_y = mid_y + ny * 38

    # Bridge lines
    bridge_color = (120, 110, 95)
    pygame.draw.line(screen, bridge_color, (int(label_x), int(label_y)), (int(v1x), int(v1y)), 2)
    pygame.draw.line(screen, bridge_color, (int(label_x), int(label_y)), (int(v2x), int(v2y)), 2)

    # Label background
    port_color = PORT_COLORS.get(port.port_type, (180, 170, 150))
    label_text = PORT_TYPE_NAMES.get(port.port_type, "3:1")
    font = get_font(14, bold=True)
    text_surface = font.render(label_text, True, WHITE)
    tw, th = text_surface.get_size()
    pad = 4
    bg_rect = pygame.Rect(label_x - tw // 2 - pad, label_y - th // 2 - pad,
                           tw + pad * 2, th + pad * 2)
    pygame.draw.rect(screen, port_color, bg_rect, border_radius=4)
    pygame.draw.rect(screen, (40, 35, 30), bg_rect, 1, border_radius=4)
    screen.blit(text_surface, (label_x - tw // 2, label_y - th // 2))


def draw_game_board(screen, game_board, offset):
    """Draw the complete game board."""
    # Calculate board center for port direction
    if game_board.tiles:
        xs = [t.x + offset[0] for t in game_board.tiles]
        ys = [t.y + offset[1] for t in game_board.tiles]
        board_center = (sum(xs) / len(xs), sum(ys) / len(ys))
    else:
        board_center = (offset[0], offset[1])

    # 1. Hex shadows + hexes
    for tile in game_board.tiles:
        draw_hex_tile(screen, tile, offset)

    # 2. Roads (under buildings)
    for edge in game_board.edges:
        x1 = edge.vertex1.x + offset[0]
        y1 = edge.vertex1.y + offset[1]
        x2 = edge.vertex2.x + offset[0]
        y2 = edge.vertex2.y + offset[1]
        if edge.structure:
            draw_road(screen, x1, y1, x2, y2, edge.structure.player.color)

    # 3. Ports
    for port in game_board.ports:
        draw_port(screen, port, offset, board_center)

    # 4. Vertices (buildings / empty dots)
    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]
        if vertex.structure:
            player_color = vertex.structure.player.color
            if isinstance(vertex.structure, City):
                draw_city(screen, x, y, player_color)
            elif isinstance(vertex.structure, Settlement):
                draw_settlement(screen, x, y, player_color)
        else:
            pygame.draw.circle(screen, (80, 75, 65), (int(x), int(y)), 3)


def draw_buildable_highlights(screen, game_board, game, offset, build_mode):
    """Highlight buildable positions for the current player."""
    highlight_color = (80, 220, 80)

    if build_mode == "SETTLEMENT":
        if game.is_initial_placement_phase() and not game.waiting_for_road:
            # Initial placement: any valid vertex
            for v in game_board.vertices:
                if v.structure is None and not any(adj.structure for adj in v.adjacent_vertices):
                    x = v.x + offset[0]
                    y = v.y + offset[1]
                    pygame.draw.circle(screen, highlight_color, (int(x), int(y)), 8, 2)
        elif game.can_trade_or_build():
            for v in game.get_buildable_vertices_for_settlements():
                x = v.x + offset[0]
                y = v.y + offset[1]
                pygame.draw.circle(screen, highlight_color, (int(x), int(y)), 8, 2)

    elif build_mode == "CITY" and game.can_trade_or_build():
        for v in game.get_buildable_vertices_for_cities():
            x = v.x + offset[0]
            y = v.y + offset[1]
            pygame.draw.circle(screen, (100, 180, 255), (int(x), int(y)), 10, 2)

    elif build_mode == "ROAD":
        if game.is_initial_placement_phase() and game.waiting_for_road:
            if game.last_settlement_vertex:
                for e in game_board.edges:
                    if (e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)):
                        x1 = e.vertex1.x + offset[0]
                        y1 = e.vertex1.y + offset[1]
                        x2 = e.vertex2.x + offset[0]
                        y2 = e.vertex2.y + offset[1]
                        pygame.draw.line(screen, highlight_color,
                                       (int(x1), int(y1)), (int(x2), int(y2)), 3)
        elif game.can_trade_or_build():
            for e in game.get_buildable_edges():
                x1 = e.vertex1.x + offset[0]
                y1 = e.vertex1.y + offset[1]
                x2 = e.vertex2.x + offset[0]
                y2 = e.vertex2.y + offset[1]
                pygame.draw.line(screen, highlight_color,
                               (int(x1), int(y1)), (int(x2), int(y2)), 3)


# ===========================================================================
# HUD / Panel Drawing
# ===========================================================================

def draw_resource_card(screen, x, y, resource_type, count, width=55, height=70):
    """Draw a single resource card."""
    color = RESOURCE_CARD_COLORS.get(resource_type, (100, 100, 100))
    name = RESOURCE_TYPE_NAMES.get(resource_type, "?")

    # Card background
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, color, rect, border_radius=6)
    # Slightly lighter inner area
    inner = pygame.Rect(x + 2, y + 2, width - 4, height - 4)
    lighter = tuple(min(c + 25, 255) for c in color)
    pygame.draw.rect(screen, lighter, inner, border_radius=5)
    # Border
    pygame.draw.rect(screen, (40, 35, 30), rect, 2, border_radius=6)

    # Resource name
    name_font = get_font(13, bold=True)
    name_surf = name_font.render(name, True, WHITE)
    name_rect = name_surf.get_rect(centerx=x + width // 2, top=y + 5)
    screen.blit(name_surf, name_rect)

    # Count (large)
    count_font = get_font(26, bold=True)
    count_surf = count_font.render(str(count), True, WHITE)
    count_rect = count_surf.get_rect(center=(x + width // 2, y + height // 2 + 5))
    screen.blit(count_surf, count_rect)


def draw_resource_panel(screen, player, x, y):
    """Draw a row of resource cards for a player."""
    resource_order = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                      ResourceType.ORE, ResourceType.SHEEP]
    card_w = 55
    spacing = 4
    for i, res_type in enumerate(resource_order):
        count = player.resources.get(res_type, 0)
        draw_resource_card(screen, x + i * (card_w + spacing), y, res_type, count)


def draw_player_info_box(screen, player, x, y, width, is_current, font, small_font):
    """Draw a compact player info box."""
    height = 90

    # Background
    bg = (50, 45, 40) if is_current else (38, 34, 30)
    pygame.draw.rect(screen, bg, (x, y, width, height), border_radius=8)

    # Border (highlighted for current player)
    if is_current:
        pygame.draw.rect(screen, player.color, (x, y, width, height), 3, border_radius=8)
    else:
        pygame.draw.rect(screen, (70, 65, 55), (x, y, width, height), 1, border_radius=8)

    # Color indicator + name
    pygame.draw.circle(screen, player.color, (x + 18, y + 20), 8)
    name_surf = font.render(player.name, True, player.color)
    screen.blit(name_surf, (x + 32, y + 11))

    # VP (prominent)
    vp = player.calculate_victory_points()
    vp_surf = font.render(f"VP: {vp}", True, GOLD)
    screen.blit(vp_surf, (x + width - 80, y + 11))

    # Building counts
    info_y = y + 40
    buildings = f"S:{len(player.settlements)} C:{len(player.cities)} R:{len(player.roads)}"
    build_surf = small_font.render(buildings, True, TEXT_DIM)
    screen.blit(build_surf, (x + 15, info_y))

    # Total resources
    total = player.get_total_resources()
    res_surf = small_font.render(f"Cards: {total}", True, TEXT_DIM)
    screen.blit(res_surf, (x + 180, info_y))

    # Achievements
    ach_y = info_y + 22
    if player.has_longest_road:
        lr = small_font.render("Longest Road", True, (255, 200, 50))
        screen.blit(lr, (x + 15, ach_y))
    if player.has_largest_army:
        la = small_font.render("Largest Army", True, (255, 200, 50))
        screen.blit(la, (x + 150, ach_y))

    return height


def draw_vp_tracker(screen, vp, target, x, y, width=200):
    """Draw a VP progress bar."""
    bar_h = 14
    # Background
    pygame.draw.rect(screen, (50, 45, 40), (x, y, width, bar_h), border_radius=4)
    # Fill
    fill_w = int((vp / target) * width) if target > 0 else 0
    fill_w = min(fill_w, width)
    if fill_w > 0:
        pygame.draw.rect(screen, GOLD, (x, y, fill_w, bar_h), border_radius=4)
    # Border
    pygame.draw.rect(screen, (80, 70, 60), (x, y, width, bar_h), 1, border_radius=4)
    # Text
    font = get_font(12, bold=True)
    text = font.render(f"{vp}/{target} VP", True, WHITE)
    text_rect = text.get_rect(center=(x + width // 2, y + bar_h // 2))
    screen.blit(text, text_rect)


def draw_button(screen, rect, text, font, enabled=True, hovered=False,
                shortcut=None, cost=None):
    """Draw a styled game button. Returns the rect for click detection."""
    x, y, w, h = rect

    if not enabled:
        bg_top = (55, 50, 45)
        bg_bot = (45, 40, 35)
        text_color = (100, 95, 85)
        border_color = (65, 60, 50)
    elif hovered:
        bg_top = (90, 75, 50)
        bg_bot = (70, 58, 38)
        text_color = WHITE
        border_color = GOLD
    else:
        bg_top = (75, 65, 48)
        bg_bot = (55, 48, 35)
        text_color = TEXT_COLOR
        border_color = (100, 90, 70)

    # Gradient-like effect: top half lighter, bottom half darker
    top_rect = pygame.Rect(x, y, w, h // 2)
    bot_rect = pygame.Rect(x, y + h // 2, w, h - h // 2)
    pygame.draw.rect(screen, bg_top, top_rect, border_top_left_radius=6, border_top_right_radius=6)
    pygame.draw.rect(screen, bg_bot, bot_rect, border_bottom_left_radius=6, border_bottom_right_radius=6)

    # Full border
    full_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, border_color, full_rect, 2, border_radius=6)

    # Text
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=(x + w // 2, y + h // 2 - (6 if cost else 0)))
    screen.blit(text_surf, text_rect)

    # Shortcut key
    if shortcut:
        sc_font = get_font(12)
        sc_color = text_color if enabled else (80, 75, 65)
        sc_surf = sc_font.render(f"[{shortcut}]", True, sc_color)
        screen.blit(sc_surf, (x + w - sc_surf.get_width() - 5, y + 3))

    # Build cost icons
    if cost and enabled:
        cost_x = x + 8
        cost_y = y + h - 14
        tiny_font = get_font(11)
        for res_type, amount in cost.items():
            dot_color = RESOURCE_CARD_COLORS.get(res_type, (100, 100, 100))
            for _ in range(amount):
                pygame.draw.circle(screen, dot_color, (cost_x + 4, cost_y + 4), 4)
                pygame.draw.circle(screen, (0, 0, 0), (cost_x + 4, cost_y + 4), 4, 1)
                cost_x += 10

    return full_rect


def draw_game_state_panel(screen, game, x, y, width, font, small_font):
    """Draw compact game state info (phase, dice, turn indicator)."""
    height = 100
    pygame.draw.rect(screen, PANEL_BG, (x, y, width, height), border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER, (x, y, width, height), 1, border_radius=8)

    cy = y + 12
    current = game.get_current_player()

    # Turn indicator
    pygame.draw.circle(screen, current.color, (x + 18, cy + 8), 8)
    turn_text = font.render(f"{current.name}'s Turn", True, current.color)
    screen.blit(turn_text, (x + 32, cy))
    cy += 28

    # Phase
    if game.is_initial_placement_phase():
        phase_label = "Setup Phase"
        if game.waiting_for_road:
            action_label = "Place a road"
        else:
            action_label = "Place a settlement"
    else:
        phase_label = game.turn_phase.replace("_", " ").title()
        if game.can_roll_dice():
            action_label = "Roll dice (D)"
        elif game.can_end_turn():
            action_label = "Build or end turn (T)"
        else:
            action_label = ""

    phase_surf = small_font.render(phase_label, True, TEXT_DIM)
    screen.blit(phase_surf, (x + 15, cy))
    cy += 20

    # Dice roll
    if game.last_dice_roll:
        dice = game.last_dice_roll
        dice_surf = font.render(f"Rolled: {dice[2]}  ({dice[0]}+{dice[1]})", True, GOLD)
        screen.blit(dice_surf, (x + 15, cy))
    elif action_label:
        act_surf = small_font.render(action_label, True, (120, 200, 120))
        screen.blit(act_surf, (x + 15, cy))

    return height


def draw_game_log(screen, messages, x, y, w, h, small_font):
    """Draw a game event log."""
    pygame.draw.rect(screen, (30, 27, 24), (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

    # Title
    title_font = get_font(14, bold=True)
    title = title_font.render("Game Log", True, TEXT_DIM)
    screen.blit(title, (x + 8, y + 4))

    # Messages (newest last, show last N that fit)
    msg_y = y + 22
    max_messages = (h - 26) // 16
    visible = messages[-max_messages:] if len(messages) > max_messages else messages

    for msg_text, msg_color in visible:
        surf = small_font.render(msg_text, True, msg_color)
        # Clip to width
        screen.blit(surf, (x + 8, msg_y), area=pygame.Rect(0, 0, w - 16, 16))
        msg_y += 16


# ===========================================================================
# Title Screen
# ===========================================================================

def draw_title_screen(screen, buttons_hovered):
    """Draw the title/menu screen. Returns list of (rect, mode_name) for buttons."""
    screen.fill((30, 22, 18))

    # Subtle pattern (diagonal lines)
    for i in range(-20, SCREEN_W + SCREEN_H, 40):
        pygame.draw.line(screen, (35, 27, 22), (i, 0), (i - SCREEN_H, SCREEN_H), 1)

    cx = SCREEN_W // 2

    # Title
    title_font = get_font(72, bold=True)
    title_surf = title_font.render("CATANTRON", True, GOLD)
    title_rect = title_surf.get_rect(center=(cx, 220))
    # Title shadow
    shadow_surf = title_font.render("CATANTRON", True, (60, 45, 20))
    screen.blit(shadow_surf, (title_rect.x + 3, title_rect.y + 3))
    screen.blit(title_surf, title_rect)

    # Subtitle
    sub_font = get_font(28)
    sub_surf = sub_font.render("AI-Powered Catan", True, (180, 160, 130))
    sub_rect = sub_surf.get_rect(center=(cx, 290))
    screen.blit(sub_surf, sub_rect)

    # Decorative line
    line_y = 330
    pygame.draw.line(screen, (80, 65, 45), (cx - 150, line_y), (cx + 150, line_y), 2)

    # Buttons
    btn_w, btn_h = 320, 60
    btn_x = cx - btn_w // 2
    btn_start_y = 380
    btn_spacing = 80

    button_font = get_font(24, bold=True)
    button_defs = [
        ("Player vs Player", "pvp"),
        ("Play vs AI", "vs_ai"),
        ("Play vs Bot", "vs_bot"),
    ]

    button_rects = []
    for i, (label, mode) in enumerate(button_defs):
        y = btn_start_y + i * btn_spacing
        hovered = buttons_hovered.get(mode, False)
        rect = draw_button(screen, (btn_x, y, btn_w, btn_h), label,
                          button_font, enabled=True, hovered=hovered)
        button_rects.append((rect, mode))

    # Footer
    footer_font = get_font(16)
    footer = footer_font.render("A school project by the Catantron team", True, (80, 70, 60))
    screen.blit(footer, footer.get_rect(center=(cx, SCREEN_H - 40)))

    return button_rects


def draw_difficulty_screen(screen, buttons_hovered):
    """Draw difficulty selection screen. Returns list of (rect, difficulty)."""
    screen.fill((30, 22, 18))

    # Pattern
    for i in range(-20, SCREEN_W + SCREEN_H, 40):
        pygame.draw.line(screen, (35, 27, 22), (i, 0), (i - SCREEN_H, SCREEN_H), 1)

    cx = SCREEN_W // 2

    # Header
    header_font = get_font(48, bold=True)
    header = header_font.render("Select Difficulty", True, GOLD)
    screen.blit(header, header.get_rect(center=(cx, 200)))

    # Difficulty buttons
    btn_w, btn_h = 280, 55
    btn_x = cx - btn_w // 2
    btn_start_y = 320
    btn_spacing = 80

    button_font = get_font(24, bold=True)
    desc_font = get_font(16)

    difficulties = [
        ("Easy", "easy", "Predictable and forgiving"),
        ("Medium", "medium", "Balanced strategy"),
        ("Hard", "hard", "Aggressive and optimal"),
    ]

    button_rects = []
    for i, (label, diff, desc) in enumerate(difficulties):
        y = btn_start_y + i * btn_spacing
        hovered = buttons_hovered.get(diff, False)
        rect = draw_button(screen, (btn_x, y, btn_w, btn_h), label,
                          button_font, enabled=True, hovered=hovered)
        # Description below button
        desc_surf = desc_font.render(desc, True, TEXT_DIM)
        screen.blit(desc_surf, desc_surf.get_rect(center=(cx, y + btn_h + 12)))
        button_rects.append((rect, diff))

    # Back button
    back_y = btn_start_y + 3 * btn_spacing + 20
    back_hovered = buttons_hovered.get("back", False)
    back_rect = draw_button(screen, (btn_x, back_y, btn_w, 45), "Back",
                           get_font(20), enabled=True, hovered=back_hovered)
    button_rects.append((back_rect, "back"))

    return button_rects


# ===========================================================================
# Robber Placement Overlay
# ===========================================================================

def draw_robber_overlay(screen, game_board, offset, robber_tile):
    """Draw highlights on valid tiles during robber placement mode."""
    for tile in game_board.tiles:
        if tile == robber_tile or tile.resource == "desert":
            continue
        tx = tile.x + offset[0]
        ty = tile.y + offset[1]
        highlight = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(highlight, (255, 220, 50, 50), (30, 30), 30)
        screen.blit(highlight, (tx - 30, ty - 30))

    # Banner at top of board area
    banner_font = get_font(28, bold=True)
    banner_surf = banner_font.render("MOVE THE ROBBER", True, (255, 80, 80))
    banner_rect = banner_surf.get_rect(center=(PANEL_X // 2, 25))
    bg_rect = banner_rect.inflate(20, 8)
    pygame.draw.rect(screen, (40, 20, 20), bg_rect, border_radius=6)
    pygame.draw.rect(screen, (255, 80, 80), bg_rect, 2, border_radius=6)
    screen.blit(banner_surf, banner_rect)


# ===========================================================================
# Dev Card Popup
# ===========================================================================

def draw_dev_card_popup(screen, card_name, description):
    """Draw a centered popup showing the purchased dev card."""
    cx = PANEL_X // 2
    cy = SCREEN_H // 2

    card_w, card_h = 280, 170
    card_x = cx - card_w // 2
    card_y = cy - card_h // 2

    # Dark overlay behind popup
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))

    # Card background
    pygame.draw.rect(screen, (45, 38, 32), (card_x, card_y, card_w, card_h), border_radius=12)
    pygame.draw.rect(screen, GOLD, (card_x, card_y, card_w, card_h), 2, border_radius=12)

    # "You got:" header
    header_font = get_font(18)
    header_surf = header_font.render("You got:", True, TEXT_DIM)
    screen.blit(header_surf, header_surf.get_rect(center=(cx, card_y + 30)))

    # Card name
    name_font = get_font(30, bold=True)
    name_surf = name_font.render(card_name, True, GOLD)
    screen.blit(name_surf, name_surf.get_rect(center=(cx, card_y + 75)))

    # Description
    desc_font = get_font(16)
    desc_surf = desc_font.render(description, True, TEXT_COLOR)
    screen.blit(desc_surf, desc_surf.get_rect(center=(cx, card_y + 115)))

    # Dismiss hint
    hint_font = get_font(13)
    hint_surf = hint_font.render("Click to dismiss", True, TEXT_DIM)
    screen.blit(hint_surf, hint_surf.get_rect(center=(cx, card_y + card_h - 18)))
