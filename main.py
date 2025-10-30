import pygame
import sys
import random
import math
from tile import Tile
from game_system import (Player, Robber, GameBoard, GameSystem, ResourceType,
                         Settlement, City, Road, DevelopmentCardType, GameConstants)

# Standard Catan numbers
NUMBER_TOKENS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]#jjjjjjjjj

# Resources
RESOURCES = ["forest"] * 4 + ["hill"] * 3 + ["field"] * 4 + ["mountain"] * 3 + ["pasture"] * 4 + ["desert"]

RESOURCE_COLORS = {
    "forest": (34, 139, 34),
    "hill": (178, 34, 34),
    "field": (218, 165, 32),
    "mountain": (169, 169, 169),
    "pasture": (144, 238, 144),
    "desert": (237, 201, 175)
}


def create_hexagonal_board(size, radius=2):
    """Create a hexagonal board with the given radius (2 for standard Catan)"""
    tiles = []

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            tiles.append(Tile(q, r, size))

    return tiles


def assign_resources_numbers(tiles, robber):
    """Assign resources and numbers with constraints"""
    all_resources = RESOURCES.copy()
    random.shuffle(all_resources)

    for i, tile in enumerate(tiles):
        if i < len(all_resources):
            tile.resource = all_resources[i]
        else:
            tile.resource = "forest"

    desert_tile = None
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            desert_tile = tile
            robber.move_to_tile(tile)
            break

    non_desert_tiles = [t for t in tiles if t.resource != "desert"]
    assign_numbers_with_constraints(non_desert_tiles)


def assign_numbers_with_constraints(tiles):
    """Assign numbers ensuring 6s and 8s are not adjacent"""
    nums = NUMBER_TOKENS.copy()
    red_numbers = [6, 8]

    sixes = [n for n in nums if n == 6]
    eights = [n for n in nums if n == 8]
    red_nums_to_place = sixes + eights

    max_attempts = 1000
    for attempt in range(max_attempts):
        for tile in tiles:
            tile.number = None

        success = True
        remaining_nums = nums.copy()

        for red_num in red_nums_to_place:
            valid_tiles = []
            for tile in tiles:
                if tile.resource != "desert" and tile.number is None:
                    if can_place_red_number(tile, red_num):
                        valid_tiles.append(tile)

            if valid_tiles:
                chosen_tile = random.choice(valid_tiles)
                chosen_tile.number = red_num
                remaining_nums.remove(red_num)
            else:
                success = False
                break

        if not success:
            continue

        remaining_tiles = [t for t in tiles if t.resource != "desert" and t.number is None]
        random.shuffle(remaining_tiles)
        random.shuffle(remaining_nums)

        for i, tile in enumerate(remaining_tiles):
            if i < len(remaining_nums):
                tile.number = remaining_nums[i]

        if all(t.number is not None for t in tiles if t.resource != "desert"):
            if is_board_valid(tiles):
                return

    print("Warning: Could not assign numbers without red adjacency.")
    random.shuffle(nums)
    for i, tile in enumerate(tiles):
        if i < len(nums):
            tile.number = nums[i]


def can_place_red_number(tile, red_number):
    """Check if a red number (6 or 8) can be placed on a tile"""
    red_numbers = [6, 8]

    for neighbor in tile.neighbors:
        if neighbor.number in red_numbers:
            return False

    return True


def validate_board(tiles):
    """Validate that no red numbers (6,8) are adjacent"""
    red_numbers = [6, 8]
    violations = []

    for tile in tiles:
        if tile.number in red_numbers:
            for neighbor in tile.neighbors:
                if neighbor.number in red_numbers:
                    violations.append(f"Red numbers adjacent: Tile ({tile.q},{tile.r}) has {tile.number}")

    return violations


def is_board_valid(tiles):
    """Check if board is valid"""
    return len(validate_board(tiles)) == 0


def compute_center_offset(tiles, screen_w, screen_h):
    if not tiles:
        return 0, 0

    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    board_cx, board_cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return screen_w / 2 - board_cx, screen_h / 2 - board_cy


def draw_game_board(screen, game_board, offset, show_coords=False):
    """Draw the game board with tiles, vertices, and edges"""
    for tile in game_board.tiles:
        color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
        tile.draw(screen, fill_color=color, offset=offset, show_coords=show_coords)

    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]

        if vertex.structure:
            player_color = vertex.structure.player.color
            if isinstance(vertex.structure, Settlement):
                pygame.draw.circle(screen, player_color, (int(x), int(y)), 8)
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 8, 2)
            elif isinstance(vertex.structure, City):
                pygame.draw.rect(screen, player_color, (x - 10, y - 10, 20, 20))
                pygame.draw.rect(screen, (0, 0, 0), (x - 10, y - 10, 20, 20), 2)
        else:
            pygame.draw.circle(screen, (100, 100, 100), (int(x), int(y)), 3)

    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        if edge.structure:
            player_color = edge.structure.player.color
            pygame.draw.line(screen, player_color, (x1, y1), (x2, y2), 4)
        else:
            pygame.draw.line(screen, (150, 150, 150), (x1, y1), (x2, y2), 1)


def find_closest_vertex(game_board, mouse_pos, offset, max_distance=15):
    """Find the closest vertex to mouse position"""
    mouse_x, mouse_y = mouse_pos
    closest_vertex = None
    closest_distance = max_distance

    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]
        distance = ((mouse_x - x) ** 2 + (mouse_y - y) ** 2) ** 0.5

        if distance < closest_distance:
            closest_distance = distance
            closest_vertex = vertex

    return closest_vertex


def find_closest_edge(game_board, mouse_pos, offset, max_distance=10):
    """Find the closest edge to mouse position"""
    mouse_x, mouse_y = mouse_pos
    closest_edge = None
    closest_distance = max_distance

    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        A = mouse_x - x1
        B = mouse_y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            distance = ((mouse_x - x1) ** 2 + (mouse_y - y1) ** 2) ** 0.5
        else:
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            distance = ((mouse_x - xx) ** 2 + (mouse_y - yy) ** 2) ** 0.5

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = edge

    return closest_edge


def draw_button(screen, x, y, width, height, text, color, hover_color, font, enabled=True):
    """Draw a button and return its rect and hover state"""
    mouse_pos = pygame.mouse.get_pos()
    rect = pygame.Rect(x, y, width, height)
    is_hovering = rect.collidepoint(mouse_pos) and enabled

    button_color = hover_color if is_hovering else color
    if not enabled:
        button_color = (60, 60, 60)

    pygame.draw.rect(screen, button_color, rect, border_radius=5)
    pygame.draw.rect(screen, (200, 200, 200) if enabled else (100, 100, 100), rect, 2, border_radius=5)

    text_color = (255, 255, 255) if enabled else (120, 120, 120)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

    return rect, is_hovering


def add_message(message, color, messages_list):
    """Add a message to the message queue with timestamp"""
    messages_list.append((message, color, pygame.time.get_ticks()))


def main():
    pygame.init()
    pygame.font.init()

    SCREEN_W, SCREEN_H = 1600, 1000
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Catan Game")
    clock = pygame.time.Clock()

    # Initialize fonts early
    font_title = pygame.font.SysFont(None, 32)
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 20)

    # Initialize UI state
    game_messages = []
    buttons = {}

    tile_size = 50

    players = [
        Player("Player 1", (255, 50, 50)),
        Player("Player 2", (50, 50, 255)),
        Player("Player 3", (255, 255, 50)),
        Player("Player 4", (255, 255, 255))
    ]

    tiles = create_hexagonal_board(tile_size, radius=2)

    for t in tiles:
        t.find_neighbors(tiles)

    robber = Robber()
    assign_resources_numbers(tiles, robber)

    game_board = GameBoard(tiles)
    game_system = GameSystem(game_board, players)
    game_system.robber = robber

    print("=== CATAN GAME STARTED ===")
    print(f"{game_system.get_current_player().name} starts")

    offset = compute_center_offset(tiles, 800, SCREEN_H)
    show_coords = False
    build_mode = "SETTLEMENT"
    show_buildable = False
    trade_mode = False

    # Trading variables
    selected_trade_partner = 0
    offering_resources = {ResourceType.WOOD: 0, ResourceType.BRICK: 0, ResourceType.WHEAT: 0, ResourceType.SHEEP: 0,
                          ResourceType.ORE: 0}
    requesting_resources = {ResourceType.WOOD: 0, ResourceType.BRICK: 0, ResourceType.WHEAT: 0, ResourceType.SHEEP: 0,
                            ResourceType.ORE: 0}

    # Robber movement
    robber_move_mode = False

    # Development card playing
    selected_dev_card = None
    show_dev_card_menu = False

    # Victory tracking
    winner = None
    show_victory_screen = False

    running = True
    while running:
        current_player = game_system.get_current_player()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    if game_system.can_roll_dice():
                        dice_result = game_system.roll_dice()
                        if dice_result:
                            print(f"Rolled: {dice_result[2]}")
                            # Automatically enter robber mode if 7 was rolled
                            if dice_result[2] == 7:
                                robber_move_mode = True
                                add_message("Move the robber!", (255, 100, 100), game_messages)
                elif event.key == pygame.K_t:
                    success, message = game_system.end_turn()
                    print(message)
                elif event.key == pygame.K_1:
                    build_mode = "SETTLEMENT"
                elif event.key == pygame.K_2:
                    build_mode = "CITY"
                elif event.key == pygame.K_3:
                    build_mode = "ROAD"
                elif event.key == pygame.K_b:
                    show_buildable = not show_buildable
                elif event.key == pygame.K_c:
                    show_coords = not show_coords
                elif event.key == pygame.K_x:
                    if game_system.can_trade_or_build():
                        success, message = game_system.try_buy_development_card()
                        print(message)
                        add_message(message, (100, 255, 100) if success else (255, 100, 100), game_messages)
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                elif event.key == pygame.K_4:
                    # Bank trade 4:1 - Quick trade wood for brick
                    if game_system.can_trade_or_build():
                        success, msg = game_system.execute_bank_trade(current_player, ResourceType.WOOD,
                                                                      ResourceType.BRICK, 4)
                        print(msg)
                        add_message(msg, (100, 255, 100) if success else (255, 100, 100), game_messages)
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                elif event.key == pygame.K_5:
                    # Toggle trade mode
                    if game_system.can_trade_or_build():
                        trade_mode = not trade_mode
                        if trade_mode:
                            print("TRADE MODE: Use arrows/WASD to adjust, ENTER to trade")
                            add_message("TRADE MODE: Use arrows/WASD", (255, 200, 255), game_messages)
                            # Reset trade amounts
                            for res in offering_resources:
                                offering_resources[res] = 0
                            for res in requesting_resources:
                                requesting_resources[res] = 0
                        else:
                            add_message("Trade mode OFF", (200, 200, 200), game_messages)
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                elif event.key == pygame.K_6:
                    # Toggle dev card menu
                    show_dev_card_menu = not show_dev_card_menu
                elif event.key == pygame.K_7:
                    # Play Knight
                    if game_system.can_trade_or_build():
                        success, msg = game_system.play_knight_card(current_player)
                        print(msg)
                        add_message(msg, (100, 255, 100) if success else (255, 100, 100), game_messages)
                        if success:
                            robber_move_mode = True
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                elif event.key == pygame.K_8:
                    # Play Year of Plenty (wood and brick)
                    if game_system.can_trade_or_build():
                        success, msg = game_system.play_year_of_plenty_card(current_player, ResourceType.WOOD,
                                                                            ResourceType.BRICK)
                        print(msg)
                        add_message(msg, (100, 255, 100) if success else (255, 100, 100), game_messages)
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                elif event.key == pygame.K_9:
                    # Play Monopoly on Wood
                    if game_system.can_trade_or_build():
                        success, msg = game_system.play_monopoly_card(current_player, ResourceType.WOOD)
                        print(msg)
                        add_message(msg, (100, 255, 100) if success else (255, 100, 100), game_messages)
                    else:
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)

                # Trade mode controls
                if trade_mode and not game_system.is_initial_placement_phase():
                    if event.key == pygame.K_UP:
                        offering_resources[ResourceType.WOOD] = min(offering_resources[ResourceType.WOOD] + 1,
                                                                    current_player.resources[ResourceType.WOOD])
                    elif event.key == pygame.K_w:
                        offering_resources[ResourceType.BRICK] = min(offering_resources[ResourceType.BRICK] + 1,
                                                                     current_player.resources[ResourceType.BRICK])
                    elif event.key == pygame.K_a:
                        offering_resources[ResourceType.WHEAT] = min(offering_resources[ResourceType.WHEAT] + 1,
                                                                     current_player.resources[ResourceType.WHEAT])
                    elif event.key == pygame.K_s:
                        offering_resources[ResourceType.SHEEP] = min(offering_resources[ResourceType.SHEEP] + 1,
                                                                     current_player.resources[ResourceType.SHEEP])
                    elif event.key == pygame.K_q:
                        offering_resources[ResourceType.ORE] = min(offering_resources[ResourceType.ORE] + 1,
                                                                   current_player.resources[ResourceType.ORE])
                    elif event.key == pygame.K_DOWN:
                        requesting_resources[ResourceType.WOOD] = min(requesting_resources[ResourceType.WOOD] + 1, 10)
                    elif event.key == pygame.K_e:
                        requesting_resources[ResourceType.BRICK] = min(requesting_resources[ResourceType.BRICK] + 1, 10)
                    elif event.key == pygame.K_d:
                        requesting_resources[ResourceType.WHEAT] = min(requesting_resources[ResourceType.WHEAT] + 1, 10)
                    elif event.key == pygame.K_f:
                        requesting_resources[ResourceType.SHEEP] = min(requesting_resources[ResourceType.SHEEP] + 1, 10)
                    elif event.key == pygame.K_z:
                        requesting_resources[ResourceType.ORE] = min(requesting_resources[ResourceType.ORE] + 1, 10)
                    elif event.key == pygame.K_LEFT:
                        # Previous trade partner
                        partners = game_system.get_available_trade_partners(current_player)
                        if partners:
                            selected_trade_partner = (selected_trade_partner - 1) % len(partners)
                    elif event.key == pygame.K_RIGHT:
                        # Next trade partner
                        partners = game_system.get_available_trade_partners(current_player)
                        if partners:
                            selected_trade_partner = (selected_trade_partner + 1) % len(partners)
                    elif event.key == pygame.K_RETURN:
                        # Execute trade
                        partners = game_system.get_available_trade_partners(current_player)
                        if partners:
                            target = partners[selected_trade_partner]
                            success, msg = game_system.execute_player_trade(current_player, target, offering_resources,
                                                                            requesting_resources)
                            print(msg)
                            if success:
                                # Reset
                                for res in offering_resources:
                                    offering_resources[res] = 0
                                for res in requesting_resources:
                                    requesting_resources[res] = 0

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # Check button clicks
                for button_name, (rect, enabled) in buttons.items():
                    if enabled and rect.collidepoint(mouse_pos):
                        if button_name == "ROLL_DICE":
                            if game_system.can_roll_dice():
                                dice_result = game_system.roll_dice()
                                if dice_result:
                                    add_message(f"Rolled: {dice_result[2]} ({dice_result[0]}+{dice_result[1]})",
                                                (255, 255, 0), game_messages)
                                    # Automatically enter robber mode if 7 was rolled
                                    if dice_result[2] == 7:
                                        robber_move_mode = True
                                        add_message("Move the robber!", (255, 100, 100), game_messages)
                        elif button_name == "END_TURN":
                            success, msg = game_system.end_turn()
                            if success:
                                game_system.update_longest_road()
                                game_system.update_largest_army()
                                winner = game_system.check_victory_conditions()
                                if winner:
                                    show_victory_screen = True
                                else:
                                    add_message(msg, (100, 255, 100), game_messages)
                        elif button_name == "BUILD_SETTLEMENT":
                            build_mode = "SETTLEMENT"
                            add_message("Settlement mode", (100, 255, 100), game_messages)
                        elif button_name == "BUILD_CITY":
                            build_mode = "CITY"
                            add_message("City mode", (100, 255, 100), game_messages)
                        elif button_name == "BUILD_ROAD":
                            build_mode = "ROAD"
                            add_message("Road mode", (100, 255, 100), game_messages)
                        elif button_name == "BUY_DEV_CARD":
                            success, msg = game_system.try_buy_development_card()
                            add_message(msg, (255, 255, 0) if success else (255, 100, 100), game_messages)
                        elif button_name == "BANK_TRADE":
                            trade_mode = not trade_mode
                            add_message("Trade mode: " + ("ON" if trade_mode else "OFF"), (255, 200, 255), game_messages)
                        elif button_name == "SHOW_BUILDABLE":
                            show_buildable = not show_buildable
                        continue

                # Robber movement mode
                if robber_move_mode:
                    clicked_tile = None
                    for tile in game_board.tiles:
                        tile_x = tile.x + offset[0]
                        tile_y = tile.y + offset[1]
                        distance = ((mouse_pos[0] - tile_x) ** 2 + (mouse_pos[1] - tile_y) ** 2) ** 0.5
                        if distance < tile.size:
                            clicked_tile = tile
                            break

                    if clicked_tile and clicked_tile != game_system.robber.position:
                        success, msg = game_system.move_robber_to_tile(clicked_tile)
                        print(msg)

                        # Steal from adjacent player
                        adjacent_players = game_system.get_players_on_tile(clicked_tile)
                        stealable = [p for p in adjacent_players if p != current_player and p.get_total_resources() > 0]

                        if stealable:
                            target = random.choice(stealable)
                            success, steal_msg = game_system.steal_random_resource(current_player, target)
                            print(steal_msg)

                        robber_move_mode = False
                        add_message("Robber moved!", (255, 100, 100), game_messages)

                elif game_system.is_initial_placement_phase():
                    if not game_system.waiting_for_road:
                        vertex = find_closest_vertex(game_board, mouse_pos, offset)
                        if vertex:
                            success, msg = game_system.try_place_initial_settlement(vertex)
                            print(msg)
                    else:
                        edge = find_closest_edge(game_board, mouse_pos, offset)
                        if edge:
                            success, msg = game_system.try_place_initial_road(edge)
                            print(msg)
                else:
                    # Check if player can build (must have rolled dice first)
                    if not game_system.can_trade_or_build():
                        add_message("Must roll dice first! (Press D)", (255, 100, 100), game_messages)
                    else:
                        if build_mode == "SETTLEMENT":
                            vertex = find_closest_vertex(game_board, mouse_pos, offset)
                            if vertex:
                                success, msg = current_player.try_build_settlement(vertex, False)
                                print(msg)
                                if success:
                                    add_message(msg, (100, 255, 100), game_messages)
                                else:
                                    add_message(msg, (255, 100, 100), game_messages)
                        elif build_mode == "CITY":
                            vertex = find_closest_vertex(game_board, mouse_pos, offset)
                            if vertex:
                                success, msg = current_player.try_build_city(vertex)
                                print(msg)
                                if success:
                                    add_message(msg, (100, 255, 100), game_messages)
                                else:
                                    add_message(msg, (255, 100, 100), game_messages)
                        elif build_mode == "ROAD":
                            edge = find_closest_edge(game_board, mouse_pos, offset)
                            if edge:
                                # Check if we're building free roads from Road Building card
                                if game_system.free_roads_remaining > 0:
                                    success, msg = game_system.try_build_free_road(edge)
                                else:
                                    success, msg = current_player.try_build_road(edge)
                                print(msg)
                                if success:
                                    add_message(msg, (100, 255, 100), game_messages)
                                else:
                                    add_message(msg, (255, 100, 100), game_messages)

        # Clear screen
        screen.fill((20, 50, 80))

        # Draw board
        draw_game_board(screen, game_board, offset, show_coords)

        # Draw messages at top of screen
        current_time = pygame.time.get_ticks()
        msg_y = 10
        for msg_text, msg_color, msg_time in game_messages[:]:
            if current_time - msg_time < 5000:  # Show for 5 seconds
                alpha = min(255, 255 - (current_time - msg_time) // 20)
                msg_surface = font.render(msg_text, True, msg_color)
                msg_surface.set_alpha(alpha)
                msg_rect = msg_surface.get_rect(center=(400, msg_y))

                # Background for readability
                bg_rect = pygame.Rect(msg_rect.x - 10, msg_rect.y - 5, msg_rect.width + 20, msg_rect.height + 10)
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
                bg_surface.set_alpha(150)
                bg_surface.fill((0, 0, 0))
                screen.blit(bg_surface, bg_rect)

                screen.blit(msg_surface, msg_rect)
                msg_y += 40
            else:
                game_messages.remove((msg_text, msg_color, msg_time))

        # Victory Screen Overlay
        if show_victory_screen and winner:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_W, SCREEN_H))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            # Victory box
            box_w, box_h = 600, 400
            box_x, box_y = (SCREEN_W - box_w) // 2, (SCREEN_H - box_h) // 2

            pygame.draw.rect(screen, (30, 30, 50), (box_x, box_y, box_w, box_h), border_radius=20)
            pygame.draw.rect(screen, (255, 215, 0), (box_x, box_y, box_w, box_h), 5, border_radius=20)

            # Title
            victory_font = pygame.font.SysFont(None, 72)
            title_text = victory_font.render("VICTORY!", True, (255, 215, 0))
            title_rect = title_text.get_rect(center=(SCREEN_W // 2, box_y + 60))
            screen.blit(title_text, title_rect)

            # Winner name
            winner_font = pygame.font.SysFont(None, 48)
            winner_text = winner_font.render(winner.name, True, winner.color)
            winner_rect = winner_text.get_rect(center=(SCREEN_W // 2, box_y + 130))
            screen.blit(winner_text, winner_rect)

            # Victory points
            vp_font = pygame.font.SysFont(None, 36)
            vp_text = vp_font.render(f"{winner.victory_points} Victory Points", True, (255, 255, 255))
            vp_rect = vp_text.get_rect(center=(SCREEN_W // 2, box_y + 180))
            screen.blit(vp_text, vp_rect)

            # Breakdown
            breakdown_font = pygame.font.SysFont(None, 24)
            breakdown_y = box_y + 220
            breakdown = winner.get_victory_point_breakdown()

            breakdown_title = breakdown_font.render("Point Breakdown:", True, (200, 200, 200))
            breakdown_title_rect = breakdown_title.get_rect(center=(SCREEN_W // 2, breakdown_y))
            screen.blit(breakdown_title, breakdown_title_rect)
            breakdown_y += 30

            for category, points in breakdown.items():
                text = breakdown_font.render(f"{category}: {points}", True, (255, 255, 255))
                text_rect = text.get_rect(center=(SCREEN_W // 2, breakdown_y))
                screen.blit(text, text_rect)
                breakdown_y += 25

            # Play again hint
            hint_font = pygame.font.SysFont(None, 20)
            hint_text = hint_font.render("Press R to play again or ESC to quit", True, (150, 150, 150))
            hint_rect = hint_text.get_rect(center=(SCREEN_W // 2, box_y + box_h - 30))
            screen.blit(hint_text, hint_rect)

            pygame.display.flip()
            continue

        if show_buildable and not game_system.is_initial_placement_phase():
            if build_mode == "SETTLEMENT":
                buildable = game_system.get_buildable_vertices_for_settlements()
                for vertex in buildable:
                    x, y = vertex.x + offset[0], vertex.y + offset[1]
                    # Pulsing effect
                    pulse = abs(math.sin(pygame.time.get_ticks() / 200)) * 3
                    pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), int(8 + pulse), 3)
            elif build_mode == "CITY":
                upgradeable = game_system.get_buildable_vertices_for_cities()
                for vertex in upgradeable:
                    x, y = vertex.x + offset[0], vertex.y + offset[1]
                    pulse = abs(math.sin(pygame.time.get_ticks() / 200)) * 3
                    pygame.draw.circle(screen, (0, 255, 255), (int(x), int(y)), int(10 + pulse), 3)
            elif build_mode == "ROAD":
                buildable = game_system.get_buildable_edges()
                for edge in buildable:
                    x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
                    x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]
                    pulse = abs(math.sin(pygame.time.get_ticks() / 200)) * 2
                    pygame.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), int(6 + pulse))

        # Draw UI Panel with better design
        panel_x = 850
        panel_width = 700

        # Draw semi-transparent background panel
        panel_surface = pygame.Surface((panel_width, SCREEN_H))
        panel_surface.set_alpha(200)
        panel_surface.fill((30, 30, 40))
        screen.blit(panel_surface, (panel_x, 0))

        y_pos = 20

        # Current Player - Large and prominent
        player_title = font_title.render(current_player.name, True, current_player.color)
        screen.blit(player_title, (panel_x + 20, y_pos))
        y_pos += 45

        # Victory Points - Big and clear with breakdown
        vp_value = current_player.calculate_victory_points()
        vp_text = font.render(f"Victory Points: {vp_value}/10", True,
                              (255, 215, 0) if vp_value >= 8 else (255, 255, 255))
        screen.blit(vp_text, (panel_x + 20, y_pos))
        y_pos += 25

        # VP Breakdown - Compact, only show non-zero
        breakdown = current_player.get_victory_point_breakdown()
        for category, points in breakdown.items():
            if points > 0:
                breakdown_text = small_font.render(f"  {category}: {points}", True, (200, 200, 200))
                screen.blit(breakdown_text, (panel_x + 30, y_pos))
                y_pos += 18

        y_pos += 7

        # Game Phase Indicator - Compact
        pygame.draw.line(screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
        y_pos += 10

        if game_system.is_initial_placement_phase():
            phase_text = "INITIAL PLACEMENT"
            phase_color = (255, 255, 0)
            phase_desc = game_system.get_current_player_needs()
        else:
            phase_text = f"TURN {game_system.turn_number}"
            phase_color = (100, 255, 100)
            if game_system.turn_phase == "ROLL_DICE":
                phase_desc = "▶ Roll dice (D)"
            else:
                phase_desc = "▶ Trade & Build (T=End)"

        phase_surface = font.render(phase_text, True, phase_color)
        screen.blit(phase_surface, (panel_x + 20, y_pos))
        y_pos += 25

        desc_surface = small_font.render(phase_desc, True, (200, 200, 200))
        screen.blit(desc_surface, (panel_x + 20, y_pos))
        y_pos += 30

        # Dice Roll Display - Compact
        if game_system.last_dice_roll and game_system.game_phase == "NORMAL_PLAY":
            pygame.draw.rect(screen, (50, 50, 60), (panel_x + 20, y_pos, panel_width - 40, 65), border_radius=10)
            pygame.draw.rect(screen, (255, 215, 0), (panel_x + 20, y_pos, panel_width - 40, 65), 3, border_radius=10)

            dice_title = small_font.render("LAST ROLL", True, (255, 215, 0))
            screen.blit(dice_title, (panel_x + 30, y_pos + 8))

            die1, die2, total = game_system.last_dice_roll

            # Draw dice visually
            dice_x = panel_x + 30
            dice_y = y_pos + 30

            # Die 1
            pygame.draw.rect(screen, (255, 255, 255), (dice_x, dice_y, 30, 30), border_radius=5)
            die1_text = font.render(str(die1), True, (0, 0, 0))
            screen.blit(die1_text, (dice_x + 10, dice_y + 5))

            # Plus sign
            plus_text = small_font.render("+", True, (255, 255, 255))
            screen.blit(plus_text, (dice_x + 38, dice_y + 5))

            # Die 2
            pygame.draw.rect(screen, (255, 255, 255), (dice_x + 60, dice_y, 30, 30), border_radius=5)
            die2_text = font.render(str(die2), True, (0, 0, 0))
            screen.blit(die2_text, (dice_x + 70, dice_y + 5))

            # Equals sign
            equals_text = small_font.render("=", True, (255, 255, 255))
            screen.blit(equals_text, (dice_x + 98, dice_y + 5))

            # Total
            total_color = (255, 0, 0) if total in [6, 8] else (255, 255, 255)
            total_text = font.render(str(total), True, total_color)
            screen.blit(total_text, (dice_x + 120, dice_y + 5))

            y_pos += 70

        pygame.draw.line(screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
        y_pos += 15

        # Action Buttons Section
        buttons.clear()

        if not game_system.is_initial_placement_phase():
            actions_title = font.render("ACTIONS", True, (100, 200, 255))
            screen.blit(actions_title, (panel_x + 20, y_pos))
            y_pos += 35

            button_width = 140
            button_height = 40
            button_spacing = 10
            button_x = panel_x + 30

            # Roll Dice Button
            can_roll = game_system.can_roll_dice()
            rect, hover = draw_button(screen, button_x, y_pos, button_width, button_height,
                                      "ROLL DICE", (0, 120, 0), (0, 180, 0), small_font, can_roll)
            buttons["ROLL_DICE"] = (rect, can_roll)

            # End Turn Button
            can_end = game_system.can_end_turn()
            rect, hover = draw_button(screen, button_x + button_width + button_spacing, y_pos, button_width,
                                      button_height,
                                      "END TURN", (120, 0, 0), (180, 0, 0), small_font, can_end)
            buttons["END_TURN"] = (rect, can_end)
            y_pos += button_height + 15

            # Building Buttons
            buttons_row = [
                ("BUILD_SETTLEMENT", "Settlement", build_mode == "SETTLEMENT"),
                ("BUILD_CITY", "City", build_mode == "CITY"),
                ("BUILD_ROAD", "Road", build_mode == "ROAD")
            ]

            button_width_small = 90
            for i, (btn_name, btn_text, is_selected) in enumerate(buttons_row):
                btn_color = (100, 100, 0) if is_selected else (60, 60, 60)
                hover_color = (150, 150, 0) if is_selected else (100, 100, 100)
                rect, hover = draw_button(screen, button_x + i * (button_width_small + 5), y_pos,
                                          button_width_small, 35, btn_text, btn_color, hover_color, small_font, True)
                buttons[btn_name] = (rect, True)
            y_pos += 45

            # Other Action Buttons
            rect, hover = draw_button(screen, button_x, y_pos, button_width, 35,
                                      "Buy Dev Card", (80, 0, 80), (120, 0, 120), small_font, True)
            buttons["BUY_DEV_CARD"] = (rect, True)

            rect, hover = draw_button(screen, button_x + button_width + button_spacing, y_pos, button_width, 35,
                                      "Trade Mode", (80, 40, 80) if trade_mode else (40, 80, 80),
                                      (120, 60, 120) if trade_mode else (60, 120, 120), small_font, True)
            buttons["BANK_TRADE"] = (rect, True)
            y_pos += 45

            # Show Buildable Toggle
            rect, hover = draw_button(screen, button_x, y_pos, button_width, 35,
                                      "Buildable: " + ("ON" if show_buildable else "OFF"),
                                      (0, 80, 0) if show_buildable else (60, 60, 60),
                                      (0, 120, 0) if show_buildable else (100, 100, 100), small_font, True)
            buttons["SHOW_BUILDABLE"] = (rect, True)
            y_pos += 50

        # Resources Section
        resources_title = font.render("RESOURCES", True, (100, 200, 255))
        screen.blit(resources_title, (panel_x + 20, y_pos))
        y_pos += 30

        resource_icons = {
            ResourceType.WOOD: "🌲",
            ResourceType.BRICK: "🧱",
            ResourceType.WHEAT: "🌾",
            ResourceType.SHEEP: "🐑",
            ResourceType.ORE: "⛰️"
        }

        for res_type, amount in current_player.resources.items():
            res_name = res_type.value.title()
            color = (255, 255, 255) if amount > 0 else (100, 100, 100)

            # Resource name and count
            text = small_font.render(f"{res_name}: {amount}", True, color)
            screen.blit(text, (panel_x + 30, y_pos))

            # Draw small bar to visualize amount
            if amount > 0:
                bar_width = min(amount * 20, 200)
                pygame.draw.rect(screen, color, (panel_x + 150, y_pos + 5, bar_width, 10))

            y_pos += 20

        y_pos += 5
        pygame.draw.line(screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
        y_pos += 10

        # Buildings Section - Compact
        buildings_title = font.render("BUILDINGS", True, (100, 200, 255))
        screen.blit(buildings_title, (panel_x + 20, y_pos))
        y_pos += 25

        buildings_text = small_font.render(
            f"Settlements:{len(current_player.settlements)}/5  Cities:{len(current_player.cities)}/4  Roads:{len(current_player.roads)}/15",
            True, (255, 255, 255))
        screen.blit(buildings_text, (panel_x + 30, y_pos))
        y_pos += 20

        # Dev Cards - Compact
        total_cards = sum(current_player.development_cards.values())
        dev_info = f"Dev Cards:{total_cards}"
        if current_player.knights_played > 0:
            dev_info += f"  Knights:{current_player.knights_played}"

        dev_text = small_font.render(dev_info, True, (255, 255, 255))
        screen.blit(dev_text, (panel_x + 30, y_pos))
        y_pos += 20

        y_pos += 5
        pygame.draw.line(screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
        y_pos += 10

        # Development Cards Display - Skip if empty to save space
        if show_dev_card_menu and any(current_player.development_cards.values()):
            dev_y = y_pos
            dev_title = small_font.render("DEV CARDS", True, (255, 200, 100))
            screen.blit(dev_title, (panel_x + 20, dev_y))
            dev_y += 20

            for card_type, count in current_player.development_cards.items():
                if count > 0:
                    card_name = card_type.value.replace('_', ' ').title()
                    card_text = small_font.render(f"{card_name}: {count}", True, (255, 255, 255))
                    screen.blit(card_text, (panel_x + 30, dev_y))
                    dev_y += 18

            y_pos = dev_y + 5

        # Free Roads Indicator - Compact
        if game_system.free_roads_remaining > 0:
            free_roads_y = y_pos
            pygame.draw.rect(screen, (0, 100, 0), (panel_x + 20, free_roads_y, panel_width - 40, 40), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 0), (panel_x + 20, free_roads_y, panel_width - 40, 40), 3, border_radius=8)

            free_roads_title = small_font.render(f"FREE ROADS: {game_system.free_roads_remaining}", True, (100, 255, 100))
            screen.blit(free_roads_title, (panel_x + 30, free_roads_y + 8))

            free_roads_desc = small_font.render("Click edges to build", True, (200, 255, 200))
            screen.blit(free_roads_desc, (panel_x + 30, free_roads_y + 24))

            y_pos = free_roads_y + 45

        # Robber Mode Indicator - Compact
        if robber_move_mode:
            robber_y = y_pos
            pygame.draw.rect(screen, (100, 0, 0), (panel_x + 20, robber_y, panel_width - 40, 40), border_radius=8)
            pygame.draw.rect(screen, (255, 0, 0), (panel_x + 20, robber_y, panel_width - 40, 40), 3, border_radius=8)

            robber_title = small_font.render("ROBBER MODE", True, (255, 100, 100))
            screen.blit(robber_title, (panel_x + 30, robber_y + 8))

            robber_desc = small_font.render("Click tile to move robber", True, (255, 200, 200))
            screen.blit(robber_desc, (panel_x + 30, robber_y + 24))

            y_pos = robber_y + 45

        pygame.draw.line(screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
        y_pos += 15

        # Controls Section - Compact 2-column layout
        controls_title = font.render("CONTROLS", True, (100, 200, 255))
        screen.blit(controls_title, (panel_x + 20, y_pos))
        y_pos += 25

        if trade_mode:
            controls = [
                ("5", "Exit", "↑WASQ", "Offer"),
                ("←→", "Partner", "↓EDFZ", "Request"),
                ("ENTER", "Trade", "", "")
            ]
        elif robber_move_mode:
            controls = [
                ("Click", "Move robber", "", "")
            ]
        elif game_system.is_initial_placement_phase():
            controls = [
                ("Click", "Place", "T", "Next")
            ]
        else:
            controls = [
                ("D", "Roll", "T", "End Turn"),
                ("1", "Settlement", "2", "City"),
                ("3", "Road", "X", "Dev Card"),
                ("B", "Buildable", "5", "Trade"),
                ("7", "Knight", "8", "Plenty"),
                ("9", "Monopoly", f"Mode: {build_mode}", "")
            ]

        for i, control_pair in enumerate(controls):
            if len(control_pair) >= 2:
                key1, desc1 = control_pair[0], control_pair[1]
                key2, desc2 = control_pair[2] if len(control_pair) > 2 else "", control_pair[3] if len(control_pair) > 3 else ""

                # Left column
                if key1 and desc1:
                    key_text = small_font.render(key1, True, (255, 255, 0))
                    screen.blit(key_text, (panel_x + 25, y_pos))
                    desc_text = small_font.render(desc1, True, (200, 200, 200))
                    screen.blit(desc_text, (panel_x + 85, y_pos))
                elif key1:
                    mode_text = small_font.render(key1, True, (100, 255, 100))
                    screen.blit(mode_text, (panel_x + 25, y_pos))

                # Right column
                if key2 and desc2:
                    key_text = small_font.render(key2, True, (255, 255, 0))
                    screen.blit(key_text, (panel_x + 350, y_pos))
                    desc_text = small_font.render(desc2, True, (200, 200, 200))
                    screen.blit(desc_text, (panel_x + 410, y_pos))
                elif key2:
                    mode_text = small_font.render(key2, True, (100, 255, 100))
                    screen.blit(mode_text, (panel_x + 350, y_pos))

                y_pos += 20

        # FLOATING TRADE POPUP - Top Right Area Overlay
        if trade_mode:
            # Position in top-right area (not edge)
            popup_width = 400
            popup_height = 300
            popup_x = SCREEN_W - popup_width - 80  # More margin from right edge
            popup_y = 50  # More margin from top

            # Semi-transparent background overlay
            overlay = pygame.Surface((SCREEN_W, SCREEN_H))
            overlay.set_alpha(100)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            # Trade popup box
            pygame.draw.rect(screen, (40, 30, 60), (popup_x, popup_y, popup_width, popup_height), border_radius=15)
            pygame.draw.rect(screen, (200, 100, 255), (popup_x, popup_y, popup_width, popup_height), 4, border_radius=15)

            trade_y = popup_y + 15

            # Title
            trade_title = font.render("TRADE MODE", True, (255, 200, 255))
            screen.blit(trade_title, (popup_x + 15, trade_y))

            # Close hint
            close_hint = small_font.render("(Press 5 to close)", True, (150, 150, 150))
            screen.blit(close_hint, (popup_x + popup_width - 140, trade_y + 5))
            trade_y += 35

            partners = game_system.get_available_trade_partners(current_player)
            if partners:
                partner = partners[selected_trade_partner]
                partner_text = small_font.render(f"Trading with: {partner.name}", True, partner.color)
                screen.blit(partner_text, (popup_x + 15, trade_y))

                partner_hint = small_font.render("(←/→ to change partner)", True, (150, 150, 150))
                screen.blit(partner_hint, (popup_x + 15, trade_y + 18))
                trade_y += 50

                # Two columns
                col1_x = popup_x + 15
                col2_x = popup_x + 210

                # YOU OFFER column
                offer_title = font.render("YOU OFFER", True, (255, 180, 180))
                screen.blit(offer_title, (col1_x, trade_y))

                request_title = font.render("YOU REQUEST", True, (180, 255, 180))
                screen.blit(request_title, (col2_x, trade_y))
                trade_y += 28

                resource_info = [
                    (ResourceType.WOOD, "Wood", "↑", "↓"),
                    (ResourceType.BRICK, "Brick", "W", "E"),
                    (ResourceType.WHEAT, "Wheat", "A", "D"),
                    (ResourceType.SHEEP, "Sheep", "S", "F"),
                    (ResourceType.ORE, "Ore", "Q", "Z")
                ]

                for res_type, name, offer_key, req_key in resource_info:
                    offer_amt = offering_resources[res_type]
                    req_amt = requesting_resources[res_type]

                    # Offer column
                    color = (255, 255, 255) if offer_amt > 0 else (120, 120, 120)
                    offer_text = small_font.render(f"[{offer_key}] {name}: {offer_amt}", True, color)
                    screen.blit(offer_text, (col1_x, trade_y))

                    # Request column
                    color = (255, 255, 255) if req_amt > 0 else (120, 120, 120)
                    req_text = small_font.render(f"[{req_key}] {name}: {req_amt}", True, color)
                    screen.blit(req_text, (col2_x, trade_y))

                    trade_y += 22

                # Execute button
                trade_y += 10
                execute_bg = pygame.Rect(popup_x + 15, trade_y, popup_width - 30, 30)
                pygame.draw.rect(screen, (80, 0, 80), execute_bg, border_radius=8)
                pygame.draw.rect(screen, (255, 255, 0), execute_bg, 3, border_radius=8)

                execute_text = font.render("Press ENTER to Execute Trade", True, (255, 255, 0))
                execute_rect = execute_text.get_rect(center=execute_bg.center)
                screen.blit(execute_text, execute_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()