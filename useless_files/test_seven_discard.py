"""
Test script to verify the 7-rolled discard mechanics work correctly
Tests automatic discarding when a player has 8+ cards and a 7 is rolled
"""

from ai_interface import AIGameEnvironment
from game_system import ResourceType
import random


def test_seven_discard_mechanics():
    """Test that the 7-rolled discard system works correctly"""
    print("="*60)
    print("TESTING 7-ROLLED DISCARD MECHANICS")
    print("="*60)

    # Test 1: Environment creation
    print("\n[TEST 1] Creating environment...")
    try:
        env = AIGameEnvironment()
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    # Test 2: Observation includes discard status
    print("\n[TEST 2] Checking discard status in observations...")
    try:
        observations = env.reset()
        obs = observations[0]

        required_discard_keys = ['waiting_for_discards', 'must_discard',
                                  'must_discard_count', 'players_discarding']
        for key in required_discard_keys:
            assert key in obs, f"Missing discard key: {key}"

        print("✅ Observation includes all discard status fields")
        print(f"   waiting_for_discards: {obs['waiting_for_discards']}")
        print(f"   must_discard: {obs['must_discard']}")
        print(f"   must_discard_count: {obs['must_discard_count']}")
        print(f"   players_discarding: {obs['players_discarding']}")
    except Exception as e:
        print(f"❌ Invalid observation structure: {e}")
        return False

    # Test 3: Give a player 8+ cards and trigger discard
    print("\n[TEST 3] Testing discard trigger with 8+ cards...")
    try:
        # Skip initial placement by advancing game state
        env.game.game_phase = "NORMAL_PLAY"
        env.game.turn_phase = "ROLL_DICE"
        env.game.dice_rolled = False

        # Give player 0 exactly 10 cards
        player = env.game.players[0]
        player.resources[ResourceType.WOOD] = 3
        player.resources[ResourceType.BRICK] = 3
        player.resources[ResourceType.WHEAT] = 2
        player.resources[ResourceType.SHEEP] = 1
        player.resources[ResourceType.ORE] = 1

        total_cards = sum(player.resources.values())
        print(f"   Player 0 has {total_cards} cards")
        assert total_cards == 10, "Player should have exactly 10 cards"

        # Mock rolling a 7
        env.game.last_dice_roll = (3, 4, 7)
        env.game.dice_rolled = True
        env.game.turn_phase = "TRADE_BUILD"

        # Manually trigger discard check (normally done by roll_dice)
        env.game.players_must_discard = env.game.get_players_who_must_discard()
        if env.game.players_must_discard:
            env.game.waiting_for_discards = True

        # Should be waiting for discards
        assert env.game.waiting_for_discards == True, "Game should be waiting for discards"
        assert player in env.game.players_must_discard, "Player 0 should need to discard"

        print(f"✅ Discard triggered correctly")
        print(f"   Players that must discard: {len(env.game.players_must_discard)}")

    except Exception as e:
        print(f"❌ Failed to trigger discard: {e}")
        return False

    # Test 4: Automatic discard execution
    print("\n[TEST 4] Testing automatic discard execution...")
    try:
        cards_before = sum(player.resources.values())
        expected_after = cards_before // 2  # Should discard half

        # Trigger automatic discards
        env._handle_automatic_discards()

        cards_after = sum(player.resources.values())

        assert cards_after == expected_after, f"Should have {expected_after} cards after discard, got {cards_after}"
        assert not env.game.waiting_for_discards, "Should no longer be waiting for discards"

        print(f"✅ Automatic discard executed correctly")
        print(f"   Cards before: {cards_before}")
        print(f"   Cards after: {cards_after}")
        print(f"   Cards discarded: {cards_before - cards_after}")

    except Exception as e:
        print(f"❌ Failed automatic discard: {e}")
        return False

    # Test 5: Verify discard math (half rounded down)
    print("\n[TEST 5] Testing discard math for different card counts...")
    try:
        test_cases = [
            (8, 4),   # 8 cards -> discard 4
            (9, 4),   # 9 cards -> discard 4 (rounds down)
            (10, 5),  # 10 cards -> discard 5
            (15, 7),  # 15 cards -> discard 7
        ]

        for total, expected_discard in test_cases:
            # Reset player cards
            env.game.players_discarded = set()
            env.game.waiting_for_discards = False

            player.resources[ResourceType.WOOD] = total
            player.resources[ResourceType.BRICK] = 0
            player.resources[ResourceType.WHEAT] = 0
            player.resources[ResourceType.SHEEP] = 0
            player.resources[ResourceType.ORE] = 0

            # Trigger discard
            env.game.players_must_discard = env.game.get_players_who_must_discard()
            env.game.waiting_for_discards = True

            cards_before = sum(player.resources.values())
            env._handle_automatic_discards()
            cards_after = sum(player.resources.values())

            actual_discarded = cards_before - cards_after
            assert actual_discarded == expected_discard, \
                f"With {total} cards, should discard {expected_discard}, but discarded {actual_discarded}"

        print(f"✅ Discard math correct for all test cases")
        for total, expected in test_cases:
            print(f"   {total} cards → discard {expected}")

    except Exception as e:
        print(f"❌ Failed discard math test: {e}")
        return False

    # Test 6: Robber movement after discards
    print("\n[TEST 6] Testing robber movement after discards...")
    try:
        # Set up test
        env.game.players_discarded = set()
        player.resources[ResourceType.WOOD] = 8
        player.resources[ResourceType.BRICK] = 0
        player.resources[ResourceType.WHEAT] = 0
        player.resources[ResourceType.SHEEP] = 0
        player.resources[ResourceType.ORE] = 0

        # Trigger discard
        env.game.players_must_discard = env.game.get_players_who_must_discard()
        env.game.waiting_for_discards = True

        robber_position_before = env.game.robber.position

        # Execute automatic discards (should also move robber)
        env._handle_automatic_discards()

        robber_position_after = env.game.robber.position

        # Robber should have moved to a different tile
        assert robber_position_after != robber_position_before, \
            "Robber should move to a different position after discards"

        print(f"✅ Robber moved automatically after discards")
        print(f"   Robber position before: {robber_position_before}")
        print(f"   Robber position after: {robber_position_after}")

    except Exception as e:
        print(f"❌ Failed robber movement test: {e}")
        return False

    # Test 7: Observation updates during discard phase
    print("\n[TEST 7] Testing observation updates during discard phase...")
    try:
        # Set up discard scenario
        env.game.players_discarded = set()
        player.resources[ResourceType.WOOD] = 10
        player.resources[ResourceType.BRICK] = 0
        player.resources[ResourceType.WHEAT] = 0
        player.resources[ResourceType.SHEEP] = 0
        player.resources[ResourceType.ORE] = 0

        env.game.players_must_discard = env.game.get_players_who_must_discard()
        env.game.waiting_for_discards = True

        # Get observation before discard
        obs_before = env.get_observation(0)
        assert obs_before['waiting_for_discards'] == True
        assert obs_before['must_discard'] == True
        assert obs_before['must_discard_count'] == 5

        # Execute discard
        env._handle_automatic_discards()

        # Get observation after discard
        obs_after = env.get_observation(0)
        assert obs_after['waiting_for_discards'] == False
        assert obs_after['must_discard'] == False
        assert obs_after['must_discard_count'] == 0

        print(f"✅ Observations update correctly during discard phase")
        print(f"   Before: waiting={obs_before['waiting_for_discards']}, must={obs_before['must_discard']}")
        print(f"   After: waiting={obs_after['waiting_for_discards']}, must={obs_after['must_discard']}")

    except Exception as e:
        print(f"❌ Failed observation update test: {e}")
        return False

    print("\n" + "="*60)
    print("✅ ALL 7-ROLLED DISCARD TESTS PASSED")
    print("="*60)
    return True


if __name__ == '__main__':
    success = test_seven_discard_mechanics()
    exit(0 if success else 1)
