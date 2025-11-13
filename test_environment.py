"""
Test script to verify the AI training environment works
No AI implementation - just tests the game environment
"""

from ai_interface import AIGameEnvironment

def test_environment():
    """Test that the environment initializes and provides observations"""
    print("="*60)
    print("TESTING CATAN AI ENVIRONMENT")
    print("="*60)

    # Test 1: Environment creation
    print("\n[TEST 1] Creating environment...")
    try:
        env = AIGameEnvironment()
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    # Test 2: Reset and get initial observations
    print("\n[TEST 2] Getting initial observations...")
    try:
        observations = env.reset()
        assert len(observations) == 4, "Should have 4 player observations"
        print(f"✅ Got {len(observations)} player observations")
    except Exception as e:
        print(f"❌ Failed to get observations: {e}")
        return False

    # Test 3: Check observation structure
    print("\n[TEST 3] Checking observation structure...")
    try:
        obs = observations[0]
        required_keys = ['is_my_turn', 'game_phase', 'my_resources', 'legal_actions']
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"
        print(f"✅ Observation has all required keys")
        print(f"   Sample observation keys: {list(obs.keys())}")
    except Exception as e:
        print(f"❌ Invalid observation structure: {e}")
        return False

    # Test 4: Check legal actions
    print("\n[TEST 4] Checking legal actions...")
    try:
        obs = observations[0]  # Player 1's observation
        legal_actions = obs['legal_actions']
        print(f"   Player 1 can: {legal_actions}")

        # During initial placement, should be able to place settlement
        assert 'place_settlement' in legal_actions or 'place_road' in legal_actions
        print(f"✅ Legal actions are provided correctly")
    except Exception as e:
        print(f"❌ Legal actions check failed: {e}")
        return False

    # Test 5: Print detailed observation for player 1
    print("\n[TEST 5] Sample observation for Player 1:")
    obs = observations[0]
    print(f"   • Is my turn: {obs['is_my_turn']}")
    print(f"   • Game phase: {obs['game_phase']}")
    print(f"   • Turn phase: {obs['turn_phase']}")
    print(f"   • My resources: {obs['my_resources']}")
    print(f"   • My buildings: {obs['my_settlements']} settlements, {obs['my_cities']} cities, {obs['my_roads']} roads")
    print(f"   • Victory points: {obs['my_victory_points']}")
    print(f"   • Legal actions: {obs['legal_actions']}")
    print(f"   • Number of opponents: {len(obs['opponents'])}")

    # Test 6: Access game internals
    print("\n[TEST 6] Accessing game state...")
    try:
        current_player = env.game.get_current_player()
        print(f"   • Current player: {current_player.name}")
        print(f"   • Total tiles: {len(env.game.game_board.tiles)}")
        print(f"   • Total ports: {len(env.game.game_board.ports)}")
        print(f"   • Total vertices: {len(env.game.game_board.vertices)}")
        print(f"   • Total edges: {len(env.game.game_board.edges)}")
        print(f"✅ Can access game internals")
    except Exception as e:
        print(f"❌ Failed to access game state: {e}")
        return False

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - ENVIRONMENT IS READY!")
    print("="*60)
    print("\nYou can now:")
    print("1. Import AIGameEnvironment from ai_interface")
    print("2. Create your AI agents")
    print("3. Start training!")
    print("\nExample:")
    print("  from ai_interface import AIGameEnvironment")
    print("  env = AIGameEnvironment()")
    print("  observations = env.reset()")
    print("  # ... your AI training loop ...")
    print("="*60)

    return True


if __name__ == "__main__":
    success = test_environment()
    exit(0 if success else 1)
