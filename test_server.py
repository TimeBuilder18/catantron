"""
Quick test to see if server starts properly
"""
import sys
import time

print("=" * 60)
print("TESTING SERVER STARTUP")
print("=" * 60)

print("\n[1/5] Importing modules...")
try:
    from game_server import GameServer
    print("    ✓ Imported game_server")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

print("\n[2/5] Creating GameServer instance...")
try:
    server = GameServer()
    print("    ✓ Created server instance")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

print("\n[3/5] Initializing game (this might take a moment)...")
try:
    start = time.time()
    server.initialize_game()
    elapsed = time.time() - start
    print(f"    ✓ Game initialized in {elapsed:.2f}s")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

print("\n[4/5] Starting server socket...")
try:
    import socket
    server.running = True
    server.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.server_socket.bind((server.host, server.port))
    server.server_socket.listen(4)
    print(f"    ✓ Server listening on {server.host}:{server.port}")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

print("\n[5/5] Testing if we can serialize game state...")
try:
    state = server.serialize_game_state(0)
    import json
    data = json.loads(state)
    print(f"    ✓ Game state serialized successfully")
    print(f"    ✓ State contains {len(data['tiles'])} tiles")
    print(f"    ✓ State contains {len(data['ports'])} ports")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Server can start properly!")
print("=" * 60)
print("\nNow waiting for 1 client to connect (10 second timeout)...")
print("In another terminal, try: python3 -c 'import socket; s=socket.socket(); s.connect((\"127.0.0.1\", 5555)); print(\"Connected!\")'")

server.server_socket.settimeout(10)
try:
    client_socket, address = server.server_socket.accept()
    print(f"\n✓ Client connected from {address}!")
    client_socket.close()
except socket.timeout:
    print("\n⚠ No client connected (timeout)")

server.server_socket.close()
print("\n✓ Test complete!")
