"""Test script to verify output suppression works correctly"""
import os
import sys
import io

# Fix Windows encoding issues with emojis
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')


class NullWriter:
    """A file-like object that discards all output - Windows compatible"""
    def write(self, text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


class SuppressOutput:
    """Context manager to suppress all stdout/stderr output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        # Use custom NullWriter instead of os.devnull for better Windows compatibility
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


print("=" * 50)
print("Testing output suppression...")
print("=" * 50)

print("\n1. Normal output (visible):")
print("   This line should be visible")

print("\n2. Suppressed output (not visible):")
with SuppressOutput():
    print("   [DEBUG] This should NOT appear")
    print("   [RULE AI] This should NOT appear")
    print("   🏠 🌾 🐑 Emoji test - should NOT appear")
    for i in range(5):
        print(f"   Loop {i} - should NOT appear")

print("   Back to normal - this SHOULD be visible")

print("\n3. Testing with emojis in normal mode:")
print("   Emoji test: 🏠 🌾 🐑 🪨 🧱 (should be visible or replaced)")

print("\n4. Testing imports with suppression:")
with SuppressOutput():
    # Simulate importing a module that prints during import
    print("This is during import - should NOT appear")

print("   Import complete - this SHOULD be visible")

print("\n" + "=" * 50)
print("Test complete! If you only see numbered sections,")
print("the suppression is working correctly.")
print("=" * 50)
