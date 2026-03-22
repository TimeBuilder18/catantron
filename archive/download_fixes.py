"""
Download fixed files directly from GitHub
No authentication needed!
"""

import urllib.request
import os

# Your GitHub repo with fixes
REPO_URL = "https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ"

# Files to download (with fixes)
FILES_TO_UPDATE = [
    "curriculum_trainer_v2.py",
    "catan_env_pytorch.py",
    "mcts.py",
    "alphazero_trainer.py",
    "FIXES_APPLIED.md",
    "UPDATE_COLAB.md",
    "COLAB_FIX.md",
]

def download_file(filename):
    """Download a file from GitHub"""
    url = f"{REPO_URL}/{filename}"
    print(f"Downloading {filename}...", end=" ")

    try:
        # Download
        urllib.request.urlretrieve(url, filename)
        print("✅")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def backup_file(filename):
    """Backup existing file"""
    if os.path.exists(filename):
        backup_name = f"{filename}.backup"
        os.rename(filename, backup_name)
        print(f"  Backed up {filename} → {backup_name}")

def main():
    print("=" * 70)
    print("DOWNLOADING FIXED FILES FROM GITHUB")
    print("=" * 70)
    print(f"Source: {REPO_URL}\n")

    success_count = 0

    for filename in FILES_TO_UPDATE:
        # Backup original
        if os.path.exists(filename) and not filename.endswith('.md'):
            backup_file(filename)

        # Download new version
        if download_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"COMPLETE: {success_count}/{len(FILES_TO_UPDATE)} files updated")
    print("=" * 70)

    if success_count == len(FILES_TO_UPDATE):
        print("\n✅ All fixes downloaded successfully!")
        print("\nYou can now run:")
        print("  python curriculum_trainer_v2.py --games-per-phase 300")
    else:
        print("\n⚠️  Some files failed to download.")
        print("Your original files have been backed up with .backup extension")
        print("\nTry again or check your internet connection.")

if __name__ == "__main__":
    main()
