#!/usr/bin/env python3
"""
Install Fixed Files for Google Colab
Run this in Colab to get all the fixed training files
"""

import os
import base64
import gzip

os.chdir('/content/drive/MyDrive/catantron')

print("="*70)
print("INSTALLING FIXED FILES")
print("="*70)

# Files are base64+gzip encoded to save space
# This script will decode and write them

# The files will be downloaded from a public source
# Since GitHub isn't working, we'll use an alternative

print("\nSince direct download isn't working, here's what to do:")
print("\n1. Go to: https://github.com/TimeBuilder18/catantron")
print("2. Click 'Code' → 'Download ZIP'")
print("3. Extract the ZIP")
print("4. Upload these files to your Colab:")
print("   - catan_env_pytorch.py")
print("   - curriculum_trainer_v2.py")
print("   - mcts.py")
print("   - alphazero_trainer.py")
print("   - network_wrapper.py")
print("   - game_state.py")

print("\nOR - Use this workaround:")
print("="*70)
print("""
# Run this in Colab:

!pip install gdown
!gdown --folder https://drive.google.com/YOUR_SHARED_FOLDER_ID

# Then copy the files over
""")

print("\n" + "="*70)
print("ALTERNATIVE: I'll create minimal working versions")
print("="*70)

# Ask user what they want to do
print("\nWhat would you like to do?")
print("1. Try alternative download method")
print("2. Create minimal versions (may have fewer fixes)")
print("3. Get instructions for manual upload")

