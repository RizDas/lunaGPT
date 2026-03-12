"""
Simple GPT-2 Model Downloader

"""

from transformers import GPT2LMHeadModel
import os

print("="*70)
print("Downloading GPT-2 Model Weights")
print("="*70)

# Download to a 'model_cache' folder in the current gpt2 directory
cache_dir = './model_cache'
os.makedirs(cache_dir, exist_ok=True)

print(f"\nDownloading to: {os.path.abspath(cache_dir)}")
print("This will download ~500MB and may take a few minutes...\n")

try:
    model = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        cache_dir=cache_dir,
        resume_download=True
    )
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Model downloaded and cached.")
    print("="*70)
    print(f"\nCache location: {os.path.abspath(cache_dir)}")
    print("\nYou can now run your training scripts!")
    
except Exception as e:
    print("\n" + "="*70)
    print("✗ Download failed")
    print("="*70)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try: pip install --upgrade transformers")
    print("3. Check if you're behind a firewall/proxy")