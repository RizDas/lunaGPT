"""
Manual GPT-2 Model Download Script
This downloads the model files directly from Hugging Face
"""

import requests
import os
from pathlib import Path

# Create directory for model
model_dir = Path('./gpt2_downloaded')
model_dir.mkdir(exist_ok=True)

print(f"Downloading GPT-2 model files to: {model_dir.absolute()}")

# Files to download from Hugging Face
files_to_download = {
    'config.json': 'https://huggingface.co/gpt2/resolve/main/config.json',
    'pytorch_model.bin': 'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin',
    'vocab.json': 'https://huggingface.co/gpt2/resolve/main/vocab.json',
    'merges.txt': 'https://huggingface.co/gpt2/resolve/main/merges.txt',
    'tokenizer.json': 'https://huggingface.co/gpt2/resolve/main/tokenizer.json',
}

def download_file(url, filename):
    """Download a file with progress indication"""
    filepath = model_dir / filename
    
    print(f"\nDownloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                total_size_mb = total_size / (1024 * 1024)
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        downloaded_mb = downloaded / (1024 * 1024)
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {downloaded_mb:.1f}/{total_size_mb:.1f} MB ({percent:.1f}%)", end='')
        
        print(f"\n  ✓ {filename} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error downloading {filename}: {e}")
        return False

# Download all files
print("="*60)
success_count = 0
for filename, url in files_to_download.items():
    if download_file(url, filename):
        success_count += 1

print("\n" + "="*60)
print(f"\nDownload complete: {success_count}/{len(files_to_download)} files downloaded")

if success_count == len(files_to_download):
    print(f"\n✓ All files downloaded successfully to: {model_dir.absolute()}")
    print("\nYou can now load the model using:")
    print(f"  model = GPT2LMHeadModel.from_pretrained('{model_dir.absolute()}')")
else:
    print("\n✗ Some files failed to download. Check your internet connection.")