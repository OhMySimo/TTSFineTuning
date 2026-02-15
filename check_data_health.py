import json
import os
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# CONFIG
JSONL_PATH = "train_with_codes.jsonl"
CHECK_AUDIO_FILES = True  # Set to False if you only want to check structure
MAX_AUDIO_CHECKS = 1000   # Check first N audio files to save time (set None for all)

def check_line(line_str):
    try:
        data = json.loads(line_str)
        
        # 1. Check Structure
        if 'audio_codes' not in data:
            return "MISSING_CODES"
        
        codes = np.array(data['audio_codes'])
        if codes.ndim != 2 or codes.shape[1] != 16:
            return f"BAD_SHAPE_{codes.shape}"
            
        if codes.shape[0] == 0:
            return "EMPTY_AUDIO"

        # 2. Check Audio File (Optional & Expensive)
        if CHECK_AUDIO_FILES and 'ref_audio' in data:
            path = data['ref_audio']
            if not os.path.exists(path):
                return f"MISSING_FILE: {path}"
            
            # Only checking existence here. 
            # Full librosa load is too slow for 100k files in this script.
            # We trust prepare_data.py did its job, but we verify the file exists.
            
        return None  # All good

    except json.JSONDecodeError:
        return "JSON_ERROR"
    except Exception as e:
        return f"UNKNOWN_ERROR: {str(e)}"

def main():
    print(f"🔍 Scanning {JSONL_PATH}...")
    
    errors = []
    line_count = 0
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📄 Found {len(lines)} lines. Checking integrity...")
    
    # Use parallel processing for speed
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(check_line, lines), total=len(lines)))
    
    for i, res in enumerate(results):
        if res is not None:
            errors.append(f"Line {i+1}: {res}")
            
    print("\n" + "="*40)
    if not errors:
        print("✅ PASSED! Data looks clean.")
        print(f"   - checked {len(lines)} samples")
        print(f"   - all have audio_codes")
        print(f"   - all ref_audio paths exist")
    else:
        print(f"❌ FAILED! Found {len(errors)} issues.")
        print("First 10 errors:")
        for e in errors[:10]:
            print(e)
        print("Fix these before training!")
    print("="*40)

if __name__ == "__main__":
    main()