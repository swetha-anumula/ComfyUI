# manual_merge.py
# A script to consolidate sharded .safetensors without using the diffusers library.

import sys
import os
import json
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from collections import defaultdict

def consolidate_without_diffusers(input_dir, output_file):
    """
    Manually reads the safetensors index, loads tensors from each shard,
    and saves them into a single consolidated file.
    """
    print("--- Starting Manual Consolidation (No Diffusers) ---")
    
    # --- Step 1: Read the index file (the "table of contents") ---
    index_path = os.path.join(input_dir, "diffusion_pytorch_model.safetensors.index.json")
    print(f"Reading index file: {index_path}")
    if not os.path.exists(index_path):
        print(f"[ERROR] Index file not found!")
        sys.exit(1)
        
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map")
    if not weight_map:
        print("[ERROR] 'weight_map' not found in the index file.")
        sys.exit(1)

    # --- Step 2: Group tensors by their shard file for efficiency ---
    # This way, we only open each shard file once.
    shards_to_load = defaultdict(list)
    for tensor_name, shard_filename in weight_map.items():
        shards_to_load[shard_filename].append(tensor_name)
    
    print(f"Found {len(weight_map)} total tensors across {len(shards_to_load)} shard files.")
    
    # --- Step 3: Load tensors from each shard and assemble them ---
    merged_state_dict = {}
    for shard_filename, tensor_names in shards_to_load.items():
        shard_path = os.path.join(input_dir, shard_filename)
        print(f"  > Loading {len(tensor_names)} tensors from {shard_filename}...")
        
        # Load the entire shard into memory (on CPU to save VRAM)
        shard_tensors = safe_load_file(shard_path, device="cpu")
        
        # Copy the required tensors from this shard to our final dictionary
        for tensor_name in tensor_names:
            merged_state_dict[tensor_name] = shard_tensors[tensor_name]

    print("All tensors have been loaded into memory.")
    
    # --- Step 4: Save the final, consolidated dictionary ---
    print(f"Saving consolidated file to: {output_file}")
    safe_save_file(merged_state_dict, output_file)
    
    print("\n--- Manual Consolidation Successful! ---")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python manual_merge.py <path_to_input_dir> <path_to_output_file.safetensors>")
        sys.exit(1)
        
    consolidate_without_diffusers(sys.argv[1], sys.argv[2])