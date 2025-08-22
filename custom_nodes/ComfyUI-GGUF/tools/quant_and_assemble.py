import json
import sys
import os
import subprocess
from gguf import GGUFReader, GGUFWriter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
MODEL_NAME = "wan2.2-low_noise"
TEMP_DIR = "/dev/shm/temp_quants"  
SUPPRESS_TYPE_WARNINGS = True 
LLAMA_TOOLS_DIR = "/shareddata/dheyo/swetha/ComfyUI/custom_nodes/ComfyUI-GGUF/tools"
LLAMA_QUANTIZE_EXEC = os.path.join(LLAMA_TOOLS_DIR, "llama.cpp/llama-quantize")
FIX_5D_SCRIPT = os.path.join(LLAMA_TOOLS_DIR, "fix_5d_tensors.py")

if MODEL_NAME == "wan2.2-high_noise":
    JSON_RECIPE_FILE = os.path.join(LLAMA_TOOLS_DIR, "hn_1.json")
    BASE_F16_FILE = "/shareddata/dheyo/swetha/ComfyUI/models_m/high_noise_4D_READY.gguf"
    FINAL_OUTPUT_FILE = "/shareddata/dheyo/swetha/ComfyUI/models_m/dheyo_v2_high_noise_final_1.gguf"
else: 
    JSON_RECIPE_FILE = os.path.join(LLAMA_TOOLS_DIR, "ln_1.json")
    BASE_F16_FILE = "/shareddata/dheyo/swetha/ComfyUI/models_m/low_noise_4D_READY.gguf"
    FINAL_OUTPUT_FILE = "/shareddata/dheyo/swetha/ComfyUI/models_m/dheyo_v2_low_noise_final_1.gguf"



def is_valid_gguf_file(path):
    """Check if a file is a valid GGUF file by reading its magic number."""
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            return magic == b'GGUF'
    except Exception as e:
        print(f"[WARN] Failed to validate GGUF file {path}: {str(e)}")
        return False

def run_command(command, description):
    """Executes a command and exits if it fails, printing the error."""
    print(f"--- Running: {description} ---")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nFATAL ERROR: Command failed with exit code {e.returncode}.")
        print(f"Command: {' '.join(command)}")
        print(f"\n--- STDERR from failed command ---")
        print(e.stderr)
        print("------------------------------------")
        sys.exit(1)

def validate_tensor(tensor, source_tensor):
    """Validate that source_tensor matches the expected tensor in shape and type."""
    if not np.array_equal(tensor.shape, source_tensor.shape):
        print(f"[ERROR] Tensor {tensor.name} shape mismatch: expected {tensor.shape}, got {source_tensor.shape}")
        return False
    try:
        tensor_type = tensor.tensor_type
        source_tensor_type = source_tensor.tensor_type
        if tensor_type != source_tensor_type and not SUPPRESS_TYPE_WARNINGS:
            print(f"[WARN] Tensor {tensor.name} type mismatch: expected {tensor_type}, got {source_tensor_type}")
    except AttributeError as e:
        print(f"[ERROR] Failed to access tensor types for {tensor.name}: {str(e)}")
        print(f"Tensor attributes: {dir(tensor)}")
        return False
    return True

def debug_kv_pairs(fields):
    """Debug key-value pairs to identify issues, handling non-array-like parts."""
    print("\n--- Debugging Key-Value Pairs ---")
    for key, val in fields.items():
        try:
            value_shape = np.shape(val.parts)
        except (ValueError, AttributeError):
            value_shape = f"Non-array (len={len(val.parts) if hasattr(val.parts, '__len__') else 'N/A'})"
        print(f"Key: {key}, Types: {val.types}, Value Shape: {value_shape}, Value: {val.parts}")
    print("------------------------------------")

def quantize_model(qt):
    """Run quantization for a single type, with caching and validation."""
    temp_file = os.path.join(TEMP_DIR, f"temp_{qt}.gguf")
    
    if os.path.exists(temp_file) and is_valid_gguf_file(temp_file):
        print(f"--- Using cached temporary {qt} model ---")
    else:
        if os.path.exists(temp_file):
            print(f"[WARN] Invalid or corrupted {qt} file found. Regenerating: {temp_file}")
            os.remove(temp_file)
        run_command([LLAMA_QUANTIZE_EXEC, BASE_F16_FILE, temp_file, qt], f"Creating temporary {qt} model")
        
    return qt, temp_file

def main():
    print("--- Starting Quantize-and-Assemble Process ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    if not os.path.exists(BASE_F16_FILE):
        print(f"[ERROR] Base GGUF file not found: {BASE_F16_FILE}. Run convert.py and fix_5d_tensors.py first.")
        sys.exit(1)

    with open(JSON_RECIPE_FILE, 'r') as f:
        recipe = json.load(f)

    print("Applying overrides: All layer normalization weights and biases to F16...")
    for i in range(40):
        recipe[f"blocks.{i}.norm1.weight"] = "F16"
        recipe[f"blocks.{i}.norm1.bias"] = "F16"
        recipe[f"blocks.{i}.norm2.weight"] = "F16"
        recipe[f"blocks.{i}.norm2.bias"] = "F16"
        recipe[f"blocks.{i}.norm3.weight"] = "F16"
        recipe[f"blocks.{i}.norm3.bias"] = "F16"
    recipe["norm.weight"] = "F16"
    recipe["norm.bias"] = "F16"
    if "patch_embedding.weight" not in recipe:
        recipe["patch_embedding.weight"] = "COPY"
    print("  > All normalization layers (weights and biases) set to F16.")
    print("  > patch_embedding.weight set to COPY if not specified.")

    used_quant_types = set(recipe.values()) - {'F16', 'F32', 'COPY'}
    unique_quant_types = sorted(list(used_quant_types))
    print(f"\nFound {len(unique_quant_types)} unique quantization types to process: {unique_quant_types}")

    temp_files = {}
    with ThreadPoolExecutor() as executor:
        temp_files = dict(executor.map(quantize_model, unique_quant_types))

    try:
        print("\n--- Assembling final mixed-precision model ---")
        base_reader = GGUFReader(BASE_F16_FILE)
        arch = base_reader.fields['general.architecture'].parts[-1].tobytes().decode('utf-8')
        writer = GGUFWriter(FINAL_OUTPUT_FILE, arch)
        
        ingredient_readers = {}
        for qt, path in temp_files.items():
            try:
                ingredient_readers[qt] = GGUFReader(path)
            except ValueError as e:
                print(f"[ERROR] Failed to read temporary file for {qt}: {str(e)}")
                print(f"File: {path}")
                sys.exit(1)
        ingredient_readers['F16'] = ingredient_readers['F32'] = ingredient_readers['COPY'] = base_reader

        debug_kv_pairs(base_reader.fields)
        print("[INFO] Setting minimal metadata manually to avoid shape mismatch.")
        try:
            writer.add_uint32("general.quantization_version", 2)
            writer.add_uint32("general.file_type", 1)
        except AttributeError:
            print("[WARN] Unable to set metadata manually; relying on GGUFWriter defaults.")
        print("\n--- Processing Tensors ---")
        patch_embedding_found = False
        for tensor in base_reader.tensors:
            print(f"Processing tensor: {tensor.name}, Shape: {tensor.shape}, Type: {tensor.tensor_type}")
            if tensor.name == "patch_embedding.weight":
                patch_embedding_found = True
            target_quant = recipe.get(tensor.name, 'COPY')
            source_quant = target_quant if target_quant in unique_quant_types else 'COPY'
            source_reader = ingredient_readers[source_quant]
            
            found = False
            for source_tensor in source_reader.tensors:
                if source_tensor.name == tensor.name:
                    if validate_tensor(tensor, source_tensor):
                        writer.add_tensor(source_tensor.name, source_tensor.data, raw_dtype=source_tensor.tensor_type)
                        found = True
                    break
            if not found:
                print(f"[ERROR] Tensor not found in source {source_quant}: {tensor.name}")
                sys.exit(1)
        
        if not patch_embedding_found:
            print("[ERROR] Tensor 'patch_embedding.weight' not found in base model. Ensure fix_5d_tensors.py was run.")
            sys.exit(1)

        print(f"\nWriting final assembled file to: {FINAL_OUTPUT_FILE}")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    finally:
        print("\n--- Cleaning up temporary files ---")
        for path in temp_files.values():
            if os.path.exists(path):
                os.remove(path)
    
    print(f"\n--- Process for {MODEL_NAME} is COMPLETE! ---")

if __name__ == "__main__":
    main()