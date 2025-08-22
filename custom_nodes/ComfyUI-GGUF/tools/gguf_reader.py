"""
A utility script to inspect the contents of a GGUF file.

This script reads a GGUF file and prints its metadata (key-value store)
and a detailed list of all tensors it contains, including their names,
shapes, and data types.

It can also optionally print a small sample of the data from a specific
tensor or generate histograms for all non-bias tensors and save them
as images.

Usage:
  # To see metadata and the full tensor list
  python3 read_gguf.py /path/to/your/model.gguf

  # To inspect the first few values of a specific tensor by name
  python3 read_gguf.py /path/to/your/model.gguf --inspect-tensor patch_embedding.weight

  # To generate histograms for all non-bias tensors and save them to a directory
  # This requires matplotlib: pip install matplotlib
  python3 read_gguf.py /path/to/your/model.gguf --histogram ./model_histograms/
"""
import sys
import os
import argparse
import numpy as np
from gguf import GGUFReader

# We conditionally import matplotlib to avoid making it a hard dependency
# for users who only want to inspect metadata.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def format_value(field):
    """Helper function to format metadata values for printing."""
    try:
        # Check if it's a string
        if len(field.parts) == 1 and field.types[0] == 5: # GGUFValueType.STRING
            return field.parts[0].tobytes().decode('utf-8', errors='ignore')
        # Check if it's a simple numeric value
        if len(field.parts) == 1 and np.isscalar(field.parts[0]):
            return field.parts[0]
        # For arrays or complex types, show the shape and type
        return f"Array(shape={np.shape(field.parts)}, types={field.types})"
    except Exception:
        return str(field.parts)

def generate_histograms(reader, output_dir):
    """Generates and saves histograms for all non-bias tensors."""
    if not MATPLOTLIB_AVAILABLE:
        print(
            "\n[ERROR] matplotlib is not installed. "
            "Please run 'pip install matplotlib' to generate histograms.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"\n--- Generating Histograms (excluding biases) ---")
    print(f"Saving images to: {output_dir}\n")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, tensor in enumerate(reader.tensors):
        # --- What has to be done (1): Filter out bias weights ---
        # We identify bias tensors by their name, which typically ends in '.bias'.
        if 'bias' in tensor.name:
            print(f"[{i:04d}] Skipping bias tensor: {tensor.name}")
            continue

        # Ensure the data is numeric before trying to create a histogram
        if not np.issubdtype(tensor.data.dtype, np.number):
            print(f"[{i:04d}] Skipping non-numeric tensor: {tensor.name} (dtype: {tensor.data.dtype})")
            continue

        print(f"[{i:04d}] Processing tensor: {tensor.name}")

        # --- How can we actually get the histogram into images ---
        data = tensor.data.flatten() # Flatten to 1D for the histogram

        plt.figure(figsize=(10, 6))
        # Use 100 bins for a reasonably detailed view.
        plt.hist(data, bins=100, color='blue', alpha=0.7)
        plt.title(f"Weight Distribution for: {tensor.name}\nShape: {tensor.shape}", fontsize=12)
        plt.xlabel("Value")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Use a log scale for the y-axis to better see the distribution,
        # especially when some values are much more frequent than others.
        plt.yscale('log')

        # --- It has to be saved to images ---
        # Sanitize the tensor name to create a valid filename.
        # Replace characters that are problematic for filesystems.
        sanitized_name = tensor.name.replace('.', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{i:04d}_{sanitized_name}.png")

        try:
            plt.savefig(output_path, bbox_inches='tight')
        except Exception as e:
            print(f"    [ERROR] Could not save histogram for {tensor.name}: {e}", file=sys.stderr)
        finally:
            # Close the plot to free up memory before the next iteration.
            plt.close()

    print("\n--- Histogram generation complete! ---")


def main():
    parser = argparse.ArgumentParser(description="Inspect a GGUF file's metadata and tensors.")
    parser.add_argument("gguf_file", help="Path to the GGUF file to inspect.")
    parser.add_argument("--inspect-tensor", help="Name or index of a tensor to print a sample of its data.", default=None)
    parser.add_argument("--histogram", metavar="OUTPUT_DIR", help="Generate histograms for all non-bias tensors and save them to the specified directory.", default=None)

    args = parser.parse_args()

    if args.inspect_tensor and args.histogram:
        print("[ERROR] --inspect-tensor and --histogram are mutually exclusive. Please choose one.", file=sys.stderr)
        sys.exit(1)

    try:
        reader = GGUFReader(args.gguf_file)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{args.gguf_file}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Not a valid GGUF file or failed to read: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- GGUF Inspector: {args.gguf_file} ---\n")

    # 1. Print Metadata (Key-Value Store)
    print("--- GGUF Metadata (Key-Value Store) ---")
    for key, field in reader.fields.items():
        print(f"- {key:<40}: {format_value(field)}")

    # 2. Print Tensor Information
    print(f"\n--- Tensor Information ({len(reader.tensors)} total) ---")
    print(f"{'Index':<7} {'Name':<50} {'Shape':<25} {'Type':<10}")
    print("-" * 100)
    for i, tensor in enumerate(reader.tensors):
        shape_str = str(tensor.shape)
        print(f"[{i:04d}]   {tensor.name:<50} {shape_str:<25} {tensor.tensor_type.name:<10}")

    # 3. Handle optional actions
    if args.inspect_tensor is not None:
        target_tensor = None
        target_id = args.inspect_tensor

        try:
            tensor_idx = int(target_id)
            if 0 <= tensor_idx < len(reader.tensors):
                target_tensor = reader.tensors[tensor_idx]
        except ValueError:
            for tensor in reader.tensors:
                if tensor.name == target_id:
                    target_tensor = tensor
                    break

        if target_tensor:
            print(f"\n--- Inspecting Tensor Data: '{target_tensor.name}' ---")
            data = target_tensor.data
            print(f"  Shape: {data.shape}")
            print(f"  Data Type (numpy): {data.dtype}")
            print(f"  Statistics:")
            print(f"    - Mean: {np.mean(data):.4f}")
            print(f"    - Min:  {np.min(data):.4f}")
            print(f"    - Max:  {np.max(data):.4f}")
            print(f"    - Std Dev: {np.std(data):.4f}")

            print("\n  Sample of Data (first few elements):")
            if data.ndim == 1:
                print(data[:16])
            elif data.ndim == 2:
                print(data[:4, :8])
            elif data.ndim >= 3:
                slicing = tuple([slice(0, 2) for _ in range(data.ndim)])
                print(data[slicing])
            else:
                print(data)
        else:
            print(f"\n[ERROR] Tensor '{target_id}' not found by name or index.", file=sys.stderr)

    # --- What has to be done: Call the new histogram function ---
    elif args.histogram is not None:
        generate_histograms(reader, args.histogram)


if __name__ == "__main__":
    main()