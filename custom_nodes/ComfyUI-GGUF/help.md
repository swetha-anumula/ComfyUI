# Wan 2.2 Dynamic Quantization: Command-Line Workflow

This guide provides the exact sequence of commands to perform an advanced, layer-wise dynamic quantization of the Wan 2.2 model.

It uses the robust **"Quantize-and-Assemble"** pipeline and assumes the following required Python scripts are present in the `tools` directory:

- `consolidate.py`
- `convert.py`
- `run_quant_assembler.py`


## Phase 1: Prerequisites \& Setup

All commands should be run from the `tools` directory.

**1. Navigate to the Tools Directory**

```bash
cd /shareddata/dheyo/swetha/ComfyUI/custom_nodes/ComfyUI-GGUF/tools
```

**2. Activate the Python Environment**

```bash
source /shareddata/dheyo/swetha/ComfyUI/comfyui/bin/activate
```

**3. Verify Required Files**
Before starting, ensure your detailed JSON recipe file exists. For example:
`high_noise_wan2.2_quant_recipe.json`

## Phase 2: The Quantization Pipeline (for high_noise model)

Execute these steps in order.

**Step 1: Consolidate the Sharded Safetensors**
This merges the multiple .safetensors files from the Hugging Face format into a single file.

```bash
python consolidate.py \
  /shareddata/dheyo/models/Wan2.2-I2V-A14B/high_noise_model/ \
  /shareddata/dheyo/swetha/ComfyUI/models_m/wan2.2-high-noise-consolidated.safetensors
```

**Step 2: Convert to Intermediate F16 GGUF**
This creates the clean F16 GGUF starting point for our main process and separates the problematic 5D tensor.

```bash
python convert.py \
  --src /shareddata/dheyo/swetha/ComfyUI/models_m/wan2.2-high-noise-consolidated.safetensors \
  --dst /shareddata/dheyo/swetha/ComfyUI/models_m/high_noise.gguf
```

*Note:* A "5D tensor" warning is expected here. This is normal.

**Step 3: Execute the Dynamic Quantization Assembler**

3.1. Configure the Assembler Script
Open `run_quant_assembler.py` and ensure the `MODEL_NAME` variable at the top is set correctly:

```python
MODEL_NAME = "wan2.2-high_noise"
```

Verify that all other file paths in the script are correct.

3.2. Run the Assembler

```bash
python run_quant_assembler.py
```

This script will automatically create temporary files, assemble the final model, clean up, and run the final `fix_5d_tensors.py` step.

## Phase 3: Processing the low_noise Model

To process the `low_noise` model, repeat the entire pipeline with the following changes:

**Consolidate:** Use the `low_noise_model` source directory and a different output name.

```bash
python consolidate.py \
  /shareddata/dheyo/models/Wan2.2-I2V-A14B/low_noise_model/ \
  /shareddata/dheyo/swetha/ComfyUI/models_m/wan2.2-low-noise-consolidated.safetensors
```

**Convert:** Use the new consolidated file and create `low_noise.gguf`.

```bash
python convert.py \
  --src /shareddata/dheyo/swetha/ComfyUI/models_m/wan2.2-low-noise-consolidated.safetensors \
  --dst /shareddata/dheyo/swetha/ComfyUI/models_m/low_noise.gguf
```

**Quantize:**
Open `run_quant_assembler.py` and change the configuration:

```python
MODEL_NAME = "wan2.2-low_noise"
```

Ensure you have a `low_noise_wan2.2_quant_recipe.json` file.

Run the assembler again:

```bash
python run_quant_assembler.py
```


## Final Output

Once both pipelines are complete, you will have your final, fully functional, and custom-quantized GGUF models located at:

```
/shareddata/dheyo/swetha/ComfyUI/models_m/dheyo_v1_high_noise_final.gguf
/shareddata/dheyo/swetha/ComfyUI/models_m/dheyo_v1_low_noise_final.gguf
```


