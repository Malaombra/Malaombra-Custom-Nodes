# Malaombra Custom Nodes

First release of **Malaombra** custom nodes for ComfyUI.

## Included Nodes

### 1) Save 4 CivitAI
- Image save node with CivitAI-friendly metadata output.
- Supports `png`, `jpeg`, and `webp` output formats.
- Preserves useful text metadata (prompt, negative prompt, sampler, cfg, seed, size, model hash).

### 2) Seed Generator
- Simple utility node to generate/pass an `INT` seed in workflows.

### 3) get-model-name-from-model
- Node that tries to resolve the model name used in the workflow by traversing prompt/upstream node data.

## Credits and Origin

- **Save 4 CivitAI** and **Seed Generator** are inspired by `comfyui-image-saver`, with small optimizations for robustness and integration.
- **get-model-name-from-model** and its model-resolution logic are original development by Malaombra.

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Find the nodes in the node add menu.

## Notes

- This is the **first release** of the package.
- The structure is modular: each node lives in its own subfolder under `nodes/`.
