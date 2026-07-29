# Malaombra Custom Nodes

 ## Installation

1. Clone this repository into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Find the nodes in the node add menu.


## Included Nodes

### 1) Save 4 CivitAI
- Image save node with CivitAI-friendly metadata output.
- Supports `png`, `jpeg`, and `webp` output formats.
- Preserves useful text metadata (prompt, negative prompt, sampler, cfg, seed, size, model hash).

### 2) Seed Generator
- Simple utility node to generate/pass an `INT` seed in workflows.

### 3) get-model-name-from-model
- Node that tries to resolve the model name used in the workflow by traversing prompt/upstream node data.

### 4) Video Combine
- Video Helper Suite based video combine node that outputs a standard ComfyUI `VIDEO`.
- Includes `show_preview` to display a preview even when `save_output` is disabled.

### 5) ComicChat Workflow Integration
- `ComicChat · Workflow Input` is a reusable, schema-validated gateway for all values supplied by ComicChat.
- Typed getter nodes expose string, integer, float, boolean, and ComfyUI combo values to any workflow.
- `ComicChat · LoRA Stack` applies the ordered LoRA selections and strengths declared in the gateway payload.
- Workflow-specific behavior remains in graph connections; the Python nodes are shared by text-to-image, upscale, video, and other workflows.

## Credits and Origin

- **Save 4 CivitAI** and **Seed Generator** are inspired by `comfyui-image-saver`, with small optimizations for robustness and integration.
- **get-model-name-from-model** and its model-resolution logic are original development.
- **Video Combine** is based on `ComfyUI-VideoHelperSuite` and requires it to be installed.

