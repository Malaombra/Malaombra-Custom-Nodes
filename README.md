# Malaombra Custom Nodes

Prima versione della libreria di nodi custom di **Malaombra** per ComfyUI.

## Cosa contiene

### 1) Save 4 CivitAI
- Nodo di salvataggio immagini con metadati compatibili con CivitAI.
- Supporta output `png`, `jpeg`, `webp`.
- Mantiene i metadati testuali utili (prompt, negative prompt, sampler, cfg, seed, size, hash modello).

### 2) Seed Generator
- Nodo utility semplice per generare/passare un seed `INT` nei workflow.

### 3) get-model-name-from-model
- Nodo che prova a risalire automaticamente al nome del modello usato nel workflow, leggendo la struttura del prompt/nodi upstream.

## Crediti e origine

- **Save 4 CivitAI** e **Seed Generator** traggono ispirazione da `comfyui-image-saver`, con alcune ottimizzazioni leggere lato robustezza e integrazione.
- **get-model-name-from-model** e la relativa logica di risoluzione del modello sono sviluppo originale di Malaombra.

## Installazione

1. Clona questa repository dentro `ComfyUI/custom_nodes/`.
2. Riavvia ComfyUI.
3. Cerca i nodi nel menu di aggiunta nodi.

## Note

- Questa è la **prima release** del pacchetto.
- La struttura è modulare: ogni nodo vive nella sua sottocartella dentro `nodes/`.
