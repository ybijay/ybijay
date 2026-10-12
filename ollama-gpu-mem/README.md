# ollama-gpu-mem

Shows how much of each loaded Ollama model is in GPU memory vs system RAM, and
cross-checks it against `nvidia-smi`. Python 3.10+, standard library only.

```
python3 ollama_gpu_mem.py                 # one-shot table
python3 ollama_gpu_mem.py --watch 2       # refresh every 2s
python3 ollama_gpu_mem.py --json          # machine-readable
python3 ollama_gpu_mem.py --host gpu-box  # remote Ollama (defaults to $OLLAMA_HOST)
```

Example:

```
MODEL         TOTAL     VRAM      RAM       PROCESSOR        CTX
llama3:8b     6.0 GiB   6.0 GiB   0.0 GiB   100% GPU         8192
mixtral:8x7b  30.0 GiB  18.0 GiB  12.0 GiB  40%/60% CPU/GPU  -

Ollama VRAM in use: 24.0 GiB
nvidia-smi, ollama processes: 24.0 GiB
GPU 0 NVIDIA GeForce RTX 4090: 23.9 GiB / 24.0 GiB (100%)
```

## Reading the numbers

- **TOTAL / VRAM** come from `size` / `size_vram` in `GET /api/ps`. TOTAL covers
  weights + KV cache + compute graph, so it is larger than the model file.
- **PROCESSOR** uses the same formula as `ollama ps`.
- All sizes are binary (GiB = 1024³), like `nvidia-smi`. Ollama's own CLI uses
  decimal GB, so `ollama ps` numbers look ~7% bigger.
- **Ollama VRAM vs nvidia-smi processes**: Ollama reports its *estimate*;
  nvidia-smi shows what the runner actually holds (including CUDA context
  overhead). A gap of a few hundred MiB is normal. A large gap points at a stale
  runner or at another app using the GPU.
- Older Ollama builds omit `size_vram`; those models show as 100% CPU.

## Tests

```
python3 -m unittest discover -s tests -v
```
