# Phase 6 commands (PowerShell, from C:\Users\mokhl\scratchlm)

Close LocalChat first (it leaves llama-server holding VRAM). Check: `nvidia-smi` must show < 1 GB used.

```
# 0. tests
.\.venv\Scripts\python.exe -m pytest -q

# 1. full validation of run A (CPU, safe any time)
.\.venv\Scripts\python.exe -m scratchlm.evaluate --checkpoint checkpoints\small-3k\best.pt

# 2. run B: small, context 256, same tokens per step (16 x 8 x 256 = 8 x 8 x 512 tokens per step, same as A)
.\.venv\Scripts\python.exe -m scratchlm.train --model small --name small-256 --block-size 256 --micro-batch-size 16 --grad-accum-steps 8 --max-steps 3000

# 3. run C: base, context 512, keep a checkpoint every 500 steps (~53 min)
.\.venv\Scripts\python.exe -m scratchlm.train --model base --name base-3k --max-steps 3000 --milestone-interval 500

# 4. compare
.\.venv\Scripts\python.exe scripts\compare_runs.py --names small-3k small-256 base-3k
.\.venv\Scripts\python.exe -m scratchlm.evaluate --checkpoint checkpoints\small-3k\best.pt checkpoints\small-256\best.pt checkpoints\base-3k\best.pt

# 5. samples
.\.venv\Scripts\python.exe scripts\sample_grid.py --checkpoint checkpoints\base-3k\best.pt
.\.venv\Scripts\python.exe -m scratchlm.generate --checkpoint checkpoints\base-3k\step_000500.pt --prompt "Once upon a time" --seed 1
.\.venv\Scripts\python.exe -m scratchlm.generate --checkpoint checkpoints\base-3k\step_001500.pt --prompt "Once upon a time" --seed 1
.\.venv\Scripts\python.exe -m scratchlm.generate --checkpoint checkpoints\base-3k\step_003000.pt --prompt "Once upon a time" --seed 1
```
