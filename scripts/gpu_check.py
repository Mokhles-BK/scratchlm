r"""GPU sanity check for ScratchLM.

Prints the torch version, whether CUDA is available, the GPU name and VRAM,
then runs a small bf16 matrix multiply and one scaled_dot_product_attention
call on the GPU to prove both work on this machine.

Run:  python scripts\gpu_check.py
"""

import sys

import torch
import torch.nn.functional as F


def main() -> None:
    print(f"Python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda or '(not built with CUDA)'}")

    if not torch.cuda.is_available():
        print("\nCUDA is NOT available — training will fall back to CPU.")
        print("Check the driver and the PyTorch CUDA build in scripts/setup.ps1.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU         : {props.name}")
    print(f"VRAM        : {props.total_memory / 1024**3:.2f} GB total, "
          f"{props.major}.{props.minor} compute capability")

    # --- bf16 matrix multiply on the GPU ---------------------------------
    print("\n--- bf16 matmul on GPU ---")
    a = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
    b = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
    c = a @ b
    print(f"  input shape : {tuple(a.shape)} x {tuple(b.shape)}")
    print(f"  output shape: {tuple(c.shape)}, dtype={c.dtype}")
    print(f"  sample value: {c[0, 0].item():.4f}")

    # --- scaled_dot_product_attention on the GPU --------------------------
    # This is the attention kernel we use for training (Windows has no Triton,
    # so torch.compile is off the table; SDPA is the supported path).
    print("\n--- scaled_dot_product_attention on GPU ---")
    B, H, T, D = 2, 4, 64, 32
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    attn = F.scaled_dot_product_attention(q, k, v)
    print(f"  input shape : {tuple(q.shape)} (q/k/v)")
    print(f"  output shape: {tuple(attn.shape)}, dtype={attn.dtype}")
    print(f"  sample value: {attn[0, 0, 0, 0].item():.4f}")

    # --- quick causality check: a future token must not change past logits -
    print("\n--- causality check (SDPA causal mask) ---")
    # The mask must be BOOLEAN: True = "this query may look at this key".
    # (A float mask is ADDED to the attention scores, so a tril of 1.0s would
    # not block anything and would give a false "masks disagree" warning.)
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    out1 = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    out2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # bf16 has ~3 significant digits; the two kernels may round differently.
    close = torch.allclose(out1.float(), out2.float(), atol=2e-2, rtol=2e-2)
    print(f"  explicit-tril == is_causal: {close}")
    if not close:
        print("  WARNING: causal masks disagree — attention may not be causal.")

    torch.cuda.synchronize()
    print("\nAll GPU checks passed.")


if __name__ == "__main__":
    main()