# One CUDA codebase, two GPU vendors

Traditionally, CUDA code only runs on NVIDIA hardware.
Porting to AMD requires a full rewrite in HIP — different APIs, different
toolchain, different maintenance burden.

**[SCALE](https://scale-lang.com)** breaks that constraint.
It is an open-source, LLVM/Clang-based CUDA compiler that compiles
*unmodified* `.cu` files for AMD GPUs, with no code changes and no HIP
anywhere in sight.

This repository shows how [pixi](https://pixi.sh) makes it trivial to build
and ship the same CUDA source for both GPU families from a single workspace.

---

## The idea

```
example/example.cu          ← one CUDA source, never touched again
        │
        ├─ pixi run -e nvcc  build   →  binary for NVIDIA  (via nvcc)
        └─ pixi run -e scale build   →  binary for AMD     (via SCALE)
```

No `#ifdef __HIP__`.  No duplicate source tree.  No separate HIP port to
maintain.

---

## How pixi wires it together

`pixi.toml` defines two environments that share the exact same source:

```toml
[feature.nvcc.dependencies]
cuda-nvcc     = "12.*"          # NVIDIA's compiler
example-basic = { path = "." }

[feature.scale.dependencies]
scale-compiler = { path = "./compiler/recipe.yaml" }   # SCALE from conda
example-basic  = { path = "." }

[environments]
nvcc  = { features = ["nvcc"]  }
scale = { features = ["scale"] }
```

Running `pixi run -e <env> benchmark` resolves the right compiler, builds
the binary, and executes it — no manual toolchain setup required.

---

## The benchmark

`example/example.cu` is a straightforward memory-bandwidth benchmark:
parallel float32 vector addition over **32 M elements (128 MiB per array)**,
timed with CUDA events over 200 iterations.

```cuda
__global__ void vectorAdd(const float* a, const float* b,
                          float* out, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
```

It reports **average kernel time** and **effective memory bandwidth**
(`3 × array_size / kernel_time` — two reads + one write), the standard
metric for streaming kernels.

---

## Running it

```bash
# Clone & enter
git clone https://github.com/wolfv/scale-example && cd scale-example

# NVIDIA GPU  (sm_89 / Ada Lovelace by default)
pixi run -e nvcc benchmark

# AMD GPU  (gfx1100 / RDNA3 by default)
pixi run -e scale benchmark
```

Change the default GPU target by editing the architecture flag in `pixi.toml`:

| Environment | Variable | Example values |
|-------------|----------|----------------|
| `nvcc`  | `CMAKE_CUDA_ARCHITECTURES` | `86` (Ampere), `89` (Ada), `90` (Hopper) |
| `scale` | target path in `build` task | `gfx1100` (RDNA3), `gfx942` (MI300X), `gfx908` (MI100) |


## Repository layout

```
scale-example/
├── pixi.toml                   # workspace — two environments
├── compiler/
│   └── recipe.yaml             # pixi-build recipe: packages SCALE from
│                               # scale-lang.com into a conda package
└── example/
    ├── CMakeLists.txt
    └── example.cu              # ← the only file that matters
```

---

## Why this matters

HIP ports are expensive.  They require rewriting API calls, testing a
separate code path, and keeping two codebases in sync forever.

SCALE sidesteps all of that.  If your project already targets NVIDIA with
CUDA, adding AMD support is now a matter of:

1. Adding `scale-compiler` as a conda dependency
2. Setting `CUDACXX` to Scale's nvcc for AMD builds
3. Shipping a second binary compiled for `gfx*`