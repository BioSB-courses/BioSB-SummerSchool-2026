# Installing shared Julia packages on SURF Research Cloud (Jupyter + Julia 1.12)

Goal: make a fixed set of Julia packages available to **every user** of the workspace,
from **any notebook**, via the existing system-wide "Julia (Global)" Jupyter kernel —
with no per-user installation or setup.

## Background (why this works)

- SURF's Julia catalog item already ships a **shared, root-owned Julia depot** at
  `/opt/julia_share`, and a system-wide Jupyter kernelspec at
  `/usr/local/share/jupyter/kernels/julia-1.12` (display name **"Julia (Global)"**).
- That kernel's `kernel.json` sets `JULIA_DEPOT_PATH` and `JULIA_LOAD_PATH` so every
  session automatically loads `/opt/julia_share/environments/v1.12/Project.toml` as
  part of its default environment — same mechanism as Julia's stdlibs.
- So: install packages into that shared environment (as root), precompile them there,
  and every notebook using that kernel gets them for free.
- SURF's "Custom Packages" plugin (apt/conda/pip) is **not** used here — it has no
  concept of Julia's package manager. This is done directly via Julia's `Pkg`.

## One-time setup steps

**1. Install/update packages into the shared depot, as root:**

```bash
sudo JULIA_DEPOT_PATH=/opt/julia_share \
  /opt/julia-1.12.6/bin/julia --project=/opt/julia_share/environments/v1.12 \
  -e 'using Pkg; Pkg.add([
        "ADTypes","CairoMakie","ComponentArrays","DataFrames","DataInterpolations",
        "DiffEqBase","Distances","ForwardDiff","IJulia","Lux","Mooncake","Optim",
        "Optimisers","Optimization","OptimizationOptimJL","OptimizationOptimisers",
        "OrdinaryDiffEq","ProgressMeter","ProgressTables","QuasiMonteCarlo",
        "StableRNGs","XGBoost","Zygote"
      ]); Pkg.precompile()'
```

(`Random` is a Julia standard library, not a registered package — no install needed,
it's always available. `IJulia` is typically already present in the shared depot as
part of the base catalog item; adding it again is a harmless no-op.)

**2. Verify no precompile errors**, and check for resolver-caused version pinning
issues (see "Troubleshooting" below) if anything fails.

**3. Confirm ownership/permissions stay root-only, read-only for everyone else:**

```bash
sudo chmod -R a+rX /opt/julia_share
```

(Typically already the case from the base catalog item — just worth a sanity check.)

## Verification

**As a normal (non-root, non-sudo) user, in a terminal:**

```bash
JULIA_DEPOT_PATH=":/opt/julia_share" \
JULIA_LOAD_PATH="@:@v#.#:@stdlib:/opt/julia_share/environments/v1.12" \
julia -e 'using ADTypes, CairoMakie, ComponentArrays, DataFrames, DataInterpolations,
                DiffEqBase, Distances, ForwardDiff, IJulia, Lux, Mooncake, Optim,
                Optimisers, Optimization, OptimizationOptimJL, OptimizationOptimisers,
                OrdinaryDiffEq, ProgressMeter, ProgressTables, QuasiMonteCarlo, Random,
                StableRNGs, XGBoost, Zygote;
          println("all good")'
```

Should print `all good` in a few seconds, with **no downloads or recompilation**.

**In Jupyter Lab (the test that actually matters):** new notebook → select kernel
**"Julia (Global)"** → run the same `using ...` line in a cell. This exercises the
exact path every workshop participant will take.

## Notes / gotchas encountered while setting this up

- **No `Manifest.toml` exists in the source repo** (only `Project.toml`), so package
  versions are resolved fresh against the current General registry each time — not
  pinned to what the workshop authors originally tested with. If the workshop
  organizers publish a `Manifest.toml` later, prefer `Pkg.instantiate()` against it
  for reproducibility instead of `Pkg.add(...)` by name.
- **Adding packages one at a time / in batches risks resolver conflicts.** We hit
  this directly: an earlier, mistaken addition of `PEtab`/`PEtabTraining` (from a
  stale fetch of the repo) pinned `LinearSolve` and `QuasiMonteCarlo` to old
  versions, which cascaded into `SciMLBase` being held back below what
  `OrdinaryDiffEqBDF` required — causing `UndefVarError: AutoDespecialize not
  defined in SciMLBase` precompile failures across six packages. Fixed via:
  ```bash
  sudo JULIA_DEPOT_PATH=/opt/julia_share \
    /opt/julia-1.12.6/bin/julia --project=/opt/julia_share/environments/v1.12 \
    -e 'using Pkg; Pkg.rm(["PEtab", "PEtabTraining"]); Pkg.update(); Pkg.precompile()'
  ```
  **Takeaway:** if precompilation fails with a similarly cryptic `UndefVarError` or
  "not defined" error deep in a SciML package, check
  `Pkg.status(; outdated=true, mode=Pkg.PKGMODE_MANIFEST)` for packages held back
  (marked `⌅`/`⌃`) and look at which direct dependency is pinning them — it's
  usually one package with an overly tight `compat` bound.
- **The source repo's `Project.toml` changed multiple times** during this setup
  (packages added/removed by the workshop organizers). Best practice: do one final,
  authoritative fetch of `Project.toml` shortly before the workshop starts, diff it
  against what's installed (`cat /opt/julia_share/environments/v1.12/Project.toml`),
  and reconcile in a single pass rather than incrementally.
- Terminal sessions do **not** automatically get `JULIA_DEPOT_PATH`/`JULIA_LOAD_PATH`
  set — only the Jupyter kernel does (baked into its `kernel.json`). Set them
  manually (as shown above) when testing from a terminal.
