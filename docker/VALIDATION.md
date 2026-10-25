# Local container validation

Validated on October 24, 2026. This records an operational smoke test and is not
scientific accuracy validation or published-result reproduction.

## Environment and scope

- Source baseline: `7dc5563300e7143582d85636bdd413d4c7688e9a`.
- Branch: `feature/containerized-placer`, independent of the editable-install branch.
- Docker Desktop 4.47.0 / Engine 28.4.0, Linux x86_64 backend on Windows/WSL2.
- NVIDIA GeForce RTX 3080, 10 GiB VRAM, driver 621.01.
- Python 3.10.16 / PyTorch 2.3.1 / CUDA 12.1 / DGL 2.4.0+cu121.
- SE(3) Transformer 1.2.0 at the Git revision pinned in the Dockerfile.

## Results

| Check | Result |
| --- | --- |
| Fresh dependency installation and image build | PASS |
| Default CLI help, without GPU access or network | PASS |
| PLACER import and bundled resource presence, offline | PASS |
| CUDA availability and an actual CUDA tensor operation | PASS |
| One-sample inference using bundled `dnHEM1_apo.pdb`, target A-149 | PASS |
| Inference with `--network none`, no source/environment mounts | PASS |
| Output directory bind mount, arbitrary UID/GID 1000:1000, `HOME=/tmp` | PASS |
| Output ownership matches selected UID | PASS |
| One CSV score row, expected columns and finite numeric scores | PASS |
| One PDB model, finite coordinates, 598 ATOM/HETATM records | PASS |
| Host Conda package versions and pip branch unchanged | PASS |

The completed local image occupied 8.35 GB (approximately 7.78 GiB) before
registry compression. The explicit Conda archives total approximately 2.98 GiB.
Docker build caches and intermediate stages consume additional local space.

The model reported 3.11 seconds for the sample, 3.88 seconds for the prediction
loop, and approximately 3.13 GB GPU memory. These are observations from this
sample, not performance or resource guarantees. Timings exclude image download,
container startup, imports, and checkpoint initialization.

Generated score columns were `label`, `model_idx`, `fape`, `lddt`, `rmsd`,
`kabsch`, `prmsd`, `plddt`, and `plddt_pde`. Numerical results and crop atom counts
can vary between stochastic runs. This apo side-chain example does not establish
ligand-docking accuracy.

Build, CLI, GPU, and inference logs and a machine-readable output summary were
saved outside the checkout. No registry image was published. Native Linux hosts,
other NVIDIA GPUs, CPU inference, ARM64/macOS, and new Python versions were
not validated.

## Runtime hardening pass, October 24, 2026

The hardened image was built and tested separately as
`placer:cuda12.1-hardened`, image
`sha256:27e5a776d15f94ea0f7683f96b88f919ec1cd9c0a6cf2d8cd5550957ad17a43e`.
It occupies 8,354,578,034 bytes (8.35 GB); runtime ownership changes did not add
another environment-sized layer. PLACER provenance/description/license labels
replaced the inherited micromamba application labels. The tested source label
was `ee19de0-dirty`, indicating the hardening changes were not committed yet.

| Hardening check | Result |
| --- | --- |
| Default runtime UID is non-root (57439) | PASS |
| Application, model resources, Python prefix, and working directory root-owned | PASS |
| Runtime user denied writes to application/environment without read-only mount enforcement | PASS |
| Read-only root filesystem and writable bounded `/tmp` tmpfs | PASS |
| Effective and bounding Linux capability sets empty | PASS |
| `NoNewPrivs` enabled and process limit 256 | PASS |
| PLACER import and CUDA tensor operation under restricted profile | PASS |
| CLI help under restricted profile without GPU/network | PASS |
| Offline GPU inference, arbitrary UID/GID 1000:1000, only output folder mounted | PASS |
| One score row, finite scores, one PDB model with finite coordinates | PASS |
| Host environment, pip branch, scientific files, and dependency lock unchanged | PASS |

The restricted run produced 595 ATOM/HETATM records. Its log and
machine-readable validation summary are outside the checkout under
`/tmp/placer-container-validation-20261024/`. This was a stochastic operational
check rather than a scientific equivalence benchmark.

Dependency scanning found outstanding critical/high findings. See
[SECURITY.md](SECURITY.md) for counts, coverage, and remediation limitations.
Runtime restriction checks passed; dependency vulnerabilities were not silently
suppressed or represented as fixed. Native Linux hosts and hardware other than
the tested Windows/WSL2 RTX 3080 setup remain unvalidated.
