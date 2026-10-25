# PLACER Linux x86_64 / NVIDIA CUDA container

This image packages the existing PLACER CLI, checkpoint, chemical dictionaries,
and an inference environment. It is independent of the pip-packaging branch and
does not change the scientific implementation.

## Hardware and operating-system requirements

| Host configuration | Scope of this image |
| --- | --- |
| Linux x86_64, NVIDIA GPU, Docker Engine and NVIDIA Container Toolkit | Intended deployment target; native Linux host not yet tested |
| Windows x86_64, NVIDIA GPU, Docker Desktop using WSL2 and Linux containers | Validated deployment target |
| Native Windows containers, WSL1, macOS, or ARM64 hosts | Unsupported by this Linux/amd64 CUDA image |
| AMD/Intel GPUs or other non-CUDA accelerators | Unsupported by this image |
| CPU-only inference | Not validated or advertised; CLI help can run without a GPU |

The container platform is `linux/amd64`; it does not become a Windows image when
launched from PowerShell. WSL2 supplies the Linux execution environment on
Windows. Enable Docker Desktop's WSL integration when running Docker commands
from a WSL terminal. Alternatively, use the Windows Docker CLI from PowerShell.

The host needs a compatible NVIDIA driver and working `--gpus all` access.
Native Linux Docker installations additionally need the NVIDIA Container Toolkit.
The CUDA 12.1 runtime is bundled: a host Conda environment or separately installed
CUDA toolkit is not required. GPU drivers remain the host administrator's
responsibility and are not installed inside this image.

Validated hardware is an RTX 3080 with 10 GiB VRAM and driver 621.01. This driver
version is the tested version, not a claimed minimum. A one-sample run reported
approximately 3.1 GB GPU memory; larger inputs and sampling options can need more.
Other NVIDIA GPU models and host configurations require their own validation.

Scientific package versions are Python 3.10.16, PyTorch 2.3.1, CUDA 12.1,
DGL 2.4.0, and OpenBabel 3.1.1. SE(3) Transformer uses NVIDIA DeepLearningExamples
commit `729963dd47e7c8bd462ad10bfac7a7b0b604e6dd`.

Host setup references:
- [Docker Desktop GPU support on Windows/WSL2](https://docs.docker.com/desktop/features/gpu/)
- [NVIDIA Container Toolkit for Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

The image includes model weights covered by the repository LICENSE. It is a
local prototype, not an official upstream image or a published registry release.
The OCI license label describes PLACER; bundled dependencies retain their own
licenses and notices. Image labels identify this fork's recipe and the upstream
application separately.

## Build

From the repository root, with a running Docker engine:

```bash
docker build --platform linux/amd64 \
  --build-arg PLACER_REVISION="$(git describe --always --dirty)" \
  --build-arg BUILD_CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t placer:cuda12.1-local .
```

From PowerShell, `docker build --platform linux/amd64 -t placer:cuda12.1-local .`
also works. Without explicit provenance arguments, the revision label is
`unknown` and the creation-time label is empty rather than inherited from the
base image. A dirty revision indicates uncommitted recipe/source changes.

The initial build downloads approximately 3 GiB of Conda packages plus the base
image and SE(3) build sources. Allow additional space for extracted libraries,
build cache, and image layers. Notebook tools and the full CUDA development
toolkit are not included. Downloads are needed at build time; inference with
bundled resources can run without network access afterward.

## Runtime hardening

Application files, weights, and the Python environment are owned by root; the
runtime process uses an unprivileged user. Package installation or source edits
belong in a new image build rather than the running container. Caches use `/tmp`.

Use the restricted launch profile below: a read-only root filesystem, a bounded
writable temporary filesystem, no Linux capabilities, no privilege escalation,
no network, and a process-count limit. The output bind mount remains writable.
These controls are Docker runtime options, not properties that a Dockerfile can
force on callers. They reduce access but do not establish a security boundary
against every malicious workload, GPU driver issue, or dependency vulnerability.
Use trusted inputs/checkpoints and expose only the needed host folders.

## Help and GPU check

The default command displays help without loading the model or requiring a GPU:

```bash
docker run --rm --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  placer:cuda12.1-local
```

Verify CUDA access before inference:

```bash
docker run --rm --gpus all --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --entrypoint /opt/conda/bin/python placer:cuda12.1-local -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The reusable runtime check also verifies UID, root ownership, dropped
capabilities, privilege restrictions, a read-only root, and writable temporary
storage. From a Linux/WSL shell at the repository root:

```bash
docker run --rm -i --gpus all --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --entrypoint /opt/conda/bin/python placer:cuda12.1-local \
  -B - --cuda < docker/check_runtime.py
```

## One-sample bundled example

From a Linux/WSL shell with Docker integration enabled:

```bash
mkdir -p container-output
docker run --rm --gpus all --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/container-output,target=/output" \
  placer:cuda12.1-local \
  --ifile /opt/placer/examples/inputs/dnHEM1_apo.pdb \
  --odir /output --target_res A-149 --nsamples 1
```

Expected outputs are `dnHEM1_apo_model.pdb` and `dnHEM1_apo.csv`. This is an
operational smoke test, not a scientific accuracy benchmark. Results are
stochastic and need not match another run numerically. The PDB represents the
modeled region rather than a replacement for the entire input structure.

Windows PowerShell users can run the same example with Docker Desktop:

```powershell
New-Item -ItemType Directory -Force .\container-output | Out-Null
$placerOutput = (Resolve-Path .\container-output).Path
docker run --rm --gpus all --network none --read-only `
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 `
  --tmpfs /tmp:rw,nosuid,nodev,size=1g `
  --mount "type=bind,source=$placerOutput,target=/output" `
  placer:cuda12.1-local `
  --ifile /opt/placer/examples/inputs/dnHEM1_apo.pdb `
  --odir /output --target_res A-149 --nsamples 1
```

## Your own structures

Mount the input folder read-only and the output folder separately. Paths passed
to PLACER must be container paths, not host paths:

```bash
docker run --rm --gpus all --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 256 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=/absolute/input-folder,target=/input,readonly" \
  --mount "type=bind,source=/absolute/output-folder,target=/output" \
  placer:cuda12.1-local \
  --ifile /input/structure.pdb --odir /output --target_res A-149 --nsamples 1
```

Use the appropriate target residue or ligand options; `A-149` is specific to the
bundled example. The output directory must exist and be writable by the chosen
container user. Linux users can select their own UID/GID as shown above for host
file ownership. `/work` is not a writable output directory: pass `--odir /output`.
`HOME=/tmp` and `XDG_CACHE_HOME=/tmp/.cache` support arbitrary non-root user IDs.

`OMP_NUM_THREADS` defaults to 1 to avoid excessive worker creation; override with
`-e OMP_NUM_THREADS=4` if needed. Adjust process and temporary-storage limits for
larger workloads only after testing. Temporary filesystem capacity is an upper
bound, not a preallocation. It contributes to memory use as files are written.
Custom weights can be mounted read-only and selected with `--weights`.
No host source checkout or environment is required at run time.

See [VALIDATION.md](VALIDATION.md) for measured checks and
[SECURITY.md](SECURITY.md) for runtime controls, scan coverage, and outstanding
dependency findings. This prototype is not security-clean.

## Dependency maintenance

`conda-linux-64.lock` contains exact package URLs and MD5 archive checksums from
the working environment's inference dependency closure. These are historical
compatibility pins, not a rolling update. Package channels must continue to
serve these archives. Every updated lock needs fresh build/runtime checks.

To regenerate from an independently validated Linux x86_64 Conda environment:

```bash
python docker/lock_environment.py \
  --prefix "$CONDA_PREFIX" --output docker/conda-linux-64.lock
```

The exporter reads Conda records and follows declared dependencies. It does not
include arbitrary pip packages; SE(3) is pinned separately in the Dockerfile.
Its upstream training entry points are outside this inference image's scope.
Git and build artifacts stay in intermediate stages. The finished environment
is copied once with root ownership rather than adding a full-size chown layer.

Update the base digest, lock, or SE(3) revision only with fresh CLI, GPU, and
restricted offline inference checks. Vulnerability scans are separate from
functional checks; a successful smoke test does not establish vulnerability
absence. No image is automatically published or host environment modified.
