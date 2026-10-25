# Container security scope and outstanding dependency findings

## Runtime changes

The image uses a non-root process and root-owned application, model resources,
and Python environment. Temporary caches use `/tmp`. The documented launch
profile additionally enables `--read-only`, `--cap-drop ALL`,
`--security-opt no-new-privileges=true`, `--network none`, a 256-process limit,
and a bounded `/tmp` tmpfs. Only the chosen output bind mount is writable.
Host input/checkpoint mounts should be read-only. The container needs NVIDIA
GPU access but does not need `--privileged`, host networking, or Docker socket
access. GPU driver security and updates remain host responsibilities.

These controls limit access; they do not fix vulnerabilities in bundled code.
Do not treat this prototype as a sandbox for untrusted checkpoints or workloads.

## Vulnerability scan, October 24, 2026

Docker Scout required account authentication, so Trivy was used without
publishing the image. Scanner image:

`aquasec/trivy@sha256:790fd66d1697395d8693d1acaa1ea702406a20f109b6fc7bac8db623eca6c06c`

The scan examined a local archive of the pre-hardening runtime image. The
hardening pass preserves the exact base image, Conda lock, and SE(3) wheel source
revision, so it does not remediate or change that dependency inventory.

| Severity | Scanner finding instances |
| --- | ---: |
| Critical | 3 |
| High | 55 |
| Medium | 78 |
| Low | 67 |
| Unknown | 1 |

Counts are package/location findings rather than unique exploitable issues.
They require triage for reachability and applicability; they are not waived.
Examples requiring attention include:

| Package | Installed version | Scanner finding | Scanner-listed fix |
| --- | --- | --- | --- |
| torch | 2.3.1 | CVE-2025-32434, critical | 2.6.0 |
| Brotli | 1.1.0 | GHSA-9v8h-qqgm-4m35, critical | 1.2.1 |
| certifi | 2024.12.14 | GHSA-6q48-6c7w-9gw2, critical | 2026.10.19 |
| dgl | 2.4.0+cu121 | GHSA-3x5x-fw77-g54c, high | Not listed |
| setuptools | 75.6.0 | CVE-2025-47273, high | 78.1.1 |

These are scanner-reported identifiers and suggested fixed versions, not
independently verified upgrade recommendations. PyTorch/DGL/CUDA versions are
coupled; replacing PyTorch alone is not an approved compatibility recipe.
Dependency remediation needs a separately validated container environment and
fresh inference tests. The host's working Conda environment must remain intact.

## Coverage and reproduction

The scan detected 78 Debian packages, 61 Python package records, and one Node.js
package record. Java archives were excluded and the Java database was not
updated because the supported workflow is Python/CUDA inference. Vulnerability
scanning was enabled; secret scanning was not part of this dependency scan.
Telemetry and dependency-identification API requests were disabled. The
vulnerability database was downloaded at scan time.

Conda-installed native libraries and statically bundled components may not be
fully identified or matched to advisories. The result is not a complete SBOM
attestation or a claim of vulnerability absence. No findings were suppressed.
The raw JSON report and summary remain outside the checkout under
`/tmp/placer-container-validation-20261024/scan/` on the validation machine.

To rescan a locally built image, save it to a dedicated temporary folder and
mount that folder read-only into the pinned scanner image, with a separate
writable reports folder and cache volume. Invoke:

```text
trivy image --cache-dir /cache --timeout 15m --scanners vuln   --skip-java-db-update --skip-files '**/*.jar' --offline-scan   --disable-telemetry --list-all-pkgs --format json   --output /reports/vulnerabilities.json --input /scan/image.tar
```

A release/publication decision should explicitly account for the outstanding
findings. Successful operational tests and root-owned files do not make this
image security-clean.
