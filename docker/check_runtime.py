"""Check a live container's ownership, restrictions, imports, and optional CUDA.

Run via stdin with the restricted profile documented in docker/README.md.
Requires no host source or environment mount and does not load the model.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() != 0, "Runtime must be non-root"
    for name in (
        "/opt/placer", "/opt/placer/PLACER.py", "/opt/placer/data",
        "/opt/placer/weights", "/opt/conda",
        "/opt/conda/lib/python3.10/site-packages", "/work",
    ):
        path = Path(name)
        assert path.stat().st_uid == 0, f"Not root-owned: {name}"
        assert not os.access(path, os.W_OK), f"Unexpected runtime write access: {name}"
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    assert int(status["CapEff"].strip(), 16) == 0, "Effective capabilities present"
    assert int(status["CapBnd"].strip(), 16) == 0, "Capability bounding set not empty"
    assert status["NoNewPrivs"].strip() == "1", "Privilege escalation not disabled"
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    assert any(row[1] == "/" and "ro" in row[3].split(",") for row in mounts), "Root filesystem not read-only"
    assert any(row[1] == "/tmp" and row[2] == "tmpfs" for row in mounts), "Missing temporary filesystem"
    limit = Path("/sys/fs/cgroup/pids.max")
    if limit.is_file():
        assert limit.read_text().strip() == "256", "Unexpected process limit"
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        (Path(directory) / "cache-probe").write_text("temporary storage works")
    import PLACER
    assert Path(PLACER.__file__).parent == Path("/opt/placer")
    result = {"result": "PASS", "uid": os.geteuid(), "read_only": True,
              "capabilities": "none", "no_new_privileges": True}
    if args.cuda:
        import torch
        assert torch.cuda.is_available(), "CUDA unavailable"
        assert torch.ones(1, device="cuda").item() == 1.0
        result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
