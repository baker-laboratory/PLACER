"""Export the installed inference dependency closure as a Linux conda lock.

Reads package records only; does not solve or modify the environment.
Review and runtime-test every regenerated lock before committing it.
"""
import argparse
import json
from pathlib import Path

ROOTS = (
    "python", "pytorch", "pytorch-cuda", "dgl", "openbabel", "numpy",
    "pandas", "scipy", "networkx", "e3nn", "opt_einsum", "pip",
    "setuptools", "wheel",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for path in (args.prefix / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["name"]] = record
    selected = set()

    def visit(name):
        if name.startswith("__") or name in selected:
            return
        if name not in records:
            raise ValueError(f"Missing installed dependency: {name}")
        selected.add(name)
        for dependency in records[name]["depends"]:
            visit(dependency.split()[0])

    for name in ROOTS:
        visit(name)
    lines = [
        "# Linux x86_64 inference packages from the validated PLACER environment.",
        "# Python 3.10.16 / PyTorch 2.3.1 / CUDA 12.1 / DGL 2.4.0.",
        "# Exact package URLs and MD5 checksums; regenerate with lock_environment.py.",
        "# SE(3) Transformer is pinned separately in Dockerfile.",
        "@EXPLICIT",
    ]
    for name in sorted(selected):
        record = records[name]
        if record.get("subdir") not in ("linux-64", "noarch"):
            raise ValueError(f"Not a Linux x86_64 package: {name}")
        url = record["url"]
        if not url.startswith("https://conda.anaconda.org/"):
            raise ValueError(f"Review unexpected package origin for {name}")
        lines.append(url + "#" + record["md5"])
    args.output.write_text("\n".join(lines) + "\n")
    size = sum(records[name].get("size", 0) for name in selected)
    print(f"Locked {len(selected)} packages; downloads: {size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
