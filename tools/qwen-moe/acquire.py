#!/usr/bin/env python3
"""Acquire the pinned Qwen MoE text baseline and official research metadata."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

OFFICIAL = "Qwen/Qwen3.6-35B-A3B"
OFFICIAL_REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
QUANT = "unsloth/Qwen3.6-35B-A3B-GGUF"
QUANT_REV = "a483e9e6cbd595906af30beda3187c2663a1118c"
NAME = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
SIZE = 22134528992
SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
CHUNK = 64 * 1024 * 1024
META = ["config.json", "model.safetensors.index.json", "README.md", "LICENSE",
        "generation_config.json", "chat_template.jinja", "tokenizer_config.json"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def acquire(root, workers, metadata_only):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquire.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        official = root / "official"
        official.mkdir(exist_ok=True)
        for name in META:
            target = official / name
            if not target.exists():
                url = f"https://huggingface.co/{OFFICIAL}/resolve/{OFFICIAL_REV}/{name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                temp = target.with_suffix(target.suffix + ".partial")
                temp.write_bytes(payload)
                os.replace(temp, target)
        manifest = dict(official_repository=OFFICIAL, official_revision=OFFICIAL_REV,
                        quant_repository=QUANT, quant_revision=QUANT_REV,
                        filename=NAME, size=SIZE, sha256=SHA256,
                        metadata={name: digest(official / name) for name in META})
        atomic_json(root / "manifest.json", manifest)
        if metadata_only:
            print(json.dumps(manifest), flush=True)
            return
        final = root / NAME
        receipt = root / "acquisition.json"
        if final.exists():
            if final.stat().st_size != SIZE or digest(final) != SHA256:
                raise RuntimeError(f"Existing model fails pinned identity: {final}")
        else:
            parts = root / ".download-parts"
            parts.mkdir(exist_ok=True)
            url = f"https://huggingface.co/{QUANT}/resolve/{QUANT_REV}/{NAME}"
            def fetch(start):
                end = min(start + CHUNK, SIZE) - 1
                path = parts / str(start)
                length = end - start + 1
                if path.exists() and path.stat().st_size == length:
                    return length
                request = urllib.request.Request(url + f"?download=true&part={start}",
                          headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                # Transient transfer failures retain completed chunks, never publish a partial model.
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=90) as response:
                            expected = f"bytes {start}-{end}/{SIZE}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise RuntimeError(f"Invalid range response: {response.status} {response.headers.get('Content-Range')}")
                            temp = path.with_suffix(".partial")
                            with temp.open("wb") as out:
                                while b := response.read(1024 * 1024):
                                    out.write(b)
                            if temp.stat().st_size != length:
                                raise RuntimeError(f"Short range at {start}")
                            os.replace(temp, path)
                            return length
                    except (OSError, RuntimeError) as error:
                        print(f"range {start} attempt {attempt + 1}: {error}", flush=True)
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            done = 0
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(fetch, start) for start in range(0, SIZE, CHUNK)]
                for future in concurrent.futures.as_completed(futures):
                    done += future.result()
                    print(f"{done}/{SIZE} bytes ({done / SIZE:.1%}), {done / max(time.monotonic() - started, .01) / 1e6:.1f} MB/s", flush=True)
            temp = final.with_suffix(final.suffix + ".partial")
            prefix = min(temp.stat().st_size // CHUNK * CHUNK, SIZE) if temp.exists() else 0
            h = hashlib.sha256()
            with temp.open("r+b" if temp.exists() else "w+b") as out:
                out.truncate(prefix)
                out.seek(0)
                remaining = prefix
                while remaining:
                    b = out.read(min(8 * 1024 * 1024, remaining))
                    if not b:
                        raise RuntimeError("Assembled prefix vanished during resume")
                    h.update(b)
                    remaining -= len(b)
                out.seek(prefix)
                for start in range(prefix, SIZE, CHUNK):
                    with (parts / str(start)).open("rb") as source:
                        while b := source.read(8 * 1024 * 1024):
                            h.update(b)
                            out.write(b)
                out.flush()
                os.fsync(out.fileno())
            if temp.stat().st_size != SIZE or h.hexdigest() != SHA256:
                raise RuntimeError("Assembled model fails pinned size/SHA256; chunks retained for repair")
            os.replace(temp, final)
            for part in parts.iterdir():
                part.unlink()
            parts.rmdir()
        atomic_json(receipt, dict(**manifest, path=str(final), completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), verified=True))
        print(f"Verified {final} sha256={SHA256}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../../data/qwen-moe"))
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 17))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    acquire(args.root, args.workers, args.metadata_only)
