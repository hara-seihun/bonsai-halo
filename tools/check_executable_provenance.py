#!/usr/bin/env python3
"""Check that each copied executable reports its own bytes, not a build-tree path."""
import hashlib
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as directory:
    paths = []
    for marker in (1, 2):
        exe = Path(directory) / f"dummy-{marker}"
        source = (
            '#include "executable_provenance.hpp"\n'
            '#include <iostream>\n'
            f'int main() {{ std::cout << executable_sha256() << " {marker}\\n"; }}\n'
        )
        subprocess.run(["c++", "-std=c++17", "-O0", "-x", "c++", "-", "-I", str(root), "-o", str(exe)],
                       input=source, text=True, check=True, timeout=10)
        paths.append(exe)
    assert hashlib.sha256(paths[0].read_bytes()).digest() != hashlib.sha256(paths[1].read_bytes()).digest()
    for marker, exe in enumerate(paths, 1):
        output = subprocess.check_output([str(exe)], text=True, timeout=5).strip()
        expected = f"{hashlib.sha256(exe.read_bytes()).hexdigest()}  {exe} {marker}"
        assert output == expected, (output, expected)
print("two copied executables report their own distinct hashes")
