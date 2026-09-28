#!/usr/bin/env python3
"""Build, retain and select the local Qwen HIP runtime without changing Bonsai."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('../../data/qwen-moe')
RUNTIMES = DATA / 'runtime'
MODEL = DATA / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def output(command):
    return subprocess.check_output([str(x) for x in command], text=True).strip()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(4 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def build(source, jobs):
    # NixOS owns the composed HIP compiler environment in the login profile.
    # Agent bash processes need not inherit it from their long-lived supervisor.
    configure = ['cmake', '-S', source, '-B', source / 'build', '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON',
        '-DGGML_HIP=ON', '-DGPU_TARGETS=gfx1151', '-DGGML_HIP_GRAPHS=ON',
        '-DGGML_HIP_NO_VMM=ON', '-DLLAMA_BUILD_TESTS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_BUILD_UI=OFF']
    for command in [configure, ['cmake', '--build', source / 'build', '--target',
                               'llama-cli', 'llama-bench', 'llama-server', 'llama-quantize', '-j', str(jobs)]]:
        run(['bash', '-lc', 'exec "$@"', 'qwen-build', *command])
    driver = source / 'build/bin/qwen-accept'
    run(['g++', '-O2', '-std=c++17', ROOT / 'tools/qwen-moe/accept.cpp',
         '-I' + str(source / 'include'), '-I' + str(source / 'ggml/include'),
         '-L' + str(source / 'build/bin'), '-Wl,-rpath,$ORIGIN', '-lllama',
         '-o', driver], env={**os.environ, 'NIX_STORE': '/nix/store'})
    # Nix's compiler wrapper injects the -L build directory into RUNPATH even
    # when $ORIGIN is explicitly requested. Remove only that build-only entry.
    patchelf = ['nix', 'shell', 'nixpkgs#patchelf', '-c', 'patchelf']
    rpath = output([*patchelf, '--print-rpath', driver]).split(':')
    build_lib = str(source / 'build/bin')
    if '$ORIGIN' not in rpath:
        raise ValueError(f'Missing relocatable qwen-accept RUNPATH: {rpath}')
    if build_lib in rpath:
        run([*patchelf, '--set-rpath', ':'.join(p for p in rpath if p != build_lib), driver])


def package(source):
    if output(['git', '-C', source, 'status', '--porcelain', '--untracked-files=no']):
        raise ValueError('Commit source before packaging the runtime')
    revision = output(['git', '-C', source, 'rev-parse', 'HEAD'])
    destination = RUNTIMES / revision
    if destination.exists():
        raise ValueError(f'Runtime already exists: {destination}')
    RUNTIMES.mkdir(parents=True, exist_ok=True)
    stage = RUNTIMES / ('.staging-' + revision)
    if stage.exists():
        raise ValueError(f'Unfinished package needs inspection: {stage}')
    stage.mkdir()
    shutil.copytree(source / 'build/bin', stage / 'bin', symlinks=True)
    shutil.copyfile(source / 'build/CMakeCache.txt', stage / 'CMakeCache.txt')
    run(['git', '-C', source, 'bundle', 'create', stage / 'source.bundle', 'HEAD'])
    files = {}
    for path in (stage / 'bin').iterdir():
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:
                if stream.read(4) != b'\x7fELF':
                    continue
            dynamic = output(['readelf', '-d', path])
            if str(source) in dynamic:
                raise ValueError(f'Artifact still depends on the task checkout: {path}; rebuild with origin RPATH')
            files[path.name] = digest(path)
    for name in ['llama-cli', 'llama-bench', 'llama-server']:
        if name not in files:
            raise ValueError(f'Missing built program {name}')
    manifest = dict(source_revision=revision,
        upstream='https://github.com/PrismML-Eng/llama.cpp',
        upstream_base='1a07bfa5f4144274c8f1c9963821dd9d9a51854b',
        source_bundle_sha256=digest(stage / 'source.bundle'),
        support_source_revision=output(['git', '-C', ROOT, 'rev-parse', 'HEAD']),
        acceptance_driver_sha256=digest(ROOT / 'tools/qwen-moe/accept.cpp'),
        cmake_cache_sha256=digest(stage / 'CMakeCache.txt'), artifacts=files,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (stage / 'runtime.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(stage, destination)
    # This remote gives the managed checkout's commits durable custody at release.
    custody = DATA / 'runtime-source.git'
    if not custody.exists():
        run(['git', 'init', '--bare', custody])
    run(['git', '-C', source, 'push', custody, f'HEAD:refs/heads/build-{revision}'])
    print(destination)


def select(revision, acceptance):
    candidate = RUNTIMES / revision
    manifest = json.loads((candidate / 'runtime.json').read_text())
    evidence = json.loads(acceptance.read_text())
    if evidence.get('source_revision') != revision or evidence.get('accepted') is not True:
        raise ValueError('Selection requires a successful receipt for this source revision')
    for name, expected in manifest['artifacts'].items():
        if digest(candidate / 'bin' / name) != expected:
            raise ValueError(f'Changed runtime artifact: {name}')
    selection = dict(source_revision=revision, acceptance=str(acceptance.resolve()),
                     acceptance_sha256=digest(acceptance),
                     default_arguments=evidence.get('default_arguments', []))
    (candidate / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    link = RUNTIMES / '.current'
    link.unlink(missing_ok=True)
    link.symlink_to(revision)
    os.replace(link, RUNTIMES / 'current')
    print(RUNTIMES / 'current')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=range(1, 9), default=6)
    p = sub.add_parser('package')
    p.add_argument('--source', type=Path, required=True)
    p = sub.add_parser('select')
    p.add_argument('revision')
    p.add_argument('--acceptance', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--runtime-max', default='300s')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plain', action='store_true', help='Disable the selected draft configuration')
    mode.add_argument('--mtp', action='store_true', help='Use the Q4 MTP candidate; batched greedy choices can differ')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.source.resolve(), args.jobs)
    elif args.action == 'package':
        package(args.source.resolve())
    elif args.action == 'select':
        select(args.revision, args.acceptance)
    else:
        binary = RUNTIMES / 'current/bin/llama-cli'
        if not binary.exists():
            raise ValueError('No accepted runtime selected')
        selection = json.loads((RUNTIMES / 'current/selection.json').read_text())
        defaults = [] if args.plain else selection['default_arguments']
        if args.mtp:
            head = DATA / 'mtp/mtp-Qwen3.6-35B-A3B-Q4_K_M.gguf'
            if not head.is_file():
                raise ValueError('Acquire and quantize the registered MTP head first')
            defaults = ['--model-draft', str(head), '--spec-type', 'draft-mtp',
                        '--spec-draft-ngl', '99', '--spec-draft-n-max', '2',
                        '--spec-draft-p-min', '0']
            print('Qwen MTP candidate: batched target arithmetic can change near-tie greedy choices.', file=sys.stderr)
        extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        command = [ROOT / 'tools/run-batch-compare', '--runtime-max', args.runtime_max,
                   '--memory-gib', '30', '--host-reserve-gib', '4', '--exec', binary,
                   '-m', MODEL, '-ngl', '99', '-fa', 'on', '-b', '512', '-ub', '256',
                   '-c', '8192', '-t', '8', '--parallel', '1', *defaults, *extra]
        os.execv(str(command[0]), [str(x) for x in command])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'qwen-runtime: {error}', file=sys.stderr)
        sys.exit(1)
