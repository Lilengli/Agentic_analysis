"""Compare CANN AIV gather and AIC direct KV load on the same sparse decode."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SOURCE_URL = 'https://gitcode.com/cann/ops-transformer.git'
SFA_HOST = Path('attention/sparse_flash_attention/op_host')
VENDOR = 'sfa_cube_probe'
ROUTE_ENV = 'SFA_BENCH_KV_ROUTE'
ROUTE_MARKER = re.compile(r'^SFA_BENCH_ROUTE=(vector|cube) TILING_KEY=([0-9]+)$')
BENCHMARK = Path(__file__).with_name('benchmark_sparse_flash_attention.py')


def patch_tiling(path):
    source = path.read_text(encoding='utf-8')
    init = re.search(
        r'void\s+SFAMlaTiling::InitParams\(\)\s*\{.*?^\}', source, re.MULTILINE | re.DOTALL)
    key = re.search(
        r'void\s+SFAMlaTiling::GenTilingKey\(\)\s*\{.*?^\}', source, re.MULTILINE | re.DOTALL)
    if init is None or key is None:
        raise RuntimeError(f'Expected SFAMlaTiling functions in {path}')

    init_body = init.group()
    key_body = key.group()
    core_line = re.search(r'^[ \t]*coreNum_\s*=\s*aicNum_\s*;', init_body, re.MULTILINE)
    if (core_line is None or 'V_TEMPLATE_MODE' not in init_body or 'C_TEMPLATE_MODE' not in init_body
            or 'V_TEMPLATE_MODE' not in key_body or 'GET_TPL_TILING_KEY' not in key_body):
        raise RuntimeError(f'Unexpected SFA dispatch in {path}')

    override = '''
    const char *route = std::getenv("SFA_BENCH_KV_ROUTE");
    if (route != nullptr && std::strcmp(route, "cube") == 0 &&
        sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize == 1) {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }

'''
    # Verify the effective tiling key once, before the benchmark starts timing.
    marker = '''
    if (std::getenv("SFA_BENCH_KV_ROUTE") != nullptr) {
        static std::atomic_flag reported = ATOMIC_FLAG_INIT;
        if (!reported.test_and_set()) {
            const char *effectiveRoute = perfMode_ == SFAPerfMode::V_TEMPLATE_MODE ? "vector" : "cube";
            std::fprintf(stderr, "SFA_BENCH_ROUTE=%s TILING_KEY=%llu\\n", effectiveRoute,
                         static_cast<unsigned long long>(tilingKey_));
            std::fflush(stderr);
        }
    }
'''
    patched_init = init_body[:core_line.start()] + override + init_body[core_line.start():]
    patched_key = key_body[:-1] + marker + '}'
    source = source.replace(init_body, patched_init, 1).replace(key_body, patched_key, 1)
    include = source.find('#include ')
    if include < 0:
        raise RuntimeError(f'No include section in {path}')
    source = (source[:include] + '#include <atomic>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n'
              + source[include:])
    path.write_text(source, encoding='utf-8')


def build_package(work, args):
    source = work / 'ops-transformer'
    source_url = str(args.op_source.resolve(strict=True)) if args.op_source else SOURCE_URL
    clone = ['git', 'clone', '--depth=1']
    if args.op_source:
        clone.append('--no-local')
    else:
        clone += ['--branch', args.cann_ref]
    subprocess.run([*clone, source_url, str(source)], check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    tilers = [path for path in (source / SFA_HOST).rglob('*.cpp')
              if 'SFAMlaTiling::InitParams' in path.read_text(encoding='utf-8')]
    if len(tilers) != 1:
        raise RuntimeError(f'Expected one SFAMlaTiling implementation under {source / SFA_HOST}, found {tilers}')
    patch_tiling(tilers[0])
    subprocess.run(
        ['bash', 'build.sh', '--pkg', '--soc=ascend910b', '--ops=sparse_flash_attention',
         f'--vendor_name={VENDOR}', f'-j{args.jobs}'],
        cwd=source, check=True,
    )
    packages = sorted((source / 'build_out').glob(f'*{VENDOR}*.run'))
    if len(packages) != 1:
        raise RuntimeError(f'Expected one {VENDOR} .run in {source / "build_out"}, found {packages}')
    install = work / 'opp'
    install.mkdir()
    env = os.environ.copy()
    env.pop('ASCEND_CUSTOM_OPP_PATH', None)
    subprocess.run(['bash', str(packages[0]), f'--install-path={install}'], check=True, env=env)
    loaders = sorted(install.glob(f'vendors/{VENDOR}*/bin/set_env.bash'))
    if len(loaders) != 1:
        raise RuntimeError(f'Expected one vendor environment script under {install}, found {loaders}')
    return loaders[0], revision


def run_case(work, loader, route, args):
    report = work / f'{route}.json'
    log = work / f'{route}.log'
    env = os.environ.copy()
    env.pop('ASCEND_CUSTOM_OPP_PATH', None)
    env[ROUTE_ENV] = route
    command = [
        'bash', '-c', 'source "$1" && shift && exec "$@"', 'sfa-ab', str(loader),
        sys.executable, '-u', str(BENCHMARK), '--impl', 'cann',
        '--seq-kv', '16384', '--topk', '2048', '--dtype', 'bf16',
        '--seed', str(args.seed), '--warmup', str(args.warmup), '--repeat', str(args.repeat),
        '--output', str(report),
    ]
    markers = set()
    with log.open('w') as log_file, subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
            marker = ROUTE_MARKER.fullmatch(line.strip())
            if marker:
                markers.add((marker[1], int(marker[2])))
        if process.wait() != 0:
            raise RuntimeError(f'{route} benchmark failed. See {log}')
    if len(markers) != 1 or next(iter(markers))[0] != route:
        raise RuntimeError(f'Could not verify the {route} tiling route. Markers: {markers}. See {log}')
    (_, key), = markers
    result = json.loads(report.read_text())
    if not result['results']['cann']['correct']:
        raise RuntimeError(f'{route} output failed reference validation. See {report}')
    result['tiling_key'] = key
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--op-source', type=Path, help='clone from a local, CANN-matched ops-transformer checkout')
    parser.add_argument('--cann-ref', default='9.1.0', help='official ops-transformer ref for the installed CANN')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('sfa_16k_cube_direct_ab.json'))
    args = parser.parse_args()
    if min(args.jobs, args.repeat) < 1 or args.warmup < 0:
        parser.error('jobs and repeat must be positive; warmup must be nonnegative')
    if not args.output.parent.is_dir():
        parser.error(f'output parent directory does not exist: {args.output.parent}')

    work = Path(tempfile.mkdtemp(prefix='sfa-cube-direct-'))
    print(f'SFA experiment directory: {work}', flush=True)
    loader, revision = build_package(work, args)
    vector = run_case(work, loader, 'vector', args)
    cube = run_case(work, loader, 'cube', args)
    if vector['tiling_key'] == cube['tiling_key']:
        raise RuntimeError(f'Both runs selected tiling key {cube["tiling_key"]}. See {work}')
    vector_us = vector['results']['cann']['event_median_us']
    cube_us = cube['results']['cann']['event_median_us']
    speedup = vector_us / cube_us
    result = {'ops_transformer_commit': revision, 'vendor_environment': str(loader), 'artifacts': str(work),
              'vector_gather': vector, 'cube_direct': cube, 'vector_over_cube_event_speedup': speedup}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'AIV gather {vector_us:.2f} us  AIC direct {cube_us:.2f} us  AIV/AIC={speedup:.3f}x', flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
