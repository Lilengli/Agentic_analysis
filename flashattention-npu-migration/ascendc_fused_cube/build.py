"""Build the mixed Cube attention on the target NPU host."""
import os
import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sysconfig
from pathlib import Path
import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths
from dependency import bundled_headers, REVISION, BUNDLE_SHA256

root = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--catlass-root', type=Path)
parser.add_argument('--legacy-sync-addr', action='store_true',
                    help='build with rtGetC2cCtrlAddr for simulator compatibility checks')
a=parser.parse_args()
base=Path('/inspire/sj-ssd3/global_user/260108540217')
external=a.catlass_root or (Path(os.environ['CATLASS_ROOT']) if os.environ.get('CATLASS_ROOT') else None)
if external and not (external/'include/catlass/gemm/tile/tile_copy.hpp').is_file():
    raise SystemExit('Explicit CATLASS root has no required headers: '+str(external))
candidates=[a.catlass_root, Path(os.environ['CATLASS_ROOT']) if os.environ.get('CATLASS_ROOT') else None,
            bundled_headers(root), base/'flash-attention-npu/csrc/catlass',
            Path('/home/gyli/flash-attention-npu/csrc/catlass')]
catlass=next((p.resolve() for p in candidates if p and (p/'include/catlass/gemm/tile/tile_copy.hpp').is_file()),None)
if catlass is None:
    raise SystemExit('CATLASS headers missing: pass --catlass-root /path/to/flash-attention-npu/csrc/catlass')
print('CATLASS:',catlass,flush=True)
home = Path(os.environ.get('ASCEND_TOOLKIT_HOME') or os.environ.get('ASCEND_HOME_PATH')
            or '/usr/local/Ascend/ascend-toolkit/latest').resolve()
arch = platform.machine() + '-linux'
npu = Path(torch_npu.__file__).resolve().parent
includes = [home/'compiler/tikcpp/include', home/arch/'tikcpp/include', home/'include',
            home/'pkg_inc', home/'runtime/include', npu/'include', Path(sysconfig.get_path('include'))]
includes += [Path(p) for p in include_paths()] + [catlass/'include']
libs = [home/'lib64', home/'compiler/lib64', home/arch/'lib64', npu/'lib']
libs += [Path(p) for p in library_paths()]
abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
obj = root/'fused_attention.o'
so = root/('fused_cube_ext' + sysconfig.get_config_var('EXT_SUFFIX'))
compile_cmd = ['bisheng', '-O2', '-x', 'asc', '--npu-arch=dav-2201',
               '--cce-auto-infer-kernel-type=false', '-fPIC', '-std=c++17',
               f'-D_GLIBCXX_USE_CXX11_ABI={abi}', '-DTORCH_EXTENSION_NAME=fused_cube_ext', '-DCATLASS_ARCH=2201', '-DK_MAX_SHAPE_DIM=0']
if a.legacy_sync_addr:
    legacy_roots = [home/'include/experiment/runtime', home/'pkg_inc/runtime',
                    home/arch/'include/experiment/runtime', home/arch/'pkg_inc/runtime']
    legacy_include = next((p for p in legacy_roots if (p/'runtime/rt_ffts.h').is_file()), None)
    if legacy_include is None:
        parser.error(f'legacy sync address requested, but runtime/rt_ffts.h was not found under {home}')
    includes.append(legacy_include)
    compile_cmd.append('-DCUBE_FA_LEGACY_SYNC_ADDR')
compile_cmd += [f'-I{p}' for p in includes] + ['-c', str(root/'fused_attention.cpp'), '-o', str(obj)]
link_cmd = ['bisheng', '--npu-arch=dav-2201', '-shared', '-fPIC', str(obj)]
link_cmd += [f'-L{p}' for p in libs] + ['-ltorch_python', '-ltorch', '-ltorch_cpu', '-lc10', '-ltorch_npu', '-lascendcl']
if a.legacy_sync_addr:
    link_cmd.append('-lruntime')
link_cmd += ['-o', str(so)]
for cmd in (compile_cmd, link_cmd):
    print('+', ' '.join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)
print('Built:', so)
manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('fused_attention.cpp', 'mixed_kernel.hpp', 'cube_tiles.hpp', 'geometry.hpp')}
manifest['catlass_root']=str(catlass)
manifest['bundled_catlass_revision']=REVISION
manifest['bundled_catlass_archive_sha256']=BUNDLE_SHA256
manifest['catlass_override']=bool(a.catlass_root or os.environ.get('CATLASS_ROOT'))
manifest['legacy_sync_addr'] = a.legacy_sync_addr
manifest['module_sha256'] = hashlib.sha256(so.read_bytes()).hexdigest()
(root/'build_manifest.json').write_text(json.dumps(manifest, indent=2))
