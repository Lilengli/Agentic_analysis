"""Compile the experimental torch extension on an Ascend 910B host."""
import os
import platform
import shlex
import subprocess
import sysconfig
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths

root = Path(__file__).resolve().parent
home = Path(os.environ.get('ASCEND_TOOLKIT_HOME') or os.environ.get('ASCEND_HOME_PATH')
            or '/usr/local/Ascend/ascend-toolkit/latest').resolve()
arch = platform.machine() + '-linux'
npu = Path(torch_npu.__file__).resolve().parent
includes = [home/'compiler/tikcpp/include', home/arch/'tikcpp/include', home/'include',
            home/'pkg_inc', home/'runtime/include', npu/'include', Path(sysconfig.get_path('include'))]
includes += [Path(p) for p in include_paths()]
libs = [home/'lib64', home/'compiler/lib64', home/arch/'lib64', npu/'lib']
libs += [Path(p) for p in library_paths()]
abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
obj = root/'flash_ascendc_probe.o'
so = root/('flash_ascendc_probe' + sysconfig.get_config_var('EXT_SUFFIX'))
compile_cmd = ['bisheng', '-O2', '-x', 'asc', '--npu-arch=dav-2201',
               '--cce-auto-infer-kernel-type=false', '-fPIC', '-std=c++17',
               f'-D_GLIBCXX_USE_CXX11_ABI={abi}', '-DTORCH_EXTENSION_NAME=flash_ascendc_probe']
compile_cmd += [f'-I{p}' for p in includes]
compile_cmd += ['-c', str(root/'flash_ascendc_probe.cpp'), '-o', str(obj)]
link_cmd = ['bisheng', '--npu-arch=dav-2201', '-shared', '-fPIC', str(obj)]
link_cmd += [f'-L{p}' for p in libs]
link_cmd += ['-ltorch_python', '-ltorch', '-ltorch_cpu', '-lc10', '-ltorch_npu', '-lascendcl', '-o', str(so)]
for command in (compile_cmd, link_cmd):
    print('+', ' '.join(shlex.quote(str(arg)) for arg in command), flush=True)
    subprocess.run([str(x) for x in command], check=True)
print('Built:', so)
