"""Build the custom online-softmax stage on the target NPU host."""
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
obj = root/'online_stage.o'
so = root/('online_stage_ext' + sysconfig.get_config_var('EXT_SUFFIX'))
compile_cmd = ['bisheng', '-O2', '-x', 'asc', '--npu-arch=dav-2201',
               '--cce-auto-infer-kernel-type=false', '-fPIC', '-std=c++17',
               f'-D_GLIBCXX_USE_CXX11_ABI={abi}', '-DTORCH_EXTENSION_NAME=online_stage_ext']
compile_cmd += [f'-I{p}' for p in includes] + ['-c', str(root/'online_stage.cpp'), '-o', str(obj)]
link_cmd = ['bisheng', '--npu-arch=dav-2201', '-shared', '-fPIC', str(obj)]
link_cmd += [f'-L{p}' for p in libs] + ['-ltorch_python', '-ltorch', '-ltorch_cpu', '-lc10', '-ltorch_npu', '-lascendcl', '-o', str(so)]
for cmd in (compile_cmd, link_cmd):
    print('+', ' '.join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)
print('Built:', so)
