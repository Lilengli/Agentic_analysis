"""Verify physical Cube QK/PV independently of mixed-kernel handshakes."""
import torch
import torch_npu  # noqa: F401
import fused_cube_ext


def run_checks():
    with torch.inference_mode():
        for bm in (32,64,128):
            for bn in (64,128):
                if bm > bn:
                    continue
                for transpose in (True,False):
                    k,n=(128,bn) if transpose else (bn,128)
                    for seed in (0,1234):
                        if seed==0:
                            a=((torch.arange(bm*k).reshape(bm,k)%17)-8).float()/8
                            rows,cols=(n,k) if transpose else (k,n)
                            b=((torch.arange(rows)[:,None]*13+torch.arange(cols)[None,:]*7)%19-9).float()/9
                            a,b=a.half(),b.half()
                        else:
                            torch.manual_seed(seed)
                            a=torch.randn((bm,k),dtype=torch.float16)
                            b=torch.randn((n,k) if transpose else (k,n),dtype=torch.float16)
                        expected=a.float()@(b.float().t() if transpose else b.float())
                        result=fused_cube_ext.product(a.to('npu:0'),b.to('npu:0'),transpose)
                        torch.npu.synchronize()
                        actual=result.cpu()
                        passed=torch.allclose(actual,expected,atol=.002,rtol=.01)
                        error=(actual-expected).abs().max().item()
                        label='QK' if transpose else 'PV'
                        print(f'Cube {label} BM={bm} BN={bn} seed={seed}: correct={passed} max_abs={error:.8g}',flush=True)
                        if not passed:
                            raise RuntimeError('Cube primitive/layout check failed; mixed attention must not run')


if __name__=='__main__':
    run_checks()
