"""probe_fa_grad.py — FA backward 定向排查（910B / MS 2.7 / CANN 25.2.1）。

背景：probe_p0_910b 发现 ops.flash_attention_score 的 backward 在 GQA 小形状
(B=2,S=64,N=4,KV=2) 下报 aclnnReduceSumGetWorkspaceSize EZ1001。
本探针在真实训练形状与"模型路径"（kv repeat 到 N 头 + sparse_mode=0）下
验证 forward+backward，并与慢注意力数值梯度对比。

用例（每例输出 PASS/FAIL + rel_err + wall）：
  PYNATIVE:
    A1 模型路径 fp16  B=8  S=2048  kv2→repeat6  sparse=0
    A2 模型路径 bf16  B=8  S=2048  kv2→repeat6  sparse=0
    B1 非GQA(kv=12)   fp16  B=8  S=2048  sparse=0（隔离 GQA 内核问题）
    C1 模型路径 fp16  B=32 S=2048（真实训练形状，仅崩溃/有限性检查）
    D1 sparse=2+因果keep-mask fp16 B=8（备用路径；自动尝试 1/0 与 -inf 两种 mask 约定）
    E1 模型路径 fp16 inner_precise=1 B=8
  GRAPH:
    F1 模型路径 fp16 B=8 S=2048（编译后跑一次，仅崩溃检查）
    F2 模型路径 fp16 B=32 S=2048

数值比较：与 SlowCell（repeat_interleave + matmul + 因果 bias + softmax）
的 value_and_grad 对比 dq/dk/dv，rel_err 阈值 fp16 1e-2 / bf16 2e-2。

用法（910B）：
    python probe_fa_grad.py
"""
import sys
import time
from pathlib import Path

import numpy as np

import mindspore as ms
from mindspore import context, nn, ops

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402


def _causal_bias(T, dtype):
    b = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
    return ms.Tensor(b, dtype)[None, None]


def make_inputs(B, T, N, KV, D, dtype):
    q = ms.Tensor(np.random.randn(B, N, T, D).astype(np.float32) * 0.5, dtype)
    k = ms.Tensor(np.random.randn(B, KV, T, D).astype(np.float32) * 0.5, dtype)
    v = ms.Tensor(np.random.randn(B, KV, T, D).astype(np.float32) * 0.5, dtype)
    return q, k, v


def slow_grads(q, k, v, N, KV, repeats, D, T, dtype):
    """慢注意力数值梯度（模型慢路径逐 op 一致）。"""
    scale = D ** -0.5
    bias = _causal_bias(T, dtype)

    class SlowCell(nn.Cell):
        def construct(self, qq, kk, vv):
            if repeats > 1:
                kk = kk.repeat_interleave(repeats, dim=1)
                vv = vv.repeat_interleave(repeats, dim=1)
            att = ops.matmul(qq * scale, kk.swapaxes(-2, -1)) + bias
            p = ops.softmax(att, axis=-1)
            return (p @ vv).sum()

    _, (dq, dk, dv) = ms.value_and_grad(SlowCell(), grad_position=(0, 1, 2))(q, k, v)
    return dq, dk, dv


class FACell(nn.Cell):
    """模型 FA 路径：kv repeat → N 头，sparse_mode=s，可选 attn_mask。"""

    def __init__(self, N, repeats, smode, scale, mask=None, inner=0):
        super().__init__()
        self.N = N
        self.repeats = repeats
        self.smode = smode
        self.scale = ms.Tensor([scale], ms.float32)
        self.mask = mask
        self.inner = inner

    def construct(self, qq, kk, vv):
        if self.repeats > 1:
            kk = kk.repeat_interleave(self.repeats, dim=1)
            vv = vv.repeat_interleave(self.repeats, dim=1)
        out = ops.flash_attention_score(
            qq, kk, vv, self.N, keep_prob=1.0, scalar_value=self.scale,
            pre_tokens=2147483647, next_tokens=0, inner_precise=self.inner,
            input_layout="BNSD", sparse_mode=self.smode, attn_mask=self.mask)
        y = out[0] if isinstance(out, tuple) else out
        return y.sum()


def run_case(name, q, k, v, N, KV, repeats, D, T, dtype, smode,
             inner=0, mask=None, check_num=True, tol=1e-2):
    t0 = time.time()
    cell = FACell(N, repeats, smode, D ** -0.5, mask=mask, inner=inner)
    try:
        _, (dq, dk, dv) = ms.value_and_grad(cell, grad_position=(0, 1, 2))(q, k, v)
        fin = (np.isfinite(dq.asnumpy()).all()
               and np.isfinite(dk.asnumpy()).all()
               and np.isfinite(dv.asnumpy()).all())
        if not fin:
            print(f"{name}: FAIL non-finite grads ({time.time()-t0:.1f}s)", flush=True)
            return False
        if check_num:
            rq, rk, rv = slow_grads(q, k, v, N, KV, repeats, D, T, dtype)
            err = max(float(np.abs(dq.asnumpy() - rq.asnumpy()).max())
                      / (float(np.abs(rq.asnumpy()).max()) + 1e-8),
                      float(np.abs(dk.asnumpy() - rk.asnumpy()).max())
                      / (float(np.abs(rk.asnumpy()).max()) + 1e-8),
                      float(np.abs(dv.asnumpy() - rv.asnumpy()).max())
                      / (float(np.abs(rv.asnumpy()).max()) + 1e-8))
            ok = err < tol
            print(f"{name}: {'PASS' if ok else 'FAIL'} rel_err={err:.4e} "
                  f"(tol {tol}) ({time.time()-t0:.1f}s)", flush=True)
            return ok
        print(f"{name}: PASS finite grads ({time.time()-t0:.1f}s)", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAIL ({str(e)[:160]!r}) ({time.time()-t0:.1f}s)", flush=True)
        return False


def main():
    device_lib.init_ms(mode="pynative")
    assert ms.get_context("device_target") == "Ascend", "Ascend only"
    print(f"MS {ms.__version__} python target=Ascend", flush=True)
    B, T, N, KV, D = 8, 2048, 12, 2, 64
    repeats = N // KV
    q, k, v = make_inputs(B, T, N, KV, D, ms.float16)
    qb, kb, vb = make_inputs(B, T, N, KV, D, ms.bfloat16)

    print("\n-- PYNATIVE --", flush=True)
    run_case("A1 model-path fp16 B8 sparse0", q, k, v, N, KV, repeats, D, T,
             ms.float16, 0)
    run_case("A2 model-path bf16 B8 sparse0", qb, kb, vb, N, KV, repeats, D, T,
             ms.bfloat16, 0, tol=2e-2)
    qn, kn, vn = make_inputs(B, T, N, N, D, ms.float16)   # kv = N（非 GQA）
    run_case("B1 non-GQA kv=12 fp16 B8 sparse0", qn, kn, vn, N, N, 1, D, T,
             ms.float16, 0)

    # 真实训练形状（B=32）：仅崩溃/有限性检查，不做慢参考（代价高）
    q32, k32, v32 = make_inputs(32, T, N, KV, D, ms.float16)
    run_case("C1 model-path fp16 B32 sparse0", q32, k32, v32, N, KV, repeats,
             D, T, ms.float16, 0, check_num=False)

    # sparse=2 + 自定义 mask（备用路径）：先试 1/0 keep 约定，再试 -inf 加法约定
    mask_keep = ms.Tensor(np.tril(np.ones((T, T), np.float32)),
                          ms.float16)
    run_case("D1 sparse2 keep-mask fp16 B8", q, k, v, N, KV, repeats, D, T,
             ms.float16, 2, mask=mask_keep)
    mask_add = ms.Tensor(np.triu(np.full((T, T), -np.inf, np.float32), k=1),
                         ms.float16)
    run_case("D2 sparse2 add-mask fp16 B8", q, k, v, N, KV, repeats, D, T,
             ms.float16, 2, mask=mask_add)

    run_case("E1 model-path fp16 inner=1 B8 sparse0", q, k, v, N, KV, repeats,
             D, T, ms.float16, 0, inner=1)

    print("\n-- GRAPH mode --", flush=True)
    context.set_context(mode=context.GRAPH_MODE)
    run_case("F1 model-path fp16 B8 sparse0 graph", q, k, v, N, KV, repeats,
             D, T, ms.float16, 0, check_num=False)
    run_case("F2 model-path fp16 B32 sparse0 graph", q32, k32, v32, N, KV,
             repeats, D, T, ms.float16, 0, check_num=False)

    print("\nFA grad probe DONE", flush=True)


if __name__ == "__main__":
    main()