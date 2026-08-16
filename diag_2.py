"""checkpoint 诊断 v4：forward 与 recompute 各自的【有序】保存张量列表 + 汇总。

v3 的缺陷：recompute 阶段的 pack 走 _recomputation_hook（未打印），且 v2 用
frame.recomputed[gid]（WeakKeyDictionary，迭代无序）——无法按执行顺序对齐
forward 53 个与 recompute 50 个。v4 在 _recomputation_hook.pack_hook 里也打印，
两个 phase 的保存都按执行顺序编号，直接 diff 定位 3 个缺失张量的精确来源。
"""
import uuid
import weakref

import torch
import torch.utils.checkpoint as ck

_FWD_N = 0
_REC_N = 0


class _Handle:
    pass


class _Holder:
    def __init__(self):
        self.handles = dict()


class _StopRecomputationError(Exception):
    pass


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref, gid):
        def pack_hook(x):
            global _REC_N
            _REC_N += 1
            gf = x.grad_fn
            gfn = type(gf).__name__ if gf is not None else None
            print("PACK REC #%d shape=%s dtype=%s reqgrad=%s grad_fn=%s"
                  % (_REC_N, tuple(x.shape), x.dtype, x.requires_grad, gfn), flush=True)
            target_frame = target_frame_ref()
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1
            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    target_frame.ignore_saved_mismatch = True
                    return x.detach()
                raise ck.CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass.")
            holder = target_frame.weak_holders[recomp_idx]()
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x.detach()
            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                    target_frame.weak_holders):
                raise _StopRecomputationError()
            return x.detach()

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)


def _internal_assert(cond):
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch.")


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            global _FWD_N
            gf = x.grad_fn
            gfn = type(gf).__name__ if gf is not None else None
            if frame.forward_completed:
                tag = "REC-OUT"
            else:
                _FWD_N += 1
                tag = "FWD"
            print("PACK %s #%d shape=%s dtype=%s reqgrad=%s grad_fn=%s"
                  % (tag, _FWD_N if tag == "FWD" else _REC_N,
                     tuple(x.shape), x.dtype, x.requires_grad, gfn), flush=True)
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                gid = int(uuid.uuid4())
            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
                try:
                    with _recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)
            _internal_assert(gid in holder.handles)
            if holder.handles[gid] is None:
                raise ck.CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case.")
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except ck.CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)


ck._checkpoint_hook = _checkpoint_hook

with open("1.py", "r", encoding="utf-8") as f:
    _code = f.read()
_code = _code.replace('if __name__ == "__main__":\n    main()', "main()")
exec(compile(_code, "1.py", "exec"))
