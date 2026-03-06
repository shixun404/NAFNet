#!/usr/bin/env python3
import argparse
import time
from collections import defaultdict

import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.profiler import ProfilerActivity, profile

from basicsr.models.archs.NAFNet_arch import NAFNet


def parse_block_list(text: str):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("block list is empty")
    return vals


def build_model(args):
    return NAFNet(
        img_channel=args.channels,
        width=args.width,
        middle_blk_num=args.middle,
        enc_blk_nums=parse_block_list(args.enc),
        dec_blk_nums=parse_block_list(args.dec),
    )


def model_stats(model, c, h, w):
    macs, params = get_model_complexity_info(
        model,
        (c, h, w),
        verbose=False,
        print_per_layer_stat=False,
    )
    return macs, params


def latency_ms(model, x, warmup, iters):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def module_breakdown(model, x, iters, topk):
    records = defaultdict(lambda: {"time_ms": 0.0, "calls": 0})
    hooks = []
    timings = {}

    use_cuda = x.is_cuda

    def pre_hook(name):
        def _hook(_m, _inp):
            if use_cuda:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                timings[name] = ev
            else:
                timings[name] = time.perf_counter()
        return _hook

    def post_hook(name):
        def _hook(_m, _inp, _out):
            if use_cuda:
                start_ev = timings.pop(name)
                end_ev = torch.cuda.Event(enable_timing=True)
                end_ev.record()
                torch.cuda.synchronize()
                elapsed = start_ev.elapsed_time(end_ev)
            else:
                elapsed = (time.perf_counter() - timings.pop(name)) * 1000.0
            records[name]["time_ms"] += elapsed
            records[name]["calls"] += 1
        return _hook

    # Leaf module breakdown gives practical optimization targets.
    for name, module in model.named_modules():
        if not name:
            continue
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            hooks.append(module.register_forward_hook(post_hook(name)))

    model.eval()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model(x)

    for h in hooks:
        h.remove()

    rows = []
    for name, item in records.items():
        calls = item["calls"]
        total = item["time_ms"]
        rows.append((name, total, total / max(calls, 1), calls))
    rows.sort(key=lambda t: t[1], reverse=True)

    print("\n[Module Time Breakdown] (top {})".format(topk))
    print("{:<50} {:>12} {:>12} {:>10}".format("module", "total_ms", "avg_ms", "calls"))
    for name, total, avg, calls in rows[:topk]:
        print("{:<50} {:>12.3f} {:>12.5f} {:>10}".format(name, total, avg, calls))


def op_breakdown(model, x, topk):
    activities = [ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if x.is_cuda:
        activities.append(ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with torch.inference_mode():
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()

    print("\n[Operator Time Breakdown] (top {})".format(topk))
    print(prof.key_averages().table(sort_by=sort_key, row_limit=topk))


def main():
    parser = argparse.ArgumentParser(description="Profile NAFNet with SIDD-style config.")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--enc", type=str, default="1,1,1,1")
    parser.add_argument("--dec", type=str, default="1,1,1,1")
    parser.add_argument("--middle", type=int, default=12)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width_px", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--module_topk", type=int, default=20)
    parser.add_argument("--op_topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = build_model(args)
    macs, params = model_stats(model, args.channels, args.height, args.width_px)

    device = torch.device(args.device)
    model = model.to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version")
        model = torch.compile(model, mode="max-autotune")

    x = torch.randn(args.batch, args.channels, args.height, args.width_px, device=device)
    if args.fp16 and device.type == "cuda":
        x = x.half()
    if args.channels_last and device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)

    lat = latency_ms(model, x, args.warmup, args.iters)

    print("[Config]")
    print(
        "width={}, enc={}, dec={}, middle={}, input=({}, {}, {}, {}), fp16={}, channels_last={}, compile={}".format(
            args.width, args.enc, args.dec, args.middle, args.batch, args.channels, args.height, args.width_px
            , args.fp16, args.channels_last, args.compile
        )
    )
    print("[Model]")
    print("macs:   {}".format(macs))
    print("params: {}".format(params))
    print("[Runtime]")
    print("avg_latency_ms: {:.4f}".format(lat))

    module_breakdown(model, x, args.iters, args.module_topk)
    op_breakdown(model, x, args.op_topk)


if __name__ == "__main__":
    main()
