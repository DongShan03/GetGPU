import sys
import time
import argparse
import gc
import os
import random

BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0
MAX_GRID_DIM = 1 << 15
MAX_BLOCK_DIM = 1024

cp = None
default_script_kernel = None

# 使用 CuPy 的 RawKernel 编译原生的 C++ CUDA 核心
kernel_code = r'''
extern "C" __global__
void default_script_kernel(char* array, unsigned long long occupy_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= occupy_size) return;

    float val = 0.0f;
    for (int k = 0; k < 2000; ++k) {
        val += k * 0.001f;
        if (k % 500 == 0) {
            array[i] = (char)(val);
        }
    }
    array[i]++;
}
'''


def init_cupy():
    global cp, default_script_kernel

    if cp is not None:
        return

    try:
        import cupy as cupy_module
    except ImportError:
        print("Error: 该脚本依赖 'cupy' 库。请通过 'pip install cupy-cudaXXX' (例如 cupy-cuda12x) 安装", file=sys.stderr)
        sys.exit(1)

    cp = cupy_module
    default_script_kernel = cp.RawKernel(kernel_code, 'default_script_kernel')


def launch_default_script(arrays, occupy_sizes, gpu_ids):
    for gid in gpu_ids:
        occupy_size = occupy_sizes[gid]
        gd = int((occupy_size + MAX_BLOCK_DIM - 1) // MAX_BLOCK_DIM)
        if gd > MAX_GRID_DIM:
            gd = MAX_GRID_DIM

        with cp.cuda.Device(gid):
            # 这里的传参类似 kernel<<<gd, max_block_dim>>>(...)
            default_script_kernel((gd,), (MAX_BLOCK_DIM,), (arrays[gid], occupy_size))


def run_default_script(arrays, occupy_sizes, total_time, gpu_ids, utilization, utilization_jitter):
    print(
        "Running default script with target utilization: "
        f"{utilization * 100:.2f}% +/- {utilization_jitter * 100:.2f}% >>>>>>>>>>>>>>>>>>>>"
    )

    start_total = time.monotonic()
    last_log_time = start_total
    next_jitter_time = start_total
    current_utilization = utilization

    while True:
        now = time.monotonic()
        if now >= next_jitter_time:
            jitter_low = max(0.05, utilization - utilization_jitter)
            jitter_high = min(1.0, utilization + utilization_jitter)
            current_utilization = random.uniform(jitter_low, jitter_high)
            next_jitter_time = now + random.uniform(3.0, 8.0)

        t1 = time.perf_counter()

        launch_default_script(arrays, occupy_sizes, gpu_ids)

        for gid in gpu_ids:
            with cp.cuda.Device(gid):
                cp.cuda.runtime.deviceSynchronize()

        t2 = time.perf_counter()
        on_time_ms = (t2 - t1) * 1000.0

        if current_utilization < 1.0 and on_time_ms > 0:
            off_time_ms = on_time_ms * (1.0 / current_utilization - 1.0)
            if off_time_ms > 1.0:
                time.sleep(off_time_ms / 1000.0)

        now = time.monotonic()
        elapsed_hours = (now - start_total) / 3600.0

        if elapsed_hours > total_time:
            break

        if (now - last_log_time) > 10.0:
            print(
                f"Occupied time: {elapsed_hours:.2f} hours "
                f"(Current Target Utilization: {current_utilization * 100:.2f}%, "
                f"Last Kernel Duration: {on_time_ms:.3f} ms)"
            )
            last_log_time = now

    # 释放内存池，等效于 C++ 的 cudaFree
    for gid in gpu_ids:
        with cp.cuda.Device(gid):
            arrays[gid] = None
    cp.get_default_memory_pool().free_all_blocks()


def process_args():
    parser = argparse.ArgumentParser(
        description="Occupy GPU memory/utilization, optionally run a command after GPUs are available."
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="GPU IDs to occupy, separated by commas or spaces. Use -1 for all GPUs.",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="GPU memory to occupy per selected GPU in GB. Defaults to 60%% of each GPU total memory.",
    )
    parser.add_argument(
        "--mem-ratio",
        type=float,
        default=0.6,
        help="GPU memory ratio to occupy when --mem-gb is not set. Default: 0.6.",
    )
    parser.add_argument(
        "--time-hours",
        type=float,
        default=12.0,
        help="Occupied time in hours for the default script. Default: 12.",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.7,
        help="Target GPU utilization for the default script, in (0.0, 1.0]. Default: 0.7.",
    )
    parser.add_argument(
        "--utilization-jitter",
        type=float,
        default=0.08,
        help="Random utilization jitter around --utilization. Default: 0.08.",
    )
    parser.add_argument(
        "--cmd",
        default=None,
        help="Optional full command string to run after GPU memory is acquired.",
    )
    args = parser.parse_args()
    init_cupy()

    total_time = args.time_hours

    # 处理 GPU IDs 字符串，支持逗号或空格分隔
    gpu_ids_str = args.gpus.replace(',', ' ')
    gpu_ids = [int(x) for x in gpu_ids_str.split() if x.strip()]
    if not gpu_ids:
        raise ValueError("GPU ID is required")

    utilization = args.utilization
    custom_cmd = args.cmd

    num_gpus = cp.cuda.runtime.getDeviceCount()

    # 处理 -1 的情况 (使用全部 GPU)
    if len(gpu_ids) == 1 and gpu_ids[0] == -1:
        gpu_ids = list(range(num_gpus))

    for gid in gpu_ids:
        if gid < 0 or gid >= num_gpus:
            print(f"Invalid GPU ID ({num_gpus} GPU in total): {gid}")
            raise ValueError("Invalid GPU ID")

    if total_time < 0:
        print(f"Occupied time must be non-negative: {total_time:.2f}")
        raise ValueError("Invalid occupied time")
    if utilization <= 0.0 or utilization > 1.0:
        print(f"Utilization must be in range (0.0, 1.0]: {utilization:.2f}")
        raise ValueError("Invalid utilization")
    if args.utilization_jitter < 0.0 or args.utilization_jitter > 1.0:
        print(f"Utilization jitter must be in range [0.0, 1.0]: {args.utilization_jitter:.2f}")
        raise ValueError("Invalid utilization jitter")
    if args.mem_ratio <= 0.0 or args.mem_ratio > 1.0:
        print(f"Memory ratio must be in range (0.0, 1.0]: {args.mem_ratio:.2f}")
        raise ValueError("Invalid memory ratio")

    occupy_sizes = {}
    occupy_mem_gb = {}
    for gid in gpu_ids:
        with cp.cuda.Device(gid):
            _, total_size = cp.cuda.runtime.memGetInfo()

        if args.mem_gb is None:
            occupy_size = int(total_size * args.mem_ratio)
        else:
            if args.mem_gb <= 0:
                print(f"GPU memory must be positive: {args.mem_gb:.2f}")
                raise ValueError("Invalid GPU memory")
            occupy_size = int(args.mem_gb * BYTES_PER_GB)

        if occupy_size > total_size:
            print(
                f"GPU-{gid}: GPU memory exceeds maximum "
                f"({total_size / BYTES_PER_GB:.2f} GB): {occupy_size / BYTES_PER_GB:.2f} GB"
            )
            raise ValueError("Exceed maximal GPU memory")

        occupy_sizes[gid] = occupy_size
        occupy_mem_gb[gid] = occupy_size / BYTES_PER_GB

    print("GPU ID: " + ",".join(map(str, gpu_ids)))
    print(
        "GPU memory (GB): "
        + ", ".join(f"GPU-{gid}={occupy_mem_gb[gid]:.2f}" for gid in gpu_ids)
    )
    print(f"Occupied time (h): {total_time:.2f}")
    print(f"Target Utilization: {utilization * 100:.2f}%")
    print(f"Utilization Jitter: +/- {args.utilization_jitter * 100:.2f}%")

    if custom_cmd is not None:
        print(f"Custom command: {custom_cmd}")

    return occupy_sizes, total_time, gpu_ids, utilization, args.utilization_jitter, custom_cmd


def allocate_mem(occupy_sizes, gpu_ids):
    arrays = {gid: None for gid in gpu_ids}
    allocated = {gid: False for gid in gpu_ids}
    cnt = 0

    while True:
        cnt += 1
        print(f"Try allocate GPU memory {cnt} times >>>>>>>>>>>>>>>>>>>>")
        all_allocated = True

        for gid in gpu_ids:
            if not allocated[gid]:
                occupy_size = occupy_sizes[gid]
                try:
                    with cp.cuda.Device(gid):
                        # 分配数组 (占用指定大小显存)
                        arrays[gid] = cp.empty(occupy_size, dtype=cp.int8)
                        free_size, total_size = cp.cuda.runtime.memGetInfo()
                        print(f"GPU-{gid}: Successfully allocate {occupy_size / BYTES_PER_GB:.2f} GB GPU memory ({free_size / BYTES_PER_GB:.2f} GB available)")
                        allocated[gid] = True
                except cp.cuda.memory.OutOfMemoryError:
                    with cp.cuda.Device(gid):
                        free_size, total_size = cp.cuda.runtime.memGetInfo()
                    print(f"GPU-{gid}: Failed to allocate {occupy_size / BYTES_PER_GB:.2f} GB GPU memory ({free_size / BYTES_PER_GB:.2f} GB available)")
                    all_allocated = False

        if all_allocated:
            break
        # 等待重试
        time.sleep(60.0)

    print("Successfully allocate memory on all GPUs!")
    return arrays


def run_custom_script(arrays, gpu_ids, custom_cmd):
    print("Running custom script >>>>>>>>>>>>>>>>>>>>")
    # 释放所有的占用内存，以便将资源交给 custom script 使用
    for gid in gpu_ids:
        with cp.cuda.Device(gid):
            arrays[gid] = None

    arrays.clear()
    gc.collect()

    for gid in gpu_ids:
        with cp.cuda.Device(gid):
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(f"GPU-{gid}: released CuPy memory pool")

    sys.stdout.flush()
    sys.stderr.flush()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    sys.stdout.flush()
    os.execvp("bash", ["bash", "-lc", custom_cmd])


def main():
    try:
        occupy_sizes, total_time, gpu_ids, utilization, utilization_jitter, custom_cmd = process_args()
        arrays = allocate_mem(occupy_sizes, gpu_ids)

        if custom_cmd is None:
            run_default_script(arrays, occupy_sizes, total_time, gpu_ids, utilization, utilization_jitter)
        else:
            run_custom_script(arrays, gpu_ids, custom_cmd)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
