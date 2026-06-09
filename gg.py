import argparse
import gc
import json
import os
import random
import signal
import shutil
import subprocess
import sys
import time

BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0
DEFAULT_GEMM_M = 8192
DEFAULT_GEMM_K = 12288
DEFAULT_GEMM_N = 16384
DEFAULT_GEMM_INNER_ITERS = 4
GEMM_CANDIDATES = (
    (1024, 2048, 2048),
    (2048, 4096, 4096),
    (4096, 4096, 8192),
    (4096, 8192, 8192),
    (8192, 8192, 8192),
    (8192, 8192, 12288),
    (8192, 12288, 16384),
    (12288, 12288, 16384),
    (12288, 16384, 16384),
    (16384, 16384, 16384),
    (16384, 16384, 24576),
    (16384, 24576, 24576),
    (24576, 24576, 24576),
)

torch = None


def init_torch():
    global torch

    if torch is not None:
        return torch

    try:
        import torch as torch_module
    except ImportError:
        print("Error: this script requires PyTorch with CUDA support.", file=sys.stderr)
        sys.exit(1)

    if not torch_module.cuda.is_available():
        print("Error: CUDA is not available in PyTorch.", file=sys.stderr)
        sys.exit(1)

    torch = torch_module
    return torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Occupy GPU memory and run a torch GEMM workload locally or through Ray jobs."
    )
    parser.add_argument(
        "--gpus",
        default="-1",
        help="GPU IDs to occupy, separated by commas or spaces. Use -1 for all visible GPUs.",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="GPU memory to occupy per selected GPU in GB. Overrides --mem-ratio.",
    )
    parser.add_argument(
        "--mem-ratio",
        type=float,
        default=0.9,
        help="GPU memory ratio to occupy when --mem-gb is not set. Default: 0.9.",
    )
    parser.add_argument(
        "--allocation-mode",
        choices=("incremental", "all-or-nothing"),
        default="incremental",
        help=(
            "incremental keeps already occupied GPUs while retrying the rest; "
            "all-or-nothing releases partial allocations when any selected GPU fails. "
            "Default: incremental."
        ),
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying failed GPU memory allocation. Default: 60.",
    )
    parser.add_argument(
        "--time-hours",
        type=float,
        default=24.0,
        help="Occupied time in hours. Default: 24.",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=1.0,
        help="Target GPU utilization in (0.0, 1.0]. Default: 1.0.",
    )
    parser.add_argument(
        "--utilization-jitter",
        type=float,
        default=0.0,
        help="Random utilization jitter around --utilization. Default: 0.0.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=300.0,
        help="Seconds between progress logs from rank 0. Use 0 to disable progress logs. Default: 300.",
    )
    parser.add_argument(
        "--cmd",
        default=None,
        help="Optional command to run after GPU memory is acquired. Memory is released before this command starts.",
    )
    parser.add_argument(
        "--gemm-shape",
        choices=("auto", "fixed"),
        default="auto",
        help="Choose GEMM shape per GPU automatically, or use --gemm-m/k/n. Default: auto.",
    )
    parser.add_argument(
        "--gemm-m",
        type=int,
        default=DEFAULT_GEMM_M,
        help=f"Torch GEMM M dimension. Default: {DEFAULT_GEMM_M}.",
    )
    parser.add_argument(
        "--gemm-k",
        type=int,
        default=DEFAULT_GEMM_K,
        help=f"Torch GEMM K dimension. Default: {DEFAULT_GEMM_K}.",
    )
    parser.add_argument(
        "--gemm-n",
        type=int,
        default=DEFAULT_GEMM_N,
        help=f"Torch GEMM N dimension. Default: {DEFAULT_GEMM_N}.",
    )
    parser.add_argument(
        "--gemm-warmup",
        type=int,
        default=20,
        help="Torch GEMM warmup iterations. Default: 20.",
    )
    parser.add_argument(
        "--gemm-sleep",
        type=float,
        default=0.0,
        help="Extra sleep seconds after each GEMM iteration. Default: 0.0.",
    )
    parser.add_argument(
        "--gemm-inner-iters",
        type=int,
        default=DEFAULT_GEMM_INNER_ITERS,
        help=(
            "Number of GEMM forward passes accumulated before one backward/sync. "
            f"Higher values reduce CPU scheduling overhead. Default: {DEFAULT_GEMM_INNER_ITERS}."
        ),
    )
    parser.add_argument(
        "--gemm-auto-target-ms",
        type=float,
        default=1000.0,
        help="Target single-iteration time used by auto shape selection. Default: 1000.",
    )
    parser.add_argument(
        "--gemm-auto-memory-ratio",
        type=float,
        default=0.75,
        help="Fraction of currently free memory allowed for auto shape candidates. Default: 0.75.",
    )
    parser.add_argument(
        "--gemm-auto-benchmark-iters",
        type=int,
        default=2,
        help="Benchmark iterations per candidate during auto shape selection. Default: 2.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=12345,
        help="Master port used by local torch.distributed.run. Default: 12345.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of Ray jobs to submit. Values greater than 1 enable Ray submit mode.",
    )
    parser.add_argument(
        "--ray-submit",
        action="store_true",
        help="Submit through Ray job submit instead of occupying local GPUs.",
    )
    parser.add_argument(
        "--ray-wait",
        action="store_true",
        help="Wait for Ray jobs and stream their logs. Default: submit jobs with --no-wait.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="GPUs reserved by each Ray job. Defaults to selected GPU count, or 8 when --gpus=-1.",
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS"),
        help="Ray jobs server address. Defaults to RAY_ADDRESS.",
    )
    parser.add_argument(
        "--working-dir",
        default=os.getcwd(),
        help="Working directory uploaded to Ray jobs. Default: current working directory.",
    )
    parser.add_argument(
        "--ray-job-prefix",
        default="gg-occupy",
        help="Prefix used in Ray job submit messages. Default: gg-occupy.",
    )
    parser.add_argument(
        "--ray-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for Ray runtime_env. Can be repeated.",
    )
    parser.add_argument(
        "--ray-pip",
        action="append",
        default=[],
        metavar="PKG[,PKG...]",
        help="Pip packages for Ray runtime_env. Can be repeated or comma separated.",
    )
    parser.add_argument("--gemm-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_gpu_ids(gpus):
    gpu_ids = [int(x) for x in gpus.replace(",", " ").split() if x.strip()]
    if not gpu_ids:
        raise ValueError("GPU ID is required")
    return gpu_ids


def selected_gpu_count(gpus):
    gpu_ids = parse_gpu_ids(gpus)
    if len(gpu_ids) == 1 and gpu_ids[0] == -1:
        return None
    return len(gpu_ids)


def validate_common_args(args):
    if args.time_hours < 0:
        raise ValueError(f"Occupied time must be non-negative: {args.time_hours:.2f}")
    if args.utilization <= 0.0 or args.utilization > 1.0:
        raise ValueError(f"Utilization must be in range (0.0, 1.0]: {args.utilization:.2f}")
    if args.utilization_jitter < 0.0 or args.utilization_jitter > 1.0:
        raise ValueError(f"Utilization jitter must be in range [0.0, 1.0]: {args.utilization_jitter:.2f}")
    if args.mem_ratio <= 0.0 or args.mem_ratio > 1.0:
        raise ValueError(f"Memory ratio must be in range (0.0, 1.0]: {args.mem_ratio:.2f}")
    if args.mem_gb is not None and args.mem_gb <= 0:
        raise ValueError(f"GPU memory must be positive: {args.mem_gb:.2f}")
    if args.retry_interval < 0.0:
        raise ValueError(f"Retry interval must be non-negative: {args.retry_interval:.2f}")
    if args.log_interval < 0.0:
        raise ValueError(f"Log interval must be non-negative: {args.log_interval:.2f}")
    if args.gemm_m <= 0 or args.gemm_k <= 0 or args.gemm_n <= 0:
        raise ValueError("GEMM dimensions must be positive")
    if args.gemm_warmup < 0:
        raise ValueError("GEMM warmup iterations must be non-negative")
    if args.gemm_sleep < 0.0:
        raise ValueError("GEMM sleep must be non-negative")
    if args.gemm_inner_iters <= 0:
        raise ValueError("GEMM inner iterations must be positive")
    if args.gemm_auto_target_ms <= 0.0:
        raise ValueError("Auto GEMM target ms must be positive")
    if args.gemm_auto_memory_ratio <= 0.0 or args.gemm_auto_memory_ratio > 1.0:
        raise ValueError("Auto GEMM memory ratio must be in range (0.0, 1.0]")
    if args.gemm_auto_benchmark_iters <= 0:
        raise ValueError("Auto GEMM benchmark iterations must be positive")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("Master port must be in range [1, 65535]")


def resolve_gpu_ids(args):
    torch_module = init_torch()
    gpu_ids = parse_gpu_ids(args.gpus)
    num_gpus = torch_module.cuda.device_count()

    if len(gpu_ids) == 1 and gpu_ids[0] == -1:
        gpu_ids = list(range(num_gpus))
    if not gpu_ids:
        raise ValueError("No visible GPU found")

    for gid in gpu_ids:
        if gid < 0 or gid >= num_gpus:
            raise ValueError(f"Invalid GPU ID ({num_gpus} visible GPU in total): {gid}")
    return gpu_ids


def get_total_memory(gid):
    torch_module = init_torch()
    with torch_module.cuda.device(gid):
        _, total_size = torch_module.cuda.mem_get_info()
    return total_size


def build_occupy_sizes(args, gpu_ids):
    occupy_sizes = {}
    occupy_mem_gb = {}

    for gid in gpu_ids:
        total_size = get_total_memory(gid)
        if args.mem_gb is None:
            occupy_size = int(total_size * args.mem_ratio)
        else:
            occupy_size = int(args.mem_gb * BYTES_PER_GB)

        if occupy_size > total_size:
            total_gb = total_size / BYTES_PER_GB
            occupy_gb = occupy_size / BYTES_PER_GB
            raise ValueError(f"GPU-{gid}: requested {occupy_gb:.2f} GB exceeds total {total_gb:.2f} GB")

        occupy_sizes[gid] = occupy_size
        occupy_mem_gb[gid] = occupy_size / BYTES_PER_GB

    return occupy_sizes, occupy_mem_gb


def release_allocated_mem(arrays, gpu_ids):
    torch_module = init_torch()

    for gid in gpu_ids:
        arrays[gid] = None
    gc.collect()

    for gid in gpu_ids:
        with torch_module.cuda.device(gid):
            torch_module.cuda.synchronize()
            torch_module.cuda.empty_cache()


def is_cuda_oom(torch_module, error):
    oom_type = getattr(torch_module.cuda, "OutOfMemoryError", RuntimeError)
    message = str(error).lower()
    return (
        isinstance(error, oom_type)
        or "out of memory" in message
        or "cublas_status_alloc_failed" in message
        or "cuda_error_out_of_memory" in message
    )


def allocate_mem(occupy_sizes, gpu_ids, allocation_mode, retry_interval):
    torch_module = init_torch()
    arrays = {gid: None for gid in gpu_ids}
    allocated = {gid: False for gid in gpu_ids}
    attempt = 0

    print(
        "Waiting for GPUs: "
        + ", ".join(f"GPU-{gid}={occupy_sizes[gid] / BYTES_PER_GB:.2f}GB" for gid in gpu_ids)
    )

    while True:
        attempt += 1
        failed = []
        round_allocated = []

        for gid in gpu_ids:
            if allocated[gid]:
                continue

            occupy_size = occupy_sizes[gid]
            try:
                with torch_module.cuda.device(gid):
                    arrays[gid] = torch_module.empty(occupy_size, dtype=torch_module.uint8, device=f"cuda:{gid}")
                    torch_module.cuda.synchronize()
                    allocated[gid] = True
                    round_allocated.append(gid)
            except Exception as error:
                if not is_cuda_oom(torch_module, error):
                    raise
                arrays[gid] = None
                with torch_module.cuda.device(gid):
                    free_size, _ = torch_module.cuda.mem_get_info()
                    torch_module.cuda.empty_cache()
                failed.append((gid, free_size))

        if all(allocated.values()):
            print("GPU memory acquired on all selected GPUs.")
            return arrays

        if allocation_mode == "all-or-nothing" and round_allocated:
            for gid in round_allocated:
                allocated[gid] = False
            release_allocated_mem(arrays, round_allocated)

        failed_text = ", ".join(f"GPU-{gid} free={free / BYTES_PER_GB:.2f}GB" for gid, free in failed)
        if failed_text:
            print(f"Attempt {attempt} pending: {failed_text}")

        if retry_interval > 0.0:
            time.sleep(retry_interval)


def build_cuda_visible_devices(gpu_ids):
    current_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_value:
        visible_devices = [item.strip() for item in current_value.split(",") if item.strip()]
        if visible_devices and max(gpu_ids) < len(visible_devices):
            return ",".join(visible_devices[gid] for gid in gpu_ids)
    return ",".join(map(str, gpu_ids))


def run_custom_script(arrays, gpu_ids, custom_cmd):
    release_allocated_mem(arrays, gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = build_cuda_visible_devices(gpu_ids)
    print(f"Starting command on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp("bash", ["bash", "-lc", custom_cmd])


class GemmMonster:
    def __init__(self, torch_module, nn, device, m, k, n):
        super().__init__()
        self.module = nn.Module()
        self.module.w1 = nn.Parameter(torch_module.randn(k, n, dtype=torch_module.float16, device=device))
        self.module.w2 = nn.Parameter(torch_module.randn(n, k, dtype=torch_module.float16, device=device))

    def __call__(self, torch_module, x):
        x = torch_module.mm(x, self.module.w1)
        x = torch_module.mm(x, self.module.w2)
        return x

    def zero_grad(self):
        self.module.zero_grad(set_to_none=True)

    def to(self, device):
        self.module.to(device)
        return self


def estimate_gemm_memory(m, k, n, inner_iters=1):
    # Rough fp16 forward/backward footprint: params, grads, input/output, saved activations.
    return 8 * k * n + inner_iters * (4 * m * k + 4 * m * n)


def cleanup_cuda(torch_module, device):
    gc.collect()
    with torch_module.cuda.device(device):
        torch_module.cuda.empty_cache()


def create_gemm_workload(torch_module, nn, device, shape):
    m, k, n = shape
    model = GemmMonster(torch_module, nn, device, m, k, n).to(device)
    x = torch_module.randn(m, k, dtype=torch_module.float16, device=device)
    return model, x


def run_gemm_step(torch_module, model, x, inner_iters=1):
    model.zero_grad()
    loss = None
    for _ in range(inner_iters):
        out = model(torch_module, x)
        current_loss = out.sum()
        loss = current_loss if loss is None else loss + current_loss
    loss.backward()
    torch_module.cuda.synchronize()


def benchmark_gemm_shape(torch_module, nn, device, shape, benchmark_iters, inner_iters):
    model = None
    x = None
    try:
        model, x = create_gemm_workload(torch_module, nn, device, shape)
        run_gemm_step(torch_module, model, x, inner_iters)

        start = time.perf_counter()
        for _ in range(benchmark_iters):
            run_gemm_step(torch_module, model, x, inner_iters)
        elapsed = time.perf_counter() - start
        return elapsed * 1000.0 / benchmark_iters
    except Exception as error:
        if not is_cuda_oom(torch_module, error):
            raise
        return None
    finally:
        del model
        del x
        cleanup_cuda(torch_module, device)


def choose_auto_gemm_shape(torch_module, nn, device, args):
    with torch_module.cuda.device(device):
        free_size, _ = torch_module.cuda.mem_get_info()
    memory_budget = int(free_size * args.gemm_auto_memory_ratio)
    candidates = [
        shape for shape in GEMM_CANDIDATES
        if estimate_gemm_memory(*shape, args.gemm_inner_iters) <= memory_budget
    ]
    if not candidates:
        candidates = [GEMM_CANDIDATES[0]]

    results = []
    for shape in candidates:
        avg_ms = benchmark_gemm_shape(
            torch_module,
            nn,
            device,
            shape,
            args.gemm_auto_benchmark_iters,
            args.gemm_inner_iters,
        )
        if avg_ms is not None:
            results.append((shape, avg_ms))
            if avg_ms >= args.gemm_auto_target_ms:
                break
        elif results:
            break

    if not results:
        raise RuntimeError("Auto GEMM shape selection failed: every candidate OOMed")

    target_ms = args.gemm_auto_target_ms
    under_target = [(shape, avg_ms) for shape, avg_ms in results if avg_ms <= target_ms]
    if under_target:
        return max(under_target, key=lambda item: item[1])
    return min(results, key=lambda item: item[1])


def resolve_gemm_shape(torch_module, nn, device, args):
    if args.gemm_shape == "fixed":
        return (args.gemm_m, args.gemm_k, args.gemm_n), None
    return choose_auto_gemm_shape(torch_module, nn, device, args)


def run_torch_gemm_worker(args):
    validate_common_args(args)
    torch_module = init_torch()
    import torch.nn as nn

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch_module.cuda.set_device(local_rank)
    device = torch_module.device(f"cuda:{local_rank}")
    shape, measured_ms = resolve_gemm_shape(torch_module, nn, device, args)
    model, x = create_gemm_workload(torch_module, nn, device, shape)

    for _ in range(args.gemm_warmup):
        run_gemm_step(torch_module, model, x, args.gemm_inner_iters)

    measured_text = "fixed" if measured_ms is None else f"{measured_ms:.1f}ms"
    print(
        f"GEMM rank {rank}/{world_size} local_rank={local_rank} "
        f"shape=({shape[0]},{shape[1]},{shape[2]}) inner_iters={args.gemm_inner_iters} measured={measured_text}"
    )

    start_total = time.monotonic()
    last_log_time = start_total
    next_jitter_time = start_total
    current_utilization = args.utilization
    iterations = 0

    while True:
        now = time.monotonic()
        if now >= next_jitter_time:
            jitter_low = max(0.05, args.utilization - args.utilization_jitter)
            jitter_high = min(1.0, args.utilization + args.utilization_jitter)
            current_utilization = random.uniform(jitter_low, jitter_high)
            next_jitter_time = now + random.uniform(3.0, 8.0)

        t1 = time.perf_counter()
        run_gemm_step(torch_module, model, x, args.gemm_inner_iters)
        t2 = time.perf_counter()

        iterations += 1
        on_time = t2 - t1
        sleep_time = args.gemm_sleep
        if current_utilization < 1.0 and on_time > 0:
            sleep_time += on_time * (1.0 / current_utilization - 1.0)
        if sleep_time > 0:
            time.sleep(sleep_time)

        now = time.monotonic()
        elapsed_hours = (now - start_total) / 3600.0
        if elapsed_hours > args.time_hours:
            break

        should_log = args.log_interval > 0 and (now - last_log_time) >= args.log_interval
        if rank == 0 and should_log:
            print(
                f"GEMM progress: {elapsed_hours:.2f}h, target_util={current_utilization * 100:.1f}%, "
                f"last_iter={on_time * 1000.0:.1f}ms, inner_iters={args.gemm_inner_iters}, iters={iterations}"
            )
            last_log_time = now


def terminate_process(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_torch_gemm_script(gpu_ids, args):
    visible_devices = build_cuda_visible_devices(gpu_ids)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices

    worker_args = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(len(gpu_ids)),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(args.master_port),
        os.path.abspath(__file__),
        "--gemm-worker",
        "--time-hours",
        str(args.time_hours),
        "--utilization",
        str(args.utilization),
        "--utilization-jitter",
        str(args.utilization_jitter),
        "--log-interval",
        str(args.log_interval),
        "--gemm-shape",
        args.gemm_shape,
        "--gemm-m",
        str(args.gemm_m),
        "--gemm-k",
        str(args.gemm_k),
        "--gemm-n",
        str(args.gemm_n),
        "--gemm-warmup",
        str(args.gemm_warmup),
        "--gemm-sleep",
        str(args.gemm_sleep),
        "--gemm-inner-iters",
        str(args.gemm_inner_iters),
        "--gemm-auto-target-ms",
        str(args.gemm_auto_target_ms),
        "--gemm-auto-memory-ratio",
        str(args.gemm_auto_memory_ratio),
        "--gemm-auto-benchmark-iters",
        str(args.gemm_auto_benchmark_iters),
    ]

    print(f"Starting GEMM on CUDA_VISIBLE_DEVICES={visible_devices}")
    sys.stdout.flush()

    proc = subprocess.Popen(worker_args, env=env)
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        terminate_process(proc)
        raise
    if return_code != 0:
        raise RuntimeError(f"torch GEMM worker exited with code {return_code}")


def parse_ray_env(ray_env):
    env_vars = {}
    for item in ray_env:
        if "=" not in item:
            raise ValueError(f"Invalid --ray-env value, expected KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --ray-env value, empty key: {item}")
        env_vars[key] = value
    return env_vars


def parse_ray_pip(ray_pip):
    packages = []
    for item in ray_pip:
        packages.extend(pkg.strip() for pkg in item.split(",") if pkg.strip())
    return packages


def script_path_in_working_dir(working_dir):
    script_path = os.path.abspath(__file__)
    working_dir = os.path.abspath(working_dir)
    try:
        rel_path = os.path.relpath(script_path, working_dir)
    except ValueError:
        return script_path
    if rel_path.startswith(".."):
        return script_path
    return rel_path


def build_child_occupy_command(args, node_idx):
    master_port = args.master_port + node_idx
    if master_port > 65535:
        master_port = 12345 + node_idx

    child_args = [
        "python3",
        script_path_in_working_dir(args.working_dir),
        "--gpus",
        "-1",
        "--mem-ratio",
        str(args.mem_ratio),
        "--time-hours",
        str(args.time_hours),
        "--utilization",
        str(args.utilization),
        "--utilization-jitter",
        str(args.utilization_jitter),
        "--log-interval",
        str(args.log_interval),
        "--allocation-mode",
        args.allocation_mode,
        "--retry-interval",
        str(args.retry_interval),
        "--gemm-shape",
        args.gemm_shape,
        "--gemm-m",
        str(args.gemm_m),
        "--gemm-k",
        str(args.gemm_k),
        "--gemm-n",
        str(args.gemm_n),
        "--gemm-warmup",
        str(args.gemm_warmup),
        "--gemm-sleep",
        str(args.gemm_sleep),
        "--gemm-inner-iters",
        str(args.gemm_inner_iters),
        "--gemm-auto-target-ms",
        str(args.gemm_auto_target_ms),
        "--gemm-auto-memory-ratio",
        str(args.gemm_auto_memory_ratio),
        "--gemm-auto-benchmark-iters",
        str(args.gemm_auto_benchmark_iters),
        "--master-port",
        str(master_port),
    ]
    if args.mem_gb is not None:
        child_args.extend(["--mem-gb", str(args.mem_gb)])
    if args.cmd is not None:
        child_args.extend(["--cmd", args.cmd])
    return child_args


def submit_ray_jobs(args):
    validate_common_args(args)
    if args.nodes <= 0:
        raise ValueError(f"Number of nodes must be positive: {args.nodes}")

    ray_bin = shutil.which("ray")
    if ray_bin is None:
        raise RuntimeError("Cannot find 'ray' command. Install Ray or run without --ray-submit/--nodes > 1.")

    if args.gpus_per_node is None:
        count = selected_gpu_count(args.gpus)
        args.gpus_per_node = count if count is not None else 8
    if args.gpus_per_node <= 0:
        raise ValueError(f"GPUs per node must be positive: {args.gpus_per_node}")

    runtime_env = {"working_dir": args.working_dir}
    env_vars = parse_ray_env(args.ray_env)
    if env_vars:
        runtime_env["env_vars"] = env_vars
    pip_packages = parse_ray_pip(args.ray_pip)
    if pip_packages:
        runtime_env["pip"] = pip_packages

    print(f"Submitting {args.nodes} Ray jobs, {args.gpus_per_node} GPUs per job.")
    processes = []

    for node_idx in range(args.nodes):
        submit_cmd = [
            ray_bin,
            "job",
            "submit",
            "--runtime-env-json",
            json.dumps(runtime_env),
            "--working-dir",
            args.working_dir,
            "--entrypoint-num-gpus",
            str(args.gpus_per_node),
        ]
        if args.ray_address:
            submit_cmd.extend(["--address", args.ray_address])
        if not args.ray_wait:
            submit_cmd.append("--no-wait")
        submit_cmd.extend(["--", *build_child_occupy_command(args, node_idx)])

        print(f"Submitted {args.ray_job_prefix}-{node_idx}")
        processes.append(subprocess.Popen(submit_cmd))

    failed = []
    for node_idx, proc in enumerate(processes):
        return_code = proc.wait()
        if return_code != 0:
            failed.append((node_idx, return_code))

    if failed:
        details = ", ".join(f"{args.ray_job_prefix}-{idx}: exit {code}" for idx, code in failed)
        raise RuntimeError(f"Ray job submit failed: {details}")


def run_local(args):
    validate_common_args(args)
    gpu_ids = resolve_gpu_ids(args)
    occupy_sizes, occupy_mem_gb = build_occupy_sizes(args, gpu_ids)

    print("GPU ID: " + ",".join(map(str, gpu_ids)))
    print("GPU memory: " + ", ".join(f"GPU-{gid}={occupy_mem_gb[gid]:.2f}GB" for gid in gpu_ids))
    print(f"Occupied time: {args.time_hours:.2f}h")

    arrays = allocate_mem(occupy_sizes, gpu_ids, args.allocation_mode, args.retry_interval)
    if args.cmd is not None:
        run_custom_script(arrays, gpu_ids, args.cmd)
        return

    try:
        run_torch_gemm_script(gpu_ids, args)
    finally:
        release_allocated_mem(arrays, gpu_ids)


def main():
    try:
        args = parse_args()
        if args.gemm_worker:
            run_torch_gemm_worker(args)
        elif args.ray_submit or args.nodes > 1:
            submit_ray_jobs(args)
        else:
            run_local(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
