# gg.py 使用说明

`gg.py` 用于抢占 GPU 并在抢到后运行计算负载。当前版本只依赖 PyTorch，不再依赖 CuPy。

## 工作流程

1. 根据 `--gpus` 选择本机可见 GPU，`-1` 表示全部可见 GPU。
2. 按 `--mem-ratio` 或 `--mem-gb` 在每张卡上分配 PyTorch tensor，占住显存。
3. 如果有卡暂时分配失败，按 `--retry-interval` 重试。
4. 抢到所有目标卡后：
   - 默认启动 `torch.distributed.run`，每张卡一个 worker。
   - 每个 worker 在自己的 GPU 上独立测量候选 GEMM shape，选择不会 OOM 且接近目标迭代耗时的 FP16 GEMM + backward 负载。
   - 如果传了 `--cmd`，会先释放抢卡 tensor，再把 `CUDA_VISIBLE_DEVICES` 设置为抢到的卡并执行命令。

脚本不会主动写额外日志文件。进度日志只由 rank 0 输出，默认每 300 秒一次，可用 `--log-interval` 调整，设置为 `0` 可关闭进度日志。

Ray 提交默认使用 `--no-wait`，提交后不持续流式打印远端 job 日志。如果需要阻塞等待 job 结束并查看 Ray 日志，加 `--ray-wait`。

## 依赖

- Python 3
- PyTorch CUDA 版本
- 多节点提交需要 Ray CLI

不需要安装 CuPy。

## 本机占用

占用本机全部可见 GPU，默认占每张卡 75% 显存，持续 24 小时：

```bash
python3 gg.py --gpus -1
```

占用指定 GPU：

```bash
python3 gg.py --gpus 0,1,2,3 --mem-ratio 0.7 --time-hours 24
```

固定每张卡占用 60GB 显存：

```bash
python3 gg.py --gpus -1 --mem-gb 60
```

降低控制台输出：

```bash
python3 gg.py --gpus -1 --log-interval 0
```

## 多节点 Ray 提交

提交 4 个 Ray job，每个 job 申请 8 张 GPU：

```bash
python3 gg.py \
  --nodes 4 \
  --gpus-per-node 8 \
  --mem-ratio 0.75 \
  --time-hours 24 \
  --ray-env NCCL_DEBUG=INFO \
  --ray-env NCCL_SOCKET_IFNAME=eth0 \
  --ray-env GLOO_SOCKET_IFNAME=eth0
```

如果需要在提交端等待任务结束：

```bash
python3 gg.py --nodes 4 --gpus-per-node 8 --ray-wait
```

如果 Ray runtime 环境没有 PyTorch，可以加：

```bash
--ray-pip torch,numpy
```

## 抢卡策略

`--allocation-mode incremental` 是默认策略。已经抢到的卡会一直保留，脚本只继续等待剩余卡。

`--allocation-mode all-or-nothing` 表示任意一张卡失败时释放本轮已经抢到的卡，下轮重新尝试，更适合需要一组卡同时空闲的场景。

## GEMM shape 自动测量

默认 `--gemm-shape auto`。每个 GPU worker 会在启动后读取当前卡剩余显存，再按候选 shape 做短 benchmark。这个测量发生在抢卡 tensor 已经占住显存之后，所以它看到的是实际可用于计算负载的剩余显存。

选择规则：

1. 先按 `--gemm-auto-memory-ratio` 过滤明显超过剩余显存预算的候选 shape。
2. 对候选 shape 从小到大做 FP16 GEMM + backward benchmark，OOM 的候选会跳过。
3. 优先选择单步耗时不超过 `--gemm-auto-target-ms` 且最接近目标耗时的 shape。
4. 如果所有可运行候选都超过目标耗时，则选择最快的可运行 shape。
5. 一旦候选达到或超过目标耗时，会停止继续探测更大的候选，避免启动阶段过长。

默认参数：

```text
--gemm-shape auto
--gemm-auto-target-ms 500
--gemm-auto-memory-ratio 0.75
--gemm-auto-benchmark-iters 2
```

这个模式适合不同节点或同一节点混用不同型号 GPU 的情况，因为每张卡都会独立决定 shape，不要求所有 rank 一致。

## 固定 GEMM shape

如果确认都是 H20 或希望复现实验，可以使用固定 shape：

```bash
python3 gg.py --gpus -1 \
  --gemm-shape fixed \
  --gemm-m 8192 \
  --gemm-k 12288 \
  --gemm-n 16384
```

如果固定 shape 导致 GEMM worker OOM，可以降低 `--mem-ratio` 或调小 `--gemm-m/k/n`。混卡场景建议使用默认自动模式。

## 抢到卡后执行自定义命令

```bash
python3 gg.py --gpus 0,1,2,3 --mem-ratio 0.7 \
  --cmd "python3 -m torch.distributed.run --nproc_per_node=4 your_script.py"
```

使用 `--cmd` 时，脚本只负责等待并确认卡可用；命令启动前会释放抢卡 tensor，把显存交给自定义任务使用。
