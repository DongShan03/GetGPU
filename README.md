# 实验室抢占显卡脚本

主文件：`gg.py`

这个脚本用于在本机或 Ray 集群中等待并占用指定 GPU 显存。抢占成功后，可以继续运行默认的 PyTorch GEMM 负载来维持 GPU 利用率，也可以释放已占用显存后执行自定义训练命令。

原版本链接：https://github.com/godweiyang/GrabGPU.git

## 依赖

需要安装支持 CUDA 的 PyTorch：

```shell
pip install torch
```

如需使用 Ray 提交多节点任务，还需要安装并配置 Ray：

```shell
pip install ray
```

## 基本使用

占用指定 GPU：

```shell
python gg.py --gpus 4,5
```

默认参数：

```text
--gpus -1              # 默认占用所有可见 GPU
--mem-ratio 0.75       # 每张指定 GPU 占用其总显存的 75%
--time-hours 24        # 默认占用 24 小时
--utilization 0.8      # 默认目标利用率 80%
--utilization-jitter 0.1
--allocation-mode incremental
--retry-interval 60
```

## 常用示例

指定固定显存占用：

```shell
python gg.py --gpus 4,5 --mem-gb 30
```

抢到 GPU 后运行训练命令：

```shell
python gg.py --gpus 4,5 --cmd "bash scripts/train.sh stage3_only"
```

执行 `--cmd` 前，脚本会释放抢占用的显存，并设置：

```shell
CUDA_VISIBLE_DEVICES=<--gpus 指定的 GPU 列表>
```

自定义占用时间和利用率：

```shell
python gg.py --gpus 4,5 --time-hours 12 --utilization 0.9
```

使用固定 GEMM 形状：

```shell
python gg.py --gpus 4,5 --gemm-shape fixed --gemm-m 8192 --gemm-k 12288 --gemm-n 16384
```

通过 Ray 提交多个任务：

```shell
python gg.py --ray-submit --nodes 2 --gpus-per-node 8 --working-dir .
```

## 参数

```text
--gpus                  GPU ID，支持逗号或空格分隔；-1 表示全部可见 GPU。
--mem-gb                每张指定 GPU 固定占用的显存 GB，不传时使用 --mem-ratio。
--mem-ratio             每张指定 GPU 占其总显存的比例，默认 0.75。
--allocation-mode       incremental 或 all-or-nothing，默认 incremental。
--retry-interval        抢占失败后的重试间隔秒数，默认 60。
--time-hours            默认 GEMM 负载运行时长，默认 24。
--utilization           目标 GPU 利用率，默认 0.8。
--utilization-jitter    利用率随机波动范围，默认 0.1。
--log-interval          rank 0 进度日志间隔秒数，默认 300；0 表示关闭。
--cmd                   抢到 GPU 后执行的一整条命令字符串。
--gemm-shape            auto 或 fixed，默认 auto。
--master-port           本机 torch.distributed.run 使用的端口，默认 12345。
--nodes                 Ray 提交任务数量，大于 1 时进入 Ray 提交模式。
--ray-submit            显式使用 Ray job submit。
--ray-wait              等待 Ray jobs 并输出日志。
--gpus-per-node         每个 Ray job 申请的 GPU 数量。
--ray-address           Ray jobs server 地址，默认读取 RAY_ADDRESS。
--working-dir           Ray 上传的工作目录，默认当前目录。
--ray-env               Ray runtime_env 环境变量，格式 KEY=VALUE，可重复。
--ray-pip               Ray runtime_env pip 依赖，可重复或逗号分隔。
```
