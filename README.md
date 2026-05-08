# 实验室抢占显卡脚本

原版本链接https://github.com/godweiyang/GrabGPU.git

## Python 版本
需要安装库 cupy-cudaXXX (根据电脑上的CUDA版本决定 cupy-cuda12x 或者 cupy-cuda13x)
使用命令

```shell
pip install cupy-cuda12x
```

`gg.py` 会先尝试在指定 GPU 上占用显存。抢占成功后：

- 如果没有传 `--cmd`，运行默认 CUDA kernel，持续占用到 `--time-hours` 结束。
- 如果传了 `--cmd`，释放占用的显存，然后执行整条自定义命令；命令结束后脚本退出。

## 使用方法

必须指定要占用哪些 GPU：

```shell
python gg.py --gpus 4,5
```

默认参数：

```text
--mem-ratio 0.6       # 每张指定 GPU 占用其总显存的 60%
--time-hours 12       # 默认占用 12 小时
--utilization 0.7     # 默认目标利用率 70%
--utilization-jitter 0.08  # 默认利用率在目标值附近做 +/-8% 小波动
```

## 常用示例

占用 GPU 4,5，各自 60% 显存，默认 12 小时，默认 70% 利用率：

```shell
python gg.py --gpus 4,5
```

指定固定占用 30 GB 显存：

```shell
python gg.py --gpus 4,5 --mem-gb 30
```

抢到 GPU 后运行训练命令：

```shell
python gg.py --gpus 4,5 --cmd "bash scripts/train.sh stage3_only"
```

执行 `--cmd` 前，`gg.py` 会自动设置：

```shell
CUDA_VISIBLE_DEVICES=<--gpus 指定的 GPU 列表>
```

因此通过 `gg.py --gpus 4,5` 调用训练脚本时，训练脚本实际也会使用 GPU 4,5。

自定义占用时间和利用率：

```shell
python gg.py --gpus 4,5 --time-hours 24 --utilization 0.8
```

关闭利用率波动：

```shell
python gg.py --gpus 4,5 --utilization-jitter 0
```

## 参数

```text
--gpus          必填。GPU ID，支持逗号或空格分隔；-1 表示全部 GPU。
--mem-gb        每张指定 GPU 固定占用的显存 GB。不传时使用 --mem-ratio。
--mem-ratio     每张指定 GPU 占其总显存的比例，默认 0.6。
--time-hours    默认 kernel 占用时长，默认 12。
--utilization   默认 kernel 目标利用率，默认 0.7。
--utilization-jitter  默认利用率随机波动范围，默认 0.08。
--cmd           可选。抢到 GPU 后执行的一整条命令字符串。
```
