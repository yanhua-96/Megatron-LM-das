# Llama2 训练脚本使用说明

本目录脚本分两层，日常只需要用第一层：

| 脚本 | 角色 |
| --- | --- |
| `run_llama.sh` | 入口：解析参数、生成 `hostfile_slots`，再通过 mpirun 在每个 rank 上拉起对应规格的 worker |
| `train_llama2_7B.sh`<br>`train_llama2_13B.sh`<br>`train_llama2_70B.sh` | 每个 rank 的训练命令，由 `run_llama.sh` 传入 `<dist-url> <dist-port>` 和训练参数后拉起，**不要手动调用** |

`--model` 决定拉起哪个 worker。模型结构、并行度和 batch 配置写在各 worker 的 `# model structure` 段，入口脚本只负责参数透传和分发。入口脚本默认使用同目录下的 `hostfile`，也可通过第一个位置参数指定其他 hostfile。

## run_llama.sh 参数

| 参数                           | 默认值          | 说明                                                                                                                                         |
| ------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model=7B\|13B\|70B`         | **7B**    | 模型规格切换，见下表                                                                                                                         |
| `--train_iters=N`            | 10              | 总训练步数。**续训时是包含已完成步数的总量**（从 300 步续到 1300 步就传 1300）                                                         |
| `--save_ckpt=true\|false`     | **false** | 是否保存 checkpoint。false 时完全不写盘；true 时保存到 `--save_checkpoint_path`                                                            |
| `--save_checkpoint_path=DIR` | `./ckpt_save` | 保存目录（仅 `save_ckpt=true` 时生效；自动 `realpath` 展开）                                                                             |
| `--load_checkpoint_path=DIR` | 无              | 续训加载目录。必须包含 `latest_checkpointed_iteration.txt`（传 checkpoint 根目录，不是 `iter_XXXXXXX` 子目录）。可单独使用：只加载不保存 |
| `--save_interval=N`          | = train_iters   | 每隔 N 步保存一次（仅 `save_ckpt=true` 时生效）。默认等于 train_iters，即只在最后一步保存                                                  |
| `--profiling=torch/hip`      | 无              | torch profiler / hip profiler                                                                                                                |
| 环境变量 `LAUNCHER`          | `torchrun`    | 可选 `mpirun`                                                                                                                              |

## --model 各规格差异

| 规格 | layers | hidden | ffn   | heads | TP | PP | MBS | GBS | max-pos | 特有参数                                                |
| ---- | ------ | ------ | ----- | ----- | -- | -- | --- | --- | ------- | ------------------------------------------------------- |
| 7B   | 32     | 4096   | 11008 | 32    | 1  | 1  | 4   | 64  | 4096    | —                                                      |
| 13B  | 40     | 5120   | 13824 | 40    | 1  | 1  | 1   | 64  | 4096    | —                                                      |
| 70B  | 80     | 8192   | 28672 | 64    | 4  | 8  | 1   | 128 | 8192    | GQA（`--group-query-attention --num-query-groups 8`） |

其余超参（seq 4096、RMSNorm、rope、swiglu、bf16、flash-attn、torch_dist ckpt 格式等）分别写在各 worker 的 `# model structure` 段。
70B 的 TP=4/PP=8 需要多卡/多机环境。

## 典型场景

### 1. 10 步冒烟测试（不保存）

```bash
bash run_llama.sh                    # 默认 7B
bash run_llama.sh --model=13B        # 13B 冒烟
```

默认 `save_ckpt=false`，跑完不落任何 checkpoint，适合验证环境和改动。

### 2. 300 步预训练（保存 checkpoint）

```bash
bash run_llama.sh --model=7B --train_iters=300 --save_ckpt=true \
    --save_checkpoint_path=./ckpt_300
```

结束时机=步数 300 处保存一次（save_interval 默认等于 train_iters）。13B/70B 换 `--model` 即可。

### 3. 从 300 步续训到 1300 步（不保存结果）

```bash
bash run_llama.sh --model=7B --train_iters=1300 \
    --load_checkpoint_path=./ckpt_300
```

不加 `--save_ckpt=true` 就默认 `false`：续训正常加载权重和优化器状态，但**跑完不写任何 checkpoint**。

要点：

- `--train_iters=1300` 是**总量**（已含前 300 步），不是“再跑 1300 步”；
- `--load_checkpoint_path` 传的是 checkpoint 根目录（含 `latest_checkpointed_iteration.txt` 的那层，不是 `iter_XXXXXXX` 子目录）；
- 续训结果也要保存的话，再加 `--save_ckpt=true --save_checkpoint_path=./ckpt_1300`（开启保存时 load 与 save 目录必须不同，脚本会校验）。

续训内置校验：tracker 文件存在且为纯数字、train_iters 必须大于 checkpoint 步数、自动导出 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`、沿用 checkpoint 里的 LR 调度（`--use-checkpoint-opt-param-scheduler`）。

### 踩坑提示

- 冒烟跑完想看 checkpoint：必须显式 `--save_ckpt=true`，默认不保存。
- 改超参改 `train_llama2_XX.sh`，改入口参数用 `run_llama.sh` 的命令行；两边都改时注意 `run_llama.sh` 透传的值会覆盖 worker 默认值。
