# 云服务器训练脚本使用说明

这些脚本用于在云服务器上运行 Jittor 点云去噪训练，并按固定间隔执行推理和评测。默认脚本目录为 `scripts/cloud/`，实际训练入口仍是 `starter_code/run.py`。

## 1. 环境准备

```bash
cd /path/to/denoise_jittor
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd starter_code
python -m pip install -r requirements.txt
cd ..
```

确认 Jittor 和 GPU：

```bash
conda activate jittor
python - <<'PY'
import jittor as jt
print(jt.__version__)
PY
nvidia-smi
```

## 2. 修改路径和参数

脚本顶部变量都可以直接改，也可以用环境变量覆盖。默认相对路径按 `starter_code/` 工作目录解释。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRAIN_DATA` | `../dataset_train` | ShapeNet 训练网格根目录 |
| `PREDICT_NOISY_DIR` | `./val_noisy` | 固定验证 noisy 点云目录 |
| `GT_DIR` | `./val_gt` | 固定验证 clean 点云目录；没有 GT 时只做格式检查 |
| `MESH_DIR` | `../dataset_train` | P2S 使用的 mesh 根目录 |
| `META_DIR` | `./val_meta` | `--p2s_normalize meta` 所需 meta |
| `PREDICT_DATALIST` | `./datalist/validate.txt` | 验证样本列表 |
| `TRAIN_CONFIG` | `configs/task/train_vm.yaml` | 训练 task 配置 |
| `PREDICT_CONFIG` | `configs/task/predict_val.yaml` | 推理 task 配置 |
| `MODEL_CONFIG` | `configs/model/vm_strong.yaml` | 模型配置 |
| `TOTAL_EPOCHS` | `100` | 总训练 epoch |
| `EVAL_INTERVAL` | `10` | 每多少 epoch 推理和评测一次 |
| `GPU_ID` | `0` | `CUDA_VISIBLE_DEVICES` |
| `P2S_NORMALIZE` | `meta` | P2S 对齐模式 |

## 3. 启动 tmux 训练

```bash
EXP_NAME=vm_strong_cloud TOTAL_EPOCHS=100 EVAL_INTERVAL=10 GPU_ID=0 \
  bash scripts/cloud/train_cloud_tmux.sh
```

启动后会输出：

```bash
tmux attach -t pc_denoise_train
```

如果同名 tmux 已存在，脚本会停止并提示你 attach、kill 或换 `SESSION_NAME`。

## 4. 查看和操作 tmux

Attach：

```bash
tmux attach -t pc_denoise_train
```

Detach：在 tmux 中按：

```text
Ctrl-b 然后按 d
```

中断训练：attach 后按 `Ctrl-c`，或在外部执行：

```bash
tmux kill-session -t pc_denoise_train
```

## 5. 查看日志

默认日志在仓库根目录：

```text
log/YYYYMMDD_HHMMSS_EXP_NAME/
├── train.log
├── eval.log
├── metrics.csv
├── eval_results.jsonl
├── command_history.txt
├── environment.txt
├── config_snapshot/
├── checkpoints/
└── predictions/
```

常用查看：

```bash
tail -f log/*_vm_strong_cloud/train.log
tail -f log/*_vm_strong_cloud/eval.log
tail -n 20 log/*_vm_strong_cloud/metrics.csv
```

## 6. 分段训练和评估逻辑

脚本不修改 trainer 代码，而是分段运行：

```text
训练 10 epoch -> checkpoint_last.pkl -> 推理 -> evaluate.py
加载上段 checkpoint_last.pkl -> 再训练 10 epoch -> 推理 -> evaluate.py
...
```

注意：当前项目 checkpoint 只保存模型权重，不保存 optimizer state 和全局 epoch state。因此这里的 resume 是权重级 resume，学习率调度会在每段重新开始。

## 7. 从 checkpoint 恢复

```bash
RESUME_CKPT=starter_code/experiments/vm_strong/checkpoint_last.pkl \
EXP_NAME=vm_resume_cloud \
bash scripts/cloud/train_cloud_tmux.sh
```

也可以直接不用 tmux：

```bash
RESUME_CKPT=starter_code/experiments/vm_strong/checkpoint_last.pkl \
bash scripts/cloud/train_eval_loop.sh
```

## 8. 单独评估某个 checkpoint

```bash
bash scripts/cloud/run_eval_once.sh \
  --checkpoint starter_code/experiments/vm_strong/checkpoint_last.pkl \
  --predict-noisy-dir ./val_noisy \
  --gt-dir ./val_gt \
  --mesh-dir ../dataset_train \
  --meta-dir ./val_meta \
  --p2s-normalize meta \
  --gpu-id 0
```

比较多个 checkpoint 时分别运行并查看 `metrics.csv`。

## 9. Dry Run 自检

不启动大模型，只检查脚本将执行什么：

```bash
bash scripts/cloud/train_eval_loop.sh --dry-run
bash scripts/cloud/run_eval_once.sh --checkpoint dummy.pkl --dry-run
```

语法检查：

```bash
bash -n scripts/cloud/train_cloud_tmux.sh
bash -n scripts/cloud/train_eval_loop.sh
bash -n scripts/cloud/run_eval_once.sh
```

## 10. 常见错误

| 错误 | 处理 |
|---|---|
| `tmux is not installed` | 安装 tmux，或使用脚本输出的普通 bash fallback 命令 |
| `No module named jittor` | `conda activate jittor` 后再运行 |
| `Expected checkpoint not found` | 检查训练是否失败、`ckpt_dir` 是否可写 |
| `未找到匹配的测试样本` | 检查 `PREDICT_NOISY_DIR`、`GT_DIR`、`PREDICT_DATALIST` 是否对应 |
| P2S 很异常 | 确认 `META_DIR` 存在，并使用 `P2S_NORMALIZE=meta` |
| 无 GT | 设置 `EVAL_ENABLED=0`，或只让脚本自动 fallback 到 `tools/check_predictions.py` |
