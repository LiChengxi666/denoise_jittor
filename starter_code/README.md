# 点云降噪赛题 Baseline

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

conda install pip # 确保使用 conda 安装的 pip，避免与系统 pip 冲突

# 安装依赖
pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

## 数据准备
1. 将训练数据 `dataset_train.tar.gz` 解压到项目根目录下：
   ```bash
   tar xzf dataset_train.tar.gz
   ```
   解压后目录：`../dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj`

2. 将测试数据 `dataset_test_noisy.zip` 解压到项目根目录下：
   ```bash
   unzip dataset_test_noisy.zip
   ```
   解压后目录：`../dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy`

## 训练
```bash
python run.py --task configs/task/train_vm.yaml
```
`train_vm.yaml` 现在默认指向 strong 模型配置（`train_strong` / `vm_strong` / `vm_strong` system），旧 baseline 已备份为 `configs/task/train_vm_baseline.yaml`。

训练权重保存在 `experiments/` 目录下。默认会保存：

| 文件 | 说明 |
|---|---|
| `experiments/vm_strong/checkpoint_last.pkl` | 最近一轮权重 |
| `experiments/vm_strong/checkpoint_best.pkl` | 验证 loss 最优权重，默认用于推理 |
| `experiments/vm_strong/checkpoint_<epoch>.pkl` | 每 10 轮和最后一轮保存一次 |

训练日志默认保存在 `logs/vm_strong/`：

| 文件 | 说明 |
|---|---|
| `train.log` | 每个 epoch 的 train/validation 摘要、lr、耗时、checkpoint 保存信息 |
| `metrics.csv` | 训练曲线，包含各项 loss、lr、epoch 耗时、best checkpoint |
| `latest.json` | 当前最新 epoch 状态，便于云端快速查看 |
| `config_snapshot.yaml` | 本次训练使用的 task/data/model/system/transform 配置快照 |

常用查看命令：

```bash
tail -f logs/vm_strong/train.log
tail -n 5 logs/vm_strong/metrics.csv
cat logs/vm_strong/latest.json
```

也可以用命令行覆盖日志和 checkpoint 目录：

```bash
python run.py --task configs/task/train_vm.yaml \
  --experiment_name vm_strong_4090_bs6 \
  --log_dir logs \
  --ckpt_dir experiments/vm_strong_4090_bs6
```

## 推理（生成提交文件）
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径，然后运行：
```bash
python run.py --task configs/task/predict_vm.yaml
```
`predict_vm.yaml` 现在默认加载 `experiments/vm_strong/checkpoint_best.pkl` 并使用 `vm_strong`。旧 baseline 推理入口已备份为 `configs/task/predict_vm_baseline.yaml`。

降噪结果保存在 `tmp_predict/` 目录下，格式为 `denoised.npy` (float32, shape (N,3))。
预测阶段使用空增强，不会对测试集 `noisy.npy` 重新采样、归一化、加噪或切训练 patch。

## 固定验证集与推理参数搜索
训练过程中的 `checkpoint_best.pkl` 仍按合成验证 loss 保存。提交前建议使用固定验证集和 CD/P2S 分数选择 checkpoint 与推理参数：

```bash
python tools/make_validation_set.py
python tools/eval_predict_grid.py --checkpoints "experiments/vm_strong/checkpoint_*.pkl"
```

`make_validation_set.py` 会额外写出 `val_meta/<synset_id>/<model_id>/meta.json`，记录生成固定验证点云时使用的原始 mesh 归一化参数。该 meta 用于排查 P2S 评测时的 mesh/point 坐标对齐问题。

网格搜索结果保存在 `experiments/vm_strong/val_grid.csv`。选择分数最高的 checkpoint 后，更新 `configs/task/predict_vm.yaml` 的 `load_ckpt` 再生成正式提交。

## P2S 与提交诊断

当前主入口固定为 `starter_code/`。如果在项目根目录也看到 `run.py`、`evaluate.py` 或 `configs/`，不要从根目录运行，避免使用旧副本。

检查 `result.zip` 或 `tmp_predict/` 的路径、shape、dtype、位移、中心和尺度统计：

```bash
python tools/diagnose_predictions.py \
  --pred_zip ../result.zip \
  --noisy_dir ../dataset_test_noisy \
  --csv_path diagnostics/predictions.csv

python tools/diagnose_predictions.py \
  --pred_dir ./tmp_predict \
  --noisy_dir ../dataset_test_noisy \
  --csv_path diagnostics/tmp_predict.csv
```

检查固定验证集 P2S 坐标对齐。`ref_gt` 是旧评测逻辑；`meta` 使用 `make_validation_set.py` 保存的原始归一化参数来同步变换 mesh；如果 clean 自身到 mesh 的 P2S 在 `ref_gt` 下很大、在 `meta` 下显著变小，说明旧本地 P2S 存在坐标错配：

```bash
python tools/diagnose_p2s_alignment.py \
  --gt_dir ./val_gt \
  --noisy_dir ./val_noisy \
  --mesh_dir ../dataset_train \
  --meta_dir ./val_meta \
  --pred_dir ./tmp_predict_val \
  --limit 10 \
  --csv_path diagnostics/p2s_alignment.csv
```

本地评测可输出逐样本 CSV，并可选择 P2S 对齐模式：

```bash
python evaluate.py \
  --pred_dir ./tmp_predict_val \
  --gt_dir ./val_gt \
  --noisy_dir ./val_noisy \
  --mesh_dir ../dataset_train \
  --meta_dir ./val_meta \
  --p2s_normalize meta \
  --csv_path diagnostics/eval_val_meta.csv \
  --workers 8
```

导出少量样本为 PLY，用 MeshLab/CloudCompare 叠加查看 noisy、pred、clean 和 mesh：

```bash
python tools/export_debug_clouds.py \
  --noisy_dir ./val_noisy \
  --gt_dir ./val_gt \
  --pred_dir ./tmp_predict_val \
  --mesh_dir ../dataset_train \
  --meta_dir ./val_meta \
  --mesh_mode meta \
  --out_dir diagnostics/debug_clouds \
  --limit 3
```

## 打包提交
```bash
cd tmp_predict
zip -r ../result.zip shapenet/
```

## 提交前检查
无 GT 时至少先检查路径、点数、dtype 和 NaN/Inf：
```bash
python tools/check_predictions.py \
    --pred_dir ./tmp_predict \
    --noisy_dir ../dataset_test_noisy
```

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./tmp_predict \
    --gt_dir ./test_gt \
    --noisy_dir ../dataset_test_noisy \
    --mesh_dir ../dataset_train \
    --workers 8
```

## 代码入口说明

请从 `starter_code/` 目录运行训练、推理和打包命令。仓库根目录下保留了旧代码/配置副本，不作为当前提交主入口。

## 云服务器训练

本地同步代码到云服务器，不同步数据集和训练产物：

```bash
rsync -avP \
  --exclude='.git' \
  --exclude='dataset_train' \
  --exclude='dataset_test_noisy' \
  --exclude='starter_code/experiments' \
  --exclude='starter_code/tmp_predict' \
  --exclude='starter_code/result.zip' \
  "/home/enovoczy/计图/" \
  ubuntu@36.103.234.94:/home/ubuntu/denoise_jittor/
```

云服务器上训练：

```bash
ssh ubuntu@36.103.234.94
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd /home/ubuntu/denoise_jittor/starter_code
python -m pip install -U pip
python -m pip install -r requirements.txt
python run.py --task configs/task/train_vm.yaml
```

4090 24GB 可优先尝试的训练参数组合：

| 场景 | 建议 |
|---|---|
| 稳妥首跑 | `batch_size=4`，`num_workers=8`，`validate_every=1`，`save_last_every=1` |
| 优先提速 | `batch_size=6`，`num_workers=12`，`validate_every=2`，`save_last_every=2` |
| 压榨速度 | `batch_size=8`，`num_workers=16`，`validate_every=5`，`save_last_every=5` |
| 显存不足 | 回退 `batch_size=4`，必要时将 `cd_loss_sample_points` 降到 `384` 做实验 |

如果训练异常：

| 现象 | 排查 |
|---|---|
| GPU 利用率低 | 增大 `num_workers`，观察 CPU/RAM；数据采样和 KDTree 可能是瓶颈 |
| CUDA OOM | 降低 `batch_size`，或实验性降低 `cd_loss_sample_points` |
| 每轮很慢 | 调大 `validate_every` / `save_last_every`，减少验证和 checkpoint IO |
| loss 出现 NaN | 降低 lr 或关闭高风险实验改动，检查 `metrics.csv` 中是哪项 loss 先异常 |
| best 长期不更新 | 查看 `train.log` 的 validation loss，并用固定验证集和 `eval_predict_grid.py` 重新选模 |

建议使用 `tmux` 防止 SSH 断开：

```bash
tmux new -s denoise_train
cd /home/ubuntu/denoise_jittor/starter_code
conda activate jittor
python run.py --task configs/task/train_vm.yaml
```

训练完成后推理、检查并打包：

```bash
cd /home/ubuntu/denoise_jittor/starter_code
python run.py --task configs/task/predict_vm.yaml
python tools/check_predictions.py --pred_dir ./tmp_predict --noisy_dir ../dataset_test_noisy
cd tmp_predict
zip -r ../result.zip shapenet/
```

把结果拉回本地：

```bash
rsync -avP \
  ubuntu@36.103.234.94:/home/ubuntu/denoise_jittor/starter_code/result.zip \
  "/home/enovoczy/计图/result.zip"
```
