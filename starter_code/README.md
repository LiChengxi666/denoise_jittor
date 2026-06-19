# 点云降噪赛题 Baseline

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
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
训练权重保存在 `experiments/` 目录下。默认会保存：

| 文件 | 说明 |
|---|---|
| `experiments/vm/checkpoint_last.pkl` | 最近一轮权重 |
| `experiments/vm/checkpoint_best.pkl` | 验证 loss 最优权重，默认用于推理 |
| `experiments/vm/checkpoint_<epoch>.pkl` | 每 10 轮和最后一轮保存一次 |

## 推理（生成提交文件）
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径，然后运行：
```bash
python run.py --task configs/task/predict_vm.yaml
```
降噪结果保存在 `tmp_predict/` 目录下，格式为 `denoised.npy` (float32, shape (N,3))。
预测阶段使用空增强，不会对测试集 `noisy.npy` 重新采样、归一化、加噪或切训练 patch。

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
  ubuntu@36.103.177.251:/home/ubuntu/denoise_jittor/
```

云服务器上训练：

```bash
ssh ubuntu@36.103.177.251
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd /home/ubuntu/denoise_jittor/starter_code
python -m pip install -U pip
python -m pip install -r requirements.txt
python run.py --task configs/task/train_vm.yaml
```

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
  ubuntu@36.103.177.251:/home/ubuntu/denoise_jittor/starter_code/result.zip \
  "/home/enovoczy/计图/result.zip"
```
