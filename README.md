# jittor_denoise

> 基于 [Jittor（计图）](https://github.com/Jittor/jittor) 的三维点云降噪方案，面向计图大赛「基于深度学习的点云降噪」赛道。  
> 队伍 **alpha奶龙** 参赛代码，核心方法为 **vm_strong**（StraightPCF 风格动态图卷积 + 多损失训练 + patch 推理融合）。

[![Framework](https://img.shields.io/badge/Framework-Jittor-blue)](https://github.com/Jittor/jittor)
[![Python](https://img.shields.io/badge/Python-3.9-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Competition-lightgrey)](#license)

---

## 目录

- [简介](#简介)
- [比赛成绩](#比赛成绩)
- [主要特性](#主要特性)
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [提交格式](#提交格式)
- [本地评测](#本地评测)
- [队伍信息](#队伍信息)
- [参考文献](#参考文献)
- [License](#license)

---

## 简介

给定含噪三维点云，模型预测每个点的三维位移向量，将噪声点「推回」物体表面附近，输出与输入同点数的降噪点云。

本仓库在官方 starter 基础上实现了 **vm_strong** 训练与推理管线，包含：

- 多层 Dynamic Edge Convolution 特征提取
- MSE + CD + 法向一致性等多目标损失
- 基于 patch 的分块推理与加权融合
- 推理超参网格搜索（最终提交配置 **E9**）

**代码入口目录：** [`starter_code/`](starter_code/)

---

## 比赛成绩

| 指标 | 数值 |
|------|------|
| A 榜总分 | **74.16** |
| CD 得分 | 62.36 |
| P2S 得分 | 85.95 |
| 排名 | 36 / — |
| 最终提交配置 | E9（`checkpoint_39.pkl`） |

最终推理超参：`step=0.97`，`momentum=0.6`，`γ=4.0`，`decay=linear`，`predict_num_steps=2`。

---

## 主要特性

| 模块 | 说明 |
|------|------|
| 模型 | `vm_strong`：动态图卷积 + MLP 位移解码 |
| 训练 | 默认 `configs/task/train_vm.yaml` → strong 数据/模型/系统配置 |
| 推理 | patch 分块 + 多步外推融合；预测阶段**空增强**，不改动测试点云 |
| 工具 | 验证集划分、推理网格搜索、提交格式检查等（见 `starter_code/tools/`） |

---

## 环境要求

- Python 3.9
- GCC / G++ ≤ 10（Jittor 编译要求）
- CUDA GPU（推荐，训练/推理）

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
cd starter_code
pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

---

## 数据准备

将官方数据集解压到**项目根目录**（与 `starter_code/` 同级）：

```bash
# 训练集：干净网格
tar xzf dataset_train.tar.gz
# → dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj

# 测试集：含噪点云
unzip dataset_test_noisy.zip
# → dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

> 数据集体积较大，未纳入本仓库；请从赛方渠道获取。

---

## 快速开始

以下命令均在 `starter_code/` 目录下执行。

### 训练

```bash
cd starter_code
python run.py --task configs/task/train_vm.yaml
```

权重默认保存至 `experiments/vm_strong/`；日志见 `logs/vm_strong/`。

### 推理（默认 strong）

```bash
python run.py --task configs/task/predict_vm.yaml
```

输出目录：`tmp_predict/shapenet/.../denoised.npy`。

### 推理（A 榜最终提交 E9）

```bash
python run.py --task configs/task/predict_vm_E9.yaml
```

需将 `checkpoint_39.pkl` 置于 `experiments/vm_strong/`（权重文件未随仓库分发）。

### 检查提交文件

```bash
python tools/check_predictions.py \
  --pred_dir ./tmp_predict \
  --noisy_dir ../dataset_test_noisy
```

### 打包提交

```bash
cd tmp_predict
zip -r ../result.zip shapenet/
```

---

## 项目结构

```
.
├── README.md                 # 本文件
├── .gitignore
└── starter_code/
    ├── run.py                # 训练 / 推理入口
    ├── evaluate.py           # 本地 CD / P2S 评测
    ├── requirements.txt
    ├── configs/
    │   ├── task/             # train_vm.yaml / predict_vm.yaml / predict_vm_E9.yaml
    │   ├── model/            # vm_strong.yaml / vm_strong_E9.yaml
    │   ├── data/
    │   ├── system/
    │   └── transform/
    ├── src/
    │   ├── model/            # vm_strong 网络
    │   ├── data/             # 数据集与增强
    │   └── system/           # 训练 / 推理逻辑
    ├── tools/                # 验证集、网格搜索、诊断脚本
    └── datalist/             # 训练 / 验证 / 测试列表
```

更详细的开发说明见 [`starter_code/README.md`](starter_code/README.md)。

---

## 提交格式

`result.zip` 根目录必须为 `shapenet/`：

```
result.zip
└── shapenet/
    └── <synset_id>/
        └── <model_id>/
            └── denoised.npy    # float32, shape (N, 3)，N 与 noisy.npy 一致
```

---

## 本地评测

需持有 GT 点云与网格（赛方提供）：

```bash
cd starter_code
python evaluate.py \
  --pred_dir ./tmp_predict \
  --gt_dir ../test_gt \
  --noisy_dir ../dataset_test_noisy \
  --mesh_dir ../dataset_train \
  --workers 8 --verbose
```

---

## 队伍信息

| 队伍 | alpha奶龙 |
|------|-----------|
| 成员 | 陈梓扬、李成蹊、黄元涛 |
| 仓库 | [GitLink — jittor_denoise](https://www.gitlink.org.cn/czy0327/jittor_denoise) |

---

## 参考文献

- Wang et al., *StraightPCF: Straight Point Cloud Filtering*, CVPR 2024.
- [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)

---

## License

本仓库代码仅供计图大赛学习与交流使用。ShapeNet 数据集使用须遵循其原始许可协议。
