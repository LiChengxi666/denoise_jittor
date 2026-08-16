# Sprint 两阶段选模流程

## CD 主线

### 第一阶段：逐 checkpoint 锚点筛选

`auto` 检测到多个 checkpoint 后只运行原始锚点配置，每个权重一次验证。

```bash
python tools/eval_predict_grid.py \
  --checkpoints "experiments/vm_sprint_cd/checkpoint_*.pkl" \
  --csv_path experiments/vm_sprint_cd/checkpoint_screen.csv
```

### 第二阶段：Top-2 完整 18 组搜索并联合导出

提供 `screen_csv` 后，`auto` 自动切换为完整 sprint 搜索。最终第一名的 checkpoint
和完整模型配置会同时导出。

```bash
python tools/eval_predict_grid.py \
  --screen_csv experiments/vm_sprint_cd/checkpoint_screen.csv \
  --screen_top_k 2 \
  --csv_path experiments/vm_sprint_cd/final_grid.csv \
  --export_best_checkpoint experiments/vm_sprint_cd/checkpoint_selected.pkl \
  --export_best_model configs/model/vm_sprint_cd_selected.yaml
```

### 正式预测

```bash
python run.py --task configs/task/predict_vm_sprint_cd.yaml
python tools/check_predictions.py \
  --pred_dir tmp_predict_sprint_cd \
  --noisy_dir ../dataset_test_noisy
```

## InfoCD 备选路线

InfoCD 不参与第一轮 CD 主线。需要评测时使用独立目录和基础模型配置：

```bash
python tools/eval_predict_grid.py \
  --checkpoints "experiments/vm_sprint_info_cd/checkpoint_*.pkl" \
  --base_model configs/model/vm_sprint_info_cd.yaml \
  --csv_path experiments/vm_sprint_info_cd/checkpoint_screen.csv

python tools/eval_predict_grid.py \
  --screen_csv experiments/vm_sprint_info_cd/checkpoint_screen.csv \
  --screen_top_k 2 \
  --base_model configs/model/vm_sprint_info_cd.yaml \
  --csv_path experiments/vm_sprint_info_cd/final_grid.csv \
  --export_best_checkpoint experiments/vm_sprint_info_cd/checkpoint_selected.pkl \
  --export_best_model configs/model/vm_sprint_info_cd_selected.yaml
```

## Checkpoint 兼容性集成测试

只有显式提供 checkpoint 时才执行加载验证；路径不存在会直接失败。

```bash
VM_COMPAT_CHECKPOINT=experiments/vm_strong/checkpoint_anchor_7187.pkl \
  python -m unittest tests.test_vm_checkpoint_integration -v
```
