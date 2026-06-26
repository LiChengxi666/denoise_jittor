# 1. 确认配置文件语法正确
```bash
cd starter_code
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/task/train_vm_enhanced.yaml'); print('task OK')"
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/model/vm_enhanced.yaml'); print('model OK')"
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/transform/vm_enhanced.yaml'); print('transform OK')"
```

# 2. 确认模型可以成功初始化
```bash
python -c "
import sys; sys.path.insert(0, '.')
from run import load
from src.model.parse import get_model
model_cfg = load('model', 'configs/model/vm_enhanced')
transform_cfg = load('transform', 'configs/transform/vm_enhanced')
m = get_model(model_config=model_cfg, transform_config=transform_cfg)
print('Model params:', sum(p.numel() for p in m.parameters()))
print('Enhanced losses:', {k:v for k,v in m.model_config.items() if any(x in k for x in ['global','p2s','normal','enabled'])})
"
```

# 3. 小规模训练 1-2 epoch 验证 loss 正常（Ctrl+C 检查无 NaN）
```bash
python run.py --task configs/task/train_vm_enhanced.yaml --experiment_name vm_enhanced_debug
```
# 看到第一个 epoch 的 train loss 输出后 Ctrl+C

# 4. 检查训练日志确认各项 loss 都有数值
```bash
cat logs/vm_enhanced_debug/train.log
cat logs/vm_enhanced_debug/latest.json
```

# 5. 正式训练
```bash
python run.py --task configs/task/train_vm_enhanced.yaml
```

# 6. 用固定验证集网格搜索最佳 checkpoint
```bash
python tools/make_validation_set.py
python tools/eval_predict_grid.py --checkpoints "experiments/vm_enhanced/checkpoint_*.pkl"
```

# 7. 推理（更新 load_ckpt 为网格搜索最佳权重后）
```bash
python run.py --task configs/task/predict_vm_enhanced.yaml
```

# 8. 提交前检查
```bash
python tools/check_predictions.py --pred_dir ./tmp_predict --noisy_dir ../dataset_test_noisy
```

# 9. 打包
```bash
cd tmp_predict && zip -r ../result.zip shapenet/
```