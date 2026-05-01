# 颜料老化预测 — 实现与模型对比

面向「第四届世界科学智能大赛——中学生赛道」颜料老化ΔE预测任务。

## 目录结构

```
yaioc26/
├── baseline_and_data/           # 组织方提供的基线与数据（只读）
├── src/
│   ├── config.py                # 路径、常量、随机种子
│   ├── data.py                  # 数据加载、分组键、提交文件写入
│   ├── features.py              # 特征工程（时间/颜色/颜料类别/交互/one-hot）
│   ├── constraints.py           # 物理后处理（t=0→0、非负、单调等渗回归）
│   ├── eval.py                  # RMSE、K折CV、时间外推CV、按类误差
│   └── models/
│       ├── base.py              # 抽象 BaseModel 接口
│       ├── rf_model.py          # 组织方 baseline + 改进 RF
│       ├── gbm_model.py         # XGBoost、LightGBM（带单调约束）
│       ├── linear_model.py      # Ridge、Polynomial
│       ├── curve_fit_model.py   # 按组 a·log(t+1)+b·t（NNLS + 经验贝叶斯收缩）
│       └── gpr_model.py         # 高斯过程回归
├── scripts/
│   ├── run_baseline.py          # 改进 RF 端到端 + 提交
│   ├── run_compare.py           # 全模型 × 两种 CV 对比
│   ├── predict.py               # 指定模型生成提交
│   ├── compare_predictions.py   # 各模型在测试集上的预测并排对比
│   └── eda.py                   # 训练集统计 + 曲线图
└── outputs/
    ├── predict_out.csv          # 最终提交（curve 模型）
    ├── metrics/                 # 对比表、CV明细、分组误差
    ├── figs/                    # 曲线图、散点图
    └── models/                  # joblib 序列化的模型
```

## 快速开始

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas scikit-learn matplotlib scipy joblib xgboost lightgbm
python scripts/eda.py            # 数据探查 + 分组统计
python scripts/run_compare.py    # 全模型对比
python scripts/predict.py --model curve   # 生成 outputs/predict_out.csv
```

## 评测结果（RMSE，越低越好）

| 模型           | 随机 5 折 CV | 时间外推 CV | 备注 |
|----------------|-------------|-------------|------|
| rf_organizer   | 0.635       | 0.920       | 组织方 baseline（仅 4 特征） |
| rf_improved    | 0.761       | 1.068       | 加上 sample/condition one-hot |
| xgb            | **0.565**   | 0.836       | 随机 CV 最佳 |
| lgbm           | 0.632       | 0.848       | |
| ridge          | 0.978       | 1.248       | |
| poly           | 1.004       | 1.410       | |
| **curve**      | 0.728       | **0.609**   | **时间外推 CV 最佳** |
| gpr            | 0.592       | 1.013       | 随机 CV 表现好但外推失败 |

**选择依据**：测试集中大多数 (sample, condition) 组合需要「超出训练最大时间点」的外推（曙红外推 12 天，翡翠绿/钴蓝外推 6 天，皮纸外推 25 天），与时间外推 CV 情形一致。按组拟合 `ΔE=a·log(t+1)+b·t` 的 `curve` 模型以 **RMSE 0.609** 领先其余模型约 27%（相对 xgb）至 43%（相对 poly），成为最终提交选择。

## 关键设计

- **特征工程**（`src/features.py`）：时间非线性（`log1p_t`、`sqrt_t`、`t²`、多个 `exp(-t/τ)`）+ 颜色派生（饱和度 `C0`、色相 `sin_h0/cos_h0`、K-M 代理 `(1-L')²/(2L')`） + 颜料类别 one-hot + `sample × aging_condition` 交互
- **Curve-Fit 物理模型**（`src/models/curve_fit_model.py`）：
  - NNLS 确保 `a,b ≥ 0`（保证 ΔE 随 t 单调递增）
  - 经验贝叶斯收缩：`coef_group = w·coef_raw + (1-w)·coef_category_prior`，其中 `w = n / (n+3)`
  - `b` 系数在 95% 分位处截断以防极端外推
  - 特别缓解了曙红（每组 3 训练点）和皮纸/humid-_heat（t_max=15→t=40）的过拟合/外推风险
- **物理后处理**（`src/constraints.py`）：
  1. 预测 clip ≥ 0
  2. 按 (sample, condition) 组用训练点作为锚，等渗回归强制单调递增
  3. 再次 clip，强制 `t=0 → 0`

## 产出文件

| 文件 | 说明 |
|------|------|
| `outputs/predict_out.csv` | 比赛提交（curve 模型，37 行单列 dietaE） |
| `outputs/metrics/model_comparison.csv` | 全模型 × CV 方案的 RMSE 对比表 |
| `outputs/metrics/per_group_errors.csv` | 时间外推 CV 下按颜料类别+条件的误差分解 |
| `outputs/metrics/predictions_all_models.csv` | 每一行测试样本 × 每个模型的预测 |
| `outputs/metrics/train_group_stats.csv` | 39 个 (sample, condition) 组的统计 |
| `outputs/metrics/test_coverage.csv` | 测试集相对训练集的外推缺口 |
| `outputs/figs/curves_*.png` | 按颜料类别分组的 ΔE-t 曲线 |
| `outputs/figs/scatter_log_t.png` | ΔE vs log(1+t) 散点图 |
| `outputs/models/*.pkl` | joblib 序列化模型 |

## 依赖

Python 3.12+；`numpy`、`pandas`、`scikit-learn`、`scipy`、`matplotlib`、`joblib`、`xgboost`、`lightgbm`。
