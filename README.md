# 二维分布生成建模：VAE 与扩散模型实验项目

## 项目简介

本项目围绕二维生成建模课程作业展开，核心目标是在统一数据集和统一评价框架下，对以下两类生成模型进行系统比较：

- VAE（Variational Autoencoder）
- Diffusion / DDPM（去噪扩散概率模型）

当前项目的正式报告以四类二维分布为实验对象：

- `gaussian_mixture`
- `ring`
- `two_moons`
- `spiral`

除主实验外，仓库还保留了若干扩展实验代码与结果，包括：

- 多随机种子统计
- 条件生成：`CVAE` 与 `Conditional Diffusion`
- 鲁棒性分析：污染数据训练
- 隐藏测试集验证
- 高维 lifted 扩展实验
- 若干增强快照与消融实验

说明：`flow` 相关代码仍保留在仓库中，但当前报告的核心比较对象是 `VAE` 与 `DDPM`。

## 当前仓库状态

- `report/main_warm.tex` 与 `report/main_warm.pdf` 是当前报告版本。
- `outputs/` 已按当前项目内容整理，保留正式结果、图表、checkpoint 和统计表。
- `outputs/FINAL_INDEX.md` 对结果目录做了中文说明。
- `README.md` 已按目前仓库结构和实验内容更新。

## 目录结构

```text
Final_Hw/
├─ data/                         # 当前使用的数据文件（.npy + metadata.json）
├─ outputs/                      # 实验结果目录
│  ├─ checkpoints/               # 模型权重
│  ├─ figures/                   # 图像结果
│  ├─ tables/                    # 指标表格与日志
│  └─ FINAL_INDEX.md             # outputs 目录中文说明
├─ report/
│  ├─ main_warm.tex              # 当前 LaTeX 报告源文件
│  ├─ main_warm.pdf              # 当前编译结果
│  └─ references.bib             # 参考文献
├─ src/
│  ├─ datasets.py                # 数据加载与污染数据构造
│  ├─ metrics.py                 # MMD / SWD / Chamfer / NLL / Mode Coverage 等指标
│  ├─ visualize.py               # 可视化工具
│  ├─ vae.py                     # VAE 模型
│  ├─ diffusion.py               # Diffusion / DDPM 模型
│  ├─ conditional_vae.py         # 条件 VAE
│  ├─ conditional_diffusion.py   # 条件 Diffusion
│  ├─ train_vae.py               # 训练单个 VAE
│  ├─ train_diffusion.py         # 训练单个 Diffusion
│  ├─ train_cvae.py              # 训练条件 VAE
│  ├─ train_conditional_diffusion.py
│  ├─ run_benchmark.py           # 四个数据集主实验
│  ├─ run_multiseed_benchmark.py # 多种子主实验
│  ├─ run_robustness.py          # 鲁棒性实验
│  ├─ run_highdim_extension.py   # 高维扩展实验
│  ├─ run_hidden_validation.py   # 隐藏测试集验证
│  ├─ latent_viz.py              # VAE 潜空间可视化
│  ├─ enhanced_snapshots.py      # Diffusion 多步快照增强图
│  ├─ collect_results.py         # 汇总 metrics.json 为 CSV
│  ├─ sample_vae.py              # 从已有 VAE checkpoint 采样
│  ├─ sample_diffusion.py        # 从已有 Diffusion checkpoint 采样
│  ├─ train_flow.py              # 历史 flow 基线
│  ├─ flow.py
│  └─ resync_metrics.py          # 基于 checkpoint 重同步指标
├─ generate_data.py              # 生成课程作业所需 .npy 数据
├─ data_generate.md              # 数据生成脚本说明
└─ 题目/                          # 题目图片
```

## 环境依赖

建议使用 Python 3.10 及以上版本。

常用依赖包括：

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pandas`

可参考：

```bash
pip install torch numpy scipy scikit-learn matplotlib pandas
```

如果需要编译报告，还需要本地 LaTeX 环境，推荐安装 `TeX Live` 或 `MiKTeX`，并确保 `xelatex` 或 `latexmk` 可用。

## 数据说明

当前仓库中 `data/` 已包含课程作业使用的数据文件：

- `train.npy`
- `test.npy`
- `train_label.npy`
- `test_label.npy`
- `hidden_test.npy`
- `hidden_test_label.npy`
- `metadata.json`

标签映射如下：

- `0`: `gaussian_mixture`
- `1`: `ring`
- `2`: `two_moons`
- `3`: `spiral`

如果你想重新生成一份同格式数据，可以运行：

```bash
python generate_data.py --output-dir data --plot
```

说明：

- 默认每类分布生成 `2000` 个训练样本、`2000` 个公开测试样本、`2000` 个隐藏测试样本。
- 当 `data/` 下已有官方 `.npy` 文件时，`src/datasets.py` 会优先读取这些文件，而不是重新随机生成。

## 快速开始

### 1. 数据分析

生成四类数据集的统计表：

```bash
python -m src.analyze_datasets --output-path outputs/tables/dataset_summary.csv
```

### 2. 单模型训练

训练单个 VAE：

```bash
python -m src.train_vae --dataset spiral --epochs 150
```

训练单个 Diffusion：

```bash
python -m src.train_diffusion --dataset spiral --epochs 250 --timesteps 80
```

### 3. 主实验

按当前默认配置，跑四个数据集上的 VAE 和 Diffusion：

```bash
python -m src.run_benchmark
```

默认主实验配置：

- VAE：`150` epochs，`latent_dim=4`
- Diffusion：`250` epochs，`timesteps=80`
- `batch_size=256`

### 4. 多种子主实验

运行多随机种子统计版本：

```bash
python -m src.run_multiseed_benchmark
```

默认随机种子为：

```text
42, 123, 456, 789, 1024
```

### 5. 条件生成

条件 VAE：

```bash
python -m src.train_cvae
```

条件 Diffusion：

```bash
python -m src.train_conditional_diffusion
```

### 6. 鲁棒性实验

```bash
python -m src.run_robustness
```

默认测试：

- 数据集：`ring`, `spiral`
- 污染比例：`0.0`, `0.05`, `0.10`
- 污染类型：`uniform`, `cluster_shift`, `heteroscedastic`

### 7. 高维扩展实验

```bash
python -m src.run_highdim_extension
```

### 8. 隐藏测试集验证

```bash
python -m src.run_hidden_validation
```

这一步会使用 `outputs/checkpoints/optional/` 里的增强版与条件模型 checkpoint，在隐藏测试集上重新评估并写出结果表。


## 结果目录说明

整理后的结果主要放在：

- `outputs/checkpoints/`
- `outputs/figures/`
- `outputs/tables/`
