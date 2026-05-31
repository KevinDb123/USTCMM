# 二维分布生成建模：VAE 与 Diffusion Model 比较

## 项目简介

本项目围绕二维分布生成建模任务，系统比较了**变分自编码器（VAE）**与**扩散概率模型（DDPM）**在 Gaussian Mixture、Ring、Two Moons、Spiral 四类典型二维分布上的生成性能。

除基线比较外，进一步扩展了残差 MLP 骨干、DDIM 采样加速、条件生成（CVAE / Conditional Diffusion）、多类型污染鲁棒性分析、隐藏测试集验证以及 8 维 lifted 高维验证。

## 目录结构

```
Final_Hw/
├── data/                    # 训练/测试/隐藏测试数据 (.npy)
├── src/                     # 模型实现与实验脚本
│   ├── vae.py               # VAE 模型
│   ├── diffusion.py         # DDPM 模型
│   ├── conditional_vae.py   # Conditional VAE
│   ├── conditional_diffusion.py  # Conditional Diffusion
│   ├── datasets.py          # 数据加载接口
│   ├── metrics.py           # MMD / SWD / Chamfer / Mode Coverage 等指标
│   ├── visualize.py         # 可视化工具
│   ├── train_vae.py         # VAE 训练脚本
│   ├── train_diffusion.py   # Diffusion 训练脚本
│   ├── run_benchmark.py     # 主实验批量运行
│   ├── run_robustness.py    # 鲁棒性实验
│   ├── run_highdim_extension.py  # 高维 lifted 验证
│   └── run_hidden_validation.py  # 隐藏测试集验证
├── outputs/
│   ├── figures/             # 生成图像
│   ├── tables/              # 结果表格
│   └── checkpoints/         # 模型权重
├── report/
│   ├── main.tex             # 最终报告 LaTeX 源码
│   └── references.bib       # 参考文献
├── generate_data.py         # 数据生成脚本
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- NumPy, Matplotlib, SciPy, scikit-learn

### 生成数据

```bash
python generate_data.py --output-dir data --plot
```

### 训练模型

```bash
# 训练 VAE
python -m src.train_vae --dataset spiral --epochs 150

# 训练 Diffusion
python -m src.train_diffusion --dataset spiral --epochs 250 --timesteps 80

# 批量运行全部主实验
python -m src.run_benchmark
```

### 运行扩展实验

```bash
# 鲁棒性分析（多类型污染）
python -m src.run_robustness

# 高维 lifted 验证
python -m src.run_highdim_extension

# 隐藏测试集验证
python -m src.run_hidden_validation
```

### 编译报告

```bash
cd report
latexmk -pdf main
```

## 评价指标

| 指标 | 用途 |
|------|------|
| MMD（最大均值差异） | 分布整体一致性 |
| SWD（切片 Wasserstein 距离） | 几何形状相似度 |
| Chamfer Distance | 局部邻域贴合度 |
| KDE-NLL / GMM-NLL | 负对数似然代理指标 |
| Mode Coverage | 多峰模式覆盖率 |
| 采样时间 | 生成效率 |

## 主要结论

- **基线轻量设定**：Diffusion 在 Ring、Spiral 等连续几何结构上占优
- **增强骨干**：VAE 在 Gaussian Mixture、Two Moons、Spiral 上取得更低 MMD，同时保持毫秒级采样
- **DDIM 加速**：在 Spiral 上将采样时间降至约 23%，质量仅有中等劣化
- **条件生成**：CVAE 平均 MMD 和速度更优，Conditional Diffusion 在 Ring/Spiral 上几何保真度更好
- **高维扩展**：cosine 调度 + 强时间条件网络可显著改善 Diffusion 高维表现

## 作者

- 黎思远 (PB23061257)
- 中国科学技术大学 · 数学建模课程
- 2026 年 5 月
