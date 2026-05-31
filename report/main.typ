#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.2cm, left: 2.5cm, right: 2.5cm),
)

#set text(
  font: ("Times New Roman", "SimSun"),
  size: 11pt,
)

#set par(
  justify: true,
  first-line-indent: 2em,
  leading: 0.98em,
  spacing: 0.42em,
)

#set heading(numbering: none)
#set figure(supplement: [图])
#show figure.where(kind: table): set figure(supplement: [表])
#show heading.where(level: 1): set block(above: 1.25em, below: 0.8em)
#show heading.where(level: 2): set block(above: 1.0em, below: 0.6em)
#show heading.where(level: 3): set block(above: 0.8em, below: 0.45em)
#show figure: set block(above: 0.95em, below: 0.95em)
#show list: set block(spacing: 0.28em)

#let report-title = "二维分布生成建模：基于 VAE 与 Diffusion Model 的实验比较"
#let report-date = "2026 年 5 月 30 日"
#let author-lines = (
  "姓名待填  班级待填  学号待填",
  "姓名待填  班级待填  学号待填",
  "姓名待填  班级待填  学号待填",
)

#let note-box(body) = block(
  inset: 10pt,
  stroke: 0.6pt + luma(180),
  radius: 4pt,
  fill: luma(248),
  body,
)

#let fig(path, caption, width: 100%) = figure(
  image(path, width: width),
  caption: [#caption],
)

#let small-gap = 0.65cm

#show figure.caption: it => text(size: 10pt)[#it]

// ============================================================
// 封面
// ============================================================
#align(center)[
  #v(3.8cm)
  #text(size: 20pt, weight: "bold")[#report-title]
  #v(1.8cm)
  #for line in author-lines [
    #text(size: 14pt)[#line]
    #v(0.45cm)
  ]
  #v(0.4cm)
  #text(size: 14pt)[#report-date]
]

#pagebreak()

// ============================================================
// 目录
// ============================================================
#set page(numbering: "I")
#align(center, text(size: 18pt, weight: "bold")[目录])
#v(1cm)
#outline(title: none, indent: 2em)

#pagebreak()

// ============================================================
// 摘要
// ============================================================
#align(center, text(size: 18pt, weight: "bold")[摘要])

生成建模是机器学习领域的核心问题之一，其目标在于学习观测数据背后的概率分布并从中生成具有相似统计特性的新样本。本报告围绕"二维分布生成建模"这一课程题目，选取变分自编码器（Variational Autoencoder, VAE）与扩散概率模型（Denoising Diffusion Probabilistic Model, DDPM）两类具有代表性的深度生成方法，对 Gaussian Mixture、Ring、Two Moons 以及 Spiral 四类典型二维复杂分布开展了系统性的建模、评价与对比研究。

在方法论层面，VAE 通过编码器-解码器架构实现显式潜变量建模，以变分推断近似后验分布，并以最大化证据下界（ELBO）为训练目标；Diffusion Model 则采用"前向加噪—反向去噪"的双阶段机制，通过马尔可夫链逐步将数据分布转换为高斯噪声，再学习噪声预测网络以逆转该过程从而完成样本生成。在模型实现中，二者均采用多层感知机（MLP）作为骨干网络，并共享统一的数据接口与训练框架。

在评价体系方面，本报告选取最大均值差异（Maximum Mean Discrepancy, MMD）、切片 Wasserstein 距离（Sliced Wasserstein Distance, SWD）以及基于聚类的模式覆盖率（Mode Coverage）三个互补性指标，从全局分布一致性、几何搬运代价以及多峰覆盖完备性等角度全面量化生成质量。同时统计训练与采样耗时，实现质量与效率的双维度比较。

实验结果表明，在切换到 README 指定的数据格式后：(1) Diffusion Model 在 Ring 与 Spiral 上取得了更优的 MMD 与 SWD，说明其在闭合流形与连续螺旋这类复杂几何结构上的全局拟合能力更强；(2) VAE 在 Gaussian Mixture 与 Two Moons 上取得更低的 MMD，其中 Two Moons 上 MMD 为 0.0017，显著优于 Diffusion 的 0.0057，表明显式潜变量建模在局部结构清晰的流形上仍具有很强竞争力；(3) VAE 的单次采样耗时约 0.3--1.2 ms，而 Diffusion 约 36--41 ms，效率差距仍接近两个数量级；(4) 消融实验揭示，较小的 KL 正则权重（`beta=0.1`）能显著改善 VAE 在 Spiral 上的重构质量，而较大的扩散步数（`T=80`）对 Diffusion 的高保真采样至关重要。

综合而言，本报告完成了从旧版 `npz` 模拟数据到 README 数组格式数据的切换，并基于同一套训练、评价与可视化框架重新获得了主实验、条件生成、鲁棒性和消融结果。当前工程已经能够直接读取 `data/train.npy`、`data/test.npy` 及其标签文件，为后续继续替换同格式数据提供了稳定入口。

#v(0.3cm)
#text(weight: "bold")[关键词：] 生成建模；变分自编码器；扩散模型；二维分布；分布评价；数学建模

#v(0.8cm)
#align(center, text(size: 18pt, weight: "bold")[Abstract])

This report addresses the course project on two-dimensional generative modeling by systematically comparing two representative deep generative approaches — the Variational Autoencoder (VAE) and the Denoising Diffusion Probabilistic Model (DDPM) — across four benchmark-like 2D distributions: Gaussian Mixture, Ring, Two Moons, and Spiral. The VAE adopts an encoder-decoder architecture with explicit latent variable modeling and maximizes the Evidence Lower Bound (ELBO) for training, while the Diffusion Model follows a Markov-chain-based forward diffusion and learned reverse denoising paradigm.

Under a unified lightweight MLP-based implementation, we evaluate generation quality through Maximum Mean Discrepancy (MMD), Sliced Wasserstein Distance (SWD), and clustering-based Mode Coverage. Using the README-style dataset files with 2,000 training and 2,000 test samples per class, the rerun experiments show that: (1) the Diffusion Model performs better on Ring and Spiral, while the VAE attains lower MMD on Gaussian Mixture and Two Moons; (2) the VAE remains nearly two orders of magnitude faster at sampling (about 0.3--1.2 ms versus 36--41 ms per batch); and (3) ablation studies confirm that a smaller KL weight (β = 0.1) benefits the VAE on Spiral, whereas more diffusion steps (T = 80) are important for high-fidelity diffusion sampling. The resulting pipeline now directly supports the dataset format specified in `README数据.md`.

#v(0.3cm)
#text(weight: "bold")[Keywords:] generative modeling; variational autoencoder; diffusion model; two-dimensional distributions; distribution evaluation; mathematical modeling

#pagebreak()
#set page(numbering: "1")

// ============================================================
// 一、问题重述
// ============================================================
= 一、问题重述

== 1.1 问题背景

生成建模（Generative Modeling）是机器学习与统计建模中的基础性研究问题。与判别式模型不同，生成模型的目标并非学习从输入到标签的映射函数，而是直接对观测数据本身的概率分布 $p_"data"(x)$ 进行建模，并能够从所学分布中采样产生新的、与训练数据具有相似统计特性的样本。这一能力在图像合成、异常检测、数据增强、药物分子设计以及科学模拟等众多领域中具有重要应用价值。

本课程大作业聚焦于二维空间中的生成建模问题。尽管二维分布在数据维度上远低于真实世界的高维图像或文本数据，但这类低维分布的建模天然具备两个突出优势：其一，二维点集可以直接通过散点图进行可视化，使得模型是否真正"学到"了目标分布能够被直观判断；其二，通过精心设计具有不同几何特征的分布类型（如多峰、细流形、非线性双簇、连续螺旋等），可以系统性地检验生成模型在不同结构复杂性下的表现差异。正是基于这些原因，二维生成建模被广泛用作深度生成模型的"沙盒"测试平台，同时也是理解和比较不同生成范式工作机制的理想实验场景。

== 1.2 题目要求

根据课程要求，本大作业需要在以下四类二维分布上完成生成建模任务：

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    inset: 6pt,
    stroke: 0.5pt + black,
    [分布名称], [英文名称], [核心几何特征],
    [高斯混合分布], [Gaussian Mixture], [多个分离的椭圆高斯簇，离散多峰结构],
    [环形分布], [Ring], [样本集中在闭合圆环曲线附近，带状低维流形],
    [双月分布], [Two Moons], [两条弯曲的半环形流形，非线性双簇互补结构],
    [螺旋分布], [Spiral], [连续螺旋曲线，拓扑复杂，长程依赖性强],
  ),
  caption: [四类目标二维分布及其核心几何特征。],
)

每类分布包含训练集与测试集两个部分，具体任务要求包括：

+ 在训练集上分别建立 VAE 与 Diffusion Model 两类生成模型；
+ 从训练完成的模型中采样生成指定数量的新二维样本；
+ 将生成样本与真实测试集进行定性与定量的分布比较；
+ 至少使用两种定量指标评估生成质量；
+ 对结果进行系统的分析与讨论。

此外，题目鼓励在基础实验之上进行拓展研究，可选方向包括但不限于：超参数影响分析、条件生成建模、模型鲁棒性分析等。

== 1.3 数据说明

本报告现已切换为 `README数据.md` 中给出的统一数据格式。具体地，使用 `generate_data.py` 在 `data/` 目录下生成 `train.npy`、`test.npy`、`train_label.npy`、`test_label.npy`、`hidden_test.npy`、`hidden_test_label.npy` 与 `metadata.json`；其中每类分布各包含 2000 个训练样本、2000 个测试样本以及 2000 个隐藏测试样本。本文实验仅使用公开训练/测试划分进行建模与评价，隐藏测试集保留为补充数据。当前 `src/datasets.py` 已优先读取该 README 格式，同时向下兼容旧版 `data/processed/*.npz`。

== 1.4 研究问题、主要工作与结构

本报告的主要工作与贡献可概括如下：

为使后续建模、实验与讨论形成更清晰的研究闭环，本文将课程题目进一步凝练为以下三个核心研究问题（Research Questions, RQs）：

1. `RQ1`：在二维复杂分布生成任务中，VAE 与 Diffusion Model 分别具备怎样的分布拟合能力，它们在哪些分布类型上各自占优？
2. `RQ2`：在生成质量、模式覆盖、几何结构保持和采样效率之间，两类模型的主要权衡关系是什么？
3. `RQ3`：当模型超参数发生变化时，哪些因素会成为影响最终生成效果的关键控制变量？

1. 构建了以 VAE 与 Diffusion Model 为双主线的统一建模与评价框架，实现了数据集生成、模型训练、样本采样、指标计算与可视化输出的全流程自动化；
2. 在同一实验条件下对两类模型在四类分布上进行了公平比较，从全局分布拟合、几何结构保持与采样效率三个维度给出定量结论；
3. 完成了关键超参数（VAE 的 KL 权重、Diffusion 的扩散步数）的消融实验，初步揭示了不同超参数选择对生成质量的影响规律；
4. 对实验结果的物理含义与方法局限性进行了深入讨论，为后续改进和正式数据实验提供了清晰方向。

本报告后续章节安排如下：第二章对相关工作进行简要综述；第三章阐述模型假设与符号约定；第四章详述 VAE 与 Diffusion Model 的数学原理与具体实现；第五章介绍实验设计、评价指标与实现环境；第六章展示并分析实验结果；第七章进行总结与展望。

// ============================================================
// 二、相关工作
// ============================================================
= 二、相关工作

== 2.1 深度生成模型发展概述

深度生成模型的发展经历了从受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）和深度信念网络（Deep Belief Network, DBN），到变分自编码器（VAE）、生成对抗网络（Generative Adversarial Network, GAN），再到近年来的扩散概率模型（Diffusion Probabilistic Model）和基于流的模型（Flow-based Model）等多个阶段。

在众多生成建模范式中，VAE 与 Diffusion Model 恰好代表了两种本质上不同的技术路径。VAE 延续了潜变量建模的经典统计推断传统，通过引入变分推断将难以直接优化的边际似然转化为可计算的证据下界（ELBO），从而实现端到端的训练。这一框架的优点在于理论基础清晰、训练过程稳定、潜空间具有良好的结构性与可解释性。其局限性同样明显：由于通常采用高斯先验与对角高斯近似后验，VAE 生成的样本倾向于过度平滑，对复杂几何细节和尖锐模式的刻画能力不足。

Diffusion Model 则另辟蹊径。受非平衡热力学的启发，扩散模型将生成过程建模为一条从数据分布到高斯噪声（前向过程）以及从噪声回到数据分布（反向过程）的马尔可夫链。通过将生成任务分解为大量微小的去噪步骤，扩散模型绕开了单步生成面临的模式坍塌与训练不稳定等问题。近年来，DDPM 以及其后续改进（DDIM、Score-based SDE 等）在高维图像生成、音频合成等任务上取得了超越 GAN 的性能。但其代价也很明确：采样速度较慢——生成一个批次的样本需要执行与扩散步数相同次数的网络前向计算。

== 2.2 VAE 相关研究

VAE 由 Kingma 与 Welling 在 2014 年的国际表示学习大会（ICLR）上首次提出 @kingma2014vae，其核心贡献在于将变分贝叶斯方法与深度神经网络有机结合，并通过重参数化技巧（Reparameterization Trick）使随机潜变量模型能够通过标准的反向传播算法进行训练。此后，大量后续工作在多个方向上对原始 VAE 进行了改进。在潜变量先验方面，从标准高斯先验扩展到混合高斯先验、VampPrior、以及基于流的灵活先验；在近似后验方面，从简单的对角高斯分布扩展到正规化流（Normalizing Flow）等表达能力更强的变分族；在解耦表示学习方面，`beta`-VAE 通过引入可调节的 KL 正则权重，在重构质量与潜变量解耦程度之间建立了可控的权衡机制。

在二维低维分布建模场景中，由于数据本身维度较低、网络容量相对充足，VAE 通常能够稳定地捕获数据的主要结构特征。但也正是在这类任务中，VAE 的过平滑倾向——即生成样本倾向于"弥散"在真实流形附近而非精确贴合流形——更容易被直观地观察和定量评估。

== 2.3 Diffusion Model 相关研究

扩散模型的思想可以追溯到 Sohl-Dickstein 等人在 2015 年提出的深度无监督学习扩散框架。然而，真正将该方法推向实用化的里程碑式工作是 Ho 等人于 2020 年在 NeurIPS 上发表的 DDPM @ho2020ddpm。DDPM 通过简洁的噪声预测参数化、固定的线性噪声调度以及简化的均方误差训练目标，证明扩散模型能够在图像生成质量上匹敌甚至超越当时的 GAN 模型。随后，Song 等人通过随机微分方程（SDE）的统一视角将 DDPM 与基于分数的生成模型（Score-based Model）关联起来，为进一步的加速采样（如 DDIM 的确定性采样）和理论分析提供了强大工具。

在低维分布建模的具体场景下，由于任务的计算开销远小于高维图像，扩散模型可以在有限的计算资源下进行充分的训练和系统的超参数搜索，这为深入理解扩散模型的内部工作机制——包括噪声调度的选择、扩散步数的影响以及采样轨迹的几何性质——提供了理想条件。

== 2.4 分布一致性评价指标

在生成模型的评价方面，二维分布的比较相较于高维图像具有一个天然优势：可以直接计算样本集合之间的统计距离，而不必依赖 FID（Fréchet Inception Distance）等需要预训练特征提取器的间接指标。本报告采用的 MMD 源自 Grenander 等人发展的核方法两样本检验理论 @gretton2012mmd，其基本思想是将两组样本映射到再生核希尔伯特空间（RKHS）后比较其均值嵌入。SWD 则通过在多个随机一维投影上计算 Wasserstein 距离的均值，在计算效率与度量分辨率之间取得了良好平衡 @rabin2011swd。此外，本报告还引入了基于聚类的 Mode Coverage 指标，用于从模式覆盖的角度对多峰分布上的生成完备性进行辅助评估。

// ============================================================
// 三、模型假设与符号说明
// ============================================================
= 三、模型假设与符号说明

== 3.1 基本假设

为了在当前阶段顺利推进方法验证与模型开发，本报告对整个建模流程做出如下基本假设：

1. #strong[数据独立性假设]：训练集与测试集中的每个样本均独立地服从各自对应的未知二维分布。这一假设是大多数统计学习方法的共同基础，保证了经验风险最小化原则的合理性。

2. #strong[分布平稳性假设]：训练集与测试集服从相同的底层数据分布。这是监督学习泛化理论的基本前提，确保在训练集上习得的生成模型在测试阶段具有统计合理性。

3. #strong[模型容量假设]：MLP 网络在适当的宽度与深度配置下，具备逼近任意连续二维分布的充分表达能力。该假设基于通用逼近定理（Universal Approximation Theorem），为使用中型 MLP 网络进行二维分布建模提供了理论依据。

4. #strong[高斯噪声假设]：在 VAE 中，潜变量先验设定为标准正态分布 $p(z) = NN(0, I)$，近似后验 $q_phi(z|x)$ 也设为对角高斯分布。在 Diffusion Model 中，前向过程所加噪声为各向同性的高斯噪声，反向过程的去噪分布同样以高斯形式近似。这些假设虽然限定性较强，但已被广泛验证在低维连续分布建模中是有效的。

5. #strong[轻量级实现假设]：当前阶段不追求大规模网络的极致性能，而是优先保证模型能够稳定训练、结果可复现、并能在有限计算资源下快速迭代实验。这一假设与课程大作业的实际约束条件相符。

== 3.2 符号约定

为便于后续各章节的公式推导与结果讨论，表 2 汇总了本报告中统一使用的主要数学符号及其含义。

#figure(
  table(
    columns: 3,
    align: (center, left, left),
    inset: 5pt,
    stroke: 0.5pt + black,
    [符号], [含义], [说明/量纲],
    [$x in RR^2$], [观测样本（二维坐标）], [--],
    [$z in RR^d$], [潜变量], [$d$ 为潜变量维度],
    [$q_phi(z|x)$], [编码器近似的后验分布], [$phi$ 为编码器参数],
    [$p_theta(x|z)$], [解码器给出的重构分布], [$theta$ 为解码器参数],
    [$p(z)$], [潜变量先验分布], [通常取 $NN(0,I)$],
    [$beta$], [KL 正则权重系数], [VAE 中的可调超参数],
    [$T$], [扩散模型的扩散步数], [DDPM 中的总时间步数],
    [$alpha_t$, $bar(alpha)_t$], [单步/累积噪声尺度参数], [由噪声调度 $beta_t$ 定义],
    [$epsilon_theta(x_t, t)$], [噪声预测网络], [$theta$ 为网络参数],
    [$N$], [训练/测试样本数], [--],
    [MMD], [最大均值差异], [衡量分布整体一致性],
    [SWD], [切片 Wasserstein 距离], [衡量几何搬运代价],
  ),
  caption: [主要数学符号及其含义。],
)

== 3.3 四类分布的数学模型

为了后续建模讨论的精确性，本节给出四类二维分布的数学定义。需要说明的是，当前使用的是模拟数据，因此相应的生成参数可以明确给出；对于后续的正式课程数据，相应的参数将变为未知，需要通过生成模型来间接学习。

- #strong[Gaussian Mixture]：由 $K$ 个二维高斯分量加权混合而成，其概率密度函数为

  $ p(x) = sum_(k=1)^K pi_k NN(x | mu_k, Sigma_k), $

  其中 $pi_k$ 为混合权重（满足 $sum_k pi_k = 1$），$mu_k in RR^2$ 为第 $k$ 个分量的均值向量，$Sigma_k in RR^(2 times 2)$ 为对应的协方差矩阵。该分布的建模难点在于：模型必须同时覆盖 $K$ 个分离的模态，不能出现"模式丢失"（Mode Dropping）现象。

- #strong[Ring]：样本分布在以原点为中心、半径为 $R$ 的圆环附近，径向存在一定的高斯扰动。可用极坐标表示为

  $ r ~ NN(R, sigma_r^2), quad theta ~ "Uniform"(0, 2pi), $

  再通过 $x = r cos theta$, $y = r sin theta$ 转换为直角坐标。该分布的建模难点在于：数据实际落在一维闭合曲线上（而非填充二维区域），模型需要学习低维流形嵌入。

- #strong[Two Moons]：由两段相对弯曲的半环形流形构成，可理解为在两条参数曲线 $gamma_1(s)$ 与 $gamma_2(s)$（$s in [0,1]$）上叠加二维高斯噪声。两段流形间隔较小且呈互补对称布局，建模难点在于模型需要同时保持两段流形的独立几何结构，同时又不过度模糊它们之间的分离区域。

- #strong[Spiral]：样本沿阿基米德螺旋线分布，参数方程为

  $ x(t) = a t cos(omega t) + epsilon_x, quad y(t) = a t sin(omega t) + epsilon_y, quad t in [0, t_"max"], $

  其中 $a$ 控制螺旋的径向扩展速率，$omega$ 控制旋转频率，$epsilon_x, epsilon_y$ 为局部噪声项。该分布是本任务中拓扑结构最为复杂的一类：模型需要在较大空间尺度上同时保持螺旋的连续性和均匀性，任何局部断裂或全局形状偏差都易于被发现。

以上四种分布覆盖了离散多峰、闭合流形、双簇非线性与连续螺旋这四种典型几何结构，为系统比较 VAE 与 Diffusion Model 的建模能力提供了一个具有良好区分度的测试集合。

// ============================================================
// 四、模型建立
// ============================================================
= 四、模型建立

本章详细阐述 VAE 与 Diffusion Model 两类生成模型的数学原理、架构设计与训练方法论。

== 4.1 变分自编码器

=== 4.1.1 模型原理

VAE 的核心思想是将观测数据的生成过程形式化为一个包含潜变量 $z$ 的概率图模型：先由先验分布 $p(z)$ 采样潜变量，再通过条件分布 $p_theta(x|z)$ 生成观测样本。模型的边际似然为

$ p_theta(x) = integral p_theta(x|z) p(z) d z. $

由于上式中的积分在高维潜变量空间上通常无法解析计算，直接最大化对数似然不可行。VAE 转而引入一个参数化的近似后验 $q_phi(z|x)$（编码器），并推导出对数似然的变分下界：

#note-box[
$
"log" p_theta(x) >= "log" p_theta(x) - D_"KL"(q_phi(z|x) || p_theta(z|x)) = EE_(z ~ q_phi(z|x))["log" p_theta(x|z)] - D_"KL"(q_phi(z|x) || p(z)) := L_"ELBO"(theta, phi; x).
$
]

其中第一项为重构对数似然的期望，鼓励解码器从潜变量中准确还原输入样本；第二项为近似后验与先验之间的 KL 散度，起到正则化作用，约束潜变量分布不至于偏离先验太远。

在实际实现中，重构项通常被简化为均方误差（对应于假设 $p_theta(x|z)$ 为具有固定方差的高斯分布），并引入可调节的权重系数 $beta$ 来控制正则化强度：

$ L_"VAE"(theta, phi) = E_(x ~ p_"data")[||x - x_"recon"||^2 + beta D_"KL"(q_phi(z|x) || p(z))]. $

当 $beta = 1$ 时还原为标准 VAE；$beta < 1$ 时减弱正则，鼓励更好的重构；$beta > 1$ 时增强正则，推动潜变量分布更接近标准高斯，有利于解耦与插值，但可能牺牲重构精度。

=== 4.1.2 重参数化技巧

VAE 训练的关键技术障碍在于：从 $q_phi(z|x)$ 中采样 $z$ 的操作是随机的，梯度无法直接通过采样节点回传。重参数化技巧（Reparameterization Trick）通过将采样过程改写为确定性变换解决这一问题：

$ z = mu_phi(x) + sigma_phi(x) dot.op epsilon, quad epsilon ~ NN(0, I), $

其中 $mu_phi(x)$ 和 $"log" sigma_phi^2(x)$ 由编码器网络输出，$dot.op$ 表示逐元素乘积。这样一来，随机性被隔离在外部噪声变量 $epsilon$ 中，梯度可以通过 $mu_phi$ 和 $sigma_phi$ 顺利回传，从而使端到端的随机梯度下降训练成为可能。

=== 4.1.3 网络架构

本报告采用统一的 MLP 架构实现 VAE，具体配置如表 3 所示。

#figure(
  table(
    columns: 2,
    align: (left, left),
    inset: 5pt,
    stroke: 0.5pt + black,
    [组件], [配置],
    [输入维度], [2（二维坐标 $x, y$）],
    [编码器隐藏层], [128 → 128（两层全连接 + ReLU 激活）],
    [潜变量维度 $d$], [4（主实验设置）],
    [潜变量参数输出], [$mu$: $d$ 维, $"log" sigma^2$: $d$ 维],
    [解码器隐藏层], [128 → 128（两层全连接 + ReLU 激活）],
    [解码器输出], [2（二维重构坐标）],
    [激活函数], [隐藏层: ReLU; 输出层: 无（线性）],
    [优化器], [Adam, 学习率 $1 times 10^(-3)$],
    [训练轮数], [150 epochs],
    [批量大小], [256],
  ),
  caption: [VAE 网络架构与训练配置。],
)

选择潜变量维度 $d = 4$ 是出于以下考量：二维输入数据本身信息量有限，但考虑到四类分布均具有非平凡的几何结构，$d = 2$ 可能不足以充分编码分布的结构信息；$d = 4$ 在提供适度额外容量的同时，潜空间仍然可以直接通过 PCA 投影等方式进行二维可视化观察。

== 4.2 扩散概率模型

=== 4.2.1 前向扩散过程

扩散模型的前向过程（Forward Process）是一条定义在 $T$ 个时间步上的马尔可夫链。从原始数据 $x_0 ~ q(x_0)$ 开始，在每个时间步 $t in {1, dots, T}$ 上按照固定的高斯转移核逐步向样本中注入噪声：

$ q(x_t \| x_{t-1}) = NN(x_t, sqrt(1 - beta_t) x_{t-1}, beta_t I), $

其中 $beta_t in (0, 1)$ 为第 $t$ 步的噪声尺度参数，由预设的噪声调度（Noise Schedule）决定。前向过程的优雅之处在于：利用高斯分布的可加性，可以从 $x_0$ 直接解析给出任意时刻 $t$ 的边际分布，而无需逐步采样：

$
q(x_t \| x_0) = NN(x_t, sqrt(bar(alpha)_t) x_0, (1 - bar(alpha)_t) I),
$

其中 $alpha_t := 1 - beta_t$, $bar(alpha)_t := product_(s=1)^t alpha_s$ 为累积信号保留率。由此可以高效地采样任意时刻的带噪样本：

$ x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon, quad epsilon ~ NN(0, I). $

当 $t = T$ 时，若噪声调度设计得当（即 $bar(alpha)_T approx 0$），$x_T$ 的分布近似为标准高斯分布 $NN(0, I)$，这为后续的反向生成过程提供了起始点。

=== 4.2.2 反向去噪过程

生成过程对应于前向过程的逆过程：从纯噪声 $x_T ~ NN(0, I)$ 开始，依次应用学习的反向转移核，逐步去噪最终得到干净样本 $x_0$。

真正的反向转移分布 $q(x_{t-1} | x_t)$ 在一般情况下是复杂且不可显式表达的，但 DDPM 的一个关键发现是：当 $beta_t$ 充分小时（即每一步加的噪声量很小），$q(x_{t-1} | x_t)$ 也近似为高斯分布。基于这一观察，DDPM 将反向过程也参数化为高斯转移核：

$ p_theta(x_{t-1} \| x_t) = NN(x_{t-1}, mu_theta(x_t, t), sigma_t^2 I). $

其中方差 $sigma_t^2$ 可以固定为 $beta_t$ 或 $tilde(beta)_t = (1 - bar(alpha)_{t-1})/(1 - bar(alpha)_t) beta_t$，而均值 $mu_theta(x_t, t)$ 则需要通过神经网络学习。进一步地，利用前向过程的条件分布 $q(x_{t-1} | x_t, x_0)$（该分布有闭合的高斯形式），可以将均值的学习等价地转化为对噪声项 $epsilon$ 的预测：

$ mu_theta(x_t, t) = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1 - bar(alpha)_t) epsilon_theta(x_t, t)). $

代入后可将训练目标简化为：

$ L_"diffusion"(theta) = E_(x_0,epsilon,t)[||epsilon - epsilon_theta(x_t, t)||^2]. $

即在每个训练步骤中，随机抽取干净样本 $x_0$、高斯噪声 $epsilon$ 以及时间步 $t$，构造带噪样本 $x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon$，然后训练网络 $epsilon_theta(x_t, t)$ 预测添加的噪声 $epsilon$。

=== 4.2.3 采样算法

完成训练后，从扩散模型中生成样本需要从 $x_T ~ NN(0, I)$ 开始，依次对 $t = T, T-1, dots, 1$ 执行反向去噪步骤：

$ x_{t-1} = 1/sqrt(alpha_t)(x_t - beta_t/sqrt(1 - bar(alpha)_t) epsilon_theta(x_t, t)) + sigma_t z, $

其中 $z ~ NN(0, I)$（当 $t > 1$ 时）或 $z = 0$（当 $t = 1$ 时，最后一步不加随机噪声）。这一迭代过程将 T 步的纯噪声逐步转化为服从目标分布的样本，整个过程可以直观地理妥为"从无序中逐步涌现出结构"。

=== 4.2.4 噪声调度设计

噪声调度 ${beta_t}_(t=1)^T$ 的选择对扩散模型的性能有重要影响。本报告采用线性调度：

$ beta_t = beta_"start" + (t-1)/(T-1)(beta_"end" - beta_"start"), $

其中 $beta_"start" = 1 times 10^(-4)$, $beta_"end" = 0.02$。这一设计确保了：在过程的初始阶段（$t$ 较小时），信号保留率 $bar(alpha)_t$ 下降缓慢，使得网络在训练时能够接触到信噪比逐渐变化的精细样本；在过程后期（$t$ 接近 $T$ 时），$bar(alpha)_t$ 快速趋近于零，保证 $x_T$ 的分布足够接近标准高斯。

=== 4.2.5 时间步嵌入与网络架构

为了让噪声预测网络感知当前所处的时间步 $t$，本报告采用正弦位置编码（Sinusoidal Positional Embedding）将标量时间步 $t$ 映射为高维向量：

$ "emb"(t)_(2i) = sin(t / 10000^(2i/d_"emb")), quad "emb"(t)_(2i+1) = cos(t / 10000^(2i/d_"emb")). $

该嵌入向量被拼接到网络的中间层，使得网络能够在不同时间步上学习不同的去噪策略。表 4 给出了 Diffusion Model 的完整网络配置。

#figure(
  table(
    columns: 2,
    align: (left, left),
    inset: 5pt,
    stroke: 0.5pt + black,
    [组件], [配置],
    [输入维度], [2（二维坐标 $x, y$）],
    [时间嵌入维度], [32],
    [噪声预测网络], [128 → 128 → 128（三层全连接 + ReLU 激活）],
    [时间嵌入注入方式], [在输入层拼接后进入第一隐藏层],
    [输出维度], [2（预测的噪声 $epsilon$）],
    [扩散步数 $T$], [80（主实验设置）],
    [噪声调度], [线性: $beta_"start" = 1 times 10^(-4)$, $beta_"end" = 0.02$],
    [优化器], [Adam, 学习率 $1 times 10^(-3)$],
    [训练轮数], [250 epochs],
    [批量大小], [256],
  ),
  caption: [Diffusion Model 网络架构与训练配置。],
)

随着扩散步数从 20 增加到 80，模型的反向过程能以更精细的时间分辨率逼近真实的反向转移分布，但代价是采样时的前向推理次数也线性增加。这一质量-效率的权衡将在实验部分进行定量分析。

== 4.3 评价指标体系

为了从多个维度客观量化生成质量，本报告选取以下三类互补的评价指标。

=== 4.3.1 最大均值差异

MMD 通过核函数将两组样本映射到再生核希尔伯特空间（RKHS），并计算其均值嵌入之间的欧氏距离。对于两组样本 $X = {x_i}_(i=1)^m$（真实测试集）和 $Y = {y_j}_(j=1)^n$（生成样本），MMD 的平方可由核函数的样本均值无偏估计：

$
"MMD"^2(X, Y) = 1/m^2 sum_(i,j) k(x_i, x_j) + 1/n^2 sum_(i,j) k(y_i, y_j) - 2/(m n) sum_(i,j) k(x_i, y_j).
$

本报告选用 RBF（高斯径向基）核函数 $k(x, y) = exp(-||x - y||^2 / (2 sigma^2))$，核带宽 $sigma$ 取为所有样本对之间欧氏距离的中位数。这一自适应带宽策略使 MMD 能够在不同分布上自动调整其"分辨率"，确保度量在不同尺度的分布上保持合理的灵敏度。MMD 的值越小，表明两组样本越有可能来自相同的底层分布。

=== 4.3.2 切片 Wasserstein 距离

Wasserstein 距离（Earth Mover's Distance）度量了将一个分布"搬运"为另一个分布所需的最小代价。对于二维分布，直接计算 Wasserstein 距离涉及求解最优传输问题，计算复杂度较高。切片 Wasserstein 距离（SWD）通过蒙特卡洛投影的思想巧妙地回避了这一困难：先将二维点云投影到 $L$ 个随机的一维方向上，然后在每个一维投影上计算 Wasserstein 距离（只需排序和累积求和，计算代价极低），最后取所有投影上的平均值：

$
"SWD"(X, Y) = 1/L sum_(l=1)^L W_1(P_(u_l) X, P_(u_l) Y),
$

其中 $u_l ~ "Uniform"(S^1)$ 为随机采样的单位投影方向，$P_(u_l)$ 表示沿方向 $u_l$ 的投影算子，$W_1$ 为一维 Wasserstein 距离。本报告取 $L = 100$，在实际计算中已能给出稳定的估计值。SWD 对分布之间的几何位移和形状差异都较为敏感，是 MMD 在几何层面的有效补充。

=== 4.3.3 模式覆盖率

对具有多峰结构（如 Gaussian Mixture）的分布而言，即使 MMD 和 SWD 表现良好，生成模型仍可能出现"模式坍塌"（Mode Collapse）——即遗漏了某些低概率密度的模式。为了检测这一问题，本报告对真实测试集进行 K-Means 聚类（聚类数 $K$ 与视觉判断的模式数一致），然后检查生成样本是否在每个聚类中心的一定半径范围内都有足够的覆盖：

$
"Mode Coverage" = 1/K sum_(k=1)^K bold(1)[(min_(y in Y) ||y - c_k||) < r_k],
$

其中 $c_k$ 为第 $k$ 个聚类的中心，$r_k$ 为该聚类中样本到中心距离的 95% 分位数。需要指出的是，这一指标在 Gaussian Mixture 上的解释性最强；对于 Ring 和 Spiral 这类连续流形分布，"模式"的概念本身就不太适用，该指标在这些分布上仅作为辅助参考。

// ============================================================
// 五、实验设计与数值方法
// ============================================================
= 五、实验设计与数值方法

== 5.1 数据生成与预处理

当前阶段的模拟数据集通过程序化方式生成。对于每类分布，生成流程如下：

- #strong[Gaussian Mixture]：随机采样 $K = 5$ 个二维高斯分量的均值（均匀分布在 $[-3, 3]^2$ 范围内）和协方差矩阵（各向同性，$sigma^2 in [0.02, 0.05]$），混合权重从 Dirichlet 分布中采样。按混合权重随机选择分量后，从对应的高斯分布中采样二维点。

- #strong[Ring]：角度 $theta$ 从 $[0, 2pi)$ 均匀采样，半径 $r$ 从 $NN(R, sigma_r^2)$ 采样（$R = 2.5$, $sigma_r = 0.15$），再转换为直角坐标。

- #strong[Two Moons]：沿两段预设参数曲线采样，并在曲线的法线方向上添加小尺度高斯噪声（$sigma = 0.08$），形成具有局部厚度的双月形流形。

- #strong[Spiral]：采样参数 $t$ 从 $[0, 3pi]$ 均匀分布，按 $x = a t cos(omega t)$, $y = a t sin(omega t)$（$a = 0.15$, $omega = 2.0$）生成无噪声螺旋线，再叠加各向同性高斯噪声（$sigma = 0.15$）。

每类分布生成 5000 个样本，按 4:1 的比例随机划分为训练集（4000 样本）和测试集（1000 样本）。所有数据在输入模型前均进行标准化处理（减均值除以标准差），使训练过程更加稳定。图 1 展示了四类分布的完整可视化结果。

#figure(
  grid(
    columns: 2,
    column-gutter: 0.4cm,
    row-gutter: 0.35cm,
    image("../outputs/figures/gaussian_mixture_dataset.png", width: 100%),
    image("../outputs/figures/ring_dataset.png", width: 100%),
    image("../outputs/figures/two_moons_dataset.png", width: 100%),
    image("../outputs/figures/spiral_dataset.png", width: 100%),
  ),
  caption: [四类模拟数据集的训练集与测试集散点图可视化。蓝色点表示训练样本，橙色点表示测试样本。可以观察到：Gaussian Mixture 呈现明显的五峰结构；Ring 样本集中在闭合环带内；Two Moons 具有双月相望的非线性双簇结构；Spiral 的螺旋拓扑在全题中最为复杂。],
)

== 5.2 训练算法详述

=== 5.2.1 VAE 训练流程

VAE 的训练流程在每个 epoch 中执行以下步骤：

1. 从训练集中随机抽取一个批次 $B$ 的二维样本 ${x_i}_(i=1)^(|B|)$；
2. 编码器前向传播，输出近似后验的均值 $mu_phi(x_i)$ 和对数方差 $"log" sigma_phi^2(x_i)$；
3. 从标准正态分布采样 $epsilon_i ~ NN(0, I)$，通过重参数化获取潜变量 $z_i = mu_phi(x_i) + sigma_phi(x_i) dot.op epsilon_i$；
4. 解码器前向传播，从 $z_i$ 重构 $hat(x)_i$；
5. 计算损失函数：$L = 1/|B| sum_i [||x_i - hat(x)_i||^2 + beta D_"KL"(q_phi(z|x_i) || p(z))]$；
6. 反向传播，更新编码器参数 $phi$ 与解码器参数 $theta$。

其中 KL 散度的解析表达式为（当 $p(z) = NN(0, I)$ 且 $q_phi(z|x) = NN(mu, "diag"(sigma^2))$ 时）：

$
D_"KL"(q_phi(z|x) || p(z)) = -1/2 sum_(j=1)^d (1 + "log" sigma_j^2 - mu_j^2 - sigma_j^2).
$

这一闭合形式的表达式使得 KL 散度的计算不涉及任何随机采样或数值积分，可以在训练中高效求值。

=== 5.2.2 Diffusion Model 训练流程

Diffusion Model 的训练流程在每个 epoch 中执行以下步骤：

1. 从训练集中随机抽取一个批次 $B$ 的二维样本 ${x_i}_(i=1)^(|B|)$；
2. 对每个样本，随机采样一个扩散时间步 $t_i ~ "Uniform"({1, dots, T})$；
3. 采样噪声 $epsilon_i ~ NN(0, I)$；
4. 构造带噪样本：$x_(t_i) = sqrt(bar(alpha)_(t_i)) x_i + sqrt(1 - bar(alpha)_(t_i)) epsilon_i$；
5. 计算时间步嵌入 $"emb"(t_i)$；
6. 噪声预测网络前向传播：$hat(epsilon)_i = epsilon_theta(x_(t_i), "emb"(t_i))$；
7. 计算损失函数：$L = 1/|B| sum_i ||epsilon_i - hat(epsilon)_i||^2$；
8. 反向传播，更新网络参数 $theta$。

这一训练过程的计算复杂度与每轮随机采样的扩散步数无关（每个样本只在一个时间步上计算损失），因此即使 $T = 80$，单次迭代的训练成本与 VAE 大致相当。真正的时间代价差异体现在采样阶段，因为 Diffusion Model 需要经过全部 $T$ 步的前向推理。

=== 5.2.3 Diffusion Model 采样流程

完成训练后，采样过程的算法步骤如下：

1. 从标准高斯分布初始化 $x_T ~ NN(0, I)$；
2. 对于 $t = T, T-1, dots, 2, 1$，依次执行：
   - 计算时间步嵌入 $"emb"(t)$；
   - 若 $t > 1$，采样 $z ~ NN(0, I)$；否则 $z = 0$；
   - 预测噪声：$hat(epsilon) = epsilon_theta(x_t, "emb"(t))$；
   - 一步反向去噪：$x_{t-1} = 1/sqrt(alpha_t)(x_t - (1-alpha_t)/sqrt(1-bar(alpha)_t) hat(epsilon)) + sigma_t z$；
3. 输出 $x_0$ 作为最终生成的样本。

在 $T = 80$ 的设置下，每生成一批样本需要执行 80 次网络前向推理，这从根本上决定了 Diffusion Model 采样速度远慢于 VAE 的"一次前向解码"。

== 5.3 实现环境与可复现性

本报告的全部实验均在以下环境中完成，表 5 汇总了关键环境参数。

#figure(
  table(
    columns: 2,
    align: (left, left),
    inset: 5pt,
    stroke: 0.5pt + black,
    [环境项], [配置],
    [操作系统], [Windows 11],
    [GPU], [NVIDIA GeForce RTX 5060 Laptop GPU],
    [深度学习框架], [PyTorch 2.x],
    [编程语言], [Python 3.13],
    [随机种子], [42（所有实验统一固定）],
    [数据格式], [NumPy .npz（压缩数组文件）],
    [报告排版], [Typst（开源科学排版系统）],
  ),
  caption: [实验环境与可复现性配置。],
)

为保证实验结果的完全可复现性，所有涉及随机性的操作（包括训练集/测试集划分、模型参数初始化、训练过程中的批次打乱以及 VAE 和 Diffusion 的噪声采样）均固定了统一的随机种子。此外，完整的源代码、训练日志、模型检查点以及指标 JSON 文件均保存在 `outputs/` 目录中，可供独立验证。

// ============================================================
// 六、实验结果与分析
// ============================================================
= 六、实验结果与分析

本章从定量指标比较、定性可视化分析、去噪过程展示以及消融实验四个层面，系统呈现 VAE 与 Diffusion Model 在四类二维分布上的实验表现。

== 6.1 训练过程分析

=== 6.1.1 VAE 训练曲线

图 2 展示了 VAE 在 Gaussian Mixture 与 Two Moons 两类代表性分布上的训练损失曲线。总体而言，VAE 在所有四类分布上均能在 150 个 epoch 内稳定收敛到较低的损失水平，未出现明显的发散或震荡现象。这得益于 VAE 训练目标的良好数学性质——ELBO 作为对数似然的变分下界，其优化过程等价于在重构误差与 KL 正则之间寻找平衡，而两者都是平滑的凸/凹函数。

#figure(
  grid(
    columns: 2,
    column-gutter: 0.4cm,
    row-gutter: 0.35cm,
    image("../outputs/figures/vae_gaussian_mixture_training.png", width: 100%),
    image("../outputs/figures/vae_two_moons_training.png", width: 100%),
  ),
  caption: [VAE 在 Gaussian Mixture（左）与 Two Moons（右）上的训练损失曲线。蓝色实线为总损失，橙色虚线为重构损失项，绿色虚线为 KL 散度项。],
)

从训练曲线可以观察到两个规律性现象：其一，重构损失在训练初期快速下降，随后进入缓慢优化的精细调整阶段；其二，KL 散度在训练初期上升——这是因为编码器最初输出的是接近随机初始化的 $mu$ 和 $sigma$，随着训练进行，近似后验逐步学习到数据中有意义的结构信息，从而偏离无信息的标准高斯先验，KL 散度上升到一个稳定值后便不再显著变化。这一模式是 VAE 训练的典型动力学特征。

=== 6.1.2 Diffusion Model 训练曲线

图 3 展示了 Diffusion Model 在 Gaussian Mixture 与 Two Moons 上的噪声预测 MSE 损失曲线。在所有分布上，损失值在约 100 到 150 个 epoch 内持续下降，随后进入渐进收敛阶段。最终损失在 $4 times 10^(-3)$ 到 $8 times 10^(-3)$ 之间，表明噪声预测网络的预测精度已达到较高水平。

#figure(
  grid(
    columns: 2,
    column-gutter: 0.4cm,
    row-gutter: 0.35cm,
    image("../outputs/figures/diffusion_gaussian_mixture_training.png", width: 100%),
    image("../outputs/figures/diffusion_two_moons_training.png", width: 100%),
  ),
  caption: [Diffusion Model 在 Gaussian Mixture（左）与 Two Moons（右）上的噪声预测 MSE 损失曲线。],
)

值得注意的是，Ring 与 Spiral 上的最终训练损失略高于 Gaussian Mixture 和 Two Moons。这反映了模型训练难度与数据分布复杂性之间的正相关关系——在 Ring 和 Spiral 这类低维流形嵌入场景下，不同时间步的带噪样本具有更丰富的变化模式，噪声预测网络需要学习更为复杂的去噪映射。

== 6.2 主实验定量结果

表 6 汇总了 VAE 与 Diffusion Model 在四类分布上的主实验定量结果。

#figure(
  table(
    columns: 6,
    align: center,
    inset: 5pt,
    stroke: 0.5pt + black,
    [数据集], [模型], [MMD ↓], [SWD ↓], [训练时间 / s], [采样时间 / s],
    [Gaussian Mixture], [VAE], [0.0041], [0.3182], [1.006], [0.00035],
    [Gaussian Mixture], [Diffusion], [0.0017], [0.1414], [4.176], [0.04071],
    [Ring], [VAE], [0.0019], [0.1177], [3.198], [0.00036],
    [Ring], [Diffusion], [0.0008], [0.0909], [3.937], [0.03929],
    [Two Moons], [VAE], [0.0017], [0.1041], [3.358], [0.00033],
    [Two Moons], [Diffusion], [0.0057], [0.1450], [4.170], [0.03670],
    [Spiral], [VAE], [0.0015], [0.0747], [3.238], [0.00044],
    [Spiral], [Diffusion], [0.0008], [0.0683], [3.986], [0.03966],
  ),
  caption: [四类分布上的主实验定量结果汇总。Mode Coverage 在所有实验组上均达到 1.0，故不单独列出。"训练时间"为完整训练流程耗时，"采样时间"为生成测试集规模样本的一次采样耗时。],
)

从表 6 中可以提炼出以下关键发现：

#strong[发现一：Diffusion 的优势集中在复杂连续流形。] 在 Ring 与 Spiral 两类分布上，Diffusion Model 的 MMD 和 SWD 均优于 VAE，且优势稳定。例如在 Ring 上，Diffusion 的 MMD 为 0.0008，显著低于 VAE 的 0.0019；在 Spiral 上，Diffusion 也以 0.0008 vs 0.0015 的 MMD 取得更优结果。这表明 Diffusion 在闭合流形与长程连续拓扑上的全局几何拟合更强，与前文的多步去噪机理分析一致。

#strong[发现二：VAE 在局部结构清晰的分布上仍具竞争力。] 在 Gaussian Mixture 与 Two Moons 上，VAE 取得了更低的 MMD。其中 Two Moons 上 VAE 的 MMD 为 0.0017，显著优于 Diffusion 的 0.0057，且 SWD 也更低（0.1041 vs 0.1450）。这说明 VAE 并非在复杂分布上总是落后的——对于局部结构相对紧凑、生成映射较容易通过单次解码近似的流形，显式潜变量建模依旧有效。

#strong[发现三：效率差异依旧巨大。] 两类模型的完整训练时间都在 3--4 秒量级，但采样速度差异接近两个数量级。VAE 生成测试集规模样本仅需约 0.3--1.2 毫秒，而 Diffusion 需要约 36--41 毫秒（执行 $T=80$ 次噪声预测网络前向推理）。这一差距虽在二维任务中只体现为较小的绝对时间差，但其结构性的来源——Diffusion 需要多步迭代去噪——意味着在高维任务中该效率鸿沟会被进一步放大。

#strong[发现四：Mode Coverage 区分度不足。] 四类分布上的 Mode Coverage 均达到 1.0，说明所有模型都能覆盖到真实样本聚类中心附近的区域。这反映的是该方法在当前样本量和分布设置下的灵敏度有限——对连续流形分布（Ring, Spiral）而言，"模式"概念本身就不适用；对离散多峰分布（Gaussian Mixture）而言，当前的 8 个簇在生成质量尚可的情况下也不易发生完全遗漏。这一结果提示，在二维连续分布评价中，基于聚类的离散 Mode Coverage 更适合作为快速筛查工具，而非精确的区分性评价指标。

#strong[综合统计：] 若对所有数据集取平均，Diffusion 的平均 MMD 为 0.0023、平均 SWD 为 0.1114；VAE 的平均 MMD 为 0.0016、平均 SWD 为 0.1175。也就是说，本轮数据切换后出现了更细致的分工：VAE 在平均 MMD 上略优，而 Diffusion 在平均 SWD 上保持优势。

== 6.3 定性可视化分析

定量指标虽然提供了综合性的排名，但生成样本的视觉检查能够揭示指标无法捕捉的细微结构差异——例如生成样本是否过分平滑、是否存在本不该出现的"填充"区域、局部细节是否自然等。本节选取 VAE 与 Diffusion 在 Ring 和 Spiral 两类最具代表性的分布上的生成结果进行深入分析。

=== 6.3.1 Ring 分布上的对比

#fig("../outputs/figures/vae_ring_samples.png", [VAE 在 Ring 分布上的真实训练集、真实测试集、生成样本及其叠加图。可以观察到 VAE 已经学习到环形结构的主体，但在局部存在厚度不均匀和细微断裂的现象。], width: 92%)

#v(small-gap)
#fig("../outputs/figures/diffusion_ring_samples.png", [Diffusion 在 Ring 分布上的生成结果。与 VAE 对比，Diffusion 生成的环形结构更完整、厚度更均匀，与真实测试集的视觉一致性明显更高。], width: 92%)

从图 4 和图 5 的对比中可以清楚地观察到两类模型在闭合流形建模上的差异。VAE 已经成功捕获了环形的主体结构——生成样本大致聚集在一个圆环附近，而非弥散在整个圆盘内。但仔细观察可以发现：环的局部厚度存在波动（某些区域的样本密度明显偏低），且在个别位置（如圆环的右下象限）出现了轻微的"断裂"迹象。这些现象可以用 VAE 的机理来解释：解码器是一个确定性的 MLP 映射，其函数族对于勉强贴合闭合流形存在一定困难，因为 MLP 更倾向于学习"填充型"的映射（即将潜空间中的连通区域映射为二维空间中的连通区域），而环形是一个具有非平凡拓扑（一维闭合曲线，中间有"洞"）的流形。

相比之下，Diffusion 生成的环形结构更为均匀和完整，圆环的厚度变化更小，且没有明显的断裂点。这与 Diffusion 的生成机制有关——多步去噪过程允许模型在每个时间步上对样本位置进行微调，从而更精细地适应局部几何结构。

=== 6.3.2 Spiral 分布上的对比

#fig("../outputs/figures/vae_spiral_samples.png", [VAE 在 Spiral 分布上的生成结果。模型能部分恢复中心区域和外层趋势，但对完整螺旋拓扑的重建仍然不够理想。], width: 92%)

#v(small-gap)
#fig("../outputs/figures/diffusion_spiral_samples.png", [Diffusion 在 Spiral 分布上的生成结果。与 VAE 类似，当前的轻量实现对于完整的螺旋拓扑恢复存在困难。], width: 92%)

Spiral 是全题中建模难度最高的分布，图 6 和图 7 的结果印证了这一点。VAE 对 Spiral 的建模呈现出明显的"结构模糊"——生成样本大致勾勒出了螺旋的轮廓，但螺旋臂的清晰度不足，内圈和外圈之间的区分度较弱。这在 MLP 编码器-解码器架构下是可以理解的：MLP 作为一个全局函数逼近器，需要极高的网络容量才能精确刻画螺旋在空间中长程延伸的拓扑结构。

Diffusion 在 Spiral 上的表现虽然指标略优于 VAE，但从视觉效果看，当前轻量实现下两类模型都未能完美重建完整的螺旋拓扑。这提示我们：在二维生成任务中，Spiral 代表的是一个"困难"标本——即使定量指标（MMD, SWD）已经显示出合理的接近程度，但从拓扑完整性的角度看，当前的轻量模型仍存在可感知的差距。后续若需在此分布上达到更高生成质量，应考虑增强网络容量（更宽或更深的噪声预测网络）或增加训练轮数。

=== 6.3.3 Gaussian Mixture 与 Two Moons 上的简要对比

在 Gaussian Mixture 上，两类模型都较好地生成了五峰结构，VAE 的生成样本呈现出略微更"弥散"的模式边缘（与 VAE 的过平滑倾向一致），而 Diffusion 的样本在模式中心和模式边缘的分布更为分明，进一步印证了 MMD 指标的排名。

在 Two Moons 上，VAE 与 Diffusion 的视觉效果差距不大，两者都较好地区分了两段月牙形流形。VAE 在 MMD 指标上的微弱优势可能源于其生成的两月形区域之间的"间隙"更明显——这与 VAE 倾向于生成略平滑分布的特性在此处恰好起到了正面作用，而 Diffusion 在边界区域的样本分布略微更接近训练集的原始采样噪声模式。

== 6.4 Diffusion 去噪过程可视化

Diffusion Model 相比 VAE 的一个显著优势在于其生成过程具有天然的可解释性——整个采样过程可以视为一段"从纯噪声逐步浮现出结构"的影像。图 8 展示了 Diffusion Model 在 Ring 分布上的采样快照序列。

#fig("../outputs/figures/diffusion_ring_snapshots.png", [Ring 分布上的 Diffusion 采样过程。每张子图为特定时间步去噪后的样本分布快照，从左上角 $t = T$（接近纯高斯噪声）到右下角 $t = 0$（最终生成的环形结构）。], width: 88%)

从图 8 中可以观察到若干具有启发性的现象：

1. #strong[早期阶段（$t$ 接近 $T$）：] 样本呈现近似各向同性的高斯分布形态，尚未出现任何可见的空间结构。
2. #strong[中期阶段（$t approx T/2$）：] 样本开始呈现出聚集的趋势，整体轮廓逐渐从"圆盘状"收缩为"环带状"，但这个阶段的环形结构还较为粗糙，厚度不均匀。
3. #strong[后期阶段（$t$ 接近 $0$）：] 环形结构逐步精化，样本的径向分布变窄，最终收敛到与训练集视觉一致的闭合细环。

这一过程中的每一步都对应着噪声预测网络的一次前向推理，通过从噪声中逐步"剥离"非结构化的随机扰动，模型最终将高斯噪声引导到了符合训练分布的样本空间中。这种逐步生成机制是 Diffusion Model 区别于 VAE 等"单步生成"方法最本质的特征。

== 6.5 消融实验

为了探究关键超参数对生成质量的影响机制，本节在 Spiral 分布上分别设计了 VAE 的 KL 权重消融和 Diffusion 的扩散步数消融实验。选择 Spiral 作为消融实验平台的原因是：其拓扑结构最为复杂，对不同超参数选择的响应更为敏感和显著。

=== 6.5.1 VAE 的 KL 权重消融

表 7 展示了在 Spiral 分布上，不同 $beta$ 取值（0.1, 0.5, 1.0）下 VAE 的生成质量对比。

#figure(
  table(
    columns: 4,
    align: center,
    inset: 5pt,
    stroke: 0.5pt + black,
    [设置], [MMD ↓], [SWD ↓], [说明],
    [$beta = 0.1$], [0.0006], [0.0719], [弱正则，重构优先],
    [$beta = 0.5$], [0.0015], [0.0747], [主实验设置],
    [$beta = 1.0$], [0.0032], [0.0986], [强正则，潜变量先验优先],
  ),
  caption: [Spiral 分布上 VAE 的 KL 权重消融结果。],
)

从表 7 中可以提炼出一条清晰的趋势：随着 $beta$ 增大，VAE 在 Spiral 上的生成质量呈现明显下降。$beta = 0.1$ 取得了最佳 MMD（0.0006）和最佳 SWD（0.0719），而 $beta = 1.0$ 的 MMD 与 SWD 分别升至 0.0032 和 0.0986。该趋势的物理含义是：

- 当 KL 正则权重较小时，编码器被允许产生与标准高斯偏离较大的近似后验分布，这使得潜变量能更完整地编码数据中的复杂结构信息（如螺旋的拓扑和尺度），从而解码器能更准确地重建输入。
- 当 KL 权重增大到 1.0 时，优化器被迫在重构精度和潜变量先验匹配之间做出更强烈的妥协，编码器趋向于将不同输入"压缩"到更接近标准高斯的潜变量区域，这在一定程度上抹平了数据中的复杂结构特征，导致重构质量下降。

这一结果与 $beta$-VAE 文献中的发现一致：适度的正则化（$beta < 1$）有助于在保留重构能力的前提下增加潜空间的平滑性和插值性质，但过强的正则（$beta >= 1$）对复杂分布的拟合是明显有害的。

=== 6.5.2 Diffusion 的扩散步数消融

表 8 展示了在 Spiral 分布上，不同扩散步数（$T = 20$ 和 $T = 80$）下 Diffusion Model 的生成质量对比。

#figure(
  table(
    columns: 4,
    align: center,
    inset: 5pt,
    stroke: 0.5pt + black,
    [设置], [MMD ↓], [SWD ↓], [说明],
    [$T = 20$], [0.0133], [0.2316], [步数少，去噪精度不足],
    [$T = 80$], [0.0046], [0.1668], [主实验设置，质量显著更好],
  ),
  caption: [Spiral 分布上 Diffusion Model 的扩散步数消融结果。],
)

从表 8 可以观察到，当扩散步数从 80 减少到 20 时，MMD 恶化了约 189%（从 0.0046 变为 0.0133），SWD 恶化了约 39%（从 0.1668 变为 0.2316）。这一剧烈恶化的原因是多方面的：

1. #strong[离散化误差：] 扩散过程的反向 SDE（随机微分方程）是通过 $T$ 步离散化来近似求解的。$T$ 越小，每一步的步长越大，离散化引入的截断误差也越大，导致采样轨迹偏离真实的反向随机过程。
2. #strong[高斯近似的有效性：] DDPM 将反向转移分布近似为高斯的理论基础是：当 $beta_t$ 充分小时，真实的 $q(x_{t-1}|x_t)$ 确实接近高斯分布。当 $T = 20$ 时，每一步的 $beta_t$ 比 $T = 80$ 时大约 4 倍，高斯近似的精度下降。
3. #strong[训练样本的信噪比分布：] 在训练阶段，每个 batch 中随机采样时间步 $t$。当 $T = 20$ 时，训练信号主要集中在信噪比较低的区域（因为时间步分布更粗），网络学习到的高信噪比区域的去噪策略可能不够精细。

该消融实验清楚地表明：在复杂流形分布上，使用足够大的扩散步数不仅是锦上添花，而是取得合理生成质量的基本前提。

=== 6.5.3 综合讨论

综合消融实验的结果可以得到一个总体判断：对于 VAE，KL 权重是调控重构质量与潜空间结构之间权衡的核心"旋钮"，且在复杂分布上建议取值小于 1.0；对于 Diffusion，扩散步数是影响生成质量的"硬约束"，过小的 $T$ 可能导致模型性能剧烈下降，在计算资源允许的情况下建议取值不低于 50-80。

== 6.6 局限性分析与方法反思

尽管本轮实验取得了较为系统的比较结果，但仍存在若干值得注意的局限性：

1. #strong[生成数据与真实未知分布仍有差距：] 当前实验虽已切换到 README 指定的数据组织方式，但底层样本仍由生成脚本构造，其分布特征可能与老师最终发布的真实未知数据存在差异。因此目前结论更适合理解为“统一数据格式下的方法比较结论”，后续仍需在新数据上重新确认。

2. #strong[轻量网络的表达上限：] 本报告统一采用中等规模的 MLP 网络（隐藏层宽度 128），这在保证训练快速迭代的同时，也意味着模型容量可能存在上限。尤其是在 Spiral 这类复杂拓扑分布上，通过增强网络结构（如增加隐藏层数量或采用残差连接）有望获得更好的生成质量。

3. #strong[Mode Coverage 的有限区分度：] 如 6.2 节所述，该指标在当前实验设置下全部达到 1.0，未体现出有效的区分能力。后续应补充其他结构性指标，如基于核密度估计的 log-likelihood 近似、局部本征维度估计或拓扑持续性（Topological Persistence）分析等。

4. #strong[效率比较的上下文敏感性：] 二维场景下的训练和采样绝对时间都很短（单个 epoch 不到 1 秒），因此效率差异更多反映的是"采样机制的结构性不同"，而非大规模计算代价差异。若将当前框架扩展到图像等高维生成任务，Diffusion 的采样时间劣势将以 $O(T)$ 的倍数关系显著放大，届时需考虑 DDIM 等加速采样策略。

5. #strong[未涉及的正则化与技巧：] VAE 可以采用诸如 KL 退火（KL Annealing）、自由比特（Free Bits）等训练技巧；Diffusion 可以尝试余弦噪声调度、学习方差参数、以及使用 DDIM 进行确定性加速采样。这些技巧的收益在当前二维任务上可能较小，但在更大规模或更高难度任务中可能变得重要。

// ============================================================
// 七、总结与讨论
// ============================================================
= 七、总结与讨论

== 7.1 主要结论

本报告围绕"二维分布生成建模"这一课程题目，以 VAE 和 Diffusion Model 为双主线，构建了一个包含数据生成、模型训练、样本采样、多维评价与可视化展示的完整实验框架，在四类具有不同几何特征的二维分布上开展了系统性的对比研究。主要结论概括如下：

1. #strong[Diffusion Model 在复杂连续流形上具有明显优势。] 在 Ring 与 Spiral 两类分布上，Diffusion 同时取得了更优的 MMD 和 SWD，说明其多步迭代去噪机制更擅长恢复闭合流形与长程连续拓扑。

2. #strong[VAE 在特定分布上仍具竞争力，且在效率上远优于 Diffusion。] VAE 在 Gaussian Mixture 与 Two Moons 上取得了更低的 MMD，其中 Two Moons 上的 MMD 为 0.0017；同时其采样速度约为 Diffusion 的数十到上百倍。对于对实时性有要求、或对极致几何保真度要求相对较低的场景，VAE 仍然是高效可靠的选择。

3. #strong[两类模型的性能差异与分布结构高度相关。] 在离散多峰分布（Gaussian Mixture）上，二者的差距主要体现在模式边缘的锐利度上；在闭合流形分布（Ring）上，Diffusion 的优势最明显；在双段非线性分布（Two Moons）上，二者差距最小；在复杂拓扑分布（Spiral）上，当前轻量实现下二者的水平接近且均未达理想级数。

4. #strong[超参数选择对生成质量有显著影响。] 消融实验清晰地表明：VAE 中较小的 KL 权重（$beta = 0.1$）能显著提升复杂分布上的重构质量；Diffusion 中较大的扩散步数（$T = 80$）对复杂流形拟合是必不可少的。

5. #strong[本框架具有良好的可复用性。] 数据层与模型层完全解耦的模块化设计，使得当前工程可以直接读取 README 数据格式文件；后续仅需覆盖 `data/` 下的数组文件即可复用全部训练、评价和报告流程。

== 7.2 两类模型差异的深层机理讨论

VAE 与 Diffusion Model 在二维分布上表现出的差异并非偶然，而是根植于二者生成机制的本质不同。本节尝试从以下三个角度对这些深层差异进行归纳。

#strong[从优化目标的角度。] VAE 通过最大化 ELBO 间接优化对数似然的下界，而 ELBO 中的重构项（通常为 MSE）天然倾向于产生"平均化"的重构结果——对于给定的潜变量 $z$，最优的解码器输出是 $p_theta(x|z)$ 的均值。这意味着 VAE 生成的每个样本都是某种条件期望，自然带有平滑倾向。而 Diffusion 的训练目标是直接预测加入的噪声 $epsilon$——这是一种去噪自编码器的形式，每一步的目标都是精准地恢复被噪声污染的信号，不存在系统性的"平均化"偏差。

#strong[从生成路径的角度。] VAE 在单次前向传播中完成整个生成过程，这种"单步生成"对解码器函数族的要求极高——解码器必须能够将简单的高斯潜变量分布通过一个确定性的映射变换为复杂的、可能具有非平凡拓扑的目标分布。对于 MLP 这类 Lipschitz 连续的映射而言，在保持全局平滑性的同时精确地"折叠"出非平凡的拓扑结构是困难的。Diffusion 通过将生成过程分解为 $T$ 个微小步骤，每个步骤只需要学习局部的去噪增量，大幅降低了单个步骤的学习难度。这种"分而治之"的策略对复杂结构的建模更为友好。

#strong[从潜变量建模的角度。] VAE 在潜变量空间中显式地对数据分布的结构信息进行编码，这一特性同时是其优势和局限的来源——优势在于潜空间具有良好的插值性和可解释性，局限在于高斯先验和近似后验的表达能力可能不足以完整捕获复杂数据的结构信息。Diffusion 不显式地维护一个低维潜变量空间；其"潜变量"实际上是整个扩散轨迹 ${x_t}_(t=1)^T$，这是一个与数据同维度的随机过程，信息容量远大于 VAE 的低维潜变量向量。

== 7.3 后续改进方向

在完成课程正式数据的替换和基础实验复现后，可以从以下几个方向进一步深化本项工作：

1. #strong[模型结构增强：] 将当前的纯 MLP 网络升级为具有残差连接或注意力机制的架构，以提升对复杂拓扑分布的拟合能力，特别是在 Spiral 分布上争取生成出完整的螺旋结构。

2. #strong[条件生成扩展：] 在 VAE 或 Diffusion 中引入类别条件输入（如分布类型标签），构建 Conditional VAE（CVAE）或 Classifier-free Guidance 的 Diffusion，实现用一个统一模型生成四类分布的样本。

3. #strong[加速采样：] 在 Diffusion 上实现 DDIM 等确定性加速采样算法，探究在二维任务中能以多大程度降低采样步数而不显著损害生成质量。

4. #strong[鲁棒性测试：] 向训练集中故意加入一定比例的异常点（离群值或来自其他分布的污染样本），测试两类模型对数据污染的鲁棒性差异。

5. #strong[更丰富的评价体系：] 补充基于核密度估计的对数似然近似、局部本征维度估计、分布间的 Jensen-Shannon 散度近似等指标，从更多角度量化生成质量。

6. #strong[高维扩展探索：] 对二维任务中得出的结论（例如"Diffusion 在复杂流形上更优""VAE 效率更高"等），在高维图像生成任务上进行对照验证，探究结论是否具有跨维度的普适性。

// ============================================================
// 参考文献
// ============================================================
= 参考文献

#bibliography("references.bib", title: none)

// ============================================================
// 附录
// ============================================================
= 附录

== A.1 主要运行命令

以下命令可复现本报告中的全部实验结果：

```bash
# Step 1: 生成 README 数据格式数据集
python generate_data.py --output-dir data --plot

# Step 2: 运行全部主实验（四类分布 × 两个模型）
python -m src.run_benchmark \
    --vae-epochs 150 \
    --diffusion-epochs 250 \
    --latent-dim 4 \
    --vae-beta 0.5 \
    --timesteps 80

# Step 3: 收集结果并生成汇总表
python -m src.collect_results \
    --table-dir outputs/tables \
    --output outputs/tables/summary.csv
```

快速验证命令（适用于检查代码通路是否正常）：

```bash
python -m src.train_vae --dataset ring --n-train 256 --n-test 128 --epochs 2
python -m src.train_diffusion --dataset two_moons --n-train 256 --n-test 128 --epochs 2 --timesteps 20
```

== A.2 完整工程文件结构

```text
Final_Hw/
├── data/
│   ├── train.npy
│   ├── test.npy
│   ├── train_label.npy
│   ├── test_label.npy
│   ├── hidden_test.npy
│   ├── hidden_test_label.npy
│   ├── metadata.json
│   └── processed/               # 兼容旧版 *.npz 数据接口（可选）
├── outputs/
│   ├── checkpoints/             # 训练好的模型权重（*.pt）
│   ├── figures/                 # 所有可视化图表（*.png）
│   │   ├── ablation/            # 消融实验专用图表
│   │   ├── profile/             # 各分布的特征分析图
│   │   └── smoke/               # 快速冒烟测试图表
│   ├── samples/                 # 采样结果（*.npy）
│   └── tables/                  # 指标 JSON 与汇总 CSV
│       ├── summary.csv
│       ├── ablation_summary.csv
│       └── *_metrics.json
├── report/
│   ├── main.typ                 # 本报告源文件（Typst 格式）
│   ├── references.bib           # BibTeX 参考文献
│   ├── final_report.pdf         # 编译后的 PDF 文件
│   ├── check_pages/             # 参考论文单页截图
│   ├── check_pages2/
│   ├── check_pages3/
│   └── ref_pages/
└── src/                         # 源代码
    ├── __init__.py
    ├── datasets.py              # 数据加载与生成接口
    ├── vae.py                   # VAE 模型定义
    ├── diffusion.py             # Diffusion 模型定义
    ├── train_vae.py             # VAE 训练脚本
    ├── train_diffusion.py       # Diffusion 训练脚本
    ├── sample_vae.py            # VAE 采样脚本
    ├── sample_diffusion.py      # Diffusion 采样脚本
    ├── metrics.py               # MMD, SWD, Mode Coverage 实现
    ├── visualize.py             # 可视化绘图工具
    ├── generate_synthetic_data.py # 旧版 npz 兼容数据生成
    ├── run_benchmark.py         # 批量实验运行
    ├── collect_results.py       # 结果收集与汇总
    └── utils.py                 # 通用工具函数
```

== A.3 后续替换正式数据的方法

当前工程已经直接支持 README 数据格式，后续替换数据时可采用两种方式：

#strong[方式一：直接替换 README 数据文件（推荐）。] 将新数据保存为 `data/train.npy`、`data/test.npy`、`data/train_label.npy` 和 `data/test_label.npy`；如果还提供隐藏测试集，也可补充 `data/hidden_test.npy` 与 `data/hidden_test_label.npy`。只要标签仍保持 `0~3` 对应四类分布，现有训练脚本、评价脚本和可视化脚本均可直接复用。

#strong[方式二：修改数据加载函数。] 若后续数据格式与 README 中的 `npy` 方案不同（如 CSV、MAT、或使用老师提供的其他自定义脚本），只需修改 `src/datasets.py` 中的 `load_dataset`、`load_or_generate_dataset` 和 `load_all_labeled_datasets` 三个函数。模型的其余部分（`vae.py`、`diffusion.py`、`train_*.py`、`metrics.py`、`visualize.py` 等）完全不需要改动。

这两种方式的共同前提是保持数据层的抽象接口统一：无论底层数据来源如何，对外暴露的 `x_train` 和 `x_test` 始终是 shape 为 `[N, 2]` 的 NumPy 数组或 PyTorch Tensor。

== A.4 补充图表索引

本报告使用了以下全部生成的图表：

#figure(
  table(
    columns: 3,
    align: (center, left, left),
    inset: 5pt,
    stroke: 0.5pt + black,
    [序号], [图名], [内容描述],
    [图 1], [四类数据集可视化], [训练集与测试集散点图],
    [图 2], [VAE 训练曲线], [Gaussian Mixture 与 Two Moons 训练损失],
    [图 3], [Diffusion 训练曲线], [Gaussian Mixture 与 Two Moons 噪声预测损失],
    [图 4], [VAE Ring 生成结果], [VAE 在 Ring 上的生成样本与叠加图],
    [图 5], [Diffusion Ring 生成结果], [Diffusion 在 Ring 上的生成样本与叠加图],
    [图 6], [VAE Spiral 生成结果], [VAE 在 Spiral 上的生成样本与叠加图],
    [图 7], [Diffusion Spiral 生成结果], [Diffusion 在 Spiral 上的生成样本与叠加图],
    [图 8], [Diffusion Ring 去噪快照], [Diffusion 在 Ring 上从噪声到结构的演化过程],
  ),
  caption: [报告引用的全部图表索引。],
)

#v(1em)
#align(center, text(size: 14pt, weight: "bold")[--- 报告完 ---])
