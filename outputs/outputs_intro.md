## outputs内容

- `checkpoints/`
  - 根目录下的 `vae_*.pt` 和 `diffusion_*.pt`：当前最终单次实验模型权重。
  - `multiseed/`：报告主实验使用的 VAE/DDPM 多随机种子 checkpoint。
  - `ablation/`、`ablation_*`：超参数消融实验的原始 checkpoint 证据。
  - `optional/`：条件生成、鲁棒性、增强快照、隐藏测试验证等拓展实验对应的 checkpoint。

- `figures/`
  - 根目录以及 `latent/`、`multiseed/`、`optional/` 下面，当前 PDF 仍在引用或对保留实验仍有支持作用的图片。

- `tables/`
  - 这些表格和日志已经基于现有 checkpoint 重新补回，不需要重新训练。
  - `summary.csv`：主实验单次结果汇总。
  - `multiseed/report_multiseed_summary.csv`：当前报告直接对应的 VAE/DDPM 多种子汇总表。
  - `multiseed/multiseed_summary.csv`：完整多种子汇总表，包含历史 `flow` 结果。
  - `optional/hidden_validation.csv`：基于已保存 checkpoint 重新生成的隐藏测试集验证结果。
