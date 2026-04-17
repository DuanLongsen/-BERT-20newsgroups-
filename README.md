# 基于 BERT 的文本二分类实践 (20newsgroups)

## 📖 项目简介
本项目基于 PyTorch 和 HuggingFace 的 `transformers` 库，使用预训练语言模型 **BERT (bert-base-uncased)** 对 `20newsgroups` 数据集进行文本二分类微调（Fine-tuning）。
分类任务目标：准确区分文本属于 `alt.atheism` (无神论) 还是 `soc.religion.christian` (基督教)。

## 🚀 核心功能与特色
- **自动数据拉取与预处理**：利用 `sklearn` 自动下载 20newsgroups 数据并进行标签编码。
- **严谨的数据集划分**：采用 8:2 比例划分训练集与验证集，并使用分层抽样 (`stratify`) 保证正负样本比例均衡。
- **自定义 PyTorch Dataset**：将纯文本自动化转化为 BERT 所需的 `input_ids` 和 `attention_mask`。
- **防止过拟合策略**：包含完整的验证集评估流程，并引入了梯度裁剪 (`clip_grad_norm_`) 与学习率预热 (`warmup`) 机制。

## 🛠️ 环境配置
建议使用 Python 3.11 及以上版本。可以通过以下命令快速安装项目所需依赖：

```bash
pip install -r requirements.txt