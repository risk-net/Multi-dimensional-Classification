# AI风险事件的多维分类


## 概述  
本部分聚焦于AI风险事件的多维分类，核心目标是构建统一的分类体系（RiskNet Taxonomy）和基准数据集，并通过代码实现模型评估。重点对比基于提示词的推理与微调大语言模型（LLMs）在多维分类任务中的性能，为AI风险事件的结构化分析提供技术支撑。


## 项目结构与文件说明  
```
Multi-dimensional-Classification/
├── README.md                    # 英文版说明
├── README_zh-CN.md             # 中文版说明（本文档）
├── data/
│   ├── classification_result/   # 模型分类结果（JSON格式）
│   ├── evaluation_result/       # 评估报告（TXT格式）
│   └── fine-tuning-data/        # SFT训练数据集及配置文件
│       ├── config/              # 微调配置文件
│       ├── train_set.json       # 训练集
│       ├── test_set.json        # 验证集
│       └── data_set_builder.ipynb  # 数据集构建脚本
└── src/
    └── evalution.py            # 评估核心脚本
```


## 🔧 安装与设置  

### 前提条件  
- Python 3.7及以上版本  
- 所需依赖包（详见requirements.txt）  


### 安装步骤  
```bash
# 克隆仓库
git clone <仓库地址>
cd Multi-dimensional-Classification

# 安装依赖
pip install -r requirements.txt
```


## 📊 数据结构  

### 1. 分类结果（`data/classification_result/`）  
该文件夹存储不同模型的分类结果（JSON格式），每个文件对应特定模型在测试集上的预测输出，具体说明如下：  

- **文件名格式**：`<模型名称>.json`（例如 `qwen3-14B-sft.json`、`kimi-k2-prompt.json`）  
- **内容**：字典列表，每个字典包含两组键值对：  
  - `label`：真实标注，包括单标签任务（实体、意图、时间、欧盟AI法案风险等级）和多标签任务（领域分类及其主/子领域）  
  - `predict`：模型预测结果，格式与 `label` 一致  
- **作用**：作为评估脚本的输入，用于计算各项性能指标  
- **可用模型**：  
  - `qwen3-14B.json` - Qwen3 14B基础模型  
  - `qwen3-14B-sft.json` - Qwen3 14B微调模型  
  - `qwen3-32B.json` - Qwen3 32B基础模型  
  - `qwen3-32B-sft.json` - Qwen3 32B微调模型  
  - `kimi-k2-prompt.json` - Kimi K2基于提示词的模型  


### 2. 评估结果（`data/evaluation_result/`）  
该文件夹存储评估脚本生成的评估报告（TXT格式），每个文件对应特定模型的性能指标，包括：  

- **文件名格式**：`<模型名称>.txt`（与 `classification_result` 中的模型名称保持一致）  
- **内容**：  
  - 单标签任务指标（实体、意图、时间、风险等级的准确率、精确率、召回率、F1分数）  
  - 多标签任务指标（主领域和子领域的汉明损失、微平均/宏平均准确率、精确率、召回率、F1分数）  


### 3. 微调数据（`data/fine-tuning-data/`）  
包含用于模型微调的数据集，具体说明如下：  

- **数据来源**：每个AI风险事件的前5名代表性新闻报道  
- **处理方式**：将标题和摘要聚合为简洁的事件摘要  
- **拆分比例**：8:2（训练:验证）  
- **格式**：用于监督学习的输入-输出对JSON格式  
- **作用**：为LLM微调提供高质量的监督数据，帮助模型学习风险相关模式，提升分类精度  


## 🔧 模型微调过程  
本项目使用LLamaFactory 0.9.4进行模型微调（该框架支持多种微调方法和模型类型），具体步骤如下：  

1. **LLamaFactory准备**  
   ```bash
   # 克隆LLamaFactory仓库
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   git checkout v0.9.4  # 切换到0.9.4版本

   # 安装依赖
   pip install -r requirements.txt
   ```  

2. **微调数据准备**  
   微调数据需转换为LLamaFactory支持的格式，示例如下：  
   ```json
   [
       {
           "instruction": "prompt内容",
           "input": "",
           "output": "response内容"
       },
       ...
   ]
   ```  
   数据集构建过程可参考 `data/fine-tuning-data/data_set_builder.ipynb`。  
   完成后，将 `data/fine-tuning-data/train_set.json` 放入LLamaFactory的 `data` 文件夹，并在 `dataset-info` 文件夹内添加对应配置文件。  

3. **微调配置**  
   创建配置文件 `data/fine-tuning-data/config/qwen3_lora_sft.yaml`，并使用该配置文件执行微调。  

执行微调：使用以下命令执行微调：
   ```bash
    lamafactory - cli train data/fine-tuning-data/config/qwen3_lora_sft.yaml
   ``` 
bash
在整个过程中，若遇到问题，可查阅LLamaFactory 官方文档或在相关技术社区寻求帮助。

## 🚀 评估使用方法  

### 运行评估  
```python
from src.evalution import evaluate_model

# 评估特定模型
evaluate_model("data/classification_result/qwen3-32B-sft.json", "qwen3-32B-sft")
```  


### 评估脚本功能  
评估脚本（`src/evalution.py`）提供以下核心功能：  

- **数据验证**：检查标签和预测的有效性  
- **全面指标**：计算各种性能指标（如准确率、F1分数等）  
- **多任务支持**：同时处理单标签和多标签分类任务  
- **详细报告**：生成包含各项指标的全面评估报告