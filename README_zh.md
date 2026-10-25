# Alpha_Crops 科研代码

对应论文：**Earth embeddings improve cross-border crop mapping under limited local-label budgets**。本目录仅整理代码、分析方案和说明文档，不包含原始数据、特征表、模型、预测结果或制图数据。

数据来源为 EuroCrops v11 和 Google Earth Engine。下载、预处理和运行步骤见 [数据说明](docs/DATA_AND_PREPROCESSING.md) 与 [实验索引](docs/EXPERIMENTS.md)。

## 使用顺序

1. 运行 `python3.12 tools/verify_release.py` 检查代码完整性，不会训练模型。
2. 运行 `python3.12 tools/configure_workspace.py --dest /绝对路径/工作目录 --gee-project 你的项目ID`，生成独立工作副本并修改副本中的路径。
3. 按 README 安装环境、获取 EuroCrops、配置 GEE，再提取和合并 AE/S2 特征。
4. 使用 `build_dataset_v2.py` 构建配对数据，按实验索引完成空间划分、训练、分析与制图。

正式确认实验保留了原来的代码哈希、数据戳和环境版本校验。新路径、新数据或新环境需要重新登记实验；配置工具不会关闭校验，也不会自动启动训练。代码归档中的原始脚本保持不变。

## 哪些代码对应正文

正文主要实验位于 `project/label_expansion/confirm/`：每次一个国家作为 target，其他十国作为 source。采样指导实验在 `guidance_G.py`；多学习器检查在 `multilearner_G.py`。

新增预算分析在 `project/results/representation_budget_20261025_v1/`。虽然目录叫 results，此代码包只包含其中的分析代码和方案。预算主结论使用全部 11 国，原来的 9 国分组保留为敏感性分析；原始确认性分析的定义不变。

当前论文制图代码在 `project/Manuscript/Paper_Alpha_Earth/0019_eleven_country_release_20261025/figure_sources/`，其调用的旧绘图模块一并保留。图表输入需要通过本地分析重新生成。

单源实验、其他试验和开发检查作为历史代码保留，未重新加入正文。`local_code/verify/` 中既有正文指标所依赖的模块，也有开发阶段的核验脚本，具体对应关系见实验索引。

## 本次整理的边界

S2 的 EVI/SAVI 保持论文实际使用的实现，不在整理代码时改算法或重跑实验。论文中已说明其与标准反射率公式的差别。

此目录可作为 GitHub 仓库内容上传；不需要再上传实验数据。尚未替作者选择代码开放许可，第三方工具的原许可证已经保留。论文 PDF 和中英文 cover letter 单独交付，不放在代码仓库中。
