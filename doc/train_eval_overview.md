# Train / Eval 全量说明（零基础版）

这份文档是给“完全没做过大模型训练/评测”的同学看的，目标是回答 3 个问题：

1. `train/` 到底在做什么，数据怎么喂给模型？
2. LoRA、DeepSpeed 是什么，为什么这里要用它们？
3. `eval/` 怎么评估模型，分数是怎么来的？

如果你只想先建立整体认知，先看第 1 节和第 2 节。

---

## 1. 一句话先看懂全流程

这个项目里的这两部分可以理解为：

- `train/`：把一个“通用多模态模型”变成“更懂你任务数据”的模型
- `eval/`：拿一批标准题（图像+问题+标准答案）测模型效果

输入输出关系：

- 训练输入：基座模型 + 训练数据描述文件（`meta_path`）+ 训练超参数
- 训练输出：LoRA 权重、checkpoint、训练日志

- 评测输入：待评估模型 + 评测 JSON + 图像目录 + DeepSeek API Key
- 评测输出：每条样本的预测记录 + 汇总指标（准确率/BERTScore）

---

## 2. 先补关键术语（看完这段再看脚本会轻松很多）

### 2.1 什么是“基座模型”？

基座模型就是没针对你任务继续训练前的原始模型，比如 InternVL3-2B / 8B。  
它“能力广”，但不一定在你的医学问答任务上最优。

### 2.2 什么是 LoRA？

LoRA（Low-Rank Adaptation）是“低秩增量训练”方法。核心思想：

- 不直接改大模型全部参数（太重、太贵）
- 在部分线性层旁边加一个小的可训练增量
- 训练时只学这个“小增量”

可粗略理解为：

- 原来参数是 `W`
- 训练后不是改 `W` 本体，而是学一个小改动 `ΔW = A * B`
- 推理时使用 `W + ΔW`

优点：

- 显存占用低
- 训练更快
- 便于切换不同任务的 LoRA 适配器

本项目脚本里 `--use_llm_lora 16` 表示 LoRA 的 rank（低秩维度）设为 16。

### 2.3 什么是 DeepSpeed？为什么配 ZeRO Stage 1？

DeepSpeed 是分布式训练加速框架。  
ZeRO（Zero Redundancy Optimizer）是它的重要内存优化策略。

这里用的是 `zero_stage1_config.json` 里的 **ZeRO Stage 1**，主要做：

- 将优化器状态分片到多卡，减少每张卡的内存压力
- 结合混合精度（fp16/bf16 auto）进一步省显存

通俗理解：  
你有 8 张卡，原来每张卡都要保存一份很大的优化器状态；ZeRO 可以把这份“大包袱”拆开分摊。

### 2.4 什么是“动态分辨率 / patch”？

图像不会直接原尺寸送进模型，而是会被切成视觉块（patch）处理。  
动态分辨率意味着：

- 按图像宽高比，选择更合适的切分网格
- 避免所有图都强行压缩成同一种形状造成信息损失

`eval/eval.py` 里有 `dynamic_preprocess`，会根据长宽比决定分块方式，并可附加一个缩略图块。

### 2.5 什么是 BERTScore？

BERTScore 是文本相似度指标，不只比字面是否完全一样，而是比语义向量接近程度。  
本项目里用 `roberta-large` 计算 F1 分数，作为开放问答质量指标之一。

---

## 3. `train/` 目录逐文件详解

`train/` 目录当前有 3 个文件：

1. `train/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh`
2. `train/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh`
3. `train/zero_stage1_config.json`

### 3.1 `internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh` 在做什么？

这是 2B 模型的训练启动脚本，本质是调用：

- `torchrun ... internvl/train/internvl_chat_finetune.py ...`

脚本关键变量：

- `GPUS`：使用多少 GPU（默认 8）
- `BATCH_SIZE`：全局 batch size（默认 64）
- `PER_DEVICE_BATCH_SIZE`：每张卡一次前向的样本数（默认 4）
- `GRADIENT_ACC`：梯度累积步数，计算式：
  - `BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS`

例子（默认值）：

- `64 / 4 / 8 = 2`
- 即每卡每步 4 条样本，累计 2 步再更新一次参数，总体等效 batch 为 64

#### 训练参数怎么理解（重点）

- `--model_name_or_path ""`
  - 这里必须填你的基座模型路径（本地路径或 Hugging Face 路径）
- `--meta_path "./shell/data/chiron-o1-2b.json"`
  - 训练数据“索引文件”的路径，告诉训练器去哪读图像和文本
- `--freeze_llm True --freeze_mlp True --freeze_backbone True`
  - 冻结大部分原始参数，只训练 LoRA 相关参数
- `--use_llm_lora 16`
  - 开启并设置 LoRA rank
- `--bf16 True`
  - 使用 bfloat16（通常对 A100/H100 很友好）
- `--learning_rate 4e-5 --weight_decay 0.01`
  - 学习率与权重衰减
- `--num_train_epochs 1`
  - 训练轮数 1（是否够要看你的数据规模）
- `--max_seq_length 8192`
  - 文本最大长度
- `--deepspeed "zero_stage1_config.json"`
  - 启用 DeepSpeed 配置

#### 输出是什么？

- checkpoint 保存在 `OUTPUT_DIR`
- 日志会 tee 到 `${OUTPUT_DIR}/training_log.txt`

注意：脚本里 `OUTPUT_DIR=''` 需要你先改成真实目录。

### 3.2 `internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh` 在做什么？

和 2B 脚本同构，只是默认资源和部分参数不同：

- 默认 `GPUS=2`、`BATCH_SIZE=16`、`PER_DEVICE_BATCH_SIZE=4`
- `meta_path` 指向 `chiron-o1-8b.json`
- `weight_decay` 是 `0.05`

适用场景：

- 你要微调 8B 基座版本
- 显存预算通常要更高

### 3.3 `zero_stage1_config.json` 每个区块在干什么？

#### `zero_optimization`

- `"stage": 1`：只做优化器状态分片（较稳，复杂度较低）
- `allgather/reduce_*`：通信相关参数，影响吞吐和稳定性
- `contiguous_gradients: true`：优化梯度内存布局

#### `fp16` 与 `bf16`

- `"enabled": "auto"`：让框架按硬件和配置自动选
- 在训练脚本已经 `--bf16 True` 的情况下，通常会走 bf16 路线

#### `optimizer`

- 类型是 `AdamW`
- `lr`、`weight_decay` 用 `"auto"`：实际值来自训练脚本参数

#### `train_batch_size` 等 `"auto"`

- 由 `per_device_train_batch_size`、GPU 数、梯度累积等共同推导

---

## 4. 训练数据“原理”要点（给完全新手）

> 这里基于当前脚本与 InternVL 常规做法解释。  
> 你项目里的 `meta_path` 实际字段，以你准备的数据文件为准。

训练并不是“只喂问题和答案”这么简单，而是要给模型一个清晰的学习格式。  
在这个项目中，你可以把每条样本抽象成：

1. 图像路径（可以 1 张，也可以多张）
2. 用户问题（question）
3. 目标输出（通常包含推理过程 + 最终答案格式）

为什么强调“最终答案格式”？  
因为评测脚本在解析输出时依赖字符串：

- `### The final answer is:`

如果训练时完全不约束输出格式，模型在评测时可能被跳过（脚本会 `continue`），导致分数失真。

实践建议：

1. 训练和评测用同一种回答模板
2. 样本里多图时保持图像顺序稳定
3. `question` 尽量保持任务风格一致（医疗问答最好统一语气）
4. 先用几十条样本小跑通，再放大规模

---

## 5. `eval/` 目录逐文件详解

`eval/` 目录当前有：

1. `eval/eval.py`
2. `eval/eval_data.json`
3. `eval/images/`（样本图像）
4. `eval/infer_img.png`（示例图片）

### 5.1 `eval/eval.py` 运行流程（逐步）

#### 第 1 步：加载模型和 tokenizer

- `AutoModel.from_pretrained(model_path, ...)`
- `AutoTokenizer.from_pretrained(model_path, ...)`

这里 `trust_remote_code=True` 说明模型仓库里有自定义实现代码。

#### 第 2 步：读取评测集 JSON

- 读取 `--vqa_json_path`
- 每条样本至少要有：`img_name`、`question`、`answer`

#### 第 3 步：图像预处理（动态切块）

- `load_image` -> `dynamic_preprocess`
- 会根据宽高比决定切块网格
- 多图样本会把每张图的 patch 拼接起来喂模型

#### 第 4 步：模型生成答案

- 调用 `model.chat(tokenizer, pixel_values, prompt, ...)`
- 输出文本里必须带：
  - `### The final answer is:`

#### 第 5 步：抽取最终答案

- 前半段当作 `reason_answer`
- 标记后的文本当作 `pred_answer`

如果没有这个标记，样本会被跳过，不计入指标。

#### 第 6 步：计算两类指标

1. 闭集准确率（CLOSED）
   - 把 `pred_answer` 与 `ground_truth` 发给 DeepSeek 判定“语义是否等价”
   - 输出 Yes/No，映射到正确/错误
2. 开放文本分（OPEN）
   - 用 `bert_score.score(...)` 算 F1

#### 第 7 步：写结果 JSON

- 每条样本会写一条详细记录
- 最后附加一个汇总对象：
  - `CLOSED Questions Acc`
  - `OPEN   Questions BERTScore-F1`

### 5.2 `eval/eval_data.json` 字段解释

当前文件是一个数组，每条样本包含：

- `id`：样本编号
- `img_name`：图像路径（字符串或字符串列表）
- `question`：问题文本（常含 `<image>` 占位）
- `reasoning`：参考推理文本（通常不直接给评测脚本使用）
- `answer`：标准答案
- `answer_type`：题型标签（例如 OPEN）

### 5.3 `eval/images/` 如何对应 `img_name`

脚本用：

- `os.path.join(image_dir, image_name)`

如果命令里 `--image_dir eval`，而 `img_name` 是 `images/197271/ct_group1/0.jpeg`，  
最终路径就是 `eval/images/197271/ct_group1/0.jpeg`。

---

## 6. 从零开始跑通：一步一步 checklist

### 6.1 跑训练前

1. 准备好 InternVL 训练环境（见官方文档）
2. 把 `train/*.sh` 里的 `OUTPUT_DIR` 改成真实目录
3. 填好 `--model_name_or_path`
4. 确认 `meta_path` 数据文件存在且字段格式正确
5. 根据显存调整 `GPUS`、`PER_DEVICE_BATCH_SIZE`、`BATCH_SIZE`

### 6.2 跑评测前

1. 安装 `bert_score`、`transformers`、`torch` 等依赖
2. 准备可加载的模型目录（`--model_path`）
3. 准备 DeepSeek API Key（脚本里的 `--api_key`）
4. 确认评测集与图像路径能对上

---

## 7. 常见坑（非常重要）

1. `OUTPUT_DIR` 为空
   - 结果：日志和权重路径异常，训练不可控
2. `--model_name_or_path` 没填
   - 结果：训练入口无法加载模型
3. 回答格式不含 `### The final answer is:`
   - 结果：评测样本被跳过，分数虚高或虚低
4. 多图路径不一致
   - 结果：`Image.open` 报错或读错图
5. batch 配置不合理
   - 结果：OOM 或训练速度很慢
6. 只看 CLOSED 不看 OPEN（或反过来）
   - 结果：误判模型真实质量

---

## 8. 最小命令示例

### 8.1 训练（示例）

```bash
cd train
GPUS=2 PER_DEVICE_BATCH_SIZE=2 BATCH_SIZE=16 bash internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh
```

### 8.2 评测（示例）

```bash
python eval/eval.py \
  --vqa_json_path eval/eval_data.json \
  --image_dir eval \
  --model_path /path/to/your/model \
  --output_path eval/results.json \
  --api_key YOUR_DEEPSEEK_API_KEY
```

## 9. 给新同学的心智模型（最终总结）

你可以把这套系统记成一句话：

- `train/` 用 LoRA 在 InternVL 上做“轻量定制学习”
- DeepSpeed 负责“把训练跑得动且更省显存”
- `eval/` 用“LLM 语义判定 + BERTScore”双轨衡量答案质量

如果你是第一次接手，先做一件事就行：  
先用极小数据（比如 20~50 条）完整跑通训练+评测闭环，再扩大规模做正式实验。

## 10. Finetune 框架怎么选：InternVL 还是 LLaMA-Factory？

这部分回答一个实际问题：  
“我要做 finetune，到底该用项目现在的 InternVL 训练链路，还是切到 LLaMA-Factory？”

### 10.1 先给结论

默认推荐：

1. 你的目标是复现或增强当前项目效果 -> 选 `InternVL` 现有链路
2. 你的目标是统一管理多模型训练平台 -> 再考虑 `LLaMA-Factory`

### 10.2 为什么当前项目默认选 InternVL 链路

当前仓库训练入口是：

- `train/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh`
- `train/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh`
- 实际调用 `internvl/train/internvl_chat_finetune.py`

注意：`internvl/train/internvl_chat_finetune.py` 不在本仓库里。  
它属于外部 `InternVL` 训练代码（见 `README.md` 的 training 部分，需要先 clone InternVL 后在其环境中运行）。

这条链路的优势是“对齐性高”：

1. 模型实现对齐
   - 直接按 InternVL 官方方式处理多模态输入与训练细节，行为最可控
2. 数据/模板对齐
   - 与当前评测约束一致（例如输出格式 `### The final answer is:`）
3. 参数语义对齐
   - 动态图像切块、冻结策略、LoRA 开关等参数含义直接对应官方实现
4. 风险最低
   - 不需要先做框架迁移，能更快进入“训练 -> 评测 -> 迭代”

### 10.3 什么时候应该考虑 LLaMA-Factory

满足下面多数条件时再考虑迁移：

1. 你要长期管理很多不同模型，而不是只训 InternVL
2. 你更在意统一实验平台（数据格式、日志、配置风格统一）
3. 你有时间做迁移对齐（数据转换、模板转换、指标回归）
4. 你接受短期内“能跑但效果不一定等价”的风险

### 10.4 决策表（实操版）

1. 如果你是“效果优先、风险最小”
   - 选：`InternVL`
   - 理由：最接近当前项目已验证链路
2. 如果你是“平台统一、长期维护优先”
   - 选：`LLaMA-Factory`
   - 理由：便于跨模型复用训练流程
3. 如果你“既要效果又要统一平台”
   - 策略：先 `InternVL` 出基线，再小规模迁移到 `LLaMA-Factory` 做 A/B

### 10.5 一套可执行的选择流程（5 步）

1. 明确本次目标
   - 复现现有效果，还是建设统一训练平台
2. 检查是否依赖 InternVL 特有细节
   - 多图输入、动态 patch、`model.chat` 行为、输出格式约束
3. 评估迁移预算
   - 是否有时间做数据与 prompt 对齐、跑回归评测
4. 选框架
   - 复现优先 -> InternVL；平台优先 -> LLaMA-Factory
5. 固化验证标准
   - 用同一评测脚本比较 `CLOSED Questions Acc` 与 `OPEN BERTScore-F1`

### 10.6 如果你决定迁移到 LLaMA-Factory，最低限度要验证什么

1. 数据等价
   - 样本数、图像路径、问答文本一致
2. 模板等价
   - 训练与推理仍能稳定产出 `### The final answer is:`
3. 训练超参近似
   - batch、学习率、epoch、LoRA rank 尽量一致
4. 评测脚本不变
   - 仍用 `eval/eval.py` 跑同一数据
5. 指标不退化
   - `CLOSED` 与 `OPEN` 至少不明显下降，再考虑切换主链路
