# Chiron-o1 项目 End2End 超详细代码索引解析

本文目标：把项目从“启动命令”到“结果产出”完整拆开，做到每个关键环节都能定位到具体代码块。

## 0. 全局入口与主链路

主链路（数据构建阶段，MICS 搜索）：

1. `src/run.py` 解析参数并加载数据
2. `src/model.py` 加载本地 intern 模型
3. `src/mics.py` 对每个样本执行多 mentor/多 intern 协同搜索
4. `src/utils.py` 完成图片定位、图片编码、远程 API 调用、评估打分辅助
5. `src/step.py` 维护推理树节点
6. 输出成功/失败 jsonl

补充链路：

1. `infer.py` 是单模型推理示例（不走 MICS）
2. `eval/eval.py` 是离线评估链路
3. `train/*.sh` 是训练启动脚本（依赖外部 InternVL 代码）

---

## 1. 启动阶段（CLI -> 运行上下文）

### 1.1 参数定义与主函数入口

- 文件：`src/run.py:90`
- 代码块：`if __name__ == "__main__":`
- 作用：定义所有 CLI 参数，完成参数校验，最后调用 `mics_start(args)`。

关键子块索引：

- 参数定义：`src/run.py:91`
- mentor/intern 最小数量校验：`src/run.py:124`
- intern 名称到路径参数映射校验：`src/run.py:117`
- 入口调用：`src/run.py:147`

输入：命令行参数
输出：`args` 对象
失败路径：参数非法时 `exit(1)`（`src/run.py:140`）

### 1.2 主流程函数 mics_start

- 文件：`src/run.py:10`
- 代码块：`def mics_start(args):`

关键子块索引：

1. 读取数据（jsonl/json）：`src/run.py:13`
2. 构建输出目录：`src/run.py:37`
3. 打开成功/失败输出文件：`src/run.py:46`
4. 分片处理（chunk）：`src/run.py:47`
5. 初始化模型：`src/run.py:55`
6. 初始化搜索器：`src/run.py:64`
7. 主循环逐样本：`src/run.py:69`
8. 单样本调用：`search_process.search(...)`（`src/run.py:73`）

输入：`args + 数据文件`
输出：`output_path` 和 `*_failed.jsonl`

---

## 2. 数据准备阶段（样本结构 -> 图片可用）

### 2.1 数据格式要求

- 示例：`src/demo_data/demo.jsonl:1`
- 核心字段：
  - `rid`
  - `messages[0].content`（问题）
  - `messages[1].content`（标准答案）
  - `images`（图片相对路径列表）
  - `presentation/age_label/gender_label/caption`

### 2.2 读取与分片

- `read_jsonl`：`src/utils.py:53`
- `split_list`：`src/utils.py:45`
- `get_chunk`：`src/utils.py:49`

### 2.3 图片路径定位与编码

- 图片定位：`src/utils.py:59` (`locate_img`)
  - 优先 `args.image_dir_path + 相对路径`：`src/utils.py:63`
  - 其次直接路径：`src/utils.py:67`
- 图片编码：`src/utils.py:30` (`encode_image`)
  - 先按最大边缩放：`resize_image_if_needed`（`src/utils.py:11`）
  - 再转 base64：`encode_to_base64`（`src/utils.py:27`）

注意：`locate_img` 的报错信息引用了 `d['image']`，但数据字段是 `images`（`src/utils.py:71`），这里是潜在 bug 点。

---

## 3. 模型初始化阶段（本地 intern 模型）

### 3.1 统一加载入口

- 文件：`src/model.py:38`
- 代码块：`def init_model(args):`

加载结果结构：

```python
model_set = {
  "qwen25_vl_7b": {"model": ..., "processor": ...},
  "qwen2_vl_7b": {"model": ..., "processor": ...},
  "internvl3_8b": {"model": ..., "processor": ...}
}
```

### 3.2 各模型加载块索引

- Qwen2.5-VL-7B：`src/model.py:59`
- Qwen2-VL-7B：`src/model.py:75`
- InternVL3-8B：`src/model.py:92`
- 空模型集告警：`src/model.py:108`

### 3.3 InternVL 多卡切分策略

- 文件：`src/model.py:12`
- 代码块：`split_model(model_path)`
- 关键逻辑：
  - 读 `num_hidden_layers`：`src/model.py:16`
  - 第一张卡预留给视觉分支（按半卡计算）：`src/model.py:17`
  - 构建 `device_map`：`src/model.py:24`

---

## 4. 搜索器初始化阶段（MICS 运行态）

### 4.1 搜索器构造

- 文件：`src/mics.py:15`
- 代码块：`MentorInternSearch.__init__`

关键索引：

- 根节点初始化：`self.root = Step(prefix_steps="")`（`src/mics.py:22`）
- 模型名单保存：`src/mics.py:24`
- 四个 OpenAI 兼容 client 初始化：`src/mics.py:29`

### 4.2 模型转发路由

- 文件：`src/mics.py:36`
- 代码块：`_call_model_forward(...)`

路由规则（靠模型名字符串匹配）：

- `'gpt'` -> `gpt_forward`（`src/mics.py:39`）
- `'gemini'` -> `gpt_forward(client4)`（`src/mics.py:43`）
- `'72'` -> `qwenplus_forward`（`src/mics.py:45`）
- `'qwen'` -> 本地 `qwenvl_forward`（`src/mics.py:48`）
- `'internvl'` -> 本地 `internvl_forward`（`src/mics.py:57`）

---

## 5. 单样本搜索阶段（核心 End2End）

### 5.1 search 主入口

- 文件：`src/mics.py:196`
- 代码块：`def search(self, data, model_set, search_file, failed_search_file):`

关键子块索引：

1. 绑定 `model_set`：`src/mics.py:198`
2. 提取 question/gt_answer：`src/mics.py:201`
3. 结构化 case_info：`process_case_info`（`src/mics.py:204`, `src/utils.py:186`）
4. 定位图片并 base64 编码：`src/mics.py:209`
5. 初始化根推理：`src/mics.py:228`
6. 初始化评分容器：`src/mics.py:232`
7. 深度循环：`src/mics.py:237`

### 5.2 每层深度的 mentor 生成

- 位置：`src/mics.py:244`

分两种情况：

1. 首层（root）：`src/mics.py:246`
  - 调 `_generate_next_step_with_mentor`：`src/mics.py:247`
  - 提取前两步：`extract_first_two_steps`（`src/mics.py:253`, `src/utils.py:301`）
2. 非首层：`src/mics.py:255`
  - 若当前路径 mentor 与本 mentor 一致，尝试复用已有 reasoning chain：`src/mics.py:256`
  - 否则重新调用 mentor 续写，并提取第一步：`extract_first_step`（`src/mics.py:271`, `src/utils.py:282`）

### 5.3 mentor 生成细节

- 文件：`src/mics.py:74`
- 代码块：`_generate_next_step_with_mentor(...)`

关键索引：

- 读取已有 reasoning 前缀：`src/mics.py:80`
- 构建 `REASONING_PROMPT`：`src/mics.py:84` + `src/prompt.py:11`
- 远程/本地模型调用：`src/mics.py:100`
- 返回 `(reasoning_prefix, response)`：`src/mics.py:111`

### 5.4 intern 评估与打分细节

- 文件：`src/mics.py:116`
- 代码块：`_evaluate_step_with_interns(...)`

评估循环结构：

1. 对每个 intern 模型：`src/mics.py:129`
2. 每个 intern 跑两个温度：`src/mics.py:148`
3. 构建 `EVALUATE_PROMPT`：`src/mics.py:138` + `src/prompt.py:34`
4. 调 intern 推理：`src/mics.py:152`
5. 抽取 `### The final answer is:`：`src/mics.py:164`
6. 构建 `JUDGE_PROMPT`：`src/mics.py:173` + `src/prompt.py:1`
7. 调 evaluator（DeepSeek）：`src/mics.py:180` + `src/utils.py:130`
8. Yes/No 转 1/-1：`src/mics.py:183` + `src/utils.py:145`
9. 分数归一化：`score = correct_count / (intern_num*2)`（`src/mics.py:190`）

### 5.5 每层决策与提前终止

- 添加子节点：`current_step.add_child_step(...)`（`src/mics.py:282`, `src/step.py:33`）
- 满分提前结束：`src/mics.py:288`
  - 多个满分 mentor 时按历史分选择：`select_best_mentor`（`src/mics.py:293`, `src/utils.py:152`）
  - 结果 `search_id='1'`：`src/mics.py:306`
- 全零提前失败：`src/mics.py:315`
- 否则选择下一步：`select_next_step`（`src/mics.py:325`, `src/utils.py:216`）

### 5.6 达到最大深度后的收尾

- 位置：`src/mics.py:334`
- 逻辑：
  - 取最终 mentor：`src/mics.py:338`
  - 如果 `reasoning_chains` 有该 mentor，就写成功 `search_id='0'`（`src/mics.py:343`）
  - 否则写失败日志（`src/mics.py:361`）

---

## 6. Step 树结构（推理路径载体）

- 文件：`src/step.py:1`

关键代码块：

1. 初始化节点并计算深度：`src/step.py:2`
2. 拼接完整文本 `self.text`：`src/step.py:21`
3. 判断终端节点：`src/step.py:27`
4. 添加子节点：`src/step.py:33`
5. 获取完整推理文本：`src/step.py:45`
6. 回溯整条路径：`src/step.py:51`

---

## 7. Prompt 系统（生成/评估/裁判）

- 文件：`src/prompt.py:1`

三类 prompt：

1. `JUDGE_PROMPT`（Yes/No 裁判模板）：`src/prompt.py:1`
2. `REASONING_PROMPT`（mentor 生成模板）：`src/prompt.py:11`
3. `EVALUATE_PROMPT`（intern 回答模板）：`src/prompt.py:34`

其在主流程中的调用位置：

- mentor 生成：`src/mics.py:84`
- intern 评估：`src/mics.py:138`
- evaluator 判定：`src/mics.py:173`

---

## 8. 本地多模态 forward 细节

### 8.1 Qwen-VL forward

- 文件：`src/qwenvl_forward.py:3`

关键块：

1. 构造 chat 消息体：`src/qwenvl_forward.py:4`
2. 将每张图片追加到 user content：`src/qwenvl_forward.py:15`
3. 模板化 + 多模态张量化：`src/qwenvl_forward.py:23`
4. `model.generate`：`src/qwenvl_forward.py:33`
5. 裁剪输入 token 并 decode：`src/qwenvl_forward.py:34`

### 8.2 InternVL forward

- 文件：`src/internvl_forward.py:83`

关键块：

1. 单图预处理到 patch tensor：`load_image`（`src/internvl_forward.py:73`）
2. 多图累计 `all_pixel_values + num_patches_list`：`src/internvl_forward.py:85`
3. 单图分支 `model.chat(...)`：`src/internvl_forward.py:93`
4. 多图分支拼接后 `model.chat(..., num_patches_list=...)`：`src/internvl_forward.py:98`

图像切块算法索引：

- `dynamic_preprocess`：`src/internvl_forward.py:35`
- 宽高比分桶选择：`find_closest_aspect_ratio`（`src/internvl_forward.py:20`）

---

## 9. 远程 API forward 与辅助函数

- 文件：`src/utils.py`

关键块索引：

1. OpenAI 兼容多模态调用：`gpt_forward`（`src/utils.py:76`）
2. Qwen API 调用：`qwenplus_forward`（`src/utils.py:103`）
3. evaluator 调用：`ds_forward`（`src/utils.py:130`）
4. 裁判输出判定：`get_correctness`（`src/utils.py:145`）
5. mentor 决策：`select_best_mentor`（`src/utils.py:152`）
6. 下一步选择：`select_next_step`（`src/utils.py:216`）
7. 图像引用文本替换：`replace_image_references`（`src/utils.py:247`）
8. 步骤提取：`extract_first_step`（`src/utils.py:282`）
9. 前两步提取：`extract_first_two_steps`（`src/utils.py:301`）

---

## 10. 输出结果结构（成功/失败）

### 10.1 成功输出字段（提前满分）

- 写入位置：`src/mics.py:297`
- 关键字段：
  - `rid/images/question/gt_answer`
  - `reasoning`
  - `scores`（每个 mentor 的每层分数）
  - `final_depth`
  - `generated_by`
  - `search_id='1'`

### 10.2 成功输出字段（达到 max_depth）

- 写入位置：`src/mics.py:345`
- 区别：`search_id='0'`

### 10.3 失败输出

失败来源与索引：

1. 图片编码失败：`src/mics.py:211`
2. 图片未找到：`src/mics.py:218`
3. 全 mentor 得分为 0：`src/mics.py:315`
4. 无有效推理路径：`src/mics.py:361`
5. 运行时异常兜底：`src/run.py:74`

---

## 11. 离线评测链路（eval）

- 文件：`eval/eval.py:95`
- 入口：`evaluate_model(...)`

完整流程索引：

1. 初始化评估模型与 tokenizer：`eval/eval.py:97`
2. 加载评测集：`eval/eval.py:106`
3. 遍历样本：`eval/eval.py:119`
4. 多图拼接并推理：`eval/eval.py:130`
5. 提取 final answer：`eval/eval.py:148`
6. DeepSeek 判对错：`eval/eval.py:159`
7. BERTScore：`eval/eval.py:167`
8. 逐条写 output：`eval/eval.py:192`
9. 汇总指标写入文件末尾：`eval/eval.py:198`

指标定义：

- 闭集准确率：`closed_correct / closed_total`
- 开放题语义分：BERTScore F1 平均

---

## 12. 单模型推理链路（infer.py）

- 文件：`infer.py:82`
- 作用：直接加载 `manglu3935/Chiron-o1-2B` 做单次/多图推理，不包含 MICS 搜索。

关键索引：

1. 图像切块预处理：`infer.py:35`
2. 模型加载：`infer.py:83`
3. 单图示例：`infer.py:93`
4. 多图示例（注释块）：`infer.py:103`

---

## 13. 训练启动链路（train/*.sh）

- `train/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh:20`
- `train/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh:20`

核心逻辑：

1. 计算梯度累积：`GRADIENT_ACC`（两个脚本的 `:6`）
2. 设置分布式环境变量：`MASTER_PORT/LAUNCHER`（`:10-13`）
3. `torchrun` 调 InternVL 训练入口：`:20-27`
4. 传 LoRA + 数据 + Deepspeed 参数：`:29-65`

注意：训练核心 Python 在外部仓库 `internvl/train/internvl_chat_finetune.py`。

---

## 14. 一张“从头到尾”执行时序图

```text
CLI(src/run.py)
  -> mics_start
    -> 读数据(read_jsonl/json)
    -> 可选分片(get_chunk)
    -> init_model(src/model.py)
    -> MentorInternSearch(args)
    -> for 每个样本:
         -> search(src/mics.py)
           -> question/gt/case_info
           -> locate_img + encode_image
           -> 初始化 root step
           -> for depth in max_depth:
                -> for mentor in mentor_models:
                     -> 生成候选 step(_generate_next_step_with_mentor)
                     -> intern 双温度评估(_evaluate_step_with_interns)
                     -> add_child_step
                -> 满分则提前写成功并 return
                -> 全零则写失败并 return
                -> select_next_step 进入下一层
           -> 到最大深度后写成功/失败
```

---

## 15. 关键风险点（带代码索引）

1. `locate_img` 报错字段名疑似写错（`image` vs `images`）
   - 位置：`src/utils.py:71`
2. `get_correctness` 只要含 `yes` 就判正确，容易误判
   - 位置：`src/utils.py:145`
3. `internvl_forward` 强依赖 CUDA（`.cuda()`）
   - 位置：`src/internvl_forward.py:89`
4. `split_model` 假设可用 GPU 数与层切分关系固定
   - 位置：`src/model.py:18`

---

## 16. 快速定位索引（按功能查文件）

- 启动入口：`src/run.py:90`
- 单样本核心搜索：`src/mics.py:196`
- mentor 生成：`src/mics.py:74`
- intern 评估：`src/mics.py:116`
- 模型初始化：`src/model.py:38`
- 树节点结构：`src/step.py:2`
- Prompt 模板：`src/prompt.py:1`
- 图片定位/编码：`src/utils.py:59`, `src/utils.py:30`
- Qwen forward：`src/qwenvl_forward.py:3`
- InternVL forward：`src/internvl_forward.py:83`
- 评测入口：`eval/eval.py:214`
- 推理 demo：`infer.py:82`

