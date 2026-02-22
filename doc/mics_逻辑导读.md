# `src/mics.py` 逻辑导读（含树状图）

这份文档的目标：让你在读 `src/mics.py` 时，不被细节淹没，先抓住“单样本怎么从输入走到输出”。

## 1. 这个文件在整个项目里的职责

`src/mics.py` 负责单样本的核心搜索循环：

- mentor 生成候选推理 step
- intern 对候选 step 打分
- 根据分数决定是否提前成功、提前失败，或继续下一层
- 把结果写入成功/失败日志（由 `run.py` 传进来的文件句柄）

核心类：`MentorInternSearch`（`src/mics.py:14`）

核心函数：

- `_call_model_forward`（`src/mics.py:42`）
- `_generate_next_step_with_mentor`（`src/mics.py:85`）
- `_evaluate_step_with_interns`（`src/mics.py:136`）
- `search`（`src/mics.py:235`）

## 2. 单样本端到端时序（你最该先记住的）

对应 `search(...)`（`src/mics.py:235`）

1. 解析输入
- 从 `data` 取 `question`、`gt_answer`、`case_info`（`src/mics.py:248`, `src/mics.py:251`, `src/mics.py:253`）
- 解析并编码图片（`src/mics.py:256`-`src/mics.py:259`）
- 图片失败直接记失败并返回（`src/mics.py:260`-`src/mics.py:273`）

2. 初始化搜索状态
- root 文本设为固定起始句（`src/mics.py:281`-`src/mics.py:283`）
- `current_step` 指向 root（`src/mics.py:283`）‘
- mentors_scores 记录每一个mentor在各层的得分历史，用于后续tie-break (select_best_mentor和结果输出里面的score字段)
- reasoning_chains 缓存每一个mentor最新的完整推理串，作用是复用已有的生成，避免重复调用mentor （同一个mentor连续被选的时候很有用）
- previous_mentors 记录最终路径每层是哪一个mentor被选中，用于tie-break，优先选择没有选过的 mentor
- 初始化 `mentors_scores`、`reasoning_chains`、`previous_mentors`（`src/mics.py:285`-`src/mics.py:290`）

3. 逐深度搜索（for depth）
- 遍历每个 mentor 生成一个候选 step（`src/mics.py:309` 起）
- 对候选 step 调 intern 评估，得到 score（`src/mics.py:343`-`src/mics.py:347`）
- 记录到候选列表，等待本层决策（`src/mics.py:351`-`src/mics.py:355`）

4. 本层决策
- 若任意 mentor score==1.0：提前成功，写成功日志并 return（`src/mics.py:362`-`src/mics.py:386`）
- 若本层全是 0 分：提前失败，写失败日志并 return（`src/mics.py:388`-`src/mics.py:395`）
- 否则选一个 best child 作为新的 `current_step`，进入下一层（`src/mics.py:397`-`src/mics.py:404`）

5. 到最大深度后的收尾
- 没提前终止则写最终结果（`src/mics.py:408`-`src/mics.py:439`）
- 若无有效路径则写失败（`src/mics.py:440`-`src/mics.py:445`）

## 3. 树状图：推理树是怎么长出来的

### 3.1 ASCII 树（概念图）

```text
root: "Let's think ..."
└── depth 0 candidates (one per mentor)
    ├── mentor A -> step_A0 (score=...)
    ├── mentor B -> step_B0 (score=...)
    └── mentor C -> step_C0 (score=...)

select_next_step(...) picks one node as current_step

current_step
└── depth 1 candidates (again one per mentor)
    ├── mentor A -> step_A1 (score=...)
    ├── mentor B -> step_B1 (score=...)
    └── mentor C -> step_C1 (score=...)

... until:
- score==1.0 => early success
- all zero => early failure
- reach max_depth => final write
```

注意：代码会把每个 mentor 的候选都挂成 child，但每层只选择一个 `current_step` 继续向下（近似 beam size = 1）。

### 3.2 Mermaid 流程树（执行视角）

```mermaid
flowchart TD
    A[search(data,...)] --> B[Parse question/answer/case/images]
    B --> C{Image OK?}
    C -- No --> C1[Write failed_search_file and return]
    C -- Yes --> D[Init root/current_step/state]
    D --> E{depth < max_depth}

    E -- Yes --> F[For each mentor: generate candidate step]
    F --> G[Evaluate candidate with interns]
    G --> H[Collect score and child node]
    H --> I{Any score == 1.0?}

    I -- Yes --> I1[Write success search_file with search_id=1]
    I1 --> Z[return]

    I -- No --> J{All scores == 0?}
    J -- Yes --> J1[Write failed_search_file]
    J1 --> Z

    J -- No --> K[select_next_step -> current_step]
    K --> E

    E -- No --> L[Finalize at max_depth]
    L --> M{Valid final mentor chain?}
    M -- Yes --> M1[Write success search_file with search_id=0]
    M -- No --> M2[Write failed_search_file]
```

## 4. mentor 生成子流程

对应 `_generate_next_step_with_mentor(...)`（`src/mics.py:85`）

1. 取当前路径前缀：`reasoning_prefix = step.get_full_reasoning()`（`src/mics.py:93`）
2. 拼 `REASONING_PROMPT`（`src/mics.py:99`-`src/mics.py:114`）
3. 调统一路由 `_call_model_forward(...)`（`src/mics.py:120`）
4. 返回 `(reasoning_prefix, response)`（`src/mics.py:130`-`src/mics.py:131`）

路由规则看 `_call_model_forward`（`src/mics.py:42`）：

- `'gpt'` -> `gpt_forward`
- `'gemini'` -> `gpt_forward`（不同 client）
- `'72'` -> `qwenplus_forward`
- `'qwen'` -> 本地 `qwenvl_forward`
- `'internvl'` -> 本地 `internvl_forward`

## 5. intern 打分子流程

对应 `_evaluate_step_with_interns(...)`（`src/mics.py:136`）

打分公式：

```text
total_evaluations = intern_model_count * 2
score = correct_count / total_evaluations
```

为什么乘 2：每个 intern 用两种温度跑两次（`temperature1`, `temperature2`，`src/mics.py:181`）。

单次评估链路：

1. intern 生成回答（`src/mics.py:185`-`src/mics.py:190`）
2. 解析 `### The final answer is:`（`src/mics.py:197`-`src/mics.py:205`）
3. evaluator（DeepSeek）判 Yes/No（`src/mics.py:210`-`src/mics.py:223`）
4. Yes 则 `correct_count += 1`（`src/mics.py:221`-`src/mics.py:222`）

## 6. 关键状态变量（读代码时盯住这几个）

在 `search(...)` 中：

- `current_step`：当前活跃路径节点（下一层从这里扩展）
- `generated_children_for_step`：本层所有候选
- `mentors_scores`：每个 mentor 的层级得分历史
- `reasoning_chains`：每个 mentor 当前完整推理串缓存
- `previous_mentors`：最终路径上已选 mentor 序列
- `full_score_mentors`：本层达到 1.0 的 mentor
- `all_zero_score`：本层是否全 0，用于提前失败

## 7. 提前终止规则（必须背下来）

成功提前终止：
- 条件：本层出现 `score == 1.0`
- 行为：立刻写成功日志，`search_id='1'`，并返回（`src/mics.py:362`-`src/mics.py:386`）

失败提前终止：
- 条件：本层所有 mentor 分数都是 0
- 行为：写失败日志并返回（`src/mics.py:388`-`src/mics.py:395`）

正常收尾：
- 条件：到 `max_depth` 仍未提前终止
- 行为：写成功日志 `search_id='0'`，或写失败（`src/mics.py:408`-`src/mics.py:445`）

## 8. 你现在可以这样验证自己是否理解

读完后你应能口述：

1. 一条样本从 `search` 进入后，先做哪三件准备？
2. 一层里 mentor 和 intern 的调用顺序是什么？
3. `score==1.0`、`all_zero_score`、`max_depth` 分别触发什么结果？
4. `current_step` 为什么每层只会更新成一个节点？

如果你愿意，下一步可以把这份文档和 `src/step.py` 结合，我再给你一版“Step 对象字段如何支撑整棵树”的补充图。

## 9. Demo：拿 `demo.jsonl` 一条样本手动跑一遍

这里用 `src/demo_data/demo.jsonl` 第 1 条（`rid=169992`）举例，目的是理解变量如何流动。  
注意：下面的 mentor/intern 文本是“示意”，不是你本地真实模型输出。

样本关键信息：

- `rid`: `169992`
- `question`: 45 岁男性，夹层样胸痛，CT 见左冠异常走行，问最可能解释
- `gt_answer`: `Anomalous interarterial course ... with potential myocardial ischemia.`
- `images`: `169992/ct_group1/0.png`

### 9.1 进入 `search(...)` 后的初始化

1. 解析问题/答案/病例信息并编码图片（`src/mics.py:248`-`src/mics.py:259`）
2. 设定：
- `current_step = root`
- `root.text = "Let's think ... step by step."`
- `mentors_scores = {mentorA: [], mentorB: [], mentorC: []}`
- `reasoning_chains = {}`
- `previous_mentors = []`

### 9.2 depth=0：每个 mentor 生成候选 step 并打分

假设 mentor 列表是：

- `chatgpt-4o-latest`
- `google/gemini-...`
- `qwen2.5-vl-72b-instruct`

对每个 mentor，代码会：

1. 调 `_generate_next_step_with_mentor(...)`
2. 返回 `(prefix_reasoning, suffix_reasoning)`
3. 首层用 `extract_first_two_steps(suffix_reasoning)` 得到 `new_step`
4. 构造 `temp_step = Step(prefix=current_step.text, step_text=new_step, ...)`
5. 调 `_evaluate_step_with_interns(temp_step, ...)` 得到 `score`

示意结果：

- mentorA `new_step`：先排除主动脉夹层，再指向冠脉异常缺血机制 -> `score=1.0`
- mentorB `new_step`：描述解剖但缺少症状机制 -> `score=0.67`
- mentorC `new_step`：回答偏泛化 -> `score=0.33`

于是：

- `mentors_scores` 变成：`{A:[1.0], B:[0.67], C:[0.33]}`
- `full_score_mentors = [A]`

### 9.3 depth=0 直接提前成功

因为出现 `score==1.0`（`src/mics.py:362` 起）：

1. 选 `best_mentor`（这里只有 A）
2. 取 `full_reasoning = reasoning_chains[A]`
3. 写 `search_file` 一条结果：
- `rid=169992`
- `reasoning=full_reasoning`
- `scores={A:[1.0], B:[0.67], C:[0.33]}`
- `final_depth='1'`
- `search_id='1'`（表示提前成功）
4. `return`，该样本结束

### 9.4 这条样本对应的“树”长什么样

```text
root
└── depth 0
    ├── mentorA step (score=1.0)  <- selected, early success
    ├── mentorB step (score=0.67)
    └── mentorC step (score=0.33)
```

因为 depth=0 就满分了，所以不会进入 depth=1。

### 9.5 如果 depth=0 没人满分会怎样

分两种：

1. 有非零分  
- 例如 `A=0.67, B=0.67, C=0.33`
- 用 `select_next_step(...)` 选一个 child 作为新 `current_step`
- 继续 depth=1

2. 全 0 分  
- `all_zero_score=True` 到层末仍不变
- 写 `failed_search_file` 并返回（`error = "All mentor scores are zero."`）

这个分叉就是 `search(...)` 的核心控制逻辑。
