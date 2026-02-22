# `src/run.py` 逻辑导读（含流程树 + demo）

这份文档帮助你理解：`run.py` 在整个系统里做什么，以及它如何把每条样本派发到 `mics.py`。

## 1. `run.py` 的角色

`src/run.py` 不是算法主体，它是“任务编排器（orchestrator）”。

它负责：

- 解析 CLI 参数
- 校验 mentor/intern 配置是否合法
- 读取数据并可选分片
- 初始化本地 intern 模型
- 为每条样本调用 `MentorInternSearch.search(...)`
- 统一写成功/失败日志

一句话：`run.py` 管“跑起来”，`mics.py` 管“怎么搜”。

## 2. 顶层执行路径

入口在 `if __name__ == "__main__":`（`src/run.py:116`）。

执行顺序：

1. 解析参数（`src/run.py:120`-`src/run.py:149`）
2. 参数校验（`src/run.py:151`-`src/run.py:179`）
3. 调 `mics_start(args)`（`src/run.py:181`）

## 3. 参数分组（快速定位）

### 3.1 I/O 与切片参数

- `--data_path`（`src/run.py:122`）
- `--image_dir_path`（`src/run.py:123`）
- `--output_path`（`src/run.py:124`）
- `--num_chunks` / `--chunk_idx`（`src/run.py:126`-`src/run.py:127`）

### 3.2 模型相关参数

本地 intern 模型路径：

- `--qwen25_vl_7b_model_path`（`src/run.py:129`）
- `--qwen2_vl_7b_model_path`（`src/run.py:130`）
- `--internvl3_8b_model_path`（`src/run.py:131`）

远程 API：

- OpenAI/Qwen/Gemini/DeepSeek 的 key + base_url（`src/run.py:133`-`src/run.py:140`）

模型列表与搜索超参：

- `--mentor_models`（`src/run.py:141`）
- `--intern_models`（`src/run.py:142`）
- `--evaluator_model`（`src/run.py:143`）
- `--max_depth` / `--temperature1` / `--temperature2`（`src/run.py:145`-`src/run.py:147`）

## 4. 参数校验逻辑（启动前的闸门）

对应 `src/run.py:151`-`src/run.py:179`。

核心约束：

- mentor 至少 2 个（`src/run.py:159`-`src/run.py:160`）
- intern 至少 1 个（`src/run.py:163`-`src/run.py:164`）
- intern 名称必须能映射到对应路径参数（`src/run.py:152`-`src/run.py:171`）

如果有错误：

- 打印错误列表 + help
- `exit(1)` 终止启动（`src/run.py:174`-`src/run.py:179`）

## 5. `mics_start(args)` 主流程拆解

对应 `src/run.py:10`。

### 5.1 读数据

- 支持 `.jsonl` 和 `.json`（`src/run.py:24`-`src/run.py:36`）
- 格式不对直接返回（`src/run.py:37`-`src/run.py:39`）

### 5.2 准备输出文件

- 成功：`output_path`
- 失败：`output_path` 同目录的 `*_failed.jsonl`（`src/run.py:50`）

### 5.3 可选分片

- `num_chunks > 1` 时只处理一个 chunk（`src/run.py:64`-`src/run.py:67`）
- `chunk_idx` 越界会写失败日志并返回（`src/run.py:68`-`src/run.py:71`）

### 5.4 初始化模型 + 搜索器

- `model_set = init_model(args)`（`src/run.py:79`）
- `search_process = MentorInternSearch(args)`（`src/run.py:87`）

### 5.5 遍历样本并派发

- for each `d in data`（`src/run.py:94`）
- 调 `search_process.search(d, model_set, search_file, failed_search_file)`（`src/run.py:98`）
- 单条样本异常不影响全局，写失败日志继续（`src/run.py:99`-`src/run.py:109`）

## 6. 流程树（run 视角）

### 6.1 ASCII 结构图

```text
main
└── parse args
    ├── validate args
    │   ├── invalid -> print help + exit(1)
    │   └── valid -> mics_start(args)
    └── mics_start
        ├── load data (.jsonl/.json)
        ├── open output + failed_output
        ├── optional get_chunk
        ├── init_model(args)
        ├── build MentorInternSearch(args)
        └── for each sample d:
            └── search_process.search(d, model_set, ...)
```

### 6.2 Mermaid 流程图

```mermaid
flowchart TD
    A[main] --> B[parse_args]
    B --> C{arg validation pass?}
    C -- No --> C1[print errors + help + exit(1)]
    C -- Yes --> D[mics_start(args)]

    D --> E[load data jsonl/json]
    E --> F{data loaded?}
    F -- No --> F1[return]
    F -- Yes --> G[open success/failed output files]

    G --> H{num_chunks > 1?}
    H -- Yes --> H1[get_chunk]
    H1 --> H2{chunk_idx valid?}
    H2 -- No --> H3[write failed log + return]
    H2 -- Yes --> I[init_model(args)]
    H -- No --> I

    I --> J[create MentorInternSearch(args)]
    J --> K[for each sample d]
    K --> L[search(d, model_set, ...)]
    L --> M{sample exception?}
    M -- Yes --> M1[write failed log and continue]
    M -- No --> K
```

## 7. 异常与失败策略（run 层）

`run.py` 的策略是：

- 启动前错误（参数不合法）直接失败退出
- 运行时单条样本错误仅记录，不中断全任务
- 文件/目录级错误在外层兜底并返回

典型失败入口：

- 数据读取失败（`src/run.py:27`-`src/run.py:36`）
- 输出目录创建失败（`src/run.py:57`-`src/run.py:59`）
- 分片索引越界（`src/run.py:68`-`src/run.py:71`）
- 模型初始化失败（`src/run.py:80`-`src/run.py:83`）
- 样本内部异常（`src/run.py:99`-`src/run.py:109`）

## 8. Demo：从一条样本进入到 `mics.search()`

这里沿用 `src/demo_data/demo.jsonl` 第一条（`rid=169992`）举例。

假设：

- `num_chunks=1`
- 参数校验通过
- `init_model(args)` 成功

执行链路：

1. `main` 解析参数后调用 `mics_start(args)`
2. `mics_start` 读入 demo 数据列表
3. 创建 `search_file` 与 `failed_search_file`
4. 初始化 `model_set`
5. 创建 `search_process = MentorInternSearch(args)`
6. 循环第一条样本 `d`，其 `rid=169992`
7. 调 `search_process.search(d, model_set, search_file, failed_search_file)`
8. 后续具体搜索逻辑转到 `src/mics.py`（见 `doc/mics_逻辑导读.md`）

## 9. 读完 `run.py` 后你应该能回答

1. 为什么说 `run.py` 是 orchestrator 而不是算法主体？
2. `num_chunks/chunk_idx` 在哪里生效，越界时怎么处理？
3. 为什么单样本异常不会中断整个任务？
4. `run.py` 和 `mics.py` 的边界到底在哪里？
