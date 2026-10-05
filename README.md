# homework_ocr — 本地离线的学生手写作答提取工具

输入学生作业扫描件 PDF，找出其中的**手写部分**并输出成文字。全部计算在本机完成，
运行期不联网。

核心分工，也是整个设计的地基：

> **OCR 不负责判断什么是学生写的。OCR 只负责识别已经筛选出来的手写区域。**

链路：

```text
空白模板 PDF ──┐
               ├─→ 页面配准（ORB + RANSAC + Homography + ECC 精配准 + 方向探测）
学生 PDF ──────┘
      ↓
对齐后的学生页
      ↓
题目 ROI（人工标一次，所有学生复用）
      ↓
光照归一化 + 自适应阈值 → 模板相减（保守抑制）
      ↓
手写 mask → 去线框残影 → 合并成文本行
      ↓
用【原图 crop】送 OCR  ← mask 只用来定位和过滤，不作为 OCR 输入
      ↓
按 mask 重叠率过滤掉印刷内容残留 → 多行拼接
      ↓
review_required 判定 → JSON / CSV / TXT / debug 图
```

---

## 1. 安装

需要 Python 3.10+。

```bash
pip install -r requirements.txt
pip install -e .
```

只想快速试一下的话，至少要装 `rapidocr` 和 `onnxruntime`（CPU 即可，无需 GPU）。

### 预下载模型

**联网环境下先跑一次**，把模型放到项目内：

```bash
homework-ocr fetch-models --out models
```

之后可以完全断网运行。`homework-ocr doctor` 会实测出站连接是否真的被拦截。

### 为什么默认用 RapidOCR 而不是 PaddleOCR

两者用的是同一批 PP-OCR 模型，但推理框架不同。选 RapidOCR（ONNXRuntime）的原因：

- Windows / macOS（含 Apple Silicon）/ Linux 通用，纯 CPU 可用，无 PaddlePaddle 依赖。
- PaddlePaddle 在 macOS 上有实打实的兼容问题：PaddleOCR ≥ 3.4 依赖 PaddlePaddle 3.1+
  才有的算子（`fused_rms_norm_ext`、`cal_aux_loss`），而 3.1+ **没有 macOS x86_64 wheel**；
  另外 paddlepaddle 3.3.1 在 macOS 26 上有导入即失败的已知问题。
  参见 [Paddle issue #78542](https://github.com/PaddlePaddle/Paddle/issues/78542)。

OCR 层是抽象接口（`OcrEngine`），PaddleOCR 适配器也写好了（`--engine paddleocr`），
想切换随时切。

---

## 2. 上手

### 方式 A：双击启动（推荐给非技术用户）

| 系统 | 操作 |
|---|---|
| Windows | 双击 `启动.bat` |
| macOS | 双击 `启动.command` |

**不需要输入任何参数。** 第一次启动会引导你做三步设置：

1. **选择空白模板** —— 选已有的，或者直接从空白 PDF 新建一个
2. **选择学生作业文件夹** —— 把所有作业 PDF 放一起，选这个文件夹
3. **结果保存到哪里** —— 留空自动决定

设置会记住，**之后每次双击就直接进主界面**。

> 「浏览…」按钮弹出的是本机的文件夹浏览器，能看到 C:\ / D:\ 和
> 「桌面 / 文档 / 下载」这些常用位置，也可以直接粘贴完整路径。
> （浏览器出于安全限制拿不到文件夹的绝对路径，所以这个选择器由程序自己实现。）

### 方式 B：命令行

```bash
homework-ocr gui
```

零参数启动。也可以显式指定，优先级高于上次设置：

```bash
homework-ocr gui --template templates/yuwen_g3_1 --input input/ --output output/
homework-ocr gui --reset     # 忘掉上次的选择，重新引导
```

命令行批处理与导出：

```bash
homework-ocr run --template templates/yuwen_g3_1 --input input/ --output output/
homework-ocr export --output output/
```

### 建模板与标注 ROI

模板是「学生还没写字之前的那份空白作业」。第一次用图形界面创建模板后，
还需要标注每道题的答案区域（每份作业只标一次）：

```bash
homework-ocr annotate --template templates/yuwen_g3_1 --suggest
```

`--suggest` 会先检测印好的横线、给出候选答案块，再在窗口里微调。
标注窗口快捷键：拖拽框选，`n` 下一个题号，`d` 删除，`[` `]` 缩放，
`PgUp`/`PgDn` 翻页，`s` 保存，`q` 退出。

### 复核界面怎么用

左右分栏：**左边的手写图才是真相，右边的文字只是参考。**

- 三种图可切换：`答案` / `手写 mask` / `整页定位`。卡在「哪些是印刷、哪些是手写」时切到 mask 看。
- 鼠标滚轮在图片上缩放，细节看不清时放大看。
- 三个按钮：
  - `确认无误` — 机器认对了，采纳原文
  - `保存并下一题` — 你改了文字，保存后自动跳下一题
  - `判为无效` — 这题没法认（比如涂成一团），留空
- 快捷键：`←` `→` 翻题，`Ctrl+Enter` 保存并下一题，`Ctrl+S` 确认无误
- 复核完的题会从「只看未复核」列表里消失；取消勾选可以**抽查**已确认的题
- 列表页的「全部确认」是批量操作，用之前请确认真的核对过

### 隐私

GUI 服务只绑定 `127.0.0.1`，不对局域网开放，出站 socket 已被封禁。
数据只在浏览器和本机之间传输，不会离开这台电脑。

**不要把 `--host` 改成 `0.0.0.0`** —— 那等于把全班作业开放给局域网。

---

## 3. 设置保存在哪里

```text
Windows  %LOCALAPPDATA%\homework_ocr\settings.json
macOS    ~/Library/Application Support/homework_ocr/settings.json
Linux    ~/.local/share/homework_ocr/settings.json
```

只存路径和端口这类偏好。模板和学生作业都留在你自己选的位置，
程序不会往自己的安装目录里塞东西。删掉这个文件就等于重置
（或者 `homework-ocr gui --reset`）。

---

## 4. 人工复核结果怎么保存

机器结果和人工结果分开存，互不覆盖：

```text
output/张三/
├── result.json      # 机器识别结果（reocr 换模型时会重写）
└── review.json      # 人工修正（永不覆盖）
```

这样 `reocr` 换 OCR 模型重跑不会冲掉已经核对过的人工劳动。导出 Excel 时两份一起读，
`最终文本` 列 = 人工修正优先，没有修正才用 OCR 结果；`复核状态` 列区分
`已确认` / `已修正` / `待复核` / `已判无效`。

---

## 5. 命令

| 命令 | 用途 |
|---|---|
| `gui` | **启动图形界面**（批处理 + 复核 + 导出） |
| `init-template` | 空白模板 PDF → `template.json` |
| `annotate` | 交互式标注 ROI |
| `run` | 批量提取 |
| `export` | 导出 Excel（含人工修正） |
| `reocr` | **只重跑 OCR**，不重跑渲染/配准/差分 |
| `fetch-models` | 预下载模型 |
| `doctor` | 环境自检（依赖 / 模型 / 中文字体 / 离线封禁实测） |
| `show` | 命令行打印结果 |
| `make-demo` | 生成自测数据 |

### `reocr`：换模型不用重跑前面

`run` 会保存 `aligned` 图和每题的答案 crop。换 OCR 模型时只跑 OCR 那一层：

```bash
homework-ocr reocr output/张三 --model-type mobile
homework-ocr reocr output/张三 --ocr-version PP-OCRv6
```

渲染/配准/差分是最慢、也最容易受参数影响的几步，避开它们能让模型对比快一个数量级。

---

## 6. 输出

```text
output/
├── homework_results_<模板id>.xlsx   # 整班汇总（export 命令生成）
└── 张三/
    ├── result.json          # 完整结构化结果（含配准分数、bbox、置信度、复核原因）
    ├── result.csv           # 表格
    ├── result.txt           # 人读的汇总
    ├── 张三.xlsx             # 单份 Excel
    ├── review.json          # 人工修正
    ├── page_001_aligned.png # 配准后的页面
    ├── page_001_mask.png    # 手写 mask
    ├── page_001_debug.jpg   # 叠加图：蓝=模板ROI 绿=手写 橙=OCR 红=需复核
    └── page_001_q1_answer.png
```

### Excel 的三个工作表

| 工作表 | 内容 |
|---|---|
| **答题明细** | 每题一行：识别文本、置信度、最终文本、复核状态、复核原因、备注。需复核的行黄底，人工改过的绿底 |
| **复核队列** | 只放还没复核的题，按优先级排。这就是当天要干活的清单 |
| **学生汇总** | 每人一行：配准成功页数、题数、需复核数，用来快速筛可疑文件 |

### debug 叠加图怎么用

出问题时先看这张图，它能立刻告诉你是哪一层坏了：

| 现象 | 大概率原因 |
|---|---|
| 模板印刷文字出现双影 / 描边 | 配准没对齐 → 调 `alignment.enable_ecc`、`ransac_threshold` |
| 整页 `alignment_failed` | 扫描质量太差 / 不是同一份模板 → 看 `alignment.reason` |
| mask 里出现大片横线、框线 | 配准残差或阈值太松 → 调 `template_dilate_px`、`adaptive_c` |
| mask 干净但 OCR 结果为空 | ROI 标错，或字迹太淡 → 核对 `template.json` |
| OCR 把拼音、题目文字也读进来了 | `ocr.min_box_overlap` 太小 → 调大 |

### `review_required` 什么时候为真

OCR 的 confidence 不等于可信度，所以复核标记是独立判定的，触发条件包括：

- 未检测到手写（可能漏答，也可能字迹太淡）
- 配准分数低于 `review.min_alignment_score`
- 笔迹碎裂（只统计**产出了文本**的区域；多行答案不算碎裂）
- 识别置信度偏低
- **有高置信度的行被 mask 判掉**（怀疑漏掉了真实作答；低置信度残影被丢弃不会报警，
  否则复核队列会被噪声淹掉）
- 墨迹覆盖率异常高（疑似涂改）

每条都会带一句可读的 `reasons`，人工扫一眼就能判断真伪。

---

## 7. 调参

所有参数集中在 `config.yaml`，每个都有行内注释说明取舍。**不确定的先别动**，
先跑一遍看 debug 图再调。最常用的几个：

| 场景 | 调什么 |
|---|---|
| 模板印刷文字残留鬼影多 | `difference.template_dilate_px` 1→2 |
| 淡铅笔字丢失 | `difference.adaptive_c` 15→10 |
| 阴影重、光照不均 | `difference.illumination: true`，必要时调大 `illumination_kernel` |
| 印刷内容混进答案 | `ocr.min_box_overlap` 调大（如 0.15→0.3） |
| 复核项太多 | 调低 `review.min_ocr_confidence`，或调大 `ocr.min_box_overlap` |
| 复核项太少（漏了问题） | 调高 `review.min_ocr_confidence`，或调小 `min_box_overlap` |

建议先拿 10~20 份真实扫描件建一个固定测试集，每次改参数都重跑，然后统计：

```text
页面配准成功率 / 手写区域召回率 / 误检率 / OCR 字符准确率 / 需人工复核比例
```

这能很快告诉你瓶颈在 CV 还是 OCR。凭肉眼调一两张图会走偏。

---

## 8. 隐私：本地计算是被强制的，不是承诺

`offline_guard` 在程序启动时**直接封禁出站 socket**，任何联网尝试都会抛异常并打印
是谁发起的。同时设置 `HF_HUB_OFFLINE` 等环境变量作为双保险。`homework-ocr doctor`
会实测并报告封禁是否生效。

需要联网的只有 `fetch-models`，它在下载完成后就回到封禁状态。

> 严格环境建议再加一层：用操作系统防火墙禁止该程序联网，
> 以及确认学生 PDF 本身没有被放进 OneDrive / iCloud 等会自动同步的目录。

---

## 9. 已知边界

- **需要空白模板。** 模板模式的精度依赖它。`difference.method: none` 和
  `detect_handwriting_no_template` 提供了无模板回退，但那是启发式的
  （靠连通域尺寸/形状/填充率猜），**误检率明显更高**，结果会明确标注。
- **手写识别本身有上限。** PP-OCRv5 在手写中文上的公开指标是 0.803
  （手写英文 0.841），印刷体是 0.945。手写是真弱项，不要期待通用模型读全。
  真要提准确率，路线是拿自己班级的真实手写数据微调识别模型。
- **数学公式是已知瓶颈。** 当前版本对 `x²`、分数、竖式、方程的识别不理想。
  建议先用 `type: drawing` 之类的方式把这类题排除在 OCR 之外，只保留 crop 供人工看。
- **选择题未做填涂判定。** `Question.type` 字段已经预留了 `choice`，
  但第一版没有实现 CV 填涂/勾选识别。
- **合成自测数据不能替代真实评估。** `make-demo` 生成的"手写"是用真实字形加随机
  扰动画出来的，字形可读，因此可以断言"OCR 结果是否等于注入的原文"；
  但它**不能**用来评估在真实学生作业上的准确率。

---

## 10. 自测

```bash
homework-ocr make-demo --out demo
homework-ocr run --template demo/template --input demo/input --output demo/output
```

会生成 3 份模拟扫描件（含倾斜、光照不均、涂改、整页倒置、漏答）和一份标准答案，
`demo/ground_truth.txt` 里是每份的预期文本，用来核对识别是否正确。
