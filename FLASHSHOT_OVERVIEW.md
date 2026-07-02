# FlashShot 项目完整概述与 Pipeline

> 当前状态：2026-07-02 | 后端 136 测试全部通过 | 前端 build 通过 | 5 commits on main

---

## 1. 产品定义

**FlashShot = AI Portrait Director（AI 写真导演系统）**

不是简单的"AI 生图工具"，而是一个稳定输出"像你本人 + 好看 + 可用"的专业写真交付系统。

**核心用户体验：**

```
上传 4-6 张自拍 → 3 秒内看到"像本人的 Hero 预览图" → 付费 → 获得一整套精选写真
```

**一句话目标：**

> 每个用户最终保存/下载的合格写真数量，是北极星指标。

---

## 2. 项目结构

```
/Users/lizeyan/Desktop/mvp_1/
├── headshot_pipeline/          # Python FastAPI 后端
│   ├── server/
│   │   ├── generation/         # NEW: ImageProvider 抽象 + ImageGateway 路由
│   │   │   ├── providers.py    # OpenRouterProvider, ChromeProvider
│   │   │   └── gateway.py      # 业务层统一入口
│   │   ├── evaluation/         # NEW: 拆出的评估系统
│   │   │   ├── evaluator.py    # VLM judge + local quality + identity similarity
│   │   │   └── agent_router.py # 纯决策逻辑状态机
│   │   ├── repair/             # NEW: 身份修复模块
│   │   │   └── identity_repair.py  # FaceSwapRepair
│   │   ├── delivery/           # NEW: 交付后处理（占位）
│   │   ├── gemini_worker.py    # Pipeline 编排器（已从 2820 行降到 1905 行）
│   │   ├── job_queue.py        # 内存队列 + session 管理
│   │   ├── router_jobs.py      # API 路由：hero-preview / unlock / generate
│   │   ├── router_sessions.py  # API 路由：session CRUD
│   │   ├── router_payment.py   # API 路由：Paddle 支付
│   │   ├── router_postprocess.py # API 路由：裁剪/换背景/高清化
│   │   ├── models.py           # Pydantic 数据模型
│   │   ├── config.py           # 配置（OpenRouter / Chrome 切换）
│   │   ├── openrouter_client.py  # OpenRouter REST API 客户端
│   │   ├── image_gateway.py    # Provider metadata + cost estimation
│   │   ├── input_quality.py    # 上传照片质量检查
│   │   ├── shot_planner.py     # 分镜规划
│   │   ├── face_swap.py        # InsightFace 换脸
│   │   ├── storage.py          # 文件存储
│   │   ├── security.py         # Token / 限流
│   │   ├── payment.py          # Paddle 支付逻辑
│   │   └── ...
│   ├── tests/                  # 136 个测试
│   └── experiments/            # 实验脚本
├── headshot-landing/           # Next.js 16 + TypeScript 前端
│   ├── src/
│   │   ├── app/create/page.tsx   # 核心流程页面
│   │   ├── components/create/  # StepGuide / StepUpload / StepStyle /
│   │   │                           # StepHeroPreview / StepGenerating / StepResults
│   │   ├── lib/api.ts            # API 客户端
│   │   ├── lib/websocket.ts      # WebSocket 实时进度
│   │   └── lib/types.ts          # TypeScript 类型定义
│   ├── public/                 # 静态资源
│   └── ...
└── gemini-image-gen-automation/  # 独立的 Gemini 自动化工具（legacy）
```

---

## 3. 完整 Pipeline（从用户上传到交付）

### 3.1 前端流程

```
Guide（自拍引导）
  ↓
Upload（上传 4-6 张照片）
  ↓
Input Quality Gate（后端检查：分辨率/模糊/人脸/角度/一致性）
  ↓
Style（选择风格：单风格 or 多风格 bundle）
  ↓
Hero Preview Generating（生成 4 张候选 → 自动筛选 → 只展示最佳 1 张）
  ← 用户看到"像我 + 好看"的 Aha Moment
  ↓
Hero Preview（展示图 + 反馈按钮：Looks like me / Not like me / Try another style）
  ↓
Payment / Unlock（免费用户看到付费墙，付费用户直接 unlock）
  ↓
Full Set Generation（每个分镜从 Identity Pack 独立生成）
  ↓
Evaluation Service（硬门槛：安全/人脸/身份/质量；软排序：审美/构图）
  ↓
Agent Router（ACCEPT / LOCAL_EDIT / IDENTITY_REPAIR / REGENERATE / DROP）
  ↓
Delivery Gate（去重检查 + AI 标注 + 高清化）
  ↓
Results（精选结果页：下载/裁剪/换背景/反馈）
```

### 3.2 后端 Pipeline 详细

```
用户上传照片
  ↓
【Input Quality Gate】input_quality.py
  - 分辨率检查
  - 模糊检测（Laplacian variance）
  - 人脸检测（Haar cascade）
  - 人脸大小/居中检查
  - 参考图之间身份一致性（InsightFace cosine）
  ↓
【Identity Pack 构建】gemini_worker.py::build_identity_pack_metadata
  - 6 张参考图按角色分类（front_neutral, front_smile, left_45, right_45, lifestyle, side_profile）
  - 主身份参考图标记
  - 临时 face template（任务级，不持久化）
  ↓
【Hero Preview】execute_hero_preview()
  - 生成 4 张候选（HERO_PREVIEW_CANDIDATE_COUNT = 4）
  - 每张候选经过 EvaluationService 评判
  - AgentRouter 选择最佳候选
  - 如需要：LOCAL_EDIT（局部修复）或 IDENTITY_REPAIR（FaceSwap）
  - 预算控制：max_regenerations=1, max_local_edits=1, max_total_api_cost=0.6
  ↓
【Full Set Generation】execute_generate_with_quality_pipeline()
  - 3 张初始候选（PIPELINE_CANDIDATE_COUNT = 3）
  - 每个 shot 从原始 Identity Pack 独立生成（星型结构，非链式 edit）
  - EvaluationService 评判：
    * VLM QA Judge（identity/face_quality/style_match/artifact/commercial_readiness）
    * Local CV Quality（分辨率/模糊/人脸/居中）
    * Identity Similarity（InsightFace cosine → 1-10 分映射）
  - AgentRouter 决策：
    * safety_fail → DROP
    * no_face → REGENERATE_FROM_ORIGINAL
    * identity_fail → REGENERATE_FROM_ORIGINAL
    * identity_gray_zone + good_composition → IDENTITY_REPAIR_ONCE
    * local_artifact + identity_pass → LOCAL_EDIT
    * all_pass → ACCEPT
  - 预算控制：max_regenerations=2, max_local_edits=2, max_identity_repairs=1
  ↓
【Delivery】job_queue.py
  - Duplicate Check（感知哈希去重）
  - AI Label（copy_with_ai_metadata）
  - Upscale（RealESRGAN / Lanczos fallback）
  - 生成下载包
  ↓
【用户反馈】
  - looks_like_me / not_like_me → 影响未来阈值
  - downloaded / selected → 记录到 feedback_summary
```

### 3.3 生成策略（核心原则）

```
❌ 禁止：自拍 → 图 A → Edit 成图 B → Edit 成图 C（链式漂移）

✅ 正确：
原始 Identity Pack
  ├── 分镜 A → 生成候选 → 质检 → 局部修复
  ├── 分镜 B → 生成候选 → 质检 → 局部修复
  └── 分镜 C → 生成候选 → 质检 → 局部修复
```

**三个统一操作：**
- `CREATE_FROM_REFERENCES` — 新分镜从原始 Identity Pack 独立生成
- `LOCAL_EDIT` — 局部瑕疵修复（手、衣领、背景杂物）
- `REGENERATE_FROM_ORIGINAL` — 人物不像或构图失败时回源重生

---

## 4. 已完成的架构升级

### Phase 1: Evaluation Service（已完成）

```
从 gemini_worker.py 拆出 evaluation/evaluator.py
├── VLM QA Judge（结构化 JSON 评分）
├── Local CV Quality Check（OpenCV：分辨率/模糊/人脸/居中）
├── Identity Similarity（InsightFace cosine → 1-10 分映射）
├── Gate Status（硬门槛判断）
└── Quality Evaluation Summary（产品级 QA 报告）
```

### Phase 2: Agent Router（已完成）

```
从 gemini_worker.py 拆出 evaluation/agent_router.py
├── decide_candidate_action() — 状态机决策
├── should_apply_identity_repair() — 灰区判断
├── select_candidate() — 候选选择
└── candidate_shortlist() — 漏斗摘要
```

### Phase 3: Image Gateway Provider 抽象（已完成）

```
新建 generation/ 模块
├── ImageProvider（抽象基类）
│   ├── create_from_references()
│   ├── local_edit()
│   ├── judge()
│   └── upscale()
├── OpenRouterProvider（包装 OpenRouterGeminiClient）
├── ChromeProvider（legacy，包装 PersistentGeminiClient）
└── ImageGateway（路由层，业务统一入口）

新建 repair/ 模块
└── FaceSwapRepair（从 gemini_worker 迁出）

结果：gemini_worker.py 不再直接调用 client.start_conversation()
       所有调用通过 gateway.create_from_references() / local_edit() / judge()
```

---

## 5. 关键指标与阈值

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Hero Preview 候选数 | 4 | 4 |
| Hero 最大重生成次数 | 1 | 1 |
| Hero 最大局部编辑次数 | 1 | 1 |
| Full Set 候选数 | 3 | 3 |
| Full Set 最大重生成次数 | 2 | 2 |
| Full Set 最大局部编辑次数 | 2 | 2 |
| Identity Pass Threshold（closeup） | 8.0 | 8.0 |
| Identity Repair Threshold（closeup） | 7.0 | 7.0 |
| Identity Cosine Accept Threshold | 0.45 | 0.45 |
| Quality Accept Threshold | 8 | 8 |
| 身份保持通过率 | ~33% | >70% |

---

## 6. 商业结构

| 套餐 | 价格 | 权益 |
|------|------|------|
| Free | $0 | 1 张低清/水印 Hero Preview |
| Starter | $9 | 1 个风格，8 张精选 |
| Standard | $19 | 3 个风格，20-30 张精选，高清下载，1 次重做 |
| Pro | $39 | 5 个风格，40-60 张精选，更多 revision，高级后处理 |
| Team | $12-29/人 | 企业统一头像，批量导出，管理后台 |

**转化路径：**

```
免费 Hero Preview → 用户产生 Aha → 付费解锁完整套图
```

---

## 7. 安全与隐私

- **Session Owner Token**：每个 session 有独立 owner token，所有操作需验证
- **图片 URL 带 Token**：`?token=` 防止未授权访问
- **支付只走 Paddle Webhook**：前端不直接提权
- **数据保留期**：7 天后删除源文件和生成文件
- **无长期人脸库**：Identity Pack 是任务级临时数据，任务结束即删除
- **无跨用户搜索**：cross_user_search = False

---

## 8. 测试覆盖

- **136 个后端测试**全部通过
- 覆盖：backend switch、resemblance loop、quality pipeline、delivery gate、payment、session rehydrate、upload consent、user feedback、shot planner、image serving、postprocess upscale
- **前端 build 通过**（Next.js 16 + TypeScript）

---

## 9. 下一步（Phase 4+）

### 短期（1-2 周）

1. **Identity Pass Rate 提升到 ≥70%**
   - 引入 pose-aware threshold（不同角度不同阈值）
   - reference weighting（正面照权重更高）
   - 输入质量检查增加 yaw/pitch/roll 角度、眼镜/刘海遮挡检测

2. **失败样本隐藏机制**
   - 低质量候选完全不展示给用户
   - 只展示"通过硬门槛 + 软排序最佳"的图

3. **成本控制系统**
   - 按套餐控制 max_total_api_cost
   - 记录每个 session 的实际成本

### 中期（2-4 周）

4. **Shot System 标准化**
   - 每个 shot 增加 pose_risk、identity_weight、expected_success_rate
   - 系统优先选择低风险 shot 做 Hero Preview

5. **三层评分系统统一**
   - Layer 1: Identity（硬门槛）
   - Layer 2: Physical Quality（客观检测）
   - Layer 3: Aesthetic（VLM 审美）
   - 决策规则：identity fail → regenerate；artifact fail → edit；aesthetic high → accept

6. **Multi-model Routing**
   - Hero face → identity-stable model
   - Full body → composition model
   - Edit → inpainting model
   - Upscale → RealESRGAN

### 长期（1-2 月）

7. **失败学习系统**
   - user "not like me" → 存储 label → 影响未来 threshold
   - 影响 shot selection 和 prompt 构造

8. **Agent Kernel 完整化**
   - Planner → Generator → Evaluator → Router → Memory
   - 从"随机生成+判断"变成"有策略的摄影导演"

9. **产品体验极致化**
   - 只展示 1 张 Hero（隐藏所有候选/修复过程）
   - "Make it more like me" / "Make it more cinematic" / "Try another angle" 按钮
   - 失败完全不可见化

---

## 10. 一句话总结

> **FlashShot 当前是一个"能生成 AI 头像的 MVP"，正在升级为"先制造 Aha Moment，再用 Identity Pack、模型路由、质量评估和受限 Agent 稳定交付高质量写真套图的 AI 写真导演系统"。**
