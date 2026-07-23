# FlashShot

**AI 写真交付系统** ("AI Portrait Director"):用户上传 4–6 张自拍 → 几秒内出一张"像本人 + 好看"的 **Hero 预览图**(Aha Moment)→ Paddle 付费解锁 → 稳定交付一整套精选写真。北极星指标 = 用户最终下载的合格写真数量。

> 本文件是**当前结构真相**(2026-07-23)。`FLASHSHOT_OVERVIEW.md` / `ARCHITECTURE_REVIEW.md` 是更早时点的叙述/评估,如与本文件冲突以本文件为准。

---

## 仓库布局

```
mvp_1/
├── headshot_pipeline/      # 后端(FastAPI)—— 产品核心
│   ├── server/             # 应用代码(分层见下)
│   ├── tests/              # 241 个 pytest 用例
│   ├── templates/          # 风格参考图(prompts.json 的 template_image 指向)
│   ├── prompts.json        # 风格/模板定义(生产硬依赖,job_queue 加载)
│   ├── deploy/             # 部署:overseas-vps/ + legacy launchd/Caddy
│   ├── experiments/        # 离线实验脚本
│   ├── legacy/             # 已废弃的 chrome-era 死代码(产品不再 import)
│   ├── persistent_client.py / watermark_remover.py  # chrome 后端耦合单元(仅 chrome 后端可达)
│   └── models/  data/  output/  .chrome_profile/    # 运行时(gitignore)
├── headshot-landing/       # 前端(Next.js 16 + React 19 + TS + Tailwind v4)
├── gemini-image-gen-automation/  # 第三方嵌套 git 仓库(gitignore,零引用)
├── FLASHSHOT_OVERVIEW.md   # 项目叙述(已订正)
└── ARCHITECTURE_REVIEW.md  # 2026-07-02 架构评估快照
```

`headshot-landing/AGENTS.md` 是前端专属的 Next.js 版本注意事项,本文件不重复。

---

## 后端分层架构(`headshot_pipeline/server/`)

```
HTTP/WS 路由   router_sessions / router_jobs / router_payment / router_postprocess / router_ws
     │
编排          job_queue.py        单 worker 队列 + session 生命周期(SQLite 持久化)+ 交付去重/标注/高清化
              gemini_worker.py    候选 pipeline 编排(hero / full_set 共用 _execute_candidate_pipeline)
     │
决策          evaluation/policy_engine.py   PolicyEngine —— 自适应(预算/风险/反馈打分),runtime 活跃路径
              evaluation/agent_router.py    AgentRouter —— 规则状态机(PolicyEngine 的内核 + 测试用)
     │
评估          evaluation/evaluator.py       VLM QA judge + 本地 CV 质检 + InsightFace 身份相似度
学习          learning/learning_layer.py    反馈→阈值标定(SQLite,保守有界增量)
     │
生成          generation/gateway.py         ImageGateway —— 业务统一入口
              generation/smart_router.py    SmartModelRouter —— 任务感知模型路由(cost×identity×latency)
              generation/providers.py       SiliconFlow / OpenRouter / Chrome(legacy) 三个 Provider
修复          repair/identity_repair.py     FaceSwapRepair(inswapper_128,本地换脸)
     │
基础设施      config.py  security.py(owner-token+限流)  payment.py(Paddle HMAC)  storage.py(SQLite)
              models.py(TIER_LIMITS/Pydantic)  shot_planner.py  delivery_label.py  ...
```

**三层评分**:硬门槛(安全/人脸/身份/质量)→ 决策 → 软排序(审美)。`aggregate_score` 权重 = identity 0.45 / face_quality 0.20 / style 0.20 / artifact 0.10 / commercial 0.05。

### 生成后端开关(`config.py` `gemini_backend`)
- **`siliconflow`(默认)**:Qwen-Image-Edit 生成 + Qwen2.5-VL 评判,无状态 HTTPS,无浏览器。
- `openrouter`:google/gemini-3.1-flash-image。
- `chrome`(legacy):驱动 gemini.google.com,易碎已弃用,仅保留为可选项。

---

## Pipeline 要点

**星型生成拓扑(反漂移硬约束)**:每个分镜都从**原始 Identity Pack** 独立生成(`CREATE_FROM_REFERENCES`),严禁 `A→edit→B→edit→C` 链式漂移;只有 `LOCAL_EDIT` / `IDENTITY_REPAIR` 才级联。

| 流程 | 候选数 | max_regen | max_edit | max_repair | max_cost |
|---|---|---|---|---|---|
| Hero Preview | 4 | 1 | 1 | 1 | $0.6 |
| Full Set | 3 | 2 | 2 | 1 | $1.0 |

---

## 定价(`server/models.py:355` `TIER_LIMITS`)

| Tier | 价格 | 风格数 | 改次数 | 证件照 | 换背景 | 高清下载 |
|---|---|---|---|---|---|---|
| free | $0 | 1 | 1 | ✗ | ✗ | ✗ |
| standard | $5 | 2 | 2 | ✓ | ✓ | ✗ |
| premium | $10 | 2 | 3 | ✓ | ✓ | ✓ |

(海外定价;$19 因太接近 ChatGPT Plus 被否决。)

---

## 安全与支付模型

- **Owner token**:每个 session 一个 256-bit 随机 token,创建时返回一次,所有 session 操作须验证。比较用 `hmac.compare_digest` 常量时间;**session 不存在也只返 401 不返 404**(防 session_id 枚举)。
- **支付唯一提权路径**:HMAC-SHA256 签名验证的 Paddle webhook(`router_payment.py::verify_paddle_signature`)。前端轮询**只读**,绝不提权。Paddle 是 Merchant of Record,价格由 price_id 服务端锁定,webhook 不校验金额。
- **数据保留**:7 天后删除源/生成文件(每小时 retention sweep)。
- 上传图片做魔术数字节校验;路径用 `safe_id` 防穿越。

---

## 如何运行

```bash
# 后端测试(用隔离 DATA_DIR,绝不碰生产 DB)
cd headshot_pipeline
DATA_DIR="$(mktemp -d)" .venv/bin/python -m pytest tests/ -q   # 241 passed

# 后端启动
.venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port 8001
# 配置走 .env(见 server/.env.example);GEMINI_BACKEND=siliconflow 需 SILICONFLOW_API_KEY

# 前端
cd headshot-landing
npm run dev      # 或 npm run build && npm start
```

健康/就绪:`GET /api/health`(进程存活)、`GET /api/ready`(生成就绪)、`GET /api/launch-ready`(付费上线门)。

---

## 部署(LIVE)

生产在美国共享 VPS `66.175.213.242`,详见 `headshot_pipeline/deploy/overseas-vps/DEPLOY_OVERSEAS_VPS.md`。

```
:443 → Caddy(只 append 一个 site block)  /api/* /ws/* → 127.0.0.1:8001 FastAPI
                                          *          → 127.0.0.1:3001 Next.js
        HTTPS egress → api.siliconflow.cn
```

> **共享主机硬约束**:这台机器还跑着 recorrect、Syntropy、Caddy(其他站点)、Tailscale、fwxt-relay。**所有改动必须 additive,绝不扰动 co-tenant**。回滚只移除 FlashShot 自身。

---

## 关键约定

- **星型生成,反漂移**:见上。
- **共享主机 additive**:部署、Caddy、systemd 都只增不改不删他者。
- **`.env` 不进仓库**;dev/prod 密钥分离,rsync 部署时 `--exclude .env`。
- **legacy/ 不被产品 import**:仅 chrome-era 死代码归档(见 `legacy/README.md`)。
