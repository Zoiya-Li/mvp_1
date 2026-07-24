# FlashShot

**AI 写真交付系统** ("AI Portrait Director"):用户上传 4–6 张自拍 → 几秒内出一张"像本人 + 好看"的 **Hero 预览图**(Aha Moment)→ 付费解锁(**Apple IAP** 为主路径 / Paddle Web 可选)→ 稳定交付一整套精选写真。北极星指标 = 用户最终下载的合格写真数量。

> 本文件是**当前结构真相**(2026-07-24 订正)。`FLASHSHOT_OVERVIEW.md` / `ARCHITECTURE_REVIEW.md` 是更早时点的叙述/评估,如与本文件冲突以本文件为准。
>
> **⚠️ 源码真相在生产,不在本仓库历史里。** 生产源站 = `38.76.165.9:/opt/flashshot`(非 git 部署快照,nginx 入口,Apple IAP 已上线,`/api/launch-ready` 绿)。本仓库长期落后生产一整套(IAP + portrait v2 + admin + 额外 evaluation/repair);**2026-07-24 已用生产快照整体覆盖对齐**(分支 `chore/reconcile-with-prod-source`)。此后"从本仓库部署"不再回滚 IAP,但生产部署仍走 38.76.165.9——改付费/上线相关代码务必先确认改的是生产真相。

---

## 仓库布局

```
mvp_1/
├── headshot_pipeline/      # 后端(FastAPI)—— 产品核心
│   ├── server/             # 应用代码(分层见下)
│   ├── tests/              # 437 个 pytest 用例(434 绿;3 个为生产既存代码/测试漂移,非覆盖引入)
│   ├── templates/          # 风格参考图(prompts.json 的 template_image 指向)
│   ├── prompts.json        # 风格/模板定义(生产硬依赖,job_queue 加载)
│   ├── config/apple-root-certs/  # Apple 根证书(IAP 收据验签)
│   ├── deploy/             # 部署脚本(注:生产实为 nginx,见"部署"章;overseas-vps/ 旧 Caddy 文档已过时)
│   ├── experiments/        # 离线实验脚本
│   ├── legacy/             # 已废弃的 chrome-era 死代码(产品不再 import)
│   ├── persistent_client.py / watermark_remover.py  # chrome 后端耦合单元(仅 chrome 后端可达)
│   └── models/  data/  output/  .chrome_profile/    # 运行时(gitignore)
├── headshot-landing/       # 前端(Next.js 16 + React 19 + TS + Tailwind v4)
├── ios/                    # iOS 原生 SwiftUI app(XcodeGen 工程 + Apple IAP + AppStore 提交材料,连生产 /api/v2;2026-07-24 从 Codex 副本合并)
├── FLASHSHOT_OVERVIEW.md   # 项目叙述(已订正)
└── ARCHITECTURE_REVIEW.md  # 2026-07-02 架构评估快照
```

`headshot-landing/AGENTS.md` 是前端专属的 Next.js 版本注意事项,本文件不重复。

---

## 后端分层架构(`headshot_pipeline/server/`)

```
HTTP/WS 路由   router_sessions / router_jobs / router_postprocess / router_ws
              router_payment(Paddle Web)/ router_portrait_v2(Apple IAP + themes/projects, /api/v2)/ router_admin(内部支持)
     │
编排          job_queue.py        单 worker 队列 + session 生命周期(SQLite 持久化)+ 交付去重/标注/高清化
              gemini_worker.py    候选 pipeline 编排(hero / full_set 共用 _execute_candidate_pipeline)
              portrait_*.py       portrait v2 域(catalog/domain/storage)+ inspiration_analyzer(灵感图分析)
     │
决策          evaluation/policy_engine.py   PolicyEngine —— 自适应(预算/风险/反馈打分),runtime 活跃路径
              evaluation/agent_router.py    AgentRouter —— 规则状态机(PolicyEngine 的内核 + 测试用)
              evaluation/recovery_planner.py + failure_taxonomy.py  失败分类 → 恢复策略
     │
评估          evaluation/evaluator.py       VLM QA judge + 本地 CV 质检 + InsightFace 身份相似度
              evaluation/set_evaluator.py   整套(多分镜)级评估
学习          learning/learning_layer.py    反馈→阈值标定(SQLite,保守有界增量)
     │
生成          generation/gateway.py         ImageGateway —— 业务统一入口
              generation/smart_router.py    SmartModelRouter —— 任务感知模型路由(cost×identity×latency)
              generation/providers.py       SiliconFlow / OpenRouter / Chrome(legacy) 三个 Provider
              openrouter_image_client.py    OpenRouter 图像 REST 客户端
修复          repair/identity_repair.py     FaceSwapRepair(inswapper_128,本地换脸)
              repair/framing_repair.py + sharpness_repair.py  构图/锐度修复
交付          delivery/ + delivery_policy.py + delivery_label.py(AI 元数据标注)+ postprocess.py + upscaler.py
     │
支付          apple_iap.py + apple_identity.py  Apple IAP 收据验签(App Store Server API + JWT 身份)
              payment.py                        Paddle HMAC webhook(Web 可选路径)
基础设施      config.py  security.py(owner-token+限流)  storage.py(SQLite)
              models.py(TIER_LIMITS/Pydantic)  shot_planner.py  ...
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
- **支付提权路径(两条,均服务端验签、前端只读)**:
  - **Apple IAP(主路径,iOS)** —— `apple_iap.py` 用 Apple App Store Server API + `config/apple-root-certs/` 验签收据;`/api/v2/auth/apple` 签发身份(`apple_identity.py`)。
  - **Paddle Web(可选)** —— HMAC-SHA256 webhook(`router_payment.py::verify_paddle_signature`),Paddle 作 Merchant of Record、price_id 服务端锁定。
  - `/api/launch-ready` 中 Paddle 标 `required_for_ios_launch:false`,故 **iOS 上线不依赖 Paddle**。
- **数据保留**:7 天后删除源/生成文件(每小时 retention sweep)。
- 上传图片做魔术数字节校验;路径用 `safe_id` 防穿越。

---

## 如何运行

```bash
# 后端测试(用隔离 DATA_DIR,绝不碰生产 DB)
cd headshot_pipeline
DATA_DIR="$(mktemp -d)" .venv/bin/python -m pytest tests/ -q   # 434 passed;3 failed 为生产既存代码/测试漂移(test_image_gateway 的 model 断言、test_health_readiness 的 generation_ready、test_shot_planner 缺 init_db),非本仓库引入

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

生产在 **`38.76.165.9`**(HK 共享 VPS,hostname `C202605310147445`,Ubuntu 5.4),源码在 `/opt/flashshot`(**非 git 部署快照**)。入口是 **nginx**(部署脚本 `/root/deploy-nginx.sh`),经 Cloudflare 橙云。

```
:443/:80 → nginx  flashshot.top → 127.0.0.1:8001 uvicorn server.main:app(FastAPI)
                                 → 127.0.0.1:3001 Next.js
HTTPS egress → api.siliconflow.cn(生成)+ Apple App Store Server(IAP 验签)
```

> 部署细节见 [`headshot_pipeline/deploy/DEPLOY_NGINX.md`](headshot_pipeline/deploy/DEPLOY_NGINX.md);`deploy/legacy/` 下为旧(US/Caddy、launchd、cloudflared)方案,已归档勿用。
> **共享主机硬约束**:这台机器还跑着 grafana.recorrect.cn、plausible.recorrect.cn 等(见 [[recorrect-no-touch]])。**所有改动必须 additive,绝不扰动 co-tenant**。回滚只移除 FlashShot 自身。

---

## 关键约定

- **星型生成,反漂移**:见上。
- **共享主机 additive**:部署、Caddy、systemd 都只增不改不删他者。
- **`.env` 不进仓库**;dev/prod 密钥分离,rsync 部署时 `--exclude .env`。
- **legacy/ 不被产品 import**:仅 chrome-era 死代码归档(见 `legacy/README.md`)。
