# FlashShot · 闪像

**AI 写真交付系统**("AI Portrait Director"):用户上传 4–6 张自拍 → 几秒出一张"像本人 + 好看"的 **Hero 预览图**(Aha Moment)→ 付费解锁(**Apple IAP** 主路径 / Paddle Web 可选)→ 稳定交付一整套精选写真。北极星 = 用户最终下载的合格写真数量。

> **结构真相以 [`CLAUDE.md`](CLAUDE.md) 为准**(后端分层、支付模型、部署拓扑、定价)。本 README 只做落地导航。

---

## 仓库布局

```
mvp_1/
├── headshot_pipeline/   后端 FastAPI(产品核心)—— server/ 分层见 CLAUDE.md
├── headshot-landing/    前端 Next.js 16 + React 19 + TS + Tailwind v4(对应生产 web/)
└── CLAUDE.md            ⭐ 当前结构真相(IAP / nginx / 437 测试 / prod=38.76.165.9)
```

历史文档:`FLASHSHOT_OVERVIEW.md`(产品叙述)、`ARCHITECTURE_REVIEW.md`(2026-07-02 评估快照)——均已让位 `CLAUDE.md`,仅供背景。

---

## 快速开始

```bash
# 后端(配置走 .env,见 server/.env.example;GEMINI_BACKEND=siliconflow 需 SILICONFLOW_API_KEY)
cd headshot_pipeline
.venv/bin/python -m pytest tests/ -q          # 434 passed(3 个 xfail 为生产既存漂移,见 CLAUDE.md)
.venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port 8001

# 前端
cd headshot-landing
npm run dev      # 或 npm run build && npm start
```

健康/就绪:`GET /api/health` · `GET /api/ready` · `GET /api/launch-ready`(付费上线门)。

---

## 部署(LIVE)

生产源码真相在 **`38.76.165.9:/opt/flashshot`**(HK,nginx,Apple IAP,`/api/launch-ready` 绿)——**非 git 部署快照**。本仓库 2026-07-24 已用生产快照整体对齐。真实部署步骤见 [`headshot_pipeline/deploy/DEPLOY_NGINX.md`](headshot_pipeline/deploy/DEPLOY_NGINX.md)(`deploy/` 下其余为历史归档)。

> ⚠️ 共享主机:38.76.165.9 同机跑 recorrect 等服务,**所有改动必须 additive**。

---

## 关键约定

- **星型生成,反漂移**:每个分镜从原始 Identity Pack 独立生成,禁链式 edit。
- **支付两条服务端验签路径**:Apple IAP(主)/ Paddle Web(可选),前端只读。
- **`.env` 不进仓库**;dev/prod 密钥分离。
- 改付费/上线相关代码前,先确认改的是**生产真相**(38.76.165.9),不是这个对齐快照。
