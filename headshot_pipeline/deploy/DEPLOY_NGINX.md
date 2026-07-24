# FlashShot 部署 · nginx on 38.76.165.9

> 2026-07-24 订正。**本文件反映生产真实拓扑。** `deploy/legacy/` 下的 US/Caddy、launchd、cloudflared 方案均已过时。
> 结构总览以仓库根 [`../../CLAUDE.md`](../../CLAUDE.md) 为准。

## 源码真相

- **生产源站**:`38.76.165.9`(HK,hostname `C202605310147445`,Ubuntu 5.4),源码在 `/opt/flashshot`——**非 git 部署快照**(无 commit 历史),独立演进。
- **部署脚本**:生产上 `/root/deploy-nginx.sh`(`4408` 字节)是 nginx 配置/部署的权威步骤来源。本文件描述经核实的拓扑;**逐步命令以该脚本为准**(改部署前先 SSH 读它)。
- **本 git 仓库**已于 2026-07-24 用 `/opt/flashshot` 快照整体对齐;但生产仍从 box 上的文件部署,不从本仓库部署。

## 拓扑(经核实)

```
浏览器 → Cloudflare(橙云)→ 38.76.165.9  nginx  :443/:80
                                 flashshot.top → 127.0.0.1:8001  uvicorn server.main:app  (FastAPI 后端)
                                               → 127.0.0.1:3001  Next.js                  (前端,对应 headshot-landing/)
        HTTPS egress → api.siliconflow.cn(生成)+ Apple App Store Server(IAP 收据验签)
```

- 入口是 **nginx**(不是 Caddy;旧 Caddy 方案见 `legacy/overseas-vps/`)。
- 后端进程:`uvicorn server.main:app`,bind `127.0.0.1:8001`。
- 前端进程:Next.js,bind `127.0.0.1:3001`。
- 配置:`/opt/flashshot/.env`(`APP_ENVIRONMENT=production`、`PAYMENT_MOCK_ENABLED=0`、`GEMINI_BACKEND=siliconflow`、`SILICONFLOW_API_KEY` set、Apple IAP 凭据 `APPLE_BUNDLE_ID`/`APPLE_APP_ID`/`APPLE_IAP_*` 等)。

## 上线门 / 健康

- `GET /api/health` —— 进程存活 + `generation_ready` + `apple_iap_ready`。
- `GET /api/ready` —— 生成就绪 + provider 自检。
- `GET /api/launch-ready` —— 付费上线门(经 CF 实测 = `launch_ready`,IAP 已配,Paddle `required_for_ios_launch:false`)。

## 共享主机硬约束(见 `recorrect-no-touch`)

38.76.165.9 同机跑 `grafana.recorrect.cn`、`plausible.recorrect.cn`、`:8888/:8889` 等服务。**FlashShot 所有改动必须 additive:nginx 只加自己的 server block,绝不改/重排他站配置;systemd/进程只动 FlashShot 自身。** 改 nginx 前先 `nginx -t` 校验。回滚只移除 FlashShot 自身。

## 更新生产代码的流程(要点)

1. SSH `root@38.76.165.9`(id_ed25519,ssh-agent,非交互可达)。SSH 偶发 "Connection closed by port 22";可靠模式:本地写脚本 → base64 → `ssh root@38.76.165.9 "echo $B64 | base64 -d | bash"`。
2. `/opt/flashshot` 是部署目录;如需推新代码,rsync 到此(**必须 `--exclude .env`**,dev/prod 密钥分离)。
3. 重启后端:`systemctl restart <flashshot-api 服务名>`(以 `/root/deploy-nginx.sh` 与实际 systemd unit 为准)。
4. 验证:`curl http://127.0.0.1:8001/api/launch-ready`(源站直连)→ `launch_ready`。

> ⚠️ 改付费/上线相关代码前务必确认:改的是 `/opt/flashshot`(生产真相),不是这个对齐快照仓库。
