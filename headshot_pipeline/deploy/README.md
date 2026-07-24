# deploy/

FlashShot 部署文档与脚本。

## 当前(权威)

- **[`DEPLOY_NGINX.md`](DEPLOY_NGINX.md)** —— 生产真实拓扑:nginx on `38.76.165.9`,`:443/:80 → 8001/3001`,Cloudflare 橙云,Apple IAP。

## 归档(`legacy/`,勿照此部署)

`legacy/` 下为**历史部署方案,均已过时,不再反映生产**:
- `overseas-vps/` —— 早期 **US VPS + Caddy** 方案(`Caddyfile.flashshot`、`flashshot-{api,web}.service`、`preflight.sh`、`DEPLOY_OVERSEAS_VPS.md`)。生产早已迁到 HK nginx。
- `com.shanxiang.*.plist` —— macOS **launchd** + Chrome 后端时代的本地常驻配置(chrome-era,已弃用)。
- `cloudflared-config.yml` —— 旧 cloudflared 隧道配置。
- `DEPLOY.md` / `setup.sh` / `status.sh` / `stop.sh` —— 旧版部署/运维脚本。

> 清理于 2026-07-24。保留作历史参考;如确认不再需要可整目录删除。
