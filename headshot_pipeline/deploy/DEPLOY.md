# 闪像 (ShanXiang) — Mac Mini 部署指南

## 概述

将 Mac Mini (M2/M2 Pro) 配置为生产服务器，通过 Cloudflare Tunnel 提供公网 HTTPS 访问。

### 架构

```
用户浏览器
    ↓ HTTPS
Cloudflare Tunnel
    ↓
Next.js (:3000) ──rewrite──> FastAPI (:8000) ──CDP──> Chrome (:9222) ──> Gemini
  首页 + /create               API + WebSocket       Selenium 自动化       AI 生成
```

三进程全部由 macOS launchd 管理，开机自启、崩溃自动重启。

---

## Phase 1: Mac Mini 环境准备

### 1.1 基础软件安装

```bash
# Xcode CLI 工具
xcode-select --install

# Homebrew（如果没装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11+
brew install python@3.11

# Node.js 20 LTS
brew install node@20

# Google Chrome
brew install --cask google-chrome

# Cloudflare Tunnel
brew install cloudflared
```

### 1.2 克隆项目

```bash
cd ~/Desktop
git clone <repo-url> mvp_1
```

目录结构：

```
~/Desktop/mvp_1/
├── headshot_pipeline/       # 后端 (FastAPI + Chrome CDP)
│   ├── server/              # FastAPI 应用
│   ├── persistent_client.py # Gemini 自动化客户端
│   ├── prompts.json         # 风格模板数据
│   ├── templates/           # 34 张模板图
│   ├── requirements.txt     # Python 依赖
│   └── deploy/              # 部署脚本 + launchd 配置
└── headshot-landing/        # 前端 (Next.js)
    ├── src/
    ├── public/images/       # 模板图片副本
    └── package.json
```

### 1.3 安装依赖

```bash
# Python
cd ~/Desktop/mvp_1/headshot_pipeline
pip3 install -r requirements.txt

# Node.js
cd ~/Desktop/mvp_1/headshot-landing
npm install
```

---

## Phase 2: Cloudflare Tunnel 配置

### 2.1 登录 Cloudflare

```bash
cloudflared tunnel login
```

会打开浏览器让你授权。选择你的域名（需要先将域名添加到 Cloudflare Dashboard）。

### 2.2 创建隧道

```bash
cloudflared tunnel create shanxiang
```

输出会显示 Tunnel ID（形如 `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`），记下来。

### 2.3 添加 DNS 记录

```bash
# 将你的域名指向隧道（按实际域名替换）
cloudflared tunnel route dns shanxiang shanxiang.ai
cloudflared tunnel route dns shanxiang www.shanxiang.ai
```

### 2.4 配置隧道

创建 `~/.cloudflared/config.yml`：

```yaml
tunnel: <你的 TUNNEL_ID>
credentials-file: /Users/<你的用户名>/.cloudflared/<TUNNEL_ID>.json

ingress:
  # 所有流量 → Next.js (:3000)
  # Next.js 的 rewrite 把 /api/* 和 /ws/* 代理到 FastAPI (:8000)
  - hostname: shanxiang.ai
    service: http://localhost:3000
  - hostname: www.shanxiang.ai
    service: http://localhost:3000
  - service: http_status:404
```

### 2.5 测试隧道

```bash
cloudflared tunnel run shanxiang
```

另开终端确认可访问：`curl https://shanxiang.ai`

---

## Phase 3: 一键部署

### 使用部署脚本

```bash
cd ~/Desktop/mvp_1/headshot_pipeline

# 一键部署（安装依赖 + 构建前端 + 生成配置 + 启动服务）
./deploy/setup.sh
```

脚本会自动：
1. 检查前置依赖（Python, Node, Chrome, cloudflared）
2. 安装 Python 依赖
3. 构建 Next.js 生产版本
4. 生成 4 个 launchd plist（替换实际路径）
5. 按顺序启动：Chrome → FastAPI → Next.js → Tunnel
6. 健康检查

### 手动管理

```bash
# 查看服务状态
./deploy/status.sh

# 停止所有服务
./deploy/stop.sh

# 重新启动
./deploy/stop.sh && ./deploy/setup.sh
```

### 直接使用 launchctl

```bash
# 启动单个服务
launchctl load ~/Library/LaunchAgents/com.shanxiang.chrome.plist
launchctl load ~/Library/LaunchAgents/com.shanxiang.api.plist
launchctl load ~/Library/LaunchAgents/com.shanxiang.web.plist
launchctl load ~/Library/LaunchAgents/com.shanxiang.tunnel.plist

# 停止单个服务
launchctl unload ~/Library/LaunchAgents/com.shanxiang.api.plist

# 重启单个服务
launchctl unload ~/Library/LaunchAgents/com.shanxiang.api.plist
launchctl load ~/Library/LaunchAgents/com.shanxiang.api.plist
```

---

## Phase 4: 首次 Gemini 登录

> **这是唯一需要手动操作的步骤。**

部署脚本启动 Chrome 后，Chrome 会以 CDP 调试模式打开一个窗口：

1. 在 Mac Mini 屏幕上找到 Chrome 窗口
2. 地址栏输入 `gemini.google.com`
3. 登录你的 Google 账号
4. 确认可以正常对话

登录完成后，Chrome 的 cookie 会保存在 `.chrome_profile/` 目录，之后重启不需要重新登录。

---

## Phase 5: Mac Mini 系统设置

### 防止休眠

```
系统设置 → 节能 → 防止自动休眠（开启）
```

显示器可以关闭，但 Mac 不能睡眠。睡眠会杀掉所有进程。

### 自动登录

```
系统设置 → 用户与群组 → 自动登录（开启）
```

确保重启后桌面会话可用（Chrome 需要 GUI 桌面环境）。

### 远程管理（可选）

```bash
# 开启 SSH
sudo systemsetup -setremotelogin on

# 开启屏幕共享（VNC）
# 系统设置 → 通用 → 共享 → 屏幕共享（开启）
```

---

## 服务端口一览

| 服务 | 端口 | 用途 |
|------|------|------|
| Chrome CDP | 9222 | Selenium 通过 CDP 控制 Chrome |
| FastAPI | 8000 | API + WebSocket，仅本地访问 |
| Next.js | 3000 | 前端页面，Cloudflare Tunnel 连接这里 |

---

## 日志

所有服务日志在 `/tmp/`：

```bash
# 查看所有日志
tail -f /tmp/shanxiang-*.log

# 单独看 API 日志
tail -f /tmp/shanxiang-api.log

# API 错误日志
tail -f /tmp/shanxiang-api-err.log

# Chrome 日志
tail -f /tmp/shanxiang-chrome.log

# Tunnel 日志
tail -f /tmp/shanxiang-tunnel.log
```

---

## 健康检查

```bash
# Chrome CDP
curl http://localhost:9222/json/version

# FastAPI
curl http://localhost:8000/api/health

# Next.js
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# 应返回 200

# 公网
curl -s -o /dev/null -w "%{http_code}" https://shanxiang.ai
# 应返回 200
```

---

## 更新部署

当代码有更新时：

```bash
cd ~/Desktop/mvp_1

# 拉取最新代码
cd headshot_pipeline && git pull
cd ../headshot-landing && git pull

# 重新构建前端（如果前端有改动）
cd ~/Desktop/mvp_1/headshot-landing
npm run build

# 重启服务
cd ~/Desktop/mvp_1/headshot_pipeline
./deploy/stop.sh && ./deploy/setup.sh
```

---

## 故障排查

### Chrome 无法启动

- 确认 `/Applications/Google Chrome.app` 存在
- 检查是否有其他 Chrome 实例占用了 9222 端口：
  ```bash
  lsof -i :9222
  kill <PID>
  ```
- 查看 Chrome 错误日志：`cat /tmp/shanxiang-chrome-err.log`

### FastAPI 连不上 Chrome

- 确认 Chrome 在运行：`curl http://localhost:9222/json/version`
- 如果 Chrome 崩溃了，重启它：
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.shanxiang.chrome.plist
  launchctl load ~/Library/LaunchAgents/com.shanxiang.chrome.plist
  sleep 3
  launchctl unload ~/Library/LaunchAgents/com.shanxiang.api.plist
  launchctl load ~/Library/LaunchAgents/com.shanxiang.api.plist
  ```

### Gemini 生成失败

- 打开 Chrome 窗口，确认 `gemini.google.com` 是否还保持登录
- 如果显示登录页面，重新登录 Google 账号
- 检查 API 日志：`tail -50 /tmp/shanxiang-api-err.log`

### 隧道无法访问

- 确认 cloudflared 在运行：`pgrep -f cloudflared`
- 检查隧道日志：`tail -50 /tmp/shanxiang-tunnel-err.log`
- 确认 DNS 记录：`dig shanxiang.ai`

### WebSocket 连接失败

- 确认 Next.js rewrite 包含 `/ws/*` 路由（`next.config.ts`）
- Cloudflare 默认支持 WebSocket，不需要额外配置
- 检查浏览器控制台是否有 `wss://` 连接错误

---

## 完整部署清单

- [ ] Mac Mini 安装 brew / python3 / node / chrome / cloudflared
- [ ] 项目 clone 到 `~/Desktop/mvp_1/`
- [ ] `pip3 install -r requirements.txt`
- [ ] `npm install && npm run build`
- [ ] Cloudflare Tunnel: login → create → route dns → config.yml
- [ ] 运行 `./deploy/setup.sh`
- [ ] Chrome 窗口登录 gemini.google.com
- [ ] 防止 Mac 休眠 + 开启自动登录
- [ ] 访问 `https://shanxiang.ai` 走完整流程验证
- [ ] 重启 Mac Mini 验证服务自动恢复
