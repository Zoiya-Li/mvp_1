#!/bin/bash
# =============================================================================
# 闪像 (ShanXiang) — Mac Mini 一键部署脚本
#
# 使用方法:
#   chmod +x deploy/setup.sh
#   ./deploy/setup.sh
#
# 前置条件:
#   - macOS (M2/M2 Pro Mac Mini)
#   - 已安装: Homebrew, Python 3.11+, Node.js 20+, Google Chrome
#   - 项目已 clone 到 ~/Desktop/mvp_1/
# =============================================================================
set -euo pipefail

# ---- 配置 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
LANDING_DIR="$(dirname "$PIPELINE_DIR")/headshot-landing"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

PYTHON="$(which python3)"
NODE="$(which node)"
NPM="$(which npm)"
CLOUDFLARED="$(which cloudflared 2>/dev/null || echo '')"
CHROME_PROFILE="$PIPELINE_DIR/.chrome_profile"

echo ""
echo "============================================"
echo "  闪像 (ShanXiang) — Mac Mini 部署"
echo "============================================"
echo ""
echo "  Pipeline:  $PIPELINE_DIR"
echo "  Landing:   $LANDING_DIR"
echo "  Python:    $PYTHON"
echo "  Node:      $NODE"
echo ""

# ---- 检查前置条件 ----
check_deps() {
    echo "📦 检查依赖..."

    if ! command -v python3 &>/dev/null; then
        echo "  ❌ Python3 未安装。运行: brew install python@3.11"
        exit 1
    fi

    if ! command -v node &>/dev/null; then
        echo "  ❌ Node.js 未安装。运行: brew install node@20"
        exit 1
    fi

    if [ ! -d "/Applications/Google Chrome.app" ]; then
        echo "  ❌ Google Chrome 未安装。运行: brew install --cask google-chrome"
        exit 1
    fi

    if [ -z "$CLOUDFLARED" ]; then
        echo "  ⚠️  cloudflared 未安装。运行: brew install cloudflared"
        echo "     隧道服务将跳过，可稍后手动配置。"
    fi

    echo "  ✓ 依赖检查通过"
}

# ---- 安装 Python 依赖 ----
install_python_deps() {
    echo ""
    echo "🐍 安装 Python 依赖..."
    pip3 install -r "$PIPELINE_DIR/requirements.txt"
    echo "  ✓ Python 依赖安装完成"
}

# ---- 构建 Next.js ----
build_frontend() {
    echo ""
    echo "🌐 构建 Next.js 前端..."
    cd "$LANDING_DIR"

    if [ ! -d "node_modules" ]; then
        echo "  安装 npm 依赖..."
        npm install
    fi

    npm run build
    echo "  ✓ 前端构建完成"
}

# ---- 生成 launchd plist ----
generate_plists() {
    echo ""
    echo "⚙️  生成 launchd 服务配置..."
    mkdir -p "$LAUNCH_AGENTS"

    local SYSTEM_PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"

    # Chrome
    sed -e "s|__CHROME_PROFILE__|$CHROME_PROFILE|g" \
        "$SCRIPT_DIR/com.shanxiang.chrome.plist" \
        > "$LAUNCH_AGENTS/com.shanxiang.chrome.plist"

    # FastAPI
    sed -e "s|__PYTHON__|$PYTHON|g" \
        -e "s|__PIPELINE_DIR__|$PIPELINE_DIR|g" \
        -e "s|__PATH__|$SYSTEM_PATH|g" \
        "$SCRIPT_DIR/com.shanxiang.api.plist" \
        > "$LAUNCH_AGENTS/com.shanxiang.api.plist"

    # Next.js
    sed -e "s|__NODE__|$NODE|g" \
        -e "s|__LANDING_DIR__|$LANDING_DIR|g" \
        -e "s|__PATH__|$SYSTEM_PATH|g" \
        "$SCRIPT_DIR/com.shanxiang.web.plist" \
        > "$LAUNCH_AGENTS/com.shanxiang.web.plist"

    # Cloudflare Tunnel
    if [ -n "$CLOUDFLARED" ]; then
        sed -e "s|__CLOUDFLARED__|$CLOUDFLARED|g" \
            "$SCRIPT_DIR/com.shanxiang.tunnel.plist" \
            > "$LAUNCH_AGENTS/com.shanxiang.tunnel.plist"
    fi

    echo "  ✓ plist 文件已生成到 $LAUNCH_AGENTS/"
}

# ---- 启动服务 ----
start_services() {
    echo ""
    echo "🚀 启动服务..."

    # 先卸载（如果已存在）
    for svc in chrome api web tunnel; do
        if [ -f "$LAUNCH_AGENTS/com.shanxiang.$svc.plist" ]; then
            launchctl unload "$LAUNCH_AGENTS/com.shanxiang.$svc.plist" 2>/dev/null || true
        fi
    done

    # 按顺序启动: Chrome → API → Web → Tunnel
    launchctl load "$LAUNCH_AGENTS/com.shanxiang.chrome.plist"
    echo "  ✓ Chrome CDP 启动 (端口 9222)"

    sleep 3  # 等 Chrome 完全启动

    launchctl load "$LAUNCH_AGENTS/com.shanxiang.api.plist"
    echo "  ✓ FastAPI 启动 (端口 8000)"

    sleep 2

    launchctl load "$LAUNCH_AGENTS/com.shanxiang.web.plist"
    echo "  ✓ Next.js 启动 (端口 3000)"

    if [ -f "$LAUNCH_AGENTS/com.shanxiang.tunnel.plist" ]; then
        launchctl load "$LAUNCH_AGENTS/com.shanxiang.tunnel.plist"
        echo "  ✓ Cloudflare Tunnel 启动"
    fi
}

# ---- 健康检查 ----
health_check() {
    echo ""
    echo "🏥 健康检查..."

    echo -n "  Chrome CDP (9222): "
    if curl -s http://localhost:9222/json/version >/dev/null 2>&1; then
        echo "✓"
    else
        echo "✗ (可能还在启动中)"
    fi

    echo -n "  FastAPI (8000): "
    if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo "✓"
    else
        echo "✗ (可能还在启动中)"
    fi

    echo -n "  Next.js (3000): "
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        echo "✓"
    else
        echo "✗ (可能还在启动中)"
    fi
}

# ---- 主流程 ----
main() {
    check_deps
    install_python_deps
    build_frontend
    generate_plists
    start_services
    health_check

    echo ""
    echo "============================================"
    echo "  ✅ 部署完成！"
    echo ""
    echo "  本地访问: http://localhost:3000"
    echo "  API 状态: http://localhost:8000/api/health"
    echo ""
    echo "  ⚠️  首次部署需要手动操作:"
    echo "  1. 打开 Chrome 窗口"
    echo "  2. 访问 gemini.google.com"
    echo "  3. 登录 Google 账号"
    echo ""
    echo "  如果使用 Cloudflare Tunnel:"
    echo "  1. 运行: cloudflared tunnel login"
    echo "  2. 运行: cloudflared tunnel create shanxiang"
    echo "  3. 运行: cloudflared tunnel route dns shanxiang <your-domain>"
    echo "  4. 编辑 ~/.cloudflared/config.yml"
    echo ""
    echo "  日志: tail -f /tmp/shanxiang-*.log"
    echo "  停止: ./deploy/stop.sh"
    echo "============================================"
}

main "$@"
