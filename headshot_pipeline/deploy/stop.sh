#!/bin/bash
# 停止所有闪像服务
set -euo pipefail

LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "⏹ 停止闪像服务..."

for svc in chrome api web tunnel; do
    plist="$LAUNCH_AGENTS/com.shanxiang.$svc.plist"
    if [ -f "$plist" ]; then
        launchctl unload "$plist" 2>/dev/null && echo "  ✓ $svc 已停止" || echo "  - $svc 未运行"
    fi
done

echo "完成。"
