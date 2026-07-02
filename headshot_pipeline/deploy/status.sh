#!/bin/bash
# 查看闪像服务状态
set -euo pipefail

echo "🔍 闪像服务状态"
echo ""

echo "  Chrome CDP (9222):"
if curl -s http://localhost:9222/json/version 2>/dev/null | grep -q "Browser"; then
    echo "    ✓ 运行中"
    curl -s http://localhost:9222/json/version 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    版本: {d.get(\"Browser\",\"?\")}')"
else
    echo "    ✗ 未运行"
fi

echo ""
echo "  FastAPI (8000):"
health=$(curl -s http://localhost:8000/api/health 2>/dev/null || echo "")
if [ -n "$health" ]; then
    echo "    ✓ 运行中"
    echo "    $health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'    队列: {d.get(\"queue_length\",0)}, 活跃会话: {d.get(\"active_session\",\"无\")}')"
else
    echo "    ✗ 未运行"
fi

echo ""
echo "  Next.js (3000):"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null | grep -q "200"; then
    echo "    ✓ 运行中"
else
    echo "    ✗ 未运行"
fi

echo ""
echo "  Cloudflare Tunnel:"
if pgrep -f "cloudflared.*tunnel.*shanxiang" >/dev/null 2>&1; then
    echo "    ✓ 运行中"
else
    echo "    ✗ 未运行"
fi

echo ""
echo "  launchd 服务:"
for svc in chrome api web tunnel; do
    status=$(launchctl list 2>/dev/null | grep "com.shanxiang.$svc" || echo "")
    if [ -n "$status" ]; then
        echo "    ✓ $svc: $status"
    else
        echo "    - $svc: 未注册"
    fi
done

echo ""
echo "  最近日志:"
echo "  --- API (最后5行) ---"
tail -5 /tmp/shanxiang-api.log 2>/dev/null || echo "  (无日志)"
echo "  --- API 错误 (最后3行) ---"
tail -3 /tmp/shanxiang-api-err.log 2>/dev/null || echo "  (无错误日志)"
