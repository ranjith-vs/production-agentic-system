#!/bin/sh
set -e

ADMIN_URL="http://apisix:9180/apisix/admin"
ADMIN_KEY="edd1c9f034335f136f87ad84b625c8f1"

echo "waiting for apisix admin api..."
until curl -s -o /dev/null -w "%{http_code}" "$ADMIN_URL/routes" -H "X-API-KEY: $ADMIN_KEY" | grep -q 200; do
  sleep 2
done

# Upstream/Service/Route split (see README "Route vs Service vs Upstream"):
# one Upstream (the backend pool), two Services (one per gateway behavior,
# each bundling its plugins + rate limit), thin Routes on top that just match
# a path and point at a Service. Adding a second provider later means a new
# Upstream + Service, not copy-pasting plugin config into more Routes.
#
# limit-count is intentionally generous (600/min, not a "reasonable-sounding"
# 30/min): a single /chat turn here fans out into 4-6 real OpenAI calls
# (embed for cache-check, planner, specialist, judge, embed for cache-store -
# more with tool calls or a 2-specialist turn), all sharing this one route.
# A tight per-IP limit that looks fine for "one user, one request" starves a
# multi-agent architecture's normal traffic almost immediately - learned this
# the hard way when a handful of real chat turns 429'd right after a rate
# limit smoke test, see README.

curl -sf -X PUT "$ADMIN_URL/upstreams/openai-upstream" \
  -H "X-API-KEY: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "roundrobin",
    "scheme": "https",
    "pass_host": "node",
    "nodes": {
      "api.openai.com:443": 1
    }
  }'
echo ""
echo "apisix upstream 'openai-upstream' created"

curl -sf -X PUT "$ADMIN_URL/services/openai-chat-svc" \
  -H "X-API-KEY: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "upstream_id": "openai-upstream",
    "plugins": {
      "ai-proxy": {
        "auth": {
          "header": {
            "Authorization": "Bearer '"$OPENAI_API_KEY"'"
          }
        },
        "model": {
          "provider": "openai",
          "name": "'"$OPENAI_MODEL"'"
        },
        "override": {
          "endpoint": "https://api.openai.com/v1/chat/completions"
        },
        "ssl_verify": false
      },
      "limit-count": {
        "count": 600,
        "time_window": 60,
        "key_type": "var",
        "key": "remote_addr",
        "rejected_code": 429
      }
    }
  }'
echo ""
echo "apisix service 'openai-chat-svc' created"

curl -sf -X PUT "$ADMIN_URL/routes/openai-chat" \
  -H "X-API-KEY: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "/v1/chat/completions",
    "methods": ["POST"],
    "service_id": "openai-chat-svc"
  }'
echo ""
echo "apisix route 'openai-chat' created"

# Raw passthrough service: the ai-proxy plugin above only supports plain
# user/assistant/system *text* messages (it 400s on tool_calls / tool-role
# messages), so anything that needs real OpenAI API fidelity - function/tool
# calling, embeddings, multi-modal - goes through this one instead. Same
# gateway, same upstream, same centralized key, no body-schema normalization.
curl -sf -X PUT "$ADMIN_URL/services/openai-raw-svc" \
  -H "X-API-KEY: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "upstream_id": "openai-upstream",
    "plugins": {
      "proxy-rewrite": {
        "regex_uri": ["^/openai/(.*)", "/$1"],
        "headers": {
          "set": {
            "Authorization": "Bearer '"$OPENAI_API_KEY"'"
          }
        }
      },
      "limit-count": {
        "count": 600,
        "time_window": 60,
        "key_type": "var",
        "key": "remote_addr",
        "rejected_code": 429
      }
    }
  }'
echo ""
echo "apisix service 'openai-raw-svc' created"

curl -sf -X PUT "$ADMIN_URL/routes/openai-raw" \
  -H "X-API-KEY: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "/openai/*",
    "methods": ["POST", "GET"],
    "service_id": "openai-raw-svc"
  }'
echo ""
echo "apisix route 'openai-raw' created"
