"""Smoke test for Phase 5 API integration."""

import sys

sys.stdout.reconfigure(line_buffering=True)

print("Testing Phase 5 API imports...")

# Test 1: Import API modules
from kg_agent.api.routes import multi_agent
from kg_agent.api.routes.agui import router as agui_router, AGUI_AVAILABLE

print("✅ API route modules import successfully")

# Test 2: Check router prefixes
assert multi_agent.router.prefix == "/multi-agent"
print("✅ Multi-agent router has correct prefix")

# Test 3: Check endpoints exist
routes = [route.path for route in multi_agent.router.routes]
# Routes include the router prefix
expected_routes = ["/multi-agent/chat", "/multi-agent/chat/stream", "/multi-agent/session",
                   "/multi-agent/session/{session_id}", "/multi-agent/sessions", "/multi-agent/status"]
for expected in expected_routes:
    assert expected in routes, f"Missing route: {expected}, available: {routes}"
print(f"✅ All expected routes present: {len(expected_routes)} routes")

# Test 4: Check AG-UI availability
print(f"✅ AG-UI available: {AGUI_AVAILABLE}")

# Test 5: Check agui routes
agui_routes = [route.path for route in agui_router.routes]
assert "/agui/info" in agui_routes
print("✅ AG-UI info endpoint exists")

# Test 6: Import from main app
from kg_agent.api.main import app

print("✅ Main app imports successfully")

# Test 7: Check that multi-agent routes are registered
app_routes = [route.path for route in app.routes]
multi_agent_routes = [r for r in app_routes if "/multi-agent" in r]
assert len(multi_agent_routes) > 0, "Multi-agent routes not registered"
print(f"✅ Multi-agent routes registered in app: {len(multi_agent_routes)} routes")

# Test 8: Test model validation
from kg_agent.api.routes.multi_agent import (
    ChatRequest,
    ChatResponse,
    SessionCreateRequest,
    SessionResponse,
    MultiAgentStatusResponse,
)

chat_req = ChatRequest(message="Hello", session_id="test", user_id="user1")
assert chat_req.message == "Hello"
print("✅ Request models work")

# Test 9: Check response models
chat_resp = ChatResponse(
    response="Test response",
    session_id="test",
    thinking_steps=[],
    execution_path=["manager"],
    total_llm_calls=1,
)
assert chat_resp.response == "Test response"
print("✅ Response models work")

# Test 10: Check session response model
session_resp = SessionResponse(
    id="session-123",
    user_id="user1",
    created_at="2025-01-01T00:00:00",
    last_active="2025-01-01T00:00:00",
    message_count=5,
    metadata={},
)
assert session_resp.message_count == 5
print("✅ Session response model works")

print("")
print("=" * 50)
print("🎉 Phase 5 all tests pass!")
print("=" * 50)

print("""
API Endpoints Available:
  POST /api/v1/multi-agent/chat        - Chat with multi-agent system
  POST /api/v1/multi-agent/chat/stream - Stream chat response (SSE)
  POST /api/v1/multi-agent/session     - Create session
  GET  /api/v1/multi-agent/session/{id}- Get session info
  DELETE /api/v1/multi-agent/session/{id} - Delete session
  GET  /api/v1/multi-agent/sessions    - List sessions
  GET  /api/v1/multi-agent/status      - System status
  GET  /api/v1/agui/info               - AG-UI integration info
""")

if AGUI_AVAILABLE:
    print("""AG-UI Endpoint (CopilotKit):
  POST /agents/kg_multi_agent          - AG-UI protocol endpoint
""")

