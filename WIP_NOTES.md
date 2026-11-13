# WIP: Claude connector (#31 - #36), backend branch

Stopped mid-task on user request (2026-11-13). Main merged at 21b19a8 before starting.

## Done (written, NOT fully tested)
| Piece | File | State |
|---|---|---|
| `assistant_actions` table + `migrate()` creates it | app/db/schema.sql, app/db/db.py | written; full pytest not run |
| #9 checks split from the insert (`plan_job`, `plan_rerun`, `insert_job`) | app/api/jobs.py | job/rerun/worker/hook tests pass (184) |
| Tool list copied from tools.json | app/assistant/tool_defs.py | equals fixture (checked) |
| .env settings, scrub() | app/assistant/settings.py | written, untested |
| Tool layer (17 tools, envelope, confirm rules, audit rows) | app/assistant/tools.py | written, untested |
| Download store for make_train_package (#35, 24 h) | app/assistant/downloads.py | written, untested |
| MCP Server + McpGate (token 401 / loopback 403) | app/assistant/mcp_server.py | written, untested (mcp 2.3.0 API probed OK) |
| Chat loop (cap, refusal, pause_turn, max_tokens, pending) | app/assistant/chat.py | written, untested |
| Routes #31 - #35 | app/assistant/routes.py | written, untested, NOT registered yet |

## Left to do
1. **app/main.py**: `create_app(..., anthropic_client=None)`; in the lifespan: `load_settings()` ->
   `app.state.assistant_settings`, `downloads.cleanup(data_dir)`, `build_server(app.state)` +
   `build_http_app(...)` -> `app.state.mcp_http_app`, `ChatManager(app.state, settings, factory)` ->
   `app.state.assistant_chat` (factory = the passed client, else `anthropic.Anthropic(api_key=...)`),
   run `server.session_manager.run()` around `yield`. Mount `McpGate(app.state)` at `/mcp` (before
   the static mount); include `app.assistant.routes.router` with prefix `/api`.
2. **app/assistant/mcp_stdio.py**: standalone launcher (imports only mcp, httpx2, stdlib); forwards
   tools/list + tools/call to `--url` with `Authorization: Bearer $AUTOLABEL_MCP_TOKEN` (when set)
   and `X-Autolabel-User: $AUTOLABEL_USER`; upstream down -> `is_error` one-line message.
3. **Tests** (no network, no real key; point `app.assistant.settings.ENV_FILE` at a temp path so the
   worktree's real `.env` is never loaded):
   - tests/test_assistant_helpers.py: test DB built from api_samples.json (project 1, classes 1-4,
     images 10-12 with real small JPEGs, boxes 101-108, jobs 5-12, hook point for image 10).
   - tests/test_assistant_tools.py: every tools.json example (ok, error.code, details, arguments,
     data keys), all-or-nothing approve, assistant_actions rows, `claude:<name>` reviewer,
     no-name MCP write refused. Note: rerun example needs job 12 removed (else job 12 is the source).
   - tests/test_assistant_mcp.py: `mcp.Client(server)` list == tools.json + annotations rule;
     `project_summary` call -> structured_content; HTTP 401 / 403 / loopback ok
     (`TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 5000))`).
   - tests/test_assistant_chat.py: every chat_fake.json scenario with a fake client.
   - tests/test_assistant_secrets.py: `sk-ant-test-LEAKME` + a token never in bodies, headers,
     caplog, assistant_actions or jobs rows.
   - tests/test_assistant_stdio.py: launcher forwards list/call to a fake upstream, both headers.
   - test_db_schema / migrate tests may need `assistant_actions` added.
4. Run the full pytest until green, then commit (not a WIP commit) and report.

## Decisions to mention in the report
- Write tool whose result is > 50,000 chars: the action is done, so the envelope is an error saying
  so and the audit row is `ok` (contract only says "error bad_request").
- Write call with no valid MCP name: audit row actor is `claude:?`.
- `set_hook_point` on an unknown image -> `bad_image_ids` (the #28 code for "image not in project").
- `add_class` with a blank description uses the name (contract: default = the name when absent).
- Destructive MCP tools get `read_only_hint: false` as well as `destructive_hint: true`.
