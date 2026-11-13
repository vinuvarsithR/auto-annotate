# WIP notes - ui, assistant sidebar (#31 - #34)

## Done (in the WIP commit)
- `app/static/assistant.js` (new): sidebar open/close (top-bar "Assistant" button, review screen only,
  remembered in localStorage), #31 status -> "Assistant not set up" panel, reviewer-name check,
  chat (#32) with collapsed tool-call list, stopped notes, links (download_url), Confirm / Cancel
  cards (#33) incl. cancel / expired / 409 / failed result, lost session (404) -> note + resend once,
  New chat (#34), refresh of image list / counts / hook after a write `ok`.
- `api.js`: real calls #31 - #34; mock driven by `docs/fixtures/assistant/chat_fake.json`
  (message = scenario's `request.message`), `?mock=1&chat=off` = no key, extra mock prompt
  "Make a training package." (tools.json example), `api._restartChats()` = server restart.
- `index.html` (sidebar markup, script list, `?v=2026-11-13.2`), `style.css` (sidebar styles).

## Mock click-test so far (Chrome, ?mock=1)
- PASS: button hidden on Projects, shown on Review; name note + Send disabled without a reviewer name.
- PASS: "How many boxes still need review?" -> 1 tool call + reply.
- PASS: "Delete box 106." -> Confirm card; Confirm -> "Confirmed: done.", box 106 gone from mock data;
  replaying the used confirm_id -> 409; Cancel -> "Cancelled"; a newer message expires the card,
  old confirm_id -> 409.
- PASS: "Look at every image." -> "3 tool calls" + tool-limit note.
- PASS: "..." -> "Claude declined this request."; "hi" -> "Claude API error 401: invalid x-api-key [redacted]".
- PASS: unknown text -> help reply; train prompt -> link to /api/assistant/downloads/... (target _blank).
- PASS (ran, log cleared after by New chat): restart + send -> lost-session path; New chat empties the log.

## Left
1. Re-check the lost-session path visibly (`api._restartChats()`, then send: expect the
   "conversation was lost" note and a new session_id) and the open "3 tool calls" list (3rd = "not run").
2. `?mock=1&chat=off`: expect "Assistant not set up" + "Claude chat is off: the server has no ANTHROPIC_API_KEY."
3. Confirm that after Confirm the review page reloaded (box 106 gone from image 11 on screen).
4. Screenshot check of the layout (browser screenshots kept timing out: the Chrome window seemed to be
   in the background, timers were throttled).
5. Run full pytest, then a final commit (not WIP) and report the hash.
6. Real-server click-test waits for the backend merge of #31 - #35.
7. Move these notes into docs/agents/ui.md and delete this file (it sits outside app/static).
