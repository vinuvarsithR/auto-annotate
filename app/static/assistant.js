// assistant.js - the assistant sidebar on the review page (contract "Claude connector", #31 - #34).
// The browser only sends text; the server runs Claude and the tools. Risky actions come back as
// `pending` items and run only when the user presses Confirm here (#33).
"use strict";

// Tools that change data (docs/fixtures/assistant/tools.json, kind "write"). After one of them
// succeeds, the review page reloads the open image, the counts and the hook points.
const WRITE_TOOLS = new Set(["add_class", "run_detection", "rerun_examples", "set_annotation_status",
  "update_annotation", "delete_annotation", "set_hook_point", "make_train_package", "push_to_roboflow"]);
const CHAT_OFF_TEXT = "Claude chat is off: the server has no ANTHROPIC_API_KEY.";
const STOPPED_NOTE = {
  tool_limit: "Stopped: the tool call limit for one message was reached. Ask again to continue.",
  max_tokens: "Stopped: the answer got too long.",
};
const OUTCOME_LABEL = { ok: "ok", error: "error", needs_confirmation: "needs Confirm", not_run: "not run" };

const chat = {
  status: null,       // #31 response, null = not loaded yet
  offText: null,      // why the chat is off (null = on)
  sessionId: null,
  sending: false,
  cards: [],          // open pending cards: {el, confirm_id, done}
};

// ------------------------------------------------------------ open / close
function setAssistantOpen(on) {
  $("assistant").hidden = !on;
  $("assistantBtn").setAttribute("aria-expanded", String(on));
  lsSet("review.assistant", on ? "1" : "0");
  if (on) {
    if (!chat.status) loadAssistantStatus();
    renderAssistant();
    if (!chat.offText) $("asInput").focus();
  }
}

async function loadAssistantStatus() {
  try {
    chat.status = await api.assistantStatus();
    chat.offText = chat.status.chat_available ? null : CHAT_OFF_TEXT;
  } catch (e) {
    console.error(e);
    chat.status = null; // try again next time the sidebar opens
    chat.offText = e instanceof ApiError && e.httpStatus === 404
      ? "This server has no assistant endpoints (#31) yet." : `Assistant status not available: ${e.message}`;
  }
  renderAssistant();
}

function setChatOff(text) {
  chat.offText = text;
  renderAssistant();
}

// ------------------------------------------------------------ rendering
function chatUser() {
  const name = reviewer();
  return name && USER_NAME_RE.test(name) ? name : null;
}

function renderAssistant() {
  const off = !!chat.offText;
  $("asOff").hidden = !off;
  $("asOffText").textContent = chat.offText || "";
  $("asLog").hidden = off;
  $("asForm").hidden = off;
  $("asNewBtn").disabled = off || chat.sending || (!chat.sessionId && !$("asLog").children.length);
  const named = !!chatUser();
  $("asNameNote").hidden = named;
  $("asInput").disabled = off || !named;
  $("asSendBtn").disabled = off || !named || chat.sending || !state.projectId;
  $("asSendBtn").textContent = chat.sending ? "Sending..." : "Send";
  $("asProject").textContent = state.project ? `Project: ${state.project.name}` : "No project";
  for (const c of chat.cards) if (!c.done) c.el.querySelectorAll("button").forEach((b) => (b.disabled = chat.sending || c.busy));
}

function logAppend(el) {
  const log = $("asLog");
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function addLine(kind, text) {
  const d = document.createElement("div");
  d.className = `as-msg ${kind}`;
  d.textContent = text;
  return logAppend(d);
}

// Text with links: http(s) URLs and the server's download paths (make_train_package -> #35).
const LINK_RE = /(https?:\/\/[^\s<>"]+|\/api\/assistant\/downloads\/[A-Za-z0-9_-]+)/g;
function linkified(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const m of text.matchAll(LINK_RE)) {
    let url = m[0];
    const trail = /[.,;:!?)]+$/.exec(url);
    if (trail) url = url.slice(0, -trail[0].length);
    frag.append(text.slice(last, m.index));
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = url;
    frag.append(a);
    last = m.index + url.length;
  }
  frag.append(text.slice(last));
  return frag;
}

const argText = (args) => JSON.stringify(args || {});

function addToolCalls(calls) {
  if (!calls.length) return;
  const d = document.createElement("details");
  d.className = "as-calls";
  const errors = calls.filter((c) => c.outcome === "error").length;
  d.innerHTML = "<summary></summary><ul></ul>";
  d.querySelector("summary").textContent =
    `${calls.length} tool call${calls.length === 1 ? "" : "s"}` + (errors ? ` (${errors} error${errors === 1 ? "" : "s"})` : "");
  for (const c of calls) {
    const li = document.createElement("li");
    li.innerHTML = '<code></code> <span></span>';
    li.querySelector("code").textContent = `${c.tool} ${argText(c.arguments)}`;
    const oc = li.querySelector("span");
    oc.className = `oc-${c.outcome}`;
    oc.textContent = `→ ${OUTCOME_LABEL[c.outcome] || c.outcome}`;
    d.querySelector("ul").appendChild(li);
  }
  logAppend(d);
}

function addPending(p) {
  const el = document.createElement("div");
  el.className = "as-pending";
  el.innerHTML = '<div class="tool"></div><p></p><div class="row"><button type="button" class="primary">Confirm</button>' +
                 '<button type="button">Cancel</button></div>';
  el.querySelector(".tool").textContent = `${p.tool} ${argText(p.arguments)}`;
  el.querySelector("p").textContent = p.effect;
  const card = { el, confirm_id: p.confirm_id, done: false, busy: false };
  const [ok, cancel] = el.querySelectorAll("button");
  ok.addEventListener("click", () => decide(card, "confirm"));
  cancel.addEventListener("click", () => decide(card, "cancel"));
  chat.cards.push(card);
  logAppend(el);
}

// Replace a card's buttons with what happened. node: text or a DOM node.
function closeCard(card, node) {
  card.done = true;
  card.el.classList.add("done");
  const row = card.el.querySelector(".row");
  const out = document.createElement("div");
  out.className = "outcome";
  out.append(node);
  row.replaceWith(out);
  chat.cards = chat.cards.filter((c) => c !== card);
}

function expireCards(why) {
  for (const c of chat.cards.slice()) closeCard(c, why);
}

function addAnswer(r) {
  addToolCalls(r.tool_calls || []);
  if (r.reply) {
    const d = document.createElement("div");
    d.className = "as-msg bot";
    d.append(linkified(r.reply));
    logAppend(d);
  }
  if (STOPPED_NOTE[r.stopped]) addLine("note", STOPPED_NOTE[r.stopped]);
  if (r.usage) $("asLog").lastElementChild.title = `${r.usage.input_tokens} input + ${r.usage.output_tokens} output tokens`;
  for (const p of r.pending || []) addPending(p);
}

// ------------------------------------------------------------ actions
async function sendMessage() {
  const text = $("asInput").value.trim();
  const user = chatUser();
  if (!text || !user || chat.sending || chat.offText || !state.projectId) return;
  chat.sending = true;
  $("asInput").value = "";
  expireCards("Expired: a newer message was sent.");
  addLine("user", text);
  const wait = addLine("wait", "Claude is working...");
  renderAssistant();
  let r = null;
  try {
    r = await sendChat(text, user);
  } finally {
    wait.remove();
    chat.sending = false;
    renderAssistant();
  }
  if (!r) return;
  chat.sessionId = r.session_id;
  addAnswer(r);
  renderAssistant();
  const wrote = (r.tool_calls || []).filter((c) => c.outcome === "ok" && WRITE_TOOLS.has(c.tool)).map((c) => c.tool);
  if (wrote.length) await refreshAfterWrite(wrote);
}

// #32 with the error rules. A lost session (server restart) starts a new one and sends again, once.
async function sendChat(text, user) {
  const body = (sid) => ({ session_id: sid, project_id: state.projectId, user, message: text });
  try {
    return await api.chat(body(chat.sessionId));
  } catch (e) {
    if (e instanceof ApiError && e.httpStatus === 404 && chat.sessionId) {
      chat.sessionId = null;
      addLine("note", "The conversation was lost (server restart). Starting a new one.");
      try { return await api.chat(body(null)); } catch (e2) { chatError(e2); return null; }
    }
    chatError(e);
    return null;
  }
}

function chatError(e) {
  console.error(e);
  if (e instanceof ApiError && e.httpStatus === 503) { setChatOff(CHAT_OFF_TEXT); return; }
  // 502 assistant_error, 409 busy / too long, 400: the server's message is meant for people.
  addLine("err", e instanceof ApiError ? e.message : String(e));
}

async function decide(card, decision) {
  const user = chatUser();
  if (card.done || card.busy || !user) { if (!user) renderAssistant(); return; }
  card.busy = true;
  renderAssistant();
  let res = null;
  try {
    res = await api.chatConfirm({ session_id: chat.sessionId, confirm_id: card.confirm_id, decision, user });
  } catch (e) {
    console.error(e);
    card.busy = false;
    if (e instanceof ApiError && e.httpStatus === 503) { setChatOff(CHAT_OFF_TEXT); return; }
    if (e instanceof ApiError && e.httpStatus === 404) {
      chat.sessionId = null;
      expireCards("The conversation was lost (server restart). Ask again in this new chat.");
      renderAssistant();
      return;
    }
    if (e instanceof ApiError && e.httpStatus === 409) { closeCard(card, `No longer valid: ${e.message}.`); return; }
    addLine("err", e instanceof ApiError ? e.message : String(e));
    renderAssistant();
    return;
  }
  card.busy = false;
  if (decision === "cancel") { closeCard(card, "Cancelled. Nothing was changed."); renderAssistant(); return; }
  const r = res.result || {};
  if (r.ok) {
    const span = document.createElement("span");
    span.append("Confirmed: done.");
    const url = r.data && r.data.download_url;
    if (url) { span.append(" "); span.append(linkified(url)); }
    closeCard(card, span);
    renderAssistant();
    if (WRITE_TOOLS.has(res.tool)) await refreshAfterWrite([res.tool]);
  } else {
    const err = r.error || {};
    closeCard(card, `Confirmed, but it failed: ${err.code || "error"}: ${err.message || "no message"}`);
    renderAssistant();
  }
}

async function newChat() {
  if (chat.sending) return;
  const sid = chat.sessionId;
  chat.sessionId = null;
  chat.cards = [];
  $("asLog").innerHTML = "";
  renderAssistant();
  if (sid) {
    try { await api.deleteChat(sid); }
    catch (e) { if (!(e instanceof ApiError && e.httpStatus === 404)) console.error(e); } // 404: already gone
  }
  $("asInput").focus();
}

// After a write: classes, counts, the image list (same page and image) and hook points.
// add_class changes the class list everywhere, so the whole project is reopened.
async function refreshAfterWrite(tools) {
  const pid = state.projectId;
  if (pid == null) return;
  if (tools.includes("add_class")) { await openProject(pid); return; }
  await refreshProgress();
  if (pid !== state.projectId) return;
  await loadImages(state.offset, Math.max(0, state.imageIndex));
  await loadHook();
}

// ------------------------------------------------------------ wiring
$("assistantBtn").addEventListener("click", () => setAssistantOpen($("assistant").hidden));
$("asCloseBtn").addEventListener("click", () => setAssistantOpen(false));
$("asNewBtn").addEventListener("click", newChat);
$("asForm").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
$("asInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendMessage(); }
});
$("asNameBtn").addEventListener("click", () => $("reviewerInput").focus());
$("reviewerInput").addEventListener("input", renderAssistant);
document.addEventListener("screen-shown", (e) => { $("assistantBtn").hidden = e.detail !== "review"; });
document.addEventListener("project-opened", renderAssistant);

(function initAssistant() {
  $("assistantBtn").hidden = state.screen !== "review";
  if (lsGet("review.assistant") === "1") setAssistantOpen(true);
  else renderAssistant();
})();
