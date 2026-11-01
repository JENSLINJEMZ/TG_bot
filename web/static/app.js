'use strict';

/* Vertex web chat — the shell is the original Vertex UI; this file replaces
   its demo script with real sessions and streaming from the local model. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const app = $('#app');
const inp = $('#in');
const send = $('#send');
const chat = $('#chat');
const sc = $('#scroll');
const rec = $('#rec');
const ttl = $('#ttl');
const fine = $('#fine');
const toastEl = $('#toast');
const connEl = $('#conn');
const comp = $('#comp');
const fileInput = $('#file');
const attachmentsEl = $('#attachments');
const toolsMenu = $('#toolsmenu');

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

const LG = '<svg class="logo" width="26" height="26"><use href="#lg"/></svg>';
const SEND_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const STOP_ICON = '<span class="stop-sq"></span>';

const state = {
  sessions: [],
  activeId: null,
  messages: [],
  busy: false,
  preset: store.get('preset') || 'core',
  controller: null,
  streamEl: null,
  streamRaw: '',
  raf: 0,
  filter: '',
  menuId: null,
  attachments: [],
  tool: '',
  uploading: 0,
  maxAttachments: 4,
};
if (!['core', 'max', 'lite'].includes(state.preset)) state.preset = 'core';

const TOKEN = new URLSearchParams(location.search).get('token') || '';
const authHeaders = () => (TOKEN ? { 'X-Web-Token': TOKEN } : {});
const baseUrl = () => location.pathname + (TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : '');
const chatUrl = id => {
  const params = new URLSearchParams();
  params.set('s', id);
  if (TOKEN) params.set('token', TOKEN);
  return `?${params.toString()}`;
};
const linkFor = id => `${location.origin}/${chatUrl(id)}`;

/* -- text helpers (same renderer as the Vertex shell) ------------------- */
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const escAttr = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function md(s) {
  const parts = String(s).split(/```/);
  return parts.map((p, i) => {
    if (i % 2) {
      const nl = p.indexOf('\n');
      const lang = (nl >= 0 ? p.slice(0, nl) : '').trim() || 'code';
      const code = (nl >= 0 ? p.slice(nl + 1) : p).replace(/\n$/, '');
      return `<div class="cb"><header><span>${esc(lang)}</span><button data-copy>Copy</button></header><pre>${esc(code)}</pre></div>`;
    }
    return p.trim().split(/\n\n+/).map(b => {
      const inl = x => esc(x).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
      if (/^- /m.test(b)) {
        return '<ul>' + b.split('\n').filter(l => l.startsWith('- ')).map(l => '<li>' + inl(l.slice(2)) + '</li>').join('') + '</ul>';
      }
      return b ? '<p>' + inl(b).replace(/\n/g, '<br>') + '</p>' : '';
    }).join('');
  }).join('');
}

const titleFor = text => {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  if (!clean) return 'New chat';
  return clean.length > 34 ? clean.slice(0, 34) + '…' : clean;
};

/* -- toast / theme / greeting ------------------------------------------ */
let toastTimer;
function toast(m) {
  toastEl.textContent = m;
  toastEl.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('on'), 1800);
}

/* -- attachments and local tools --------------------------------------- */
const TOOL_LABELS = {
  describe: 'Describe image',
  summarize: 'Summarize file',
  removebg: 'Remove background',
  time: 'Time & date',
  calc: 'Calculator',
};

function attachmentUrl(a) {
  const raw = a && a.url ? a.url : `/api/uploads/${encodeURIComponent(a && a.id || '')}/content`;
  if (!TOKEN) return raw;
  return raw + (raw.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(TOKEN);
}

function attSize(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KB';
  return (n / (1024 * 1024)).toFixed(1) + ' MB';
}

function attsHtml(atts) {
  return (atts || []).map(a => {
    const url = attachmentUrl(a);
    const name = escAttr(a.name || 'attachment');
    if (a.kind === 'image') {
      return `<a class="attimg" href="${escAttr(url)}" target="_blank" rel="noopener"><img src="${escAttr(url)}" alt="${name}" loading="lazy"></a>`;
    }
    return `<a class="filelink" href="${escAttr(url)}" target="_blank" rel="noopener"><span>▤</span><span>${esc(a.name || 'file')}</span><small>${attSize(a.size)}</small></a>`;
  }).join('');
}

function renderAttachments() {
  if (!attachmentsEl) return;
  const has = state.attachments.length > 0 || !!state.tool;
  attachmentsEl.hidden = !has;
  if (!has) { attachmentsEl.innerHTML = ''; return; }
  const parts = [];
  if (state.tool) {
    parts.push(`<div class="atchip toolchip"><span class="fico">✦</span><span class="aname">${esc(TOOL_LABELS[state.tool] || state.tool)}</span><button data-toolclear aria-label="Turn tool off">×</button></div>`);
  }
  for (const a of state.attachments) {
    const thumb = a.kind === 'image'
      ? `<img src="${escAttr(attachmentUrl(a))}" alt="">`
      : `<span class="fico">▤</span>`;
    parts.push(`<div class="atchip">${thumb}<span class="aname">${esc(a.name || 'attachment')}</span><button data-remove="${escAttr(a.id)}" aria-label="Remove attachment">×</button></div>`);
  }
  attachmentsEl.innerHTML = parts.join('');
}

function hasImageAttachment() {
  return state.attachments.some(a => a.kind === 'image');
}

function hasTextAttachment() {
  return state.attachments.some(a => a.kind === 'text');
}

function canSend() {
  if (state.busy || state.uploading) return false;
  const text = inp.value.trim();
  if (state.tool === 'time') return true;
  if (state.tool === 'calc') return !!text;
  if (state.tool === 'describe' || state.tool === 'removebg') return hasImageAttachment();
  if (state.tool === 'summarize') return hasTextAttachment();
  return !!text || state.attachments.length > 0;
}

function setTool(tool) {
  state.tool = TOOL_LABELS[tool] ? tool : '';
  $$('.toolsmenu .mi[data-tool]').forEach(b => b.classList.toggle('on', (b.dataset.tool || '') === state.tool));
  inp.placeholder = state.tool === 'describe' ? 'Describe an attached image…'
    : state.tool === 'summarize' ? 'Ask about an attached file…'
    : state.tool === 'removebg' ? "Background colour or 'transparent' (optional)…"
    : state.tool === 'time' ? 'Press send for the current local time…'
    : state.tool === 'calc' ? 'Enter an expression, e.g. 12 * (4 + 3)…'
    : 'How can I help you today?';
  renderAttachments();
  grow();
}

async function prepareImage(file) {
  if (!file || !file.type.startsWith('image/') || typeof createImageBitmap !== 'function') {
    return file;
  }
  let bitmap;
  try { bitmap = await createImageBitmap(file); } catch { return file; }
  const max = 1600;
  const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height));
  if (scale >= 1) { bitmap.close?.(); return file; }
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) { bitmap.close?.(); return file; }
  const png = file.type === 'image/png' || /\.png$/i.test(file.name || '');
  if (!png) {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  if (typeof canvas.toBlob !== 'function') return file;
  const type = png ? 'image/png' : 'image/jpeg';
  const blob = await new Promise(resolve => canvas.toBlob(resolve, type, 0.86));
  if (!blob) return file;
  const name = (file.name || 'image').replace(/\.[^.]+$/, '') + (png ? '.png' : '.jpg');
  try { return new File([blob], name, { type, lastModified: Date.now() }); }
  catch { return file; }
}

async function addFiles(files) {
  const list = [...(files || [])];
  if (!list.length) return;
  const room = state.maxAttachments - state.attachments.length;
  if (room <= 0) { toast(`You can attach up to ${state.maxAttachments} files`); return; }
  if (list.length > room) toast(`Only ${room} more attachment${room === 1 ? '' : 's'} can be added`);
  for (const file of list.slice(0, room)) {
    state.uploading++;
    try {
      const form = new FormData();
      const prepared = await prepareImage(file);
      form.append('file', prepared, prepared.name || file.name);
      const res = await fetch('/api/uploads', {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      });
      if (!res.ok) {
        let msg = res.statusText || `HTTP ${res.status}`;
        try { const j = await res.json(); if (j && j.error) msg = j.error; } catch {}
        throw new Error(msg);
      }
      const data = await res.json();
      state.attachments.push(data.upload);
      renderAttachments();
      grow();
    } catch (e) {
      toast(e.message || 'Could not upload that file');
    } finally {
      state.uploading--;
    }
  }
  if (state.uploading === 0) { inp.focus(); grow(); }
}

async function removeAttachment(id) {
  const index = state.attachments.findIndex(a => a.id === id);
  if (index < 0) return;
  const [attachment] = state.attachments.splice(index, 1);
  renderAttachments();
  grow();
  try { await api(`/api/uploads/${encodeURIComponent(attachment.id)}`, { method: 'DELETE' }); } catch {}
}

function clearComposerAttachments() {
  state.attachments = [];
  state.tool = '';
  renderAttachments();
  setTool('');
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  store.set('theme', t);
  $('#tlabel').textContent = t === 'dark' ? 'Dark mode' : 'Light mode';
}
setTheme(store.get('theme') || (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light'));
$('#theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');

const hour = new Date().getHours();
$('#greet').textContent = hour < 5 ? 'Burning the midnight oil' : hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';

const chips = [
  ['✎', 'Write', 'Help me write a short, punchy product announcement'],
  ['🎓', 'Learn', 'Explain how neural networks learn, simply'],
  ['</>', 'Code', 'Write a debounce function in JavaScript'],
  ['☕', 'Life stuff', 'Plan a calm Sunday routine'],
  ['✦', 'Surprise me', 'Surprise me with something interesting'],
];
$('#chips').innerHTML = chips.map((c, i) => `<button class="chip" style="animation-delay:${.35 + i * .08}s" data-p="${escAttr(c[2])}"><span>${c[0]}</span>${c[1]}</button>`).join('');

/* -- connection -------------------------------------------------------- */
function setConn(s) {
  connEl.className = 'conn ' + s;
  connEl.title = s === 'online' ? 'Local model ready' : s === 'offline' ? 'Server unreachable' : 'Connecting…';
}

async function checkHealth() {
  try {
    const h = await api('/api/health');
    setConn(h.enabled ? 'online' : 'offline');
    if (Number.isFinite(h.max_attachments) && h.max_attachments > 0) {
      state.maxAttachments = h.max_attachments;
    }
    if (h.model) {
      const tag = $('#modeltag');
      if (tag) tag.textContent = h.model;
      if (h.enabled) connEl.title = h.model + ' ready';
    }
    return h;
  } catch {
    setConn('offline');
    return null;
  }
}

/* -- api --------------------------------------------------------------- */
async function api(path, opts = {}) {
  const { headers, ...rest } = opts;
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(headers || {}) },
    ...rest,
  });
  if (!res.ok) {
    let msg = res.statusText || `HTTP ${res.status}`;
    try { const j = await res.json(); if (j && j.error) msg = j.error; } catch {}
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  return res.json();
}

/* -- sessions ---------------------------------------------------------- */
async function loadSessions() {
  try {
    const data = await api('/api/sessions');
    state.sessions = data.sessions || [];
    renderRec();
  } catch {
    setConn('offline');
  }
}

function renderRec() {
  const q = state.filter.toLowerCase();
  const list = state.sessions.filter(s => !q || String(s.title).toLowerCase().includes(q));
  if (!list.length) {
    rec.innerHTML = `<div class="ci" style="cursor:default;color:var(--mu)">${state.sessions.length ? 'No matches' : 'No chats yet'}</div>`;
    return;
  }
  rec.innerHTML = list.map((s, i) => `
    <div class="ci${s.id === state.activeId ? ' on' : ''}" data-id="${escAttr(s.id)}" title="${escAttr(s.title)}" style="animation-delay:${Math.min(i, 12) * .03}s">
      <span>${esc(s.title)}</span>
      <i class="dots" data-menu="${escAttr(s.id)}" title="Chat options" aria-label="Chat options">•••</i>
    </div>`).join('');
}

function setChatting(v) {
  document.body.classList.toggle('chatting', v);
  app.classList.toggle('chatting', v);
  fine.style.display = v ? 'block' : 'none';
}

function reset(push = true) {
  if (state.busy) { toast('Stop the current reply first'); return; }
  state.activeId = null;
  state.messages = [];
  chat.innerHTML = '';
  clearComposerAttachments();
  setChatting(false);
  ttl.textContent = 'New chat';
  if (push) history.pushState({}, '', baseUrl());
  renderRec();
  sc.scrollTop = 0;
  inp.focus();
  if (innerWidth <= 800) app.classList.add('closed');
}

async function loadSession(id, push = true) {
  if (state.busy) { toast('Stop the current reply first'); return; }
  try {
    const { session } = await api(`/api/sessions/${encodeURIComponent(id)}`);
    state.activeId = session.id;
    state.messages = session.messages || [];
    clearComposerAttachments();
    ttl.textContent = session.title;
    chat.innerHTML = '';
    state.messages.forEach(m => addMessage(m.role, m.content, { messageId: m.id, animate: false, attachments: m.attachments || [] }));
    setChatting(state.messages.length > 0);
    if (push) history.pushState({}, '', chatUrl(session.id));
    renderRec();
    sc.scrollTop = sc.scrollHeight;
    if (innerWidth <= 800) app.classList.add('closed');
    return session;
  } catch (e) {
    toast(e.message || 'Could not open that chat');
    setConn('offline');
  }
}

/* -- message elements -------------------------------------------------- */
function actsHtml() {
  return [
    'copy|M8 8h11v11H8zM5 15V5h10',
    'retry|M4 12a8 8 0 1 0 3-6.2M4 4v4h4',
    'up|M7 11v9H4v-9zM7 11l4-8c2 0 3 1 3 3v3h5l-1 8H7',
    'down|M7 13V4H4v9zM7 13l4 8c2 0 3-1 3-3v-3h5l-1-8H7',
  ].map(s => {
    const [k, d] = s.split('|');
    return `<button class="ib" data-a="${k}" aria-label="${k}"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg></button>`;
  }).join('');
}

function addMessage(role, content, opts = {}) {
  const attachments = opts.attachments || [];
  if (role === 'user') {
    const d = document.createElement('div');
    d.className = 'msg u';
    const atts = attachments.length ? `<div class="atts">${attsHtml(attachments)}</div>` : '';
    const body = content ? `<div class="b">${esc(content)}</div>` : '';
    d.innerHTML = atts + body;
    chat.appendChild(d);
    if (opts.animate !== false) sc.scrollTop = sc.scrollHeight;
    return d;
  }
  const d = document.createElement('div');
  d.className = 'msg a';
  const atts = attachments.length ? `<div class="atts">${attsHtml(attachments)}</div>` : '';
  d.innerHTML = `${LG}<div style="flex:1"><div class="body">${md(content)}</div>${atts}<div class="acts"></div></div>`;
  d._raw = content;
  d._id = opts.messageId || null;
  const acts = d.querySelector('.acts');
  acts.innerHTML = actsHtml();
  requestAnimationFrame(() => acts.classList.add('show'));
  chat.appendChild(d);
  if (opts.animate !== false) sc.scrollTop = sc.scrollHeight;
  return d;
}

function addAssistantStream() {
  const d = document.createElement('div');
  d.className = 'msg a think';
  d.innerHTML = `${LG}<div style="flex:1"><div class="body"><span class="caret"></span></div><div class="acts"></div></div>`;
  chat.appendChild(d);
  sc.scrollTop = sc.scrollHeight;
  return d;
}

function scheduleStreamRender() {
  if (state.raf) return;
  // Long code replies can grow to tens of KB; repainting markdown at 60 fps
  // makes the tab sluggish, so coalesce chunks into ~12 renders per second.
  state.raf = setTimeout(() => {
    state.raf = 0;
    const el = state.streamEl;
    if (!el) return;
    const body = el.querySelector('.body');
    body.innerHTML = md(state.streamRaw);
    const last = body.lastElementChild;
    if (last && last.tagName !== 'DIV') last.classList.add('caret');
    else body.insertAdjacentHTML('beforeend', '<span class="caret"></span>');
    const near = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
    if (near) sc.scrollTop = sc.scrollHeight;
    $('#jump').hidden = near;
  }, 80);
}

function finalizeStream(el, raw, messageId, attachments = []) {
  if (!el) return;
  el.classList.remove('think');
  const body = el.querySelector('.body');
  body.innerHTML = md(raw || '');
  el._raw = raw || '';
  if (messageId) el._id = messageId;
  const wrap = body.parentElement;
  let atts = wrap.querySelector('.atts');
  if (attachments && attachments.length) {
    if (!atts) {
      atts = document.createElement('div');
      atts.className = 'atts';
      body.insertAdjacentElement('afterend', atts);
    }
    atts.innerHTML = attsHtml(attachments);
  } else if (atts) {
    atts.remove();
  }
  const acts = el.querySelector('.acts');
  acts.innerHTML = actsHtml();
  requestAnimationFrame(() => acts.classList.add('show'));
  sc.scrollTop = sc.scrollHeight;
}

function showError(el, msg) {
  if (!el) return;
  const wrap = el.querySelector('.body').parentElement;
  const err = document.createElement('div');
  err.className = 'err';
  err.textContent = msg;
  const retry = document.createElement('button');
  retry.className = 'retry';
  retry.textContent = 'Retry';
  retry.onclick = () => { err.remove(); retry.remove(); regenerate(); };
  wrap.appendChild(err);
  wrap.appendChild(retry);
  sc.scrollTop = sc.scrollHeight;
}

function setSendMode(streaming) {
  if (streaming) {
    send.innerHTML = STOP_ICON;
    send.classList.add('ready', 'stop');
    send.setAttribute('aria-label', 'Stop');
    send.title = 'Stop generating';
  } else {
    send.innerHTML = SEND_ICON;
    send.classList.remove('stop');
    send.classList.toggle('ready', canSend());
    send.setAttribute('aria-label', 'Send');
    send.title = '';
  }
}

/* -- SSE --------------------------------------------------------------- */
async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = 'message';
        const dataLines = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (!dataLines.length) continue;
        let data;
        try { data = JSON.parse(dataLines.join('\n')); } catch { data = {}; }
        onEvent(event, data);
      }
    }
  } finally {
    try { reader.releaseLock(); } catch {}
  }
}

async function runStream(sessionId, payload, el) {
  state.streamEl = el;
  state.streamRaw = '';
  state.busy = true;
  setSendMode(true);
  const controller = new AbortController();
  state.controller = controller;

  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) {
      let msg = res.statusText || `HTTP ${res.status}`;
      try { const j = await res.json(); if (j && j.error) msg = j.error; } catch {}
      throw new Error(msg);
    }
    if (!res.body) throw new Error('Streaming is not supported by this browser');
    setConn('online');

    await readSSE(res.body, (event, data) => {
      if (event === 'meta') {
        if (data.message_id) el._id = data.message_id;
      } else if (event === 'token') {
        state.streamRaw += data.text || '';
        scheduleStreamRender();
      } else if (event === 'done') {
        finalizeStream(el, data.content || state.streamRaw, data.message_id, data.attachments || []);
        if (data.title) {
          ttl.textContent = data.title;
          const s = state.sessions.find(x => x.id === sessionId);
          if (s) s.title = data.title;
          renderRec();
        }
      } else if (event === 'error') {
        throw new Error(data.error || 'The model stopped unexpectedly');
      }
    });

    if (el.classList.contains('think')) finalizeStream(el, state.streamRaw, el._id);
    return true;
  } catch (err) {
    if (err && err.name === 'AbortError') {
      if (!state.streamRaw) {
        el.classList.remove('think');
        el.querySelector('.body').innerHTML = '<p style="color:var(--mu)">Stopped.</p>';
      } else {
        finalizeStream(el, state.streamRaw, el._id);
      }
      toast('Stopped');
    } else {
      finalizeStream(el, state.streamRaw, el._id);
      showError(el, (err && err.message) || 'Something went wrong');
      if (err instanceof TypeError) setConn('offline');
    }
    return false;
  } finally {
    state.busy = false;
    state.controller = null;
    state.streamEl = null;
    setSendMode(false);
    loadSessions();
    inp.focus();
  }
}

$('#chips').addEventListener('click', e => {
  const chip = e.target.closest('[data-p]');
  if (!chip) return;
  inp.value = chip.dataset.p;
  grow();
  inp.focus();
});

/* -- composer ---------------------------------------------------------- */
function grow() {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 220) + 'px';
  if (!state.busy) send.classList.toggle('ready', canSend());
}
inp.oninput = grow;
inp.onkeydown = e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!state.busy) go();
  }
};
send.onclick = () => { if (state.busy) stop(); else go(); };

function stop() {
  if (state.controller) state.controller.abort();
}

function displayForSend(text) {
  if (state.tool === 'removebg') {
    if (!text) return 'Remove background';
    return /^remove background/i.test(text) ? text : `Remove background: ${text}`;
  }
  if (text) return text;
  if (state.tool === 'time') return 'What time is it?';
  if (state.tool === 'describe') return 'Describe the attached image';
  if (state.tool === 'summarize') return 'Summarize the attached file';
  if (hasImageAttachment() && hasTextAttachment()) return 'Analyze the attached files';
  if (hasImageAttachment()) return 'Describe the attached image';
  if (hasTextAttachment()) return 'Summarize the attached file';
  return 'New chat';
}

async function go() {
  const v = inp.value.trim();
  if (state.busy) return;
  if (state.uploading) { toast('Wait for the upload to finish'); return; }
  if (state.tool === 'calc' && !v) { toast('Enter an expression to calculate'); return; }
  if ((state.tool === 'describe' || state.tool === 'removebg') && !hasImageAttachment()) {
    toast('Attach an image first'); return;
  }
  if (state.tool === 'summarize' && !hasTextAttachment()) {
    toast('Attach a text file first'); return;
  }
  if (!v && !state.attachments.length && state.tool !== 'time') {
    toast('Type a message or attach a file'); return;
  }

  const display = displayForSend(v);
  const outgoing = {
    content: v,
    preset: state.preset,
    attachments: state.attachments.map(a => a.id),
    tool: state.tool || undefined,
  };
  const localAtts = state.attachments.slice();

  // Hold the busy flag across session creation too, so a double click cannot
  // start two streams before runStream() gets a chance to set it.
  state.busy = true;
  setSendMode(true);

  let sessionId = state.activeId;
  let created = false;
  if (!sessionId) {
    try {
      const { session } = await api('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: titleFor(display), preset: state.preset }),
      });
      sessionId = session.id;
      state.activeId = session.id;
      state.sessions.unshift(session);
      history.pushState({}, '', chatUrl(session.id));
      renderRec();
      created = true;
    } catch (e) {
      state.busy = false;
      setSendMode(false);
      toast(e.message || 'Could not start a chat');
      setConn('offline');
      return;
    }
  }

  setChatting(true);
  if (created) ttl.textContent = titleFor(display);
  inp.value = '';
  addMessage('user', display, { attachments: localAtts });
  clearComposerAttachments();
  grow();
  const el = addAssistantStream();
  const clientId = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  await runStream(sessionId, { ...outgoing, client_id: clientId }, el);
}

async function regenerate() {
  if (state.busy || !state.activeId) return;
  const assistants = [...chat.querySelectorAll('.msg.a')];
  const last = assistants[assistants.length - 1];
  if (last) last.remove();
  const el = addAssistantStream();
  await runStream(state.activeId, { regenerate: true, preset: state.preset }, el);
}

/* -- copy / share ------------------------------------------------------ */
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    return true;
  } catch {
    toast('Could not copy');
    return false;
  }
}

$('#share').onclick = async () => {
  if (!state.activeId) { toast('Start a chat first'); return; }
  if (await copyText(linkFor(state.activeId))) toast('Link copied');
};

/* -- model menu -------------------------------------------------------- */
function setPreset(p) {
  state.preset = p;
  store.set('preset', p);
  $$('.mi[data-m]').forEach(x => x.classList.toggle('on', x.dataset.m === p));
  const label = $(`.mi[data-m="${p}"] b`);
  if (label) $('#mn').textContent = label.textContent;
}

$('#mb').onclick = e => {
  e.stopPropagation();
  toolsMenu.classList.remove('open');
  $('#menu').classList.toggle('open');
};
$$('.mi[data-m]').forEach(b => b.onclick = e => {
  e.stopPropagation();
  setPreset(b.dataset.m);
  $('#menu').classList.remove('open');
});
setPreset(state.preset);

/* -- sidebar / search -------------------------------------------------- */
const toggleSidebar = () => app.classList.toggle('closed');
$('#close').onclick = toggleSidebar;
$('#open').onclick = toggleSidebar;
$('#scrim').onclick = toggleSidebar;
if (innerWidth <= 800) app.classList.add('closed');

$('#new').onclick = () => reset();
$('#searchbtn').onclick = () => {
  const wrap = $('#searchwrap');
  wrap.hidden = !wrap.hidden;
  if (!wrap.hidden) $('#search').focus();
  else { $('#search').value = ''; state.filter = ''; renderRec(); }
};
$('#search').oninput = e => { state.filter = e.target.value.trim(); renderRec(); };

$('#attach').onclick = () => fileInput.click();
fileInput.onchange = () => {
  addFiles(fileInput.files);
  fileInput.value = '';
};
$('#tools').onclick = e => {
  e.stopPropagation();
  $('#menu').classList.remove('open');
  toolsMenu.classList.toggle('open');
};
$$('.toolsmenu .mi[data-tool]').forEach(button => {
  button.onclick = e => {
    e.stopPropagation();
    setTool(button.dataset.tool || '');
    toolsMenu.classList.remove('open');
    inp.focus();
  };
});
attachmentsEl.addEventListener('click', e => {
  const remove = e.target.closest('[data-remove]');
  if (remove) { removeAttachment(remove.dataset.remove); return; }
  if (e.target.closest('[data-toolclear]')) setTool('');
});
comp.addEventListener('dragover', e => {
  if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) {
    e.preventDefault();
    comp.style.borderColor = 'var(--ac)';
  }
});
comp.addEventListener('dragleave', () => { comp.style.borderColor = ''; });
comp.addEventListener('drop', e => {
  if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
  e.preventDefault();
  comp.style.borderColor = '';
  addFiles(e.dataTransfer.files);
});
inp.addEventListener('paste', e => {
  if (!e.clipboardData || !e.clipboardData.files || !e.clipboardData.files.length) return;
  e.preventDefault();
  addFiles(e.clipboardData.files);
});
$('#projects').onclick = () => toast('Projects are not available in this build');
$('#artifacts').onclick = () => toast('Artifacts are not available in this build');

/* -- session rows / menu ----------------------------------------------- */
rec.addEventListener('click', e => {
  const dots = e.target.closest('.dots');
  if (dots) { e.stopPropagation(); openSessionMenu(dots.dataset.menu, dots); return; }
  const row = e.target.closest('.ci[data-id]');
  if (row) loadSession(row.dataset.id);
});

function openSessionMenu(id, anchor) {
  state.menuId = id;
  const m = $('#sessmenu');
  m.classList.add('open');
  const r = anchor.getBoundingClientRect();
  const w = 190;
  const h = m.offsetHeight || 132;
  const left = Math.max(8, Math.min(r.right - w, innerWidth - w - 8));
  let top = r.bottom + 6;
  if (top + h > innerHeight - 8) top = r.top - h - 6;
  m.style.left = `${left}px`;
  m.style.top = `${Math.max(8, top)}px`;
}

function closeSessionMenu() {
  $('#sessmenu').classList.remove('open');
  state.menuId = null;
}

$('#sessmenu').addEventListener('click', async e => {
  const btn = e.target.closest('[data-sm]');
  if (!btn) return;
  const id = state.menuId;
  const action = btn.dataset.sm;
  closeSessionMenu();
  if (!id) return;
  if (action === 'rename') startRename(id);
  else if (action === 'link') {
    if (await copyText(linkFor(id))) toast('Link copied');
  } else if (action === 'delete') {
    await deleteSession(id);
  }
});

function startRename(id) {
  const row = rec.querySelector(`.ci[data-id="${CSS.escape(id)}"]`);
  if (!row) return;
  const span = row.querySelector('span');
  const current = span.textContent;
  const input = document.createElement('input');
  input.value = current;
  input.className = 'rename';
  span.replaceWith(input);
  input.focus();
  input.select();
  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (!title || title === current) { renderRec(); return; }
    try {
      const { session } = await api(`/api/sessions/${encodeURIComponent(id)}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      });
      const s = state.sessions.find(x => x.id === id);
      if (s) s.title = session.title;
      if (state.activeId === id) ttl.textContent = session.title;
      renderRec();
    } catch (e) {
      toast(e.message || 'Rename failed');
      renderRec();
    }
  };
  input.onblur = commit;
  input.onkeydown = e => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    if (e.key === 'Escape') { e.preventDefault(); done = true; renderRec(); }
  };
}

async function deleteSession(id) {
  if (state.busy) { toast('Stop the current reply first'); return; }
  if (!confirm('Delete this chat?')) return;
  try {
    await api(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
    state.sessions = state.sessions.filter(s => s.id !== id);
    if (state.activeId === id) reset();
    else renderRec();
    toast('Chat deleted');
  } catch (e) {
    toast(e.message || 'Delete failed');
  }
}

/* -- chat actions ------------------------------------------------------ */
chat.addEventListener('click', e => {
  const c = e.target.closest('[data-copy]');
  if (c) {
    copyText(c.closest('.cb').querySelector('pre').textContent);
    c.textContent = 'Copied!';
    setTimeout(() => { c.textContent = 'Copy'; }, 1400);
    return;
  }
  const b = e.target.closest('[data-a]');
  if (!b) return;
  const k = b.dataset.a;
  const msg = b.closest('.msg');
  if (k === 'copy') { copyText(msg._raw || ''); toast('Copied to clipboard'); }
  else if (k === 'retry') {
    const assistants = [...chat.querySelectorAll('.msg.a')];
    if (assistants[assistants.length - 1] !== msg) {
      toast('Only the latest reply can be regenerated');
      return;
    }
    regenerate();
  }
  else {
    b.classList.toggle('on');
    toast('Feedback is not stored in this build');
  }
});

/* -- misc -------------------------------------------------------------- */
document.addEventListener('click', e => {
  if (!e.target.closest('.r')) $('#menu').classList.remove('open');
  if (!e.target.closest('.l')) toolsMenu.classList.remove('open');
  if (!e.target.closest('#sessmenu') && !e.target.closest('.dots')) closeSessionMenu();
});

$('#jump').onclick = () => { sc.scrollTop = sc.scrollHeight; };
sc.addEventListener('scroll', () => {
  const near = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
  $('#jump').hidden = near || !document.body.classList.contains('chatting');
});

addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'b') { e.preventDefault(); toggleSidebar(); }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); reset(); }
  if (e.key === 'Escape') {
    $('#menu').classList.remove('open');
    toolsMenu.classList.remove('open');
    closeSessionMenu();
    if (state.busy) stop();
  }
});

addEventListener('popstate', () => {
  const id = new URLSearchParams(location.search).get('s');
  if (id) loadSession(id, false);
  else reset(false);
});

/* -- boot -------------------------------------------------------------- */
(async function init() {
  await checkHealth();
  await loadSessions();
  const id = new URLSearchParams(location.search).get('s');
  if (id) await loadSession(id, false);
  inp.focus();
  setInterval(checkHealth, 15000);
})();
