/* Shared helpers. Kept deliberately small — three pages do not need a framework,
   and a build step would be one more thing between a fresh clone and a demo. */

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const params = new URLSearchParams(location.search);

export async function api(path, { method = 'GET', body, token, form } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload = body;
  if (form) payload = form;
  else if (body && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!res.ok) throw new ApiError(data.detail || `request failed (${res.status})`, res.status, data);
  return data;
}

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

export function notice(host, message, kind = '') {
  const box = el('div', { class: `notice ${kind}` }, message);
  host.replaceChildren(box);
  return box;
}

export function clearNotice(host) { host.replaceChildren(); }

export function plural(n, one, many = null) {
  return `${n.toLocaleString()} ${n === 1 ? one : (many || one + 's')}`;
}

export function bytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function when(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString(undefined,
    { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

export async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = el('textarea', { style: 'position:fixed;opacity:0' });
    ta.value = text;
    document.body.append(ta); ta.select();
    document.execCommand('copy'); ta.remove();
  }
  if (button) {
    const original = button.textContent;
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = original; }, 1400);
  }
}

/** Wire a click-or-drop file zone. `onFiles` receives a File array. */
export function dropZone(zone, input, onFiles) {
  zone.addEventListener('click', (event) => {
    if (event.target.tagName !== 'INPUT') input.click();
  });
  input.addEventListener('change', () => {
    if (input.files.length) onFiles([...input.files]);
    input.value = '';
  });
  for (const type of ['dragenter', 'dragover']) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add('over');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove('over');
    });
  }
  zone.addEventListener('drop', (event) => {
    const files = [...(event.dataTransfer?.files || [])]
      .filter((f) => f.type.startsWith('image/'));
    if (files.length) onFiles(files);
  });
}

export function lightbox(src, actions = []) {
  const box = el('div', { class: 'lightbox', onclick: (e) => { if (e.target === box) box.remove(); } },
    el('button', { class: 'close', onclick: () => box.remove(), 'aria-label': 'Close' }, '×'),
    el('img', { src }),
    actions.length ? el('div', { class: 'bar-actions' }, ...actions) : null,
  );
  const onKey = (event) => {
    if (event.key === 'Escape') { box.remove(); document.removeEventListener('keydown', onKey); }
  };
  document.addEventListener('keydown', onKey);
  document.body.append(box);
  return box;
}

/** Shrink a selfie before upload: a 12 MP phone photo is ~5 MB on a hotel
 *  Wi-Fi, and the detector never sees more than ~1280 px anyway. */
export async function shrinkImage(file, maxSide = 1280, quality = 0.9) {
  if (!file.type.startsWith('image/')) return file;
  if (file.size < 400_000) return file;
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    if (scale === 1) return file;
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', quality));
    bitmap.close?.();
    return blob ? new File([blob], file.name.replace(/\.\w+$/, '') + '.jpg', { type: 'image/jpeg' }) : file;
  } catch {
    return file; // any failure just means we upload the original
  }
}
