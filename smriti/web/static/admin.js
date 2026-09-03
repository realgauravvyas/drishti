/* The organiser flow: create an album, upload it, watch it index, share it. */

import { $, api, bytes, copy, dropZone, el, lightbox, notice, params,
         plural, when } from '/static/common.js';

const STORE = 'smriti.events';         // { [eventId]: {token, name} }
const UPLOAD_BATCH = 8;                // files per request: keeps each POST small
const POLL_MS = 1500;

const state = { eventId: null, token: null, event: null, poll: null, uploading: false };

// ------------------------------------------------------------ credentials --
const remembered = () => { try { return JSON.parse(localStorage.getItem(STORE) || '{}'); } catch { return {}; } };
const remember = (id, token, name) => {
  const all = remembered();
  all[id] = { token, name, at: Date.now() };
  try { localStorage.setItem(STORE, JSON.stringify(all)); } catch { /* private mode */ }
};
const forget = (id) => {
  const all = remembered();
  delete all[id];
  try { localStorage.setItem(STORE, JSON.stringify(all)); } catch { /* ignore */ }
};

function show(step) {
  for (const id of ['step-create', 'step-dash', 'step-token']) {
    $(`#${id}`).classList.toggle('hide', id !== step);
  }
}

// -------------------------------------------------------------------- boot --
async function boot() {
  try {
    const info = await api('/api/engine');
    $('#engine-badge').textContent = `${info.name} · ${info.dim}-d · match ≥ ${info.threshold}`;
  } catch { /* badge is cosmetic */ }

  const id = params.get('event');
  const token = params.get('token') || (id ? remembered()[id]?.token : null);
  if (id && token) return open(id, token);
  if (id) { show('step-token'); return; }

  showCreate();
}

function showCreate() {
  show('step-create');
  const all = Object.entries(remembered());
  $('#known-events').replaceChildren(
    ...(all.length ? [
      'Your albums on this device: ',
      ...all.flatMap(([id, meta], i) => [
        i ? ' · ' : '',
        el('a', { href: `/admin.html?event=${id}` }, meta.name || id),
      ]),
    ] : []),
  );
}

async function open(id, token) {
  state.eventId = id;
  state.token = token;
  try {
    state.event = await api(`/api/events/${id}`, { token });
  } catch (error) {
    show('step-token');
    notice($('#token-msg'), error.message, 'bad');
    return;
  }
  remember(id, token, state.event.name);
  // Keep the token out of the address bar (and out of screenshots and history).
  history.replaceState(null, '', `/admin.html?event=${id}`);
  renderDash();
  show('step-dash');
  startPolling();
  refreshGallery();
  refreshSearches();
}

// --------------------------------------------------------------- creation --
$('#create-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#create-btn');
  button.disabled = true;
  const form = new FormData();
  form.append('name', $('#name').value);
  form.append('notes', $('#notes').value);
  form.append('retention_days', $('#retention').value);
  form.append('allow_download', $('#allow-download').checked ? 'true' : 'false');
  try {
    const created = await api('/api/events', { method: 'POST', form });
    remember(created.event_id, created.admin_token, $('#name').value);
    await open(created.event_id, created.admin_token);
    notice($('#upload-msg'),
      'Album created. Save this page — the organiser token is stored in this ' +
      'browser only, and the server keeps just a hash of it.', 'good');
  } catch (error) {
    notice($('#create-msg'), error.message, 'bad');
    button.disabled = false;
  }
});

$('#token-go').addEventListener('click', () => {
  const token = $('#token-input').value.trim();
  if (token) open(params.get('event'), token);
});
$('#new-event').addEventListener('click', () => {
  stopPolling();
  history.replaceState(null, '', '/admin.html');
  showCreate();
});

// -------------------------------------------------------------- dashboard --
function renderDash() {
  const event = state.event;
  const shareUrl = `${location.origin}/find.html?code=${event.share_code}`;
  $('#dash-name').textContent = event.name;
  $('#dash-sub').textContent =
    `Created ${when(event.created_at)} · engine ${event.engine} · ` +
    (event.expires_at ? `auto-deletes ${when(event.expires_at)}` : 'no auto-delete');
  $('#share-code').textContent = event.share_code;
  $('#share-url').textContent = shareUrl;
  $('#copy-url').onclick = (e) => copy(shareUrl, e.target);
  renderStats(event.stats);
}

function renderStats(stats) {
  const cells = [
    ['Photos', stats.photos_total],
    ['Indexed', stats.photos_indexed],
    ['Pending', stats.photos_pending],
    ['Failed', stats.photos_failed],
    ['Faces', stats.faces],
    ['Searches', stats.searches],
    ['On disk', bytes(stats.bytes)],
  ];
  $('#stats').replaceChildren(...cells.map(([label, value]) =>
    el('div', { class: 'stat' },
      el('b', {}, typeof value === 'number' ? value.toLocaleString() : value),
      el('span', {}, label))));
  const total = Math.max(1, stats.photos_total);
  const done = stats.photos_indexed + stats.photos_failed;
  $('#index-bar').style.width = `${Math.round(100 * done / total)}%`;
}

function startPolling() {
  stopPolling();
  state.poll = setInterval(async () => {
    try {
      const progress = await api(`/api/events/${state.eventId}/progress`, { token: state.token });
      renderStats(progress);
      // "Nothing pending" is not "finished" while batches are still uploading:
      // a fast indexer drains between batches, and stopping there would freeze
      // the progress bar until the last batch landed.
      if (progress.done && !state.uploading) stopPolling();
    } catch { stopPolling(); }
  }, POLL_MS);
}
function stopPolling() { if (state.poll) { clearInterval(state.poll); state.poll = null; } }

// ----------------------------------------------------------------- upload --
async function upload(files) {
  const box = $('#upload-progress');
  const bar = $('#upload-bar');
  const text = $('#upload-text');
  box.classList.remove('hide');
  $('#upload-msg').replaceChildren();

  const totals = { queued: 0, duplicates: 0, rejected: 0 };
  const rejects = [];
  state.uploading = true;
  startPolling();          // show indexing progress *while* the upload runs

  for (let i = 0; i < files.length; i += UPLOAD_BATCH) {
    const batch = files.slice(i, i + UPLOAD_BATCH);
    const form = new FormData();
    for (const file of batch) form.append('files', file, file.name);
    try {
      const result = await api(`/api/events/${state.eventId}/photos`,
                               { method: 'POST', form, token: state.token });
      totals.queued += result.queued;
      totals.duplicates += result.duplicates;
      totals.rejected += result.rejected;
      rejects.push(...result.results.filter((r) => r.status === 'rejected'));
    } catch (error) {
      notice($('#upload-msg'), `Upload failed: ${error.message}`, 'bad');
      break;
    }
    const done = Math.min(files.length, i + batch.length);
    bar.style.width = `${Math.round(100 * done / files.length)}%`;
    text.textContent = `Uploaded ${done} of ${files.length}…`;
  }

  state.uploading = false;
  text.textContent =
    `${plural(totals.queued, 'photo')} added` +
    (totals.duplicates ? `, ${totals.duplicates} already in the album` : '') +
    (totals.rejected ? `, ${totals.rejected} rejected` : '') + '.';
  if (rejects.length) {
    notice($('#upload-msg'),
      `Rejected: ${rejects.slice(0, 5).map((r) => `${r.file} (${r.detail})`).join('; ')}` +
      (rejects.length > 5 ? ` and ${rejects.length - 5} more` : ''), 'bad');
  }
  startPolling();
  setTimeout(refreshGallery, 800);
}

dropZone($('#drop'), $('#file'), upload);

// ---------------------------------------------------------------- gallery --
async function refreshGallery() {
  if (!state.eventId) return;
  try {
    const data = await api(`/api/events/${state.eventId}/photos?limit=300`, { token: state.token });
    $('#gallery-count').textContent = data.photos.length ? `· ${data.photos.length} shown` : '';
    $('#gallery').replaceChildren(...data.photos.map(photoTile));
    if (!data.photos.length) {
      $('#gallery').replaceChildren(el('p', { class: 'muted' }, 'Nothing uploaded yet.'));
    }
  } catch (error) {
    notice($('#upload-msg'), error.message, 'bad');
  }
}

function photoTile(photo) {
  const label = photo.state === 'indexed'
    ? (photo.n_faces ? `${photo.n_faces} ${photo.n_faces === 1 ? 'face' : 'faces'}` : 'no faces')
    : photo.state;
  return el('div', {
    class: 'tile',
    title: `${photo.name}\n${photo.width}×${photo.height} · ${bytes(photo.bytes)}` +
           (photo.error ? `\n${photo.error}` : ''),
    onclick: () => openAdminPhoto(photo),
  },
    el('img', { src: photo.thumb_url, alt: photo.name, loading: 'lazy' }),
    el('div', { class: 'state' }, label),
  );
}

function openAdminPhoto(photo) {
  const box = lightbox(photo.original_url, [
    el('button', {
      class: 'danger', style: 'background:rgba(255,255,255,.14)',
      onclick: async () => {
        if (!confirm(`Delete ${photo.name}? This also removes its faces from the index.`)) return;
        await api(`/api/events/${state.eventId}/photos/${photo.photo_id}`,
                  { method: 'DELETE', token: state.token });
        box.remove();
        refreshGallery();
      },
    }, 'Delete photo'),
  ]);
}

$('#refresh-gallery').addEventListener('click', refreshGallery);

// ------------------------------------------------------------- re-indexing --
$('#reindex').addEventListener('click', async () => {
  if (!confirm('Re-detect every face in the album? Existing search results stay valid; this just rebuilds the index.')) return;
  const result = await api(`/api/events/${state.eventId}/reindex`,
                           { method: 'POST', token: state.token });
  notice($('#upload-msg'), `Re-queued ${plural(result.requeued, 'photo')}.`, 'good');
  startPolling();
});

// ----------------------------------------------------------------- people --
$('#load-people').addEventListener('click', async () => {
  const host = $('#people');
  host.replaceChildren(el('div', { class: 'row' }, el('i', { class: 'spin dark' }), 'Grouping faces…'));
  try {
    const data = await api(`/api/events/${state.eventId}/people`, { token: state.token });
    if (data.skipped) {
      host.replaceChildren(el('p', { class: 'muted' },
        `Skipped: ${plural(data.faces, 'face')} is above the clustering limit.`));
      return;
    }
    host.replaceChildren(
      el('p', { class: 'muted' },
        `About ${plural(data.n_people, 'distinct person', 'distinct people')} across ` +
        `${plural(data.faces, 'face')}. ${data.singletons} appear in only one photo. ` +
        `Grouped with ${data.method} at similarity ≥ ${data.threshold}. ` +
        `This tends to over-count slightly — it will split a person before it merges two.`),
      el('div', { class: 'grid', style: 'grid-template-columns:repeat(auto-fill,minmax(110px,1fr))' },
        ...data.top.map((person) => el('div', { class: 'tile', title: `${person.faces} faces` },
          el('img', { src: person.thumb_url, alt: '', loading: 'lazy' }),
          el('div', { class: 'state' }, `${person.photos} photos`)))),
    );
  } catch (error) {
    notice(host, error.message, 'bad');
  }
});

// --------------------------------------------------------------- searches --
async function refreshSearches() {
  try {
    const data = await api(`/api/events/${state.eventId}/searches`, { token: state.token });
    if (!data.searches.length) return;
    $('#searches').replaceChildren(
      el('table', {},
        el('thead', {}, el('tr', {},
          el('th', {}, 'When'), el('th', {}, 'Selfies'),
          el('th', {}, 'Matches'), el('th', {}, 'Best score'), el('th', {}, 'Time'))),
        el('tbody', {}, ...data.searches.slice(0, 15).map((row) => el('tr', {},
          el('td', {}, when(row.ts)),
          el('td', {}, String(row.n_queries)),
          el('td', {}, String(row.n_matches)),
          el('td', {}, row.top_score.toFixed(2)),
          el('td', {}, `${Math.round(row.ms)} ms`))))),
      el('p', { class: 'muted', style: 'margin:10px 0 0' },
        'No faces or selfies are stored — only that a search happened.'));
  } catch { /* non-critical panel */ }
}

// ----------------------------------------------------------------- delete --
$('#delete-event').addEventListener('click', async () => {
  const name = state.event.name;
  if (prompt(`Type the album name to confirm permanent deletion:\n\n${name}`) !== name) return;
  try {
    await api(`/api/events/${state.eventId}`, { method: 'DELETE', token: state.token });
    forget(state.eventId);
    stopPolling();
    history.replaceState(null, '', '/admin.html');
    showCreate();
    notice($('#create-msg'), `"${name}" and all of its photos were deleted.`, 'good');
  } catch (error) {
    notice($('#delete-msg'), error.message, 'bad');
  }
});

boot();
