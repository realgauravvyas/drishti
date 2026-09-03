/* The guest flow: selfie in, your photos out. */

import { $, api, ApiError, dropZone, el, lightbox, notice, params,
         plural, shrinkImage } from '/static/common.js';

const CODE = (params.get('code') || '').toUpperCase();
const MAX_SELFIES = 3;

const state = {
  event: null,
  selfies: [],        // {file, url}
  matches: [],
  picked: new Set(),
};

const TIERS = [
  { key: 'sure',   label: 'Definitely you',  hint: 'High-confidence matches.' },
  { key: 'likely', label: 'Probably you',    hint: 'Good matches — worth a look.' },
  { key: 'maybe',  label: 'Might be you',    hint: 'Weaker matches: a side profile, someone far back, or not you at all.' },
];

function show(step) {
  for (const id of ['boot', 'step-selfie', 'step-searching', 'step-results']) {
    $(`#${id}`).classList.toggle('hide', id !== step);
  }
}

// ---------------------------------------------------------------- boot ----
async function boot() {
  if (!CODE) { location.href = '/index.html'; return; }
  try {
    state.event = await api(`/api/events/by-code/${encodeURIComponent(CODE)}`);
  } catch (error) {
    show('boot');
    notice($('#boot'), error.message, 'bad');
    return;
  }
  const event = state.event;
  $('#event-name').textContent = event.name;
  $('#title').textContent = event.name;
  $('#subtitle').textContent =
    `${plural(event.photos_indexed, 'photo')} ready to search. Add a selfie and we'll pull out the ones with you in them.`;

  if (event.photos_pending > 0) {
    notice($('#pending-warning'),
      `The organiser is still adding photos — ${plural(event.photos_pending, 'photo')} left to process. ` +
      `You can search now and check back later for the rest.`);
  }
  if (event.photos_total === 0) {
    notice($('#pending-warning'), 'No photos have been uploaded to this event yet.', 'bad');
  }
  show('step-selfie');
}

// -------------------------------------------------------------- selfies ---
function renderSelfies() {
  const host = $('#selfies');
  host.replaceChildren(...state.selfies.map((item, i) =>
    el('div', { class: 'tile' },
      el('img', { src: item.url, alt: '' }),
      el('div', {
        class: 'check', style: 'background:var(--danger);color:#fff;border-color:var(--danger)',
        title: 'Remove',
        onclick: () => { URL.revokeObjectURL(item.url); state.selfies.splice(i, 1); renderSelfies(); },
      }, '×'),
    )));
  $('#search').disabled = state.selfies.length === 0;
  $('#search').textContent = state.selfies.length > 1
    ? `Find my photos (${state.selfies.length} selfies)` : 'Find my photos';
}

async function addFiles(files) {
  const room = MAX_SELFIES - state.selfies.length;
  if (room <= 0) {
    notice($('#selfie-msg'), `Three selfies is the limit — remove one first.`, 'bad');
    return;
  }
  for (const raw of files.slice(0, room)) {
    const file = await shrinkImage(raw);
    state.selfies.push({ file, url: URL.createObjectURL(file) });
  }
  $('#selfie-msg').replaceChildren();
  renderSelfies();
}

// --------------------------------------------------------------- search ---
async function runSearch() {
  show('step-searching');
  $('#searching-sub').textContent =
    `Searching ${plural(state.event.photos_indexed, 'photo')} from ${state.event.name}.`;

  const form = new FormData();
  for (const item of state.selfies) form.append('selfies', item.file, item.file.name);

  let result;
  try {
    result = await api(`/api/events/by-code/${encodeURIComponent(CODE)}/search`,
                       { method: 'POST', form });
  } catch (error) {
    show('step-selfie');
    notice($('#selfie-msg'),
      error instanceof ApiError && error.status === 429
        ? 'Too many searches from this device — wait a minute and try again.'
        : error.message, 'bad');
    return;
  }

  if (result.no_face_in_selfie) {
    show('step-selfie');
    notice($('#selfie-msg'), result.message, 'bad');
    return;
  }

  state.matches = result.matches;
  state.picked = new Set(result.matches.filter((m) => m.tier === 'sure').map((m) => m.photo_id));
  renderResults(result);
  show('step-results');
}

function renderResults(result) {
  const n = result.count;
  $('#results-title').textContent = n ? `${plural(n, 'photo')} of you` : 'No photos found';
  $('#results-sub').textContent = n
    ? `Searched ${plural(result.searched_faces, 'face')} across ${plural(result.searched_photos, 'photo')} in ${result.ms} ms. ` +
      `Photos we're confident about are pre-selected.`
    : `We compared your face against ${plural(result.searched_faces, 'face')} and found nothing above the matching threshold. ` +
      `Try a clearer, front-facing selfie — or you may simply not be in this album.`;

  $('#actionbar').classList.toggle('hide', n === 0);
  $('#download').classList.toggle('hide', !result.allow_download);

  const host = $('#tiers');
  host.replaceChildren();
  for (const tier of TIERS) {
    const items = state.matches.filter((m) => m.tier === tier.key);
    if (!items.length) continue;
    host.append(
      el('div', { class: 'tier' },
        el('h2', {}, tier.label),
        el('span', { class: `pill ${tier.key}` }, String(items.length)),
      ),
      el('p', { class: 'muted', style: 'margin:-6px 0 12px' }, tier.hint),
      el('div', { class: 'grid' }, ...items.map(tile)),
    );
  }
  updatePicked();
}

function tile(match) {
  const node = el('div', {
    class: 'tile' + (state.picked.has(match.photo_id) ? ' picked' : ''),
    onclick: (event) => {
      if (event.shiftKey) return openPhoto(match);
      state.picked.has(match.photo_id)
        ? state.picked.delete(match.photo_id)
        : state.picked.add(match.photo_id);
      node.classList.toggle('picked');
      updatePicked();
    },
  },
    el('img', { src: match.thumb_url, alt: match.name, loading: 'lazy' }),
    el('div', { class: 'check' }, '✓'),
    el('div', { class: 'score', title: `cosine similarity ${match.score}` }, match.score.toFixed(2)),
    match.faces_in_photo > 1
      ? el('div', { class: 'state' }, `${match.faces_in_photo} people`) : null,
  );
  node.addEventListener('dblclick', () => openPhoto(match));
  return node;
}

function openPhoto(match) {
  lightbox(match.original_url, [
    el('a', { class: 'btn', href: `${match.original_url}&download=true` }, 'Download this photo'),
  ]);
}

function updatePicked() {
  const n = state.picked.size;
  $('#picked-count').textContent = n ? `${n} selected` : 'Tap a photo to select it';
  $('#download').disabled = n === 0;
  $('#download').textContent = n ? `Download ${n}` : 'Download selected';
}

async function downloadZip() {
  const chosen = state.matches.filter((m) => state.picked.has(m.photo_id));
  if (!chosen.length) return;
  const button = $('#download');
  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Building ZIP…';

  const form = new FormData();
  form.append('photo_ids', chosen.map((m) => m.photo_id).join(','));
  form.append('tokens', chosen.map((m) => m.token).join(','));

  try {
    const res = await fetch(`/api/events/by-code/${encodeURIComponent(CODE)}/download`,
                            { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail || 'download failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const link = el('a', { href: url, download: `${state.event.name}-photos.zip` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
    notice($('#results-msg'), `Downloaded ${plural(chosen.length, 'photo')}.`, 'good');
  } catch (error) {
    notice($('#results-msg'), error.message, 'bad');
  } finally {
    button.disabled = false;
    button.textContent = original;
    updatePicked();
  }
}

// ----------------------------------------------------------------- wire ---
dropZone($('#drop'), $('#file'), addFiles);
$('#camera').addEventListener('click', () => $('#file-camera').click());
$('#file-camera').addEventListener('change', (event) => {
  if (event.target.files.length) addFiles([...event.target.files]);
  event.target.value = '';
});
$('#search').addEventListener('click', runSearch);
$('#again').addEventListener('click', () => {
  $('#selfie-msg').replaceChildren();
  show('step-selfie');
});
$('#download').addEventListener('click', downloadZip);
$('#select-all').addEventListener('click', () => {
  state.matches.forEach((m) => state.picked.add(m.photo_id));
  document.querySelectorAll('#tiers .tile').forEach((t) => t.classList.add('picked'));
  updatePicked();
});
$('#select-none').addEventListener('click', () => {
  state.picked.clear();
  document.querySelectorAll('#tiers .tile').forEach((t) => t.classList.remove('picked'));
  updatePicked();
});

boot();
