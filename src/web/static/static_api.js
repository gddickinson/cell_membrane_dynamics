// Static-site data layer: answers api() GETs from the JSON exported by scripts/build_static.py. The analysis itself
// runs in the browser (engine_worker.js), so the only thing that needs the local Python server is POST /analyse
// (the Python pipeline, with optional Lucas-Kanade optical flow on the PIEZO1 channel).
export const NEEDS_SERVER = 'needs the full app running locally (Python server): see About, "Run the web app"';

const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body || clean === 'analyse') throw new Error('The Python analysis ' + NEEDS_SERVER + '.');
  return file(clean);
}
