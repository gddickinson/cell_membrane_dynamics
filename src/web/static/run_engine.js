// Run the browser engine in a module worker (falls back to the main thread where workers are unavailable).
import { analyseSequence } from './dynamics_engine.js';

export function runBrowser(masks, w, h, opts, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    let worker = null;
    try { worker = new Worker(new URL('./engine_worker.js', import.meta.url), { type: 'module' }); } catch { worker = null; }
    if (!worker) {
      try { resolve(analyseSequence(masks, w, h, { ...opts, onFrame: onProgress })); } catch (e) { reject(e); }
      return;
    }
    worker.onmessage = e => {
      const d = e.data;
      if ('progress' in d) onProgress(d.progress);
      else { worker.terminate(); if (d.error) reject(new Error(d.error)); else resolve(d.result); }
    };
    worker.onerror = e => { worker.terminate(); reject(new Error(e.message || 'engine worker failed')); };
    worker.postMessage({ masks, w, h, opts });
  });
}
