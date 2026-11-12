// Module worker: runs dynamics_engine.analyseSequence off the main thread and reports progress per frame.
import { analyseSequence } from './dynamics_engine.js';

self.onmessage = e => {
  const { masks, w, h, opts } = e.data;
  try {
    const res = analyseSequence(masks, w, h, { ...opts, onFrame: i => self.postMessage({ progress: i }) });
    self.postMessage({ result: res });
  } catch (err) {
    self.postMessage({ error: String(err && err.message || err) });
  }
};
