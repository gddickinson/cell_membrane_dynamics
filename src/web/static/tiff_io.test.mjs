// Node test for tiff_io.js on the files from scripts/make_tiff_fixtures.py, plus a write/read round trip.
// Run: node src/web/static/tiff_io.test.mjs   (tests/test_web.py also checks that tifffile reads writeTiff8 output)
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { readTiff, writeTiff8, TiffError } from './tiff_io.js';

const dir = fileURLToPath(new URL('../../../tests/fixtures/tiff/', import.meta.url));
const meta = JSON.parse(readFileSync(dir + 'meta.json'));
let fails = 0, checks = 0;
const ok = (cond, msg) => { checks++; if (!cond) { fails++; console.error('FAIL', msg); } };
const buf = f => { const b = readFileSync(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const near = (a, b) => (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) < 1e-6 * Math.max(1, b));

for (const [name, m] of Object.entries(meta)) {
  if (m.error) {
    let err = null;
    try { readTiff(buf(dir + name)); } catch (e) { err = e; }
    ok(err instanceof TiffError && /compressed/.test(err.message), `${name}: expected a 'compressed' error`);
    continue;
  }
  const t = readTiff(buf(dir + name));
  ok(t.w === 20 && t.h === 16 && t.frames.length === 3, `${name}: shape ${t.frames.length} x ${t.h} x ${t.w}`);
  const want = (f, y, x) => (m.dtype === 'uint8' ? (20 * y + x + 7 * f) % 256 : 1000 * f + 20 * y + x);
  let bad = 0;
  t.frames.forEach((fr, f) => { for (let y = 0; y < 16; y++) for (let x = 0; x < 20; x++) if (fr[y * 20 + x] !== want(f, y, x)) bad++; });
  ok(bad === 0, `${name}: ${bad} wrong pixels`);
  ok(near(t.umPerPx, m.umPerPx), `${name}: umPerPx ${t.umPerPx} vs ${m.umPerPx}`);
  ok(near(t.frameS, m.frameS), `${name}: frameS ${t.frameS} vs ${m.frameS}`);
}

const frames = [0, 1].map(k => Uint8Array.from({ length: 12 * 10 }, (_, i) => (i * 3 + k * 50) % 256));
const bytes = writeTiff8({ w: 12, h: 10, frames }, { umPerPx: 0.1, frameS: 2 });
const back = readTiff(bytes.buffer);
ok(back.frames.length === 2 && back.frames[1].every((v, i) => v === frames[1][i]), 'round trip pixels');
ok(near(back.umPerPx, 0.1) && back.frameS === 2, `round trip scale ${back.umPerPx} ${back.frameS}`);
if (process.argv[2]) writeFileSync(process.argv[2], bytes);

console.log(`${checks - fails}/${checks} TIFF checks passed`);
process.exit(fails ? 1 : 0);
