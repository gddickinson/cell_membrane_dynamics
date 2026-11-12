// Run-length decoding of binary masks as written by src/web/demo_data.py rle_encode (row-major runs, starting with
// a run of zeros that may be 0). Returns a Uint8Array of 0/255.
export function rleDecode(runs, size) {
  const out = new Uint8Array(size);
  let pos = 0, on = false;
  for (const n of runs) { if (on) out.fill(255, pos, pos + n); pos += n; on = !on; }
  return out;
}
