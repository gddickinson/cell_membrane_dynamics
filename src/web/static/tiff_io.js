// Small pure-JS TIFF reader/writer for 2D image sequences (no library). Reads baseline, uncompressed, one sample per
// pixel, 8/16/32-bit integer or 32/64-bit float, little- or big-endian, every page of the IFD chain; a tifffile
// "shaped" file that stores the stack contiguously behind a single IFD is split using its {"shape": [...]} description.
// Pixel size (µm/px) and frame interval (s) come from the file when it has them: ImageJ description (unit + X
// resolution, finterval), OME-XML (PhysicalSizeX, TimeIncrement), or X resolution in cm. Compressed files are
// refused with a clear message (the local app reads those with tifffile).

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };
const UNIT_UM = { micron: 1, microns: 1, um: 1, 'µm': 1, '\\u00B5m': 1, '\\u00b5m': 1, nm: 1e-3, nanometer: 1e-3, mm: 1e3 };
export class TiffError extends Error {}

function readIfd(dv, off, le) {
  const n = dv.getUint16(off, le), tags = {};
  for (let i = 0; i < n; i++) {
    const e = off + 2 + i * 12, tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le), count = dv.getUint32(e + 4, le);
    const at = (TYPE_SIZE[type] || 1) * count <= 4 ? e + 8 : dv.getUint32(e + 8, le);
    if (type === 2) { tags[tag] = new TextDecoder('latin1').decode(new Uint8Array(dv.buffer, at, count)).replace(/\0+$/, ''); continue; }
    const vals = [];
    for (let k = 0; k < count; k++) {
      if (type === 3 || type === 8) vals.push(dv.getUint16(at + 2 * k, le));
      else if (type === 4 || type === 9) vals.push(dv.getUint32(at + 4 * k, le));
      else if (type === 5) vals.push(dv.getUint32(at + 8 * k, le) / dv.getUint32(at + 8 * k + 4, le));
      else if (type === 1 || type === 7) vals.push(dv.getUint8(at + k));
      else vals.push(0);
    }
    tags[tag] = vals;
  }
  return { tags, next: dv.getUint32(off + 2 + n * 12, le) };
}

function sampleGetter(dv, bits, fmt, le) {
  if (bits === 8) return fmt === 2 ? o => dv.getInt8(o) : o => dv.getUint8(o);
  if (bits === 16) return fmt === 2 ? o => dv.getInt16(o, le) : o => dv.getUint16(o, le);
  if (bits === 32) return fmt === 3 ? o => dv.getFloat32(o, le) : fmt === 2 ? o => dv.getInt32(o, le) : o => dv.getUint32(o, le);
  return o => dv.getFloat64(o, le);
}

// readTiff(ArrayBuffer) -> { w, h, frames: [Float32Array], bits, umPerPx, frameS, source }
export function readTiff(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength < 8) throw new TiffError('Not a TIFF file.');
  const bo = dv.getUint16(0);
  if (bo !== 0x4949 && bo !== 0x4d4d) throw new TiffError('Not a TIFF file.');
  const le = bo === 0x4949, magic = dv.getUint16(2, le);
  if (magic === 43) throw new TiffError('BigTIFF files are not supported in the browser: use the local app.');
  if (magic !== 42) throw new TiffError('Not a TIFF file.');
  const ifds = [];
  for (let off = dv.getUint32(4, le), guard = 0; off && guard < 100000; guard++) {
    const { tags, next } = readIfd(dv, off, le);
    ifds.push(tags); off = next;
  }
  const first = ifds[0], w = first[256][0], h = first[257][0];
  const bits = (first[258] || [1])[0], fmt = (first[339] || [1])[0], bpp = bits / 8;
  if ((first[259] || [1])[0] !== 1) throw new TiffError('This TIFF is compressed; the browser reads uncompressed TIFFs only. Use the local app (Python, tifffile) or save it uncompressed in Fiji.');
  // separate sample planes (planar configuration 2, e.g. a 3- or 4-frame stack tifffile guessed to be RGB) are read
  // as frames, as the Python loader does; interleaved RGB is refused
  const spp = (first[277] || [1])[0], planar = (first[284] || [1])[0];
  if (spp !== 1 && planar !== 2) throw new TiffError('RGB TIFFs are not supported: save a single-channel (grey) stack.');
  if (![8, 16, 32, 64].includes(bits)) throw new TiffError(`${bits}-bit images are not supported (8, 16, 32 or 64 only).`);
  const get = sampleGetter(dv, bits, fmt, le), desc = typeof first[270] === 'string' ? first[270] : '';
  const readPlane = (tags, start = null, planes = 1) => {
    const out = new Float32Array(w * h * planes);
    if (start !== null) { for (let i = 0; i < out.length; i++) out[i] = get(start + i * bpp); return out; }
    const offs = tags[273], counts = tags[279];
    let i = 0;
    for (let s = 0; s < offs.length && i < out.length; s++) {
      const end = offs[s] + (counts ? counts[s] : out.length * bpp);
      for (let o = offs[s]; o < end && i < out.length; o += bpp) out[i++] = get(o);
    }
    return out;
  };
  let frames = ifds.filter(t => t[256][0] === w && t[257][0] === h).map(t => readPlane(t));
  if (spp > 1) {
    const all = readPlane(first, null, spp);
    frames = Array.from({ length: spp }, (_, k) => all.slice(k * w * h, (k + 1) * w * h));
  }
  let source = 'plain TIFF', umPerPx = null, frameS = null, channels = 1;
  if (desc.startsWith('{"shape"')) {
    let shape = [];
    try { shape = JSON.parse(desc).shape || []; } catch { shape = []; }
    const n = shape.slice(0, -2).reduce((a, b) => a * b, 1);
    if (frames.length === 1 && n > 1) {   // contiguous: planes follow the first one
      const start = first[273][0];
      if (start + n * w * h * bpp > buf.byteLength) throw new TiffError('The file is shorter than its shape says.');
      frames = Array.from({ length: n }, (_, k) => readPlane(null, start + k * w * h * bpp));
    }
    source = `tifffile array ${shape.join(' × ')}`;
  }
  const res = first[282] ? first[282][0] : 0, resUnit = (first[296] || [2])[0];
  if (desc.startsWith('ImageJ=')) {
    const kv = Object.fromEntries(desc.split('\n').map(l => l.split('=')).filter(p => p.length === 2).map(([k, v]) => [k.trim(), v.trim()]));
    channels = +kv.channels || 1; source = 'ImageJ TIFF';
    const f = UNIT_UM[kv.unit];
    if (f && res > 0) umPerPx = f / res;
    if (+kv.finterval > 0) frameS = +kv.finterval;
  } else if (desc.includes('<OME')) {
    const px = desc.match(/<(?:\w+:)?Pixels\b[^>]*>/);
    if (px) {
      const a = k => { const m = px[0].match(new RegExp(`\\b${k}="([^"]*)"`)); return m ? m[1] : null; };
      channels = +a('SizeC') || 1; source = 'OME-TIFF';
      if (a('PhysicalSizeX')) umPerPx = +a('PhysicalSizeX') * (UNIT_UM[a('PhysicalSizeXUnit') || 'µm'] ?? 1);
      if (+a('TimeIncrement') > 0) frameS = +a('TimeIncrement');
    }
  } else if (res > 0 && resUnit === 3) { umPerPx = 1e4 / res; source += ' (resolution in cm)'; }
  if (channels > 1) throw new TiffError(`The file has ${channels} channels: save the mask and the PIEZO1 channel as separate single-channel stacks.`);
  return { w, h, frames, bits, umPerPx, frameS, source };
}

// writeTiff8({ w, h, frames: [Uint8Array] }, { umPerPx, frameS }) -> Uint8Array: an uncompressed 8-bit ImageJ stack
// with the pixel size (unit=micron, X/YResolution) and frame interval (finterval) in the file.
export function writeTiff8({ w, h, frames }, { umPerPx = 1, frameS = 1 } = {}) {
  const N = frames.length;
  const desc = `ImageJ=1.11a\nimages=${N}\nframes=${N}\nunit=micron\nfinterval=${frameS}\nloop=false\n\0`;
  const NT = 13, ifdSize = 2 + NT * 12 + 4, extra = 16, plane = w * h;
  const descOff = 8, metaStart = descOff + desc.length + (desc.length % 2), dataStart = metaStart + N * (ifdSize + extra);
  const buf = new ArrayBuffer(dataStart + N * plane), dv = new DataView(buf), u8 = new Uint8Array(buf);
  dv.setUint16(0, 0x4949); dv.setUint16(2, 42, true); dv.setUint32(4, metaStart, true);
  for (let i = 0; i < desc.length; i++) u8[descOff + i] = desc.charCodeAt(i) & 0x7f;
  const den = Math.max(1, Math.round(umPerPx * 1e6));
  for (let p = 0; p < N; p++) {
    const ifd = metaStart + p * (ifdSize + extra), ext = ifd + ifdSize, img = dataStart + p * plane;
    const tags = [[256, 4, 1, w], [257, 4, 1, h], [258, 3, 1, 8], [259, 3, 1, 1], [262, 3, 1, 1], [270, 2, desc.length, descOff],
      [273, 4, 1, img], [277, 3, 1, 1], [278, 4, 1, h], [279, 4, 1, plane], [282, 5, 1, ext], [283, 5, 1, ext + 8], [296, 3, 1, 1]];
    dv.setUint16(ifd, tags.length, true);
    tags.forEach(([tag, type, count, val], i) => {
      const o = ifd + 2 + i * 12;
      dv.setUint16(o, tag, true); dv.setUint16(o + 2, type, true); dv.setUint32(o + 4, count, true);
      if (type === 3) dv.setUint16(o + 8, val, true); else dv.setUint32(o + 8, val, true);
    });
    dv.setUint32(ifd + 2 + tags.length * 12, p === N - 1 ? 0 : ifd + ifdSize + extra, true);
    dv.setUint32(ext, 1e6, true); dv.setUint32(ext + 4, den, true); dv.setUint32(ext + 8, 1e6, true); dv.setUint32(ext + 12, den, true);
    u8.set(frames[p], img);
  }
  return u8;
}
