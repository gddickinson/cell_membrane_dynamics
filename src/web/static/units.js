// Unit conversion and number formatting. The engines (JS and Python) work in px, px/s and nm^-1 like the repo's code;
// every number shown on a page goes through here so it carries µm, µm/s or µm^-1 and the pixel size it used.
export const CLASS_COLORS = { expanding: '#ff5a6e', retracting: '#4f8dff', flowing: '#3ddc84', stationary: '#9aa3b5' };
export const CLASSES = ['expanding', 'retracting', 'flowing', 'stationary'];

export const um = (px, u) => px * u;                 // px -> µm
export const um2 = (px2, u) => px2 * u * u;          // px^2 -> µm^2
export const umPerS = (pxs, u) => pxs * u;           // px/s -> µm/s
export const perUm = nm => nm * 1000;                // nm^-1 -> µm^-1

export function f(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (+v.toPrecision(digits)).toString();
}
export const pfmt = p => (p === null || p === undefined || Number.isNaN(p) ? '–' : p < 1e-4 ? p.toExponential(1) : p.toFixed(4));

// Arc length (px) at each point of a closed contour, starting at 0
export function arcPositions(c) {
  const out = [0];
  for (let i = 1; i < c.length; i++) out.push(out[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
  return out;
}
