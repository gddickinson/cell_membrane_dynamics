// Pearson correlation with the two-sided p-value of scipy.stats.pearsonr, in pure JS (no library).
// p = 2 * I_{(1-|r|)/2}(n/2 - 1, n/2 - 1): the sf of the beta distribution scipy uses for r under H0.

// log Gamma (Lanczos, g = 7, n = 9): ~1e-15 relative accuracy for x > 0
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LG[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Continued fraction for the incomplete beta function (modified Lentz)
function betacf(a, b, x) {
  const TINY = 1e-300, EPS = 1e-16;
  let c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 10000; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

// Regularised incomplete beta I_x(a, b)
export function betainc(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log1p(-x);
  if (x < (a + 1) / (a + b + 2)) return Math.exp(lbt) * betacf(a, b, x) / a;
  return 1 - Math.exp(lbt) * betacf(b, a, 1 - x) / b;
}

const mean = v => { let s = 0; for (const x of v) s += x; return s / v.length; };

// scipy.stats.pearsonr(x, y): { r, p } (NaN when either input is constant or n < 2)
export function pearson(x, y) {
  const n = x.length;
  if (n < 2 || y.length !== n) return { r: NaN, p: NaN };
  const mx = mean(x), my = mean(y);
  const xm = Array.from(x, v => v - mx), ym = Array.from(y, v => v - my);
  let ax = 0, ay = 0;
  for (let i = 0; i < n; i++) { ax = Math.max(ax, Math.abs(xm[i])); ay = Math.max(ay, Math.abs(ym[i])); }
  if (!(ax > 0) || !(ay > 0)) return { r: NaN, p: NaN };
  for (let i = 0; i < n; i++) { xm[i] /= ax; ym[i] /= ay; }
  let nx = 0, ny = 0;
  for (let i = 0; i < n; i++) { nx += xm[i] * xm[i]; ny += ym[i] * ym[i]; }
  nx = Math.sqrt(nx); ny = Math.sqrt(ny);
  let r = 0;
  for (let i = 0; i < n; i++) r += (xm[i] / nx) * (ym[i] / ny);
  r = Math.max(-1, Math.min(1, r));
  if (n === 2) return { r, p: 1 };
  const ab = n / 2 - 1;
  return { r, p: Math.min(1, 2 * betainc(ab, ab, (1 - Math.abs(r)) / 2)) };
}
