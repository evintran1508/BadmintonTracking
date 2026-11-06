// Homography inversion and court <-> image projection utilities

/**
 * Invert a 3x3 matrix given as a flat 9-element row-major array.
 * @param {number[]} m
 * @returns {number[]}
 */
export function invert3x3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) {
    throw new Error("Singular homography matrix");
  }
  const invDet = 1.0 / det;
  return [
    A * invDet, D * invDet, G * invDet,
    B * invDet, E * invDet, H * invDet,
    C * invDet, F * invDet, I * invDet,
  ];
}

/**
 * Project court coordinates (meters) to image pixel coordinates using H_img_to_court.
 * @param {number[]} hImgToCourt - 9-element row-major homography (image px -> court m)
 * @param {number} x_m
 * @param {number} y_m
 * @returns {[number, number]}
 */
export function courtToPx(hImgToCourt, x_m, y_m) {
  const hInv = invert3x3(hImgToCourt);
  const w = hInv[6] * x_m + hInv[7] * y_m + hInv[8];
  const px = (hInv[0] * x_m + hInv[1] * y_m + hInv[2]) / w;
  const py = (hInv[3] * x_m + hInv[4] * y_m + hInv[5]) / w;
  return [px, py];
}

/**
 * Sample a circle of radius `radius_m` on the court around `(cx_m, cy_m)`
 * and project it to image polygon points (perspective-foreshortened ellipse).
 * @param {number[]} hImgToCourt
 * @param {number} cx_m
 * @param {number} cy_m
 * @param {number} [radius_m=0.45]
 * @param {number} [segments=24]
 * @returns {Array<[number, number]>}
 */
export function projectFootRingPx(hImgToCourt, cx_m, cy_m, radius_m = 0.45, segments = 24) {
  const pts = [];
  for (let k = 0; k < segments; k++) {
    const theta = (2 * Math.PI * k) / segments;
    const x = cx_m + radius_m * Math.cos(theta);
    const y = cy_m + radius_m * Math.sin(theta);
    pts.push(courtToPx(hImgToCourt, x, y));
  }
  return pts;
}

/**
 * Solve 4-point homography (px -> court_m) using Gaussian elimination and evaluate mean reprojection error across all points.
 * @param {Array<{px: [number, number], court_m: [number, number]}>} calibPoints
 * @returns {{H: number[], reprojErrorM: number}}
 */
export function solveCourtCalibration(calibPoints) {
  if (calibPoints.length < 4) {
    return { H: [1, 0, 0, 0, 1, 0, 0, 0, 1], reprojErrorM: 999.0 };
  }
  const corners = calibPoints.length >= 6
    ? [calibPoints[0], calibPoints[1], calibPoints[4], calibPoints[5]]
    : calibPoints.slice(0, 4);
  const A = [];
  const b = [];
  for (const pt of corners) {
    const [x, y] = pt.px;
    const [u, v] = pt.court_m;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const n = 8;
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    }
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [b[i], b[maxRow]] = [b[maxRow], b[i]];
    const piv = A[i][i] || 1e-12;
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / piv;
      b[k] -= f * b[i];
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
    }
  }
  const h = new Array(8).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) sum -= A[i][j] * h[j];
    h[i] = sum / (A[i][i] || 1e-12);
  }
  const H = [...h, 1.0];

  let errSum = 0;
  for (const pt of calibPoints) {
    const [x, y] = pt.px;
    const w = H[6] * x + H[7] * y + H[8];
    const ux = (H[0] * x + H[1] * y + H[2]) / w;
    const vy = (H[3] * x + H[4] * y + H[5]) / w;
    errSum += Math.hypot(ux - pt.court_m[0], vy - pt.court_m[1]);
  }
  return {
    H,
    reprojErrorM: Number((errSum / calibPoints.length).toFixed(4)),
  };
}


