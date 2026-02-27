

function guessGitHubContext(){
  try{
    const host = window.location.hostname || "";
    const path = (window.location.pathname || "/").split("/").filter(Boolean);
    if (host.endsWith("github.io") && path.length >= 1){
      const user = host.split(".")[0];
      const repo = path[0];
      return {user, repo};
    }
  }catch(e){}
  return null;
}

function updateRepoLink(fallbackRepo){
  const a = document.getElementById("repoLink");
  if (!a) return;
  const ctx = guessGitHubContext();
  if (ctx){
    const url = `https://github.com/${ctx.user}/${ctx.repo}`;
    a.href = url;
    a.textContent = url;
  }else if (fallbackRepo){
    a.href = `https://github.com/<your-username>/${fallbackRepo}`;
  }
}

// LabAgentBench interactive demo (Branin)
// Pure client-side: random search vs (text-prior) GP-LCB.

const DOMAIN = { x: [-5, 10], y: [0, 15] };

let landscapeBase = null;

function fmt(x, digits=3){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function clamp(x, lo, hi){
  return Math.min(Math.max(x, lo), hi);
}

// -----------------------
// Deterministic RNG (LCG)
// -----------------------
function LCG(seed){
  let s = (seed >>> 0) + 1;
  return () => {
    // Numerical Recipes
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randUniform(rng, lo, hi){
  return lo + (hi-lo) * rng();
}

// -----------------------
// Branin function (minimisation)
// -----------------------
function branin(x, y){
  const pi = Math.PI;
  const a = 1.0;
  const b = 5.1/(4*pi*pi);
  const c = 5/pi;
  const r = 6.0;
  const s = 10.0;
  const t = 1/(8*pi);
  return a*Math.pow(y - b*x*x + c*x - r, 2) + s*(1 - t)*Math.cos(x) + s;
}

// -----------------------
// Hint parsing
// -----------------------
function parseHint(text){
  const hint = { xBounds: null, yBounds: null, center: null, raw: text || "" };
  const s = (text || "").toLowerCase();

  // x in [a,b]
  const rx = /x\s*(?:in)?\s*[\[\(]\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*[\]\)]/i;
  const ry = /y\s*(?:in)?\s*[\[\(]\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*[\]\)]/i;
  const mx = s.match(rx);
  const my = s.match(ry);
  if (mx){
    const a = parseFloat(mx[1]), b = parseFloat(mx[3]);
    hint.xBounds = [Math.min(a,b), Math.max(a,b)];
  }
  if (my){
    const a = parseFloat(my[1]), b = parseFloat(my[3]);
    hint.yBounds = [Math.min(a,b), Math.max(a,b)];
  }

  // near (a,b)
  const rnear = /near\s*[\[\(]?\s*(-?\d+(\.\d+)?)\s*[,;]\s*(-?\d+(\.\d+)?)\s*[\]\)]?/i;
  const mn = s.match(rnear);
  if (mn){
    hint.center = [parseFloat(mn[1]), parseFloat(mn[3])];
  }

  return hint;
}

function boundsFromHint(hint){
  let xb = DOMAIN.x.slice();
  let yb = DOMAIN.y.slice();

  if (hint && hint.xBounds){
    xb = [clamp(hint.xBounds[0], DOMAIN.x[0], DOMAIN.x[1]), clamp(hint.xBounds[1], DOMAIN.x[0], DOMAIN.x[1])];
  }
  if (hint && hint.yBounds){
    yb = [clamp(hint.yBounds[0], DOMAIN.y[0], DOMAIN.y[1]), clamp(hint.yBounds[1], DOMAIN.y[0], DOMAIN.y[1])];
  }

  // If no explicit bounds but a center is given, create a window
  if ((!hint.xBounds || !hint.yBounds) && hint.center){
    const wx = 0.25*(DOMAIN.x[1]-DOMAIN.x[0]);
    const wy = 0.25*(DOMAIN.y[1]-DOMAIN.y[0]);
    xb = [clamp(hint.center[0]-wx, DOMAIN.x[0], DOMAIN.x[1]), clamp(hint.center[0]+wx, DOMAIN.x[0], DOMAIN.x[1])];
    yb = [clamp(hint.center[1]-wy, DOMAIN.y[0], DOMAIN.y[1]), clamp(hint.center[1]+wy, DOMAIN.y[0], DOMAIN.y[1])];
  }

  // Ensure increasing
  xb = [Math.min(xb[0], xb[1]), Math.max(xb[0], xb[1])];
  yb = [Math.min(yb[0], yb[1]), Math.max(yb[0], yb[1])];

  // Avoid degenerate
  if (Math.abs(xb[1]-xb[0]) < 1e-6) xb = DOMAIN.x.slice();
  if (Math.abs(yb[1]-yb[0]) < 1e-6) yb = DOMAIN.y.slice();

  return {x: xb, y: yb};
}

function samplePoint(rng, bounds){
  return [randUniform(rng, bounds.x[0], bounds.x[1]), randUniform(rng, bounds.y[0], bounds.y[1])];
}

// -----------------------
// Minimal linear algebra (Gauss‑Jordan)
// -----------------------
function invertMatrix(A){
  const n = A.length;
  // Deep copy
  const M = A.map(row => row.slice());
  // Identity
  const I = Array.from({length:n}, (_,i) => Array.from({length:n}, (_,j) => (i===j?1:0)));

  for (let i=0;i<n;i++){
    // pivot
    let pivot = M[i][i];
    let pivotRow = i;
    for (let r=i+1;r<n;r++){
      if (Math.abs(M[r][i]) > Math.abs(pivot)){
        pivot = M[r][i];
        pivotRow = r;
      }
    }
    if (Math.abs(pivot) < 1e-12){
      // singular; add jitter and retry by nudging diagonal
      M[i][i] += 1e-6;
      pivot = M[i][i];
    }
    if (pivotRow !== i){
      [M[i], M[pivotRow]] = [M[pivotRow], M[i]];
      [I[i], I[pivotRow]] = [I[pivotRow], I[i]];
    }

    // normalize row
    const invP = 1.0 / pivot;
    for (let j=0;j<n;j++){
      M[i][j] *= invP;
      I[i][j] *= invP;
    }

    // eliminate others
    for (let r=0;r<n;r++){
      if (r === i) continue;
      const factor = M[r][i];
      if (Math.abs(factor) < 1e-14) continue;
      for (let j=0;j<n;j++){
        M[r][j] -= factor * M[i][j];
        I[r][j] -= factor * I[i][j];
      }
    }
  }
  return I;
}

function matVecMul(A, v){
  const n = A.length;
  const out = new Array(n).fill(0);
  for (let i=0;i<n;i++){
    let s = 0;
    for (let j=0;j<v.length;j++){
      s += A[i][j] * v[j];
    }
    out[i] = s;
  }
  return out;
}

function dot(a,b){
  let s = 0;
  for (let i=0;i<a.length;i++) s += a[i]*b[i];
  return s;
}

// -----------------------
// GP surrogate (RBF) + LCB acquisition
// -----------------------
function rbfKernel(x1, x2, ell=2.0, sigma=1.0){
  const dx = x1[0]-x2[0];
  const dy = x1[1]-x2[1];
  const d2 = dx*dx + dy*dy;
  return sigma*sigma*Math.exp(-0.5*d2/(ell*ell));
}

function gpFit(X, y, noise=1e-6){
  const n = X.length;
  const K = Array.from({length:n}, () => new Array(n).fill(0));
  for (let i=0;i<n;i++){
    for (let j=0;j<n;j++){
      K[i][j] = rbfKernel(X[i], X[j]) + (i===j ? noise : 0.0);
    }
  }
  const Kinv = invertMatrix(K);
  const alpha = matVecMul(Kinv, y);
  return {Kinv, alpha, X, y};
}

function gpPredict(model, xStar){
  const n = model.X.length;
  const k = new Array(n);
  for (let i=0;i<n;i++) k[i] = rbfKernel(xStar, model.X[i]);
  const mu = dot(k, model.alpha);

  const v = matVecMul(model.Kinv, k);
  const kxx = rbfKernel(xStar, xStar) + 1e-8;
  let var = kxx - dot(k, v);
  if (!Number.isFinite(var) || var < 1e-10) var = 1e-10;
  return {mu, var};
}

// -----------------------
// Algorithms
// -----------------------
function runRandom(iters, seed, bounds){
  const rng = LCG(seed);
  const X = [];
  const y = [];
  for (let i=0;i<iters;i++){
    const p = samplePoint(rng, bounds);
    X.push(p);
    y.push(branin(p[0], p[1]));
  }
  return {X, y};
}

function runGPLCB(iters, seed, bounds, kappa=2.0){
  const rng = LCG(seed);
  const X = [];
  const y = [];

  const n0 = Math.min(5, iters);
  for (let i=0;i<n0;i++){
    const p = samplePoint(rng, bounds);
    X.push(p);
    y.push(branin(p[0], p[1]));
  }

  for (let t=n0;t<iters;t++){
    const model = gpFit(X, y);

    // candidate pool
    const candN = 600;
    let bestAcq = Infinity;
    let bestX = null;

    for (let i=0;i<candN;i++){
      const p = samplePoint(rng, bounds);
      const pr = gpPredict(model, p);
      const acq = pr.mu - kappa*Math.sqrt(pr.var); // LCB for minimisation
      if (acq < bestAcq){
        bestAcq = acq;
        bestX = p;
      }
    }

    X.push(bestX);
    y.push(branin(bestX[0], bestX[1]));
  }

  return {X, y};
}

// -----------------------
// Visualisation (canvas)
// -----------------------
function valueToColor(v, vmin, vmax){
  const t = clamp((v - vmin) / (vmax - vmin + 1e-9), 0, 1);
  // blue -> purple -> orange
  const hue = 220 - 180*t; // 220 (blue) to 40 (orange)
  const sat = 75;
  const light = 30 + 25*(1-t);
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function drawLandscapeBase(){
  const canvas = document.getElementById("landscape");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;

  // grid
  const nx = 110, ny = 55;
  const dx = w / nx;
  const dy = h / ny;

  // compute min/max on grid
  let vmin = Infinity, vmax = -Infinity;
  const grid = [];
  for (let iy=0;iy<ny;iy++){
    const row = [];
    const yv = DOMAIN.y[0] + (iy/(ny-1))*(DOMAIN.y[1]-DOMAIN.y[0]);
    for (let ix=0;ix<nx;ix++){
      const xv = DOMAIN.x[0] + (ix/(nx-1))*(DOMAIN.x[1]-DOMAIN.x[0]);
      const v = branin(xv, yv);
      row.push(v);
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
    grid.push(row);
  }

  // draw
  for (let iy=0;iy<ny;iy++){
    for (let ix=0;ix<nx;ix++){
      const v = grid[iy][ix];
      ctx.fillStyle = valueToColor(v, vmin, vmax);
      ctx.fillRect(ix*dx, h - (iy+1)*dy, dx+1, dy+1);
    }
  }

  // axes labels
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText(`x ∈ [${DOMAIN.x[0]}, ${DOMAIN.x[1]}]`, 10, 18);
  ctx.fillText(`y ∈ [${DOMAIN.y[0]}, ${DOMAIN.y[1]}]`, 10, 36);

  landscapeBase = ctx.getImageData(0,0,w,h);
}

function xyToPix(p, canvas){
  const w = canvas.width, h = canvas.height;
  const x = (p[0] - DOMAIN.x[0])/(DOMAIN.x[1]-DOMAIN.x[0]) * w;
  const y = h - (p[1] - DOMAIN.y[0])/(DOMAIN.y[1]-DOMAIN.y[0]) * h;
  return [x,y];
}

function drawLandscapeWithPoints(points){
  const canvas = document.getElementById("landscape");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;

  if (!landscapeBase) drawLandscapeBase();
  ctx.putImageData(landscapeBase, 0, 0);

  // points
  for (let i=0;i<points.length;i++){
    const p = points[i];
    const [x,y] = xyToPix(p, canvas);
    ctx.beginPath();
    ctx.arc(x, y, (i === points.length-1) ? 5 : 3, 0, 2*Math.PI);
    ctx.fillStyle = (i === points.length-1) ? "rgba(97,208,149,0.95)" : "rgba(232,236,255,0.85)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawProgress(bestSoFar){
  const canvas = document.getElementById("progress");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.fillRect(0,0,w,h);

  const pad = 45;
  const innerW = w - 2*pad;
  const innerH = h - 2*pad;

  const ymin = Math.min(...bestSoFar);
  const ymax = Math.max(...bestSoFar);
  const ypad = 0.08*(ymax - ymin + 1e-9);
  const y0 = ymin - ypad;
  const y1 = ymax + ypad;

  // axes
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h-pad);
  ctx.lineTo(w-pad, h-pad);
  ctx.stroke();

  // labels
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillText("iteration", w/2 - 28, h-14);
  ctx.save();
  ctx.translate(16, h/2 + 38);
  ctx.rotate(-Math.PI/2);
  ctx.fillText("best f(x,y) (lower is better)", 0, 0);
  ctx.restore();

  // line
  ctx.strokeStyle = "rgba(122,162,255,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i=0;i<bestSoFar.length;i++){
    const x = pad + (i/(bestSoFar.length-1))*innerW;
    const y = pad + (1 - (bestSoFar[i]-y0)/(y1-y0))*innerH;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // last marker
  const lx = pad + innerW;
  const ly = pad + (1 - (bestSoFar[bestSoFar.length-1]-y0)/(y1-y0))*innerH;
  ctx.beginPath();
  ctx.arc(lx, ly, 4, 0, 2*Math.PI);
  ctx.fillStyle = "rgba(97,208,149,0.95)";
  ctx.fill();
}

// -----------------------
// UI plumbing
// -----------------------
function updateValLabel(id){
  const el = document.getElementById(id);
  const out = document.getElementById(`${id}_val`);
  if (!el || !out) return;
  out.textContent = (el.type === "range") ? el.value : el.value;
}

function bestSoFarFromY(y){
  const best = [];
  let cur = Infinity;
  for (const v of y){
    cur = Math.min(cur, v);
    best.push(cur);
  }
  return best;
}

function run(){
  const algo = document.getElementById("algo").value;
  const iters = parseInt(document.getElementById("iters").value);
  const seed = parseInt(document.getElementById("seed").value);
  const hintText = document.getElementById("hint").value;

  const hint = parseHint(hintText);
  const bounds = (algo.startsWith("text_")) ? boundsFromHint(hint) : {x: DOMAIN.x.slice(), y: DOMAIN.y.slice()};

  const info = document.getElementById("runInfo");
  info.innerHTML = `Running <b>${algo}</b> for ${iters} iters. Sampling bounds: x ∈ [${fmt(bounds.x[0],2)}, ${fmt(bounds.x[1],2)}], y ∈ [${fmt(bounds.y[0],2)}, ${fmt(bounds.y[1],2)}].`;

  let res;
  if (algo === "random" || algo === "text_random"){
    res = runRandom(iters, seed, bounds);
  }else if (algo === "gp_lcb" || algo === "text_gp_lcb"){
    res = runGPLCB(iters, seed, bounds, 2.0);
  }else{
    res = runRandom(iters, seed, bounds);
  }

  const best = bestSoFarFromY(res.y);
  const bestVal = Math.min(...res.y);
  const bestIdx = res.y.indexOf(bestVal);
  const bestX = res.X[bestIdx];

  document.getElementById("bestVal").textContent = fmt(bestVal, 3);
  document.getElementById("bestX").textContent = `(${fmt(bestX[0], 3)}, ${fmt(bestX[1], 3)})`;

  drawLandscapeWithPoints(res.X);
  drawProgress(best);

  info.innerHTML = info.innerHTML + `<br/>Best found: <b>${fmt(bestVal,3)}</b> at x = (${fmt(bestX[0],3)}, ${fmt(bestX[1],3)}).`;
}

function reset(){
  document.getElementById("algo").value = "text_gp_lcb";
  document.getElementById("iters").value = "30";
  document.getElementById("seed").value = "0";
  document.getElementById("hint").value = "x in [-5, 0], y in [10, 15]";
  ["algo","iters","seed"].forEach(updateValLabel);
  document.getElementById("runInfo").textContent = "Ready.";
  document.getElementById("bestVal").textContent = "—";
  document.getElementById("bestX").textContent = "—";
  drawLandscapeWithPoints([]);
  drawProgress([0,0]);
}

function main(){
  drawLandscapeBase();
  reset();

  document.getElementById("runBtn").addEventListener("click", run);
  document.getElementById("resetBtn").addEventListener("click", reset);

  document.getElementById("algo").addEventListener("change", () => updateValLabel("algo"));
  document.getElementById("iters").addEventListener("input", () => updateValLabel("iters"));
  document.getElementById("seed").addEventListener("input", () => updateValLabel("seed"));

  // set initial labels
  ["algo","iters","seed"].forEach(updateValLabel);
}

main();

    updateRepoLink("labagentbench");
