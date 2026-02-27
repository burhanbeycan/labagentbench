diff --git a/docs/assets/app.js b/docs/assets/app.js
index 7e1b33d174d39ebac8ec9530e9dfa2a8b123c00a..8b617948ef23c3ed8b571b7c6d22868795b7395e 100644
--- a/docs/assets/app.js
+++ b/docs/assets/app.js
@@ -216,53 +216,53 @@ function rbfKernel(x1, x2, ell=2.0, sigma=1.0){
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
-  let var = kxx - dot(k, v);
-  if (!Number.isFinite(var) || var < 1e-10) var = 1e-10;
-  return {mu, var};
+  let variance = kxx - dot(k, v);
+  if (!Number.isFinite(variance) || variance < 1e-10) variance = 1e-10;
+  return {mu, var: variance};
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
@@ -371,89 +371,91 @@ function drawLandscapeWithPoints(points){
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
 
-  const ymin = Math.min(...bestSoFar);
-  const ymax = Math.max(...bestSoFar);
+  const series = (bestSoFar && bestSoFar.length) ? bestSoFar : [0];
+  const ymin = Math.min(...series);
+  const ymax = Math.max(...series);
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
-  for (let i=0;i<bestSoFar.length;i++){
-    const x = pad + (i/(bestSoFar.length-1))*innerW;
-    const y = pad + (1 - (bestSoFar[i]-y0)/(y1-y0))*innerH;
+  const denom = Math.max(series.length - 1, 1);
+  for (let i=0;i<series.length;i++){
+    const x = pad + (i/denom)*innerW;
+    const y = pad + (1 - (series[i]-y0)/(y1-y0))*innerH;
     if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
   }
   ctx.stroke();
 
   // last marker
-  const lx = pad + innerW;
-  const ly = pad + (1 - (bestSoFar[bestSoFar.length-1]-y0)/(y1-y0))*innerH;
+  const lx = pad + ((series.length - 1) / denom) * innerW;
+  const ly = pad + (1 - (series[series.length-1]-y0)/(y1-y0))*innerH;
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
@@ -485,42 +487,44 @@ function run(){
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
+  updateRepoLink("labagentbench");
   drawLandscapeBase();
   reset();
 
   document.getElementById("runBtn").addEventListener("click", run);
   document.getElementById("resetBtn").addEventListener("click", reset);
 
   document.getElementById("algo").addEventListener("change", () => updateValLabel("algo"));
   document.getElementById("iters").addEventListener("input", () => updateValLabel("iters"));
   document.getElementById("seed").addEventListener("input", () => updateValLabel("seed"));
+  document.getElementById("hint").addEventListener("keydown", (event) => {
+    if (event.key === "Enter") run();
+  });
 
   // set initial labels
   ["algo","iters","seed"].forEach(updateValLabel);
 }
 
 main();
-
-    updateRepoLink("labagentbench");
