// 사용: node tests/js/trainer_parity.js <fixture.json>
// 픽스처: {arch, weights, x, y, B, T, lr, step, steps, expected:{loss, grads, updated, eval_loss?}}
const fs = require("fs");
const path = require("path");
const L = require(path.join(__dirname, "..", "..", "web", "trainer.js"));

const fx = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const w = {};
for (const k in fx.weights) w[k] = Float32Array.from(fx.weights[k]);
const model = L.loadWeights(fx.arch, w);
const x = Int32Array.from(fx.x), y = Int32Array.from(fx.y);
const r = L.forwardLoss(model, x, y, fx.B, fx.T);
const grads = L.backward(model, r.cache);

function maxRel(got, exp) {
  let worst = 0;
  for (const k in exp) {
    const a = got[k], e = exp[k];
    if (!a || a.length !== e.length) return Infinity;
    for (let i = 0; i < e.length; i++) {
      const rel = Math.abs(a[i] - e[i]) / Math.max(Math.abs(e[i]), 1e-3);
      if (rel > worst) worst = rel;
    }
  }
  return worst;
}
const e = fx.expected;
const lossErr = Math.abs(r.loss - e.loss);
const gradErr = maxRel(grads, e.grads);
const opt = L.newOptState(model);
L.adamwStep(model, grads, opt, fx.step || 1, L.lrAt(fx.step || 1, fx.steps, fx.lr));
const updErr = maxRel(model.params, e.updated);
const out = { loss: r.loss, loss_abs_err: lossErr, max_rel_grad_err: gradErr, max_rel_update_err: updErr };
if (e.eval_loss != null) {
  const ev = L.evaluate(L.loadWeights(fx.arch, w), Int32Array.from(fx.eval_ids), 2, fx.B, fx.T, 1234);
  out.eval_loss = ev;
}
out.ok = lossErr <= 1e-3 && gradErr <= 1e-2 && updErr <= 1e-2;
console.log(JSON.stringify(out));
process.exit(out.ok ? 0 : 1);
