// node tests/js/trainer_unit.test.js — 프레임워크 없이 assert 만 사용
const assert = require("assert");
const path = require("path");
const L = require(path.join(__dirname, "..", "..", "web", "trainer.js"));

const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const small = { d_model: 32, n_layer: 2, n_head: 4, seq_len: 16, vocab_size: 24 };

test("paramCount = SPEC 공식, tensorSpecs 합과 일치", () => {
  for (const a of [small, { d_model: 128, n_layer: 2, n_head: 4, seq_len: 64, vocab_size: 2000 }]) {
    const V = a.vocab_size, d = a.d_model, T = a.seq_len, Ln = a.n_layer;
    const f = V * d + T * d + Ln * (12 * d * d + 13 * d) + 2 * d;
    assert.strictEqual(L.paramCount(a), f);
    assert.strictEqual(L.tensorSpecs(a).reduce((s, t) => s + t.size, 0), f);
  }
});

test("tensorSpecs 순서와 shape", () => {
  const s = L.tensorSpecs(small).map((t) => t.name);
  assert.deepStrictEqual(s.slice(0, 4), ["tok.weight", "pos.weight", "blocks.0.ln1.weight", "blocks.0.ln1.bias"]);
  assert.strictEqual(s[4], "blocks.0.qkv.weight");
  assert.strictEqual(s[12], "blocks.0.mlp.2.weight"); assert.strictEqual(s[13], "blocks.0.mlp.2.bias");
  assert.strictEqual(s[14], "blocks.1.ln1.weight");
  assert.deepStrictEqual(s.slice(-2), ["ln.weight", "ln.bias"]);
  assert.deepStrictEqual(L.tensorSpecs(small)[4].shape, [96, 32]);
  assert.deepStrictEqual(L.tensorSpecs(small)[10].shape, [128, 32]); assert.deepStrictEqual(L.tensorSpecs(small)[12].shape, [32, 128]);
});

test("createModel 초기화(LN 1/0, bias 0, 가중치 std≈0.02) 와 재현성", () => {
  const m = L.createModel(small, 5), m2 = L.createModel(small, 5), m3 = L.createModel(small, 6);
  assert.ok(m.params["ln.weight"].every((v) => v === 1));
  assert.ok(m.params["blocks.1.qkv.bias"].every((v) => v === 0));
  const w = m.params["tok.weight"];
  let s2 = 0; for (const v of w) s2 += v * v;
  assert.ok(Math.abs(Math.sqrt(s2 / w.length) - 0.02) < 0.003);
  assert.deepStrictEqual(Array.from(m.params["tok.weight"].slice(0, 8)), Array.from(m2.params["tok.weight"].slice(0, 8)));
  assert.notDeepStrictEqual(Array.from(m.params["tok.weight"].slice(0, 8)), Array.from(m3.params["tok.weight"].slice(0, 8)));
});

test("erf 정확도 (알려진 값)", () => {
  const ref = [[0.5, 0.5204998778130465], [1, 0.8427007929497149], [2, 0.9953222650189527], [3, 0.9999779095030014]];
  for (const [x, e] of ref) {
    assert.ok(Math.abs(L.erf(x) - e) < 1e-8, "erf(" + x + ")");
    assert.ok(Math.abs(L.erf(-x) + e) < 1e-8);
  }
  assert.strictEqual(L.erf(0), 0);
});

test("직렬화 왕복(가중치, Adam m/v/t, rng)", () => {
  const m = L.createModel(small, 1), opt = L.newOptState(m), rng = L.makeRng(9);
  const ids = new Int32Array(300).map((_, i) => 4 + (i % 7));
  const io = {};
  for (let s = 1; s <= 3; s++) L.trainStep(m, opt, ids, 2, 8, rng, s, 10, 3e-3, io);
  const st = L.serializeState(m, opt, rng);
  assert.strictEqual(st.opt.t, 3);
  const m2 = L.createModel(small, 99), rng2 = L.makeRng(0);
  const r = L.deserializeState(m2, st, rng2);
  for (const t of m.specs) {
    assert.deepStrictEqual(m2.params[t.name], m.params[t.name]);
    assert.deepStrictEqual(r.opt.m[t.name], opt.m[t.name]);
    assert.deepStrictEqual(r.opt.v[t.name], opt.v[t.name]);
  }
  assert.strictEqual(r.opt.t, 3);
  assert.strictEqual(rng2(), rng());
  assert.strictEqual(L.transferList(st).length, 3 * m.specs.length);
  // 복원 후 이어서 학습하면 원본과 같은 궤적
  const a = L.trainStep(m, opt, ids, 2, 8, rng, 4, 10, 3e-3, io);
  const b = L.trainStep(m2, r.opt, ids, 2, 8, rng2, 4, 10, 3e-3, {});
  assert.ok(Math.abs(a - b) < 1e-6);
});

test("getBatch 형태·정렬·범위", () => {
  const ids = Int32Array.from({ length: 100 }, (_, i) => i);
  const rng = L.makeRng(3);
  let maxStart = 0, minStart = 1e9;
  for (let k = 0; k < 300; k++) {
    const { x, y } = L.getBatch(ids, 5, 10, rng);
    assert.strictEqual(x.length, 50); assert.strictEqual(y.length, 50);
    for (let r = 0; r < 5; r++) {
      for (let t = 0; t < 10; t++) assert.strictEqual(y[r * 10 + t], x[r * 10 + t] + 1);
      for (let t = 1; t < 10; t++) assert.strictEqual(x[r * 10 + t], x[r * 10] + t);
      maxStart = Math.max(maxStart, x[r * 10]); minStart = Math.min(minStart, x[r * 10]);
    }
  }
  assert.strictEqual(maxStart, 100 - 10 - 1); // n-T-1 까지 나옴
  assert.ok(minStart <= 2);
});

test("generate: EOS 에서 멈추고 특수 토큰을 내지 않는다", () => {
  const m = L.createModel(small, 2);
  // EOS(3)가 압도적으로 높게 만든다 -> sampleTopK 는 EOS 허용(종료 신호)
  const tok = m.params["tok.weight"], d = small.d_model;
  const rng = L.makeRng(1);
  const out = L.generate(m, [2, 5, 6], 20, 0.8, 5, rng);
  assert.ok(out.length <= 20);
  for (const id of out) assert.ok(id >= 4 && id < small.vocab_size);
  // 강제로 EOS 우세: 마지막 LN bias 를 크게 조정하는 대신 logits 직접 검사
  const lg = new Float32Array(small.vocab_size).fill(-5); lg[3] = 10;
  assert.strictEqual(L.sampleTopK(lg, 5, 0.8, rng), 3);
  lg[3] = -5; lg[1] = 100; lg[0] = 100;
  for (let i = 0; i < 50; i++) assert.ok(L.sampleTopK(lg, 5, 1.0, rng) >= 3);
  assert.deepStrictEqual(L.generate(m, [2], 0, 0.8, 5, rng), []);
  // 학습한 뒤 EOS 가 항상 뒤따르는 패턴이면 생성이 짧게 끝난다
  const pat = []; for (let i = 0; i < 400; i++) pat.push(2, 4, 5, 6, 3);
  const ids = Int32Array.from(pat), opt = L.newOptState(m), io = {}, r2 = L.makeRng(4);
  for (let s = 1; s <= 120; s++) L.trainStep(m, opt, ids, 8, 8, r2, s, 120, 5e-3, io);
  const g = L.generate(m, [2, 4], 30, 0.01, 3, L.makeRng(5));
  assert.deepStrictEqual(g, [5, 6], "학습된 패턴 4 5 6 <eos>");
});

test("lrAt 스케줄", () => {
  const lr = 3e-3, steps = 100; // warmup 10
  assert.ok(Math.abs(L.lrAt(1, steps, lr) - lr / 10) < 1e-12);
  assert.ok(Math.abs(L.lrAt(10, steps, lr) - lr) < 1e-12);
  assert.ok(Math.abs(L.lrAt(100, steps, lr) - 0.1 * lr) < 1e-12);
  assert.ok(L.lrAt(55, steps, lr) < lr && L.lrAt(55, steps, lr) > 0.1 * lr);
  assert.ok(Math.abs(L.lrAt(1, 5, lr) - lr) < 1e-12); // steps<10 -> warmup 1
  assert.ok(Math.abs(L.lrAt(1, 5000, lr) - lr / 100) < 1e-12); // warmup 상한 100
});

test("30단계 학습이 손실을 낮춘다 (반복 말뭉치)", () => {
  const V = 16, arch = { d_model: 32, n_layer: 2, n_head: 4, seq_len: 16, vocab_size: V };
  const m = L.createModel(arch, 1), opt = L.newOptState(m), rng = L.makeRng(2);
  const pat = []; for (let i = 0; i < 300; i++) pat.push(4 + (i % 5), 9, 10, 11);
  const ids = Int32Array.from(pat), io = {};
  const before = L.evaluate(m, ids, 4, 8, 16);
  let first = null, lastLoss = null;
  for (let s = 1; s <= 30; s++) {
    const l = L.trainStep(m, opt, ids, 8, 16, rng, s, 30, 5e-3, io);
    if (first == null) first = l; lastLoss = l;
  }
  const after = L.evaluate(m, ids, 4, 8, 16);
  assert.ok(after < before - 0.5, "eval " + before + " -> " + after);
  assert.ok(lastLoss < first - 0.5);
});

// ---- Worker 메시지 하네스 ----
function harness() {
  const msgs = [];
  const handler = L.makeHandler((m) => msgs.push(m));
  const waitFor = async (pred, ms = 20000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { const f = msgs.find(pred); if (f) return f; await sleep(5); }
    throw new Error("메시지 대기 시간 초과");
  };
  return { msgs, handler, waitFor };
}
function corpusIds(n, V) { const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = 4 + ((i * 7 + (i >> 3)) % (V - 4)); return a; }

test("worker: init/train/checkpoint/done/export/eval/generate/bench", async () => {
  const h = harness();
  const ids = corpusIds(2000, 24);
  await h.handler({ data: { type: "init", arch: small, weights: null, itos: null, corpus: ids, evalIds: ids.slice(0, 400), seed: 3 } });
  const ready = await h.waitFor((m) => m.type === "ready");
  assert.strictEqual(ready.params, L.paramCount(small));
  await h.handler({ data: { type: "bench", steps: 4, batch: 2, seqLen: 8 } });
  const bn = h.msgs.find((m) => m.type === "bench");
  assert.ok(bn.msPerStep > 0 && bn.tokensPerSec > 0);
  await h.handler({ data: { type: "train", steps: 10, batch: 4, seqLen: 8, lr: 3e-3, startStep: 0, totalPlanned: 10, checkpointEvery: 4 } });
  const prog = h.msgs.filter((m) => m.type === "progress");
  assert.strictEqual(prog.length, 10);
  assert.strictEqual(prog[9].step, 10); assert.strictEqual(prog[9].eta, 0);
  const cps = h.msgs.filter((m) => m.type === "checkpoint");
  assert.deepStrictEqual(cps.map((c) => c.step), [4, 8]);
  assert.strictEqual(cps[0].state.opt.t, 4);
  const done = h.msgs.find((m) => m.type === "done");
  assert.strictEqual(done.step, 10);
  await h.handler({ data: { type: "eval", batches: 2, batch: 4, seqLen: 8 } });
  assert.ok(isFinite(h.msgs.find((m) => m.type === "eval").loss));
  await h.handler({ data: { type: "export" } });
  const w = h.msgs.find((m) => m.type === "weights").weights;
  assert.strictEqual(Object.keys(w).length, L.tensorSpecs(small).length);
  await h.handler({ data: { type: "generate", prompt: Int32Array.of(2, 4), maxNew: 5, temperature: 0.8, topK: 5, seed: 1 } });
  const gen = h.msgs.find((m) => m.type === "generated");
  assert.strictEqual(h.msgs.filter((m) => m.type === "token").length, gen.ids.length);
  assert.ok(!h.msgs.some((m) => m.type === "error"), JSON.stringify(h.msgs.filter((m) => m.type === "error")));
});

test("worker: 가중치로 init 하면 bench 가 원본을 바꾸지 않는다", async () => {
  const h = harness();
  const base = L.createModel(small, 11);
  const w = {}; for (const t of base.specs) w[t.name] = base.params[t.name].slice();
  await h.handler({ type: "init", arch: small, weights: w, corpus: corpusIds(1000, 24), evalIds: corpusIds(200, 24) });
  await h.handler({ type: "bench", steps: 3, batch: 2, seqLen: 8 });
  await h.handler({ type: "export" });
  const out = h.msgs.find((m) => m.type === "weights").weights;
  for (const k in w) assert.deepStrictEqual(out[k], w[k]);
});

test("worker: pause / resume / stop", async () => {
  const h = harness();
  await h.handler({ data: { type: "init", arch: small, corpus: corpusIds(3000, 24), evalIds: corpusIds(300, 24), seed: 1 } });
  const run = h.handler({ data: { type: "train", steps: 400, batch: 4, seqLen: 8, lr: 3e-3, startStep: 0, totalPlanned: 400, checkpointEvery: 0 } });
  await h.waitFor((m) => m.type === "progress" && m.step >= 3);
  await h.handler({ data: { type: "pause" } });
  await sleep(60);
  const n1 = h.msgs.filter((m) => m.type === "progress").length;
  await sleep(150);
  const n2 = h.msgs.filter((m) => m.type === "progress").length;
  assert.strictEqual(n1, n2, "일시정지 중에는 진행 없음");
  assert.ok(!h.msgs.some((m) => m.type === "done" || m.type === "stopped"));
  await h.handler({ data: { type: "resume" } });
  await h.waitFor((m) => m.type === "progress" && m.step > n2);
  await h.handler({ data: { type: "stop" } });
  await run;
  const st = h.msgs.find((m) => m.type === "stopped");
  assert.ok(st && st.step >= n2 && st.step < 400);
  assert.strictEqual(st.state.opt.t, st.step);
  assert.ok(st.state.weights["tok.weight"] instanceof Float32Array);
  assert.ok(!h.msgs.some((m) => m.type === "done"));
  // 일시정지 상태에서 stop 도 풀려야 한다
  const run2 = h.handler({ data: { type: "train", steps: 400, batch: 4, seqLen: 8, lr: 3e-3, startStep: st.step, totalPlanned: 800 } });
  await sleep(30);
  await h.handler({ data: { type: "pause" } });
  await sleep(50);
  await h.handler({ data: { type: "stop" } });
  await run2;
  assert.strictEqual(h.msgs.filter((m) => m.type === "stopped").length, 2);
});

test("worker: 오류는 한국어 error 메시지", async () => {
  const h = harness();
  await h.handler({ data: { type: "train", steps: 1 } });
  assert.ok(/[가-힣]/.test(h.msgs.find((m) => m.type === "error").message));
  await h.handler({ data: { type: "init", arch: small, corpus: new Int32Array(5), evalIds: null } });
  await h.handler({ data: { type: "train", steps: 1, batch: 2, seqLen: 8 } });
  assert.strictEqual(h.msgs.filter((m) => m.type === "error").length, 2);
});

(async () => {
  let fail = 0;
  for (const [name, fn] of tests) {
    try { await fn(); console.log("ok   - " + name); }
    catch (e) { fail++; console.log("FAIL - " + name + "\n  " + (e && e.stack || e)); }
  }
  console.log((tests.length - fail) + "/" + tests.length + " passed");
  process.exit(fail ? 1 : 0);
})();
