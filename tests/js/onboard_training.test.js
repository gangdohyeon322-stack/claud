// node tests/js/onboard_training.test.js — Onboard.lbw / Onboard.training 단위 + 가짜/실제(인프로세스) Worker 통합 테스트
const assert = require("assert");
const path = require("path");
const WEB = path.join(__dirname, "..", "..", "web");

// ---- 최소 브라우저 환경 ----
const listeners = {};
const doc = {
  visibilityState: "visible",
  addEventListener(t, f) { (listeners[t] = listeners[t] || []).push(f); },
  removeEventListener(t, f) { listeners[t] = (listeners[t] || []).filter((x) => x !== f); },
};
globalThis.document = doc;
let battery = { charging: true, level: 0.9 };
const navStub = {
  userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 7) Mobile", deviceMemory: 8, hardwareConcurrency: 8, maxTouchPoints: 5,
  storage: { estimate: async () => ({ quota: 1e9, usage: 1e6 }) },
  wakeLock: { request: async () => ({ release() { navStub._released = true; } }) },
  getBattery: async () => battery,
};
Object.defineProperty(globalThis, "navigator", { value: navStub, configurable: true, writable: true });

const Onboard = require(path.join(WEB, "onboard.js"));
const LBTrain = require(path.join(WEB, "trainer.js"));
const T = Onboard.training;

// ---- 메모리 kv (Onboard.db 를 덮어씀) ----
const mem = new Map();
Onboard.db.get = async (k) => (mem.has(k) ? mem.get(k) : undefined);
Onboard.db.set = async (k, v) => { mem.set(k, v); };
Onboard.db.del = async (k) => { mem.delete(k); };
Onboard.db.setMany = async (o) => { for (const k of Object.keys(o)) { if (o[k] === undefined) mem.delete(k); else mem.set(k, o[k]); } };

const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms = 8000) {
  const t0 = Date.now();
  while (!fn()) { if (Date.now() - t0 > ms) throw new Error("until 시간 초과"); await sleep(5); }
}

function rows(n, gen) { return Array.from({ length: n }, (_, i) => ({ c: i, d: i, t: "", x: gen(i), u: "", s: "link" })); }
async function putSnapshot(list) { mem.set("snapshot", { v: 2, chunks: list, index: null, saved_at: "2026-11-15T00:00:00Z", count: list.length }); }
const TEXT = "가나다라마바사 아자차카타파하 abcdefg ";
function bigRows(n = 40) { return rows(n, (i) => TEXT.repeat(6) + (i % 7)); }

// 실제 trainer.js 핸들러를 같은 프로세스에서 Worker 처럼 쓴다
function inProcWorker(log) {
  const w = { onmessage: null, terminated: false };
  const handler = LBTrain.makeHandler((msg) => {
    if (w.terminated) return;
    log && log.push(msg);
    setTimeout(() => w.onmessage && w.onmessage({ data: msg }), 0);
  });
  w.postMessage = (msg) => { if (!w.terminated) { log && log.push({ _sent: msg.type }); handler({ data: msg }); } };
  w.terminate = () => { w.terminated = true; };
  return w;
}
// 대본대로 답하는 가짜 Worker
function scriptedWorker(script) {
  const w = { sent: [], onmessage: null, terminated: false };
  const reply = (m) => setTimeout(() => !w.terminated && w.onmessage && w.onmessage({ data: m }), 0);
  w.reply = reply;
  w.postMessage = (msg) => { w.sent.push(msg); script(msg, reply, w); };
  w.terminate = () => { w.terminated = true; };
  return w;
}
const smallArch = { d_model: 32, n_layer: 1, n_head: 4, seq_len: 16, vocab_size: 100 };

test("lbw: build/parse 왕복", () => {
  const arch = { d_model: 32, n_layer: 1, n_head: 4, seq_len: 16, vocab_size: 10 };
  const specs = T._tensorSpecs(arch);
  const tensors = {};
  let k = 0;
  for (const s of specs) { const n = s.shape.reduce((a, b) => a * b, 1); tensors[s.name] = Float32Array.from({ length: n }, () => (k++ % 97) / 7 - 3); }
  const header = { format: "lbw", arch, itos: ["<pad>", "<unk>", "<bos>", "<eos>", "가", "a", "b", "c", "d", "e"], tensors: specs,
    parent: null, total_steps: 5, trained_on: "device" };
  const buf = Onboard.lbw.build(header, tensors);
  assert.ok(buf instanceof ArrayBuffer);
  const u8 = new Uint8Array(buf);
  assert.strictEqual(String.fromCharCode(...u8.slice(0, 8)), "LBWEIGHT");
  assert.strictEqual(new DataView(buf).getUint32(8, true), 1);
  const back = Onboard.lbw.parse(buf);
  assert.deepStrictEqual(back.header.itos, header.itos);
  assert.strictEqual(back.header.total_steps, 5);
  assert.deepStrictEqual(back.header.tensors.map((t) => t.name), specs.map((s) => s.name));
  for (const s of specs) assert.deepStrictEqual(Array.from(back.tensors[s.name]), Array.from(tensors[s.name]));
  // 한글이 든 헤더라 payload 시작이 4의 배수가 아니어도 읽혀야 한다
  const hl = new DataView(buf).getUint32(12, true);
  assert.ok((16 + hl) % 4 !== undefined);
});

test("lbw: 잘못된 파일은 한국어 오류", () => {
  const arch = { d_model: 4, n_layer: 1, n_head: 1, seq_len: 4, vocab_size: 5 };
  const header = { arch, itos: [], tensors: [{ name: "a", shape: [3] }] };
  const good = Onboard.lbw.build(header, { a: Float32Array.of(1, 2, 3) });
  // 페이로드 길이 불일치 (잘림)
  assert.throws(() => Onboard.lbw.parse(good.slice(0, good.byteLength - 4)), /크기가 맞지 않습니다/);
  // 페이로드가 더 긴 경우
  const longer = new Uint8Array(good.byteLength + 4); longer.set(new Uint8Array(good));
  assert.throws(() => Onboard.lbw.parse(longer.buffer), /크기가 맞지 않습니다/);
  // 매직
  const bad = good.slice(0); new Uint8Array(bad)[0] = 0x58;
  assert.throws(() => Onboard.lbw.parse(bad), /가중치 파일/);
  // 버전
  const v2 = good.slice(0); new DataView(v2).setUint32(8, 2, true);
  assert.throws(() => Onboard.lbw.parse(v2), /버전/);
  assert.throws(() => Onboard.lbw.parse(new ArrayBuffer(4)), /작거나/);
  // 빌드 시 텐서 누락/크기 오류
  assert.throws(() => Onboard.lbw.build(header, {}), /텐서가 없습니다/);
  assert.throws(() => Onboard.lbw.build(header, { a: Float32Array.of(1) }), /크기/);
});

test("prepareCorpus: 어휘 생성(특수 토큰 선두, 빈도순), BOS/EOS, holdout", async () => {
  mem.clear();
  await putSnapshot(rows(40, (i) => (i % 2 ? "aab" : "abc") + "가"));
  const r = await T.prepareCorpus({ itos: null, vocabSize: 50 });
  assert.deepStrictEqual(r.itos.slice(0, 4), ["<pad>", "<unk>", "<bos>", "<eos>"]);
  // 빈도: a(60) , 가(40), b(40), c(20) -> a 다음 가/b 는 코드포인트 순 (b < 가)
  assert.deepStrictEqual(r.itos.slice(4), ["a", "b", "가", "c"]);
  assert.strictEqual(r.unkRate, 0);
  // 40조각 중 5% -> period 20 -> 인덱스 19, 39 가 평가용(2조각)
  const chunkLen = (s) => s.length + 2;
  assert.strictEqual(r.eval.length, chunkLen("aab가") + chunkLen("abc가"));
  assert.strictEqual(r.train.length + r.eval.length, 40 * 6);
  assert.strictEqual(r.train[0], 2); assert.strictEqual(r.train[5], 3); assert.strictEqual(r.train[6], 2);
  assert.strictEqual(r.eval[0], 2); assert.strictEqual(r.eval[r.eval.length - 1], 3);
  assert.strictEqual(r.chars, 40 * 4);
  // 결정적
  const r2 = await T.prepareCorpus({ itos: null, vocabSize: 50 });
  assert.deepStrictEqual(Array.from(r2.train), Array.from(r.train));
  assert.deepStrictEqual(Array.from(r2.eval), Array.from(r.eval));
  assert.ok(r.train instanceof Int32Array && r.eval instanceof Int32Array);
});

test("prepareCorpus: vocabSize 상한과 고정 itos 의 unkRate", async () => {
  mem.clear();
  await putSnapshot(rows(20, () => "aaaabbbcc가"));
  const small = await T.prepareCorpus({ itos: null, vocabSize: 6 }); // 특수 4 + 글자 2
  assert.deepStrictEqual(small.itos, ["<pad>", "<unk>", "<bos>", "<eos>", "a", "b"]);
  assert.ok(Math.abs(small.unkRate - 3 / 10) < 1e-9, "c, c, 가 -> unk: " + small.unkRate);
  const fixed = ["<pad>", "<unk>", "<bos>", "<eos>", "a", "b", "c"];
  const f = await T.prepareCorpus({ itos: fixed });
  assert.deepStrictEqual(f.itos, fixed);
  assert.ok(Math.abs(f.unkRate - 1 / 10) < 1e-9);
  assert.ok(Array.from(f.train).includes(1)); // UNK id
  assert.ok(Array.from(f.train).every((v) => v >= 0 && v < fixed.length));
});

test("prepareCorpus: 자료 없음은 한국어 오류, 조각이 적으면 마지막 하나만 평가용", async () => {
  mem.clear();
  await assert.rejects(T.prepareCorpus({}), /저장된 자료가 없습니다/);
  const r = T._buildCorpus(rows(3, (i) => "abc" + i), { holdout: 0.05 });
  assert.strictEqual(r.evalChunks, 1);
  const none = T._buildCorpus(rows(5, () => "abc"), { holdout: 0 });
  assert.strictEqual(none.eval.length, 0);
});

test("benchmark 등급 경계 (_grade)", () => {
  assert.strictEqual(T._grade(0), "good");
  assert.strictEqual(T._grade(1000), "good");
  assert.strictEqual(T._grade(1000.1), "slow");
  assert.strictEqual(T._grade(5000), "slow");
  assert.strictEqual(T._grade(5000.1), "bad");
});

test("benchmark: 가짜 Worker 로 결과 계산·라벨·종료", async () => {
  let w;
  T._setWorkerFactory(() => (w = scriptedWorker((m, reply) => {
    if (m.type === "init") reply({ type: "ready", params: 123 });
    if (m.type === "bench") reply({ type: "bench", msPerStep: 1500, tokensPerSec: 341 });
  })));
  const prog = [];
  const r = await T.benchmark({ arch: smallArch, batch: 8, seqLen: 64, onProgress: (p) => prog.push(p) });
  assert.strictEqual(r.grade, "slow");
  assert.strictEqual(r.label, "느림");
  assert.strictEqual(r.est100Sec, 150);
  assert.strictEqual(r.est300Sec, 450);
  assert.ok(w.terminated);
  const bm = w.sent.find((m) => m.type === "bench");
  assert.strictEqual(bm.steps, 10); assert.strictEqual(bm.batch, 8); assert.strictEqual(bm.seqLen, 16); // seq_len 로 제한
  assert.strictEqual(w.sent[0].type, "init"); assert.strictEqual(w.sent[0].weights, null);
  assert.ok(prog.length >= 2 && prog[prog.length - 1] === 1);
  // 오류 전파
  T._setWorkerFactory(() => scriptedWorker((m, reply) => { if (m.type === "init") reply({ type: "error", message: "초기화 실패" }); }));
  await assert.rejects(T.benchmark({ arch: smallArch }), /초기화 실패/);
  T._setWorkerFactory(null);
});

test("세션: 가짜 Worker — 체크포인트 저장 형식, 완료 결과, .lbw 내보내기", async () => {
  mem.clear(); T._reset();
  await putSnapshot(bigRows());
  const log = [];
  let w;
  T._setWorkerFactory(() => (w = scriptedWorker((m, reply) => {
    log.push(m.type);
    const arch = (w.arch = w.arch || m.arch);
    const fakeW = () => Object.fromEntries(T._tensorSpecs(w.arch).map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1)).fill(0.5)]));
    if (m.type === "init") { w.init = m; reply({ type: "ready", params: 1 }); }
    if (m.type === "eval") reply({ type: "eval", loss: log.filter((x) => x === "eval").length === 1 ? 4.5 : 3.2 });
    if (m.type === "train") {
      w.train = m;
      let i = m.startStep;
      const tick = () => {
        i++;
        reply({ type: "progress", step: i, loss: 4 - i * 0.01, msPerStep: 10, eta: (m.totalPlanned - i) / 100 });
        if (i % m.checkpointEvery === 0 && i < m.totalPlanned) {
          const f = fakeW(); const z = Object.fromEntries(Object.keys(f).map((k) => [k, new Float32Array(f[k].length)]));
          reply({ type: "checkpoint", step: i, state: { weights: f, opt: { m: z, v: z, t: i }, rng: 77 } });
        }
        if (i < m.startStep + m.steps) setTimeout(tick, 1); else reply({ type: "done", step: i, trainLoss: 3.9 });
      };
      setTimeout(tick, 1);
    }
    if (m.type === "export") reply({ type: "weights", weights: fakeW() });
    if (m.type === "generate") reply({ type: "generated", ids: [4, 5, 6] });
  })));
  const states = [];
  const st = await T.start({ mode: "scratch", arch: { d_model: 32, n_layer: 1, n_head: 4, seq_len: 16, vocab_size: 200 }, steps: 45, batch: 2, seqLen: 8,
    lr: 3e-3, checkpointEvery: 20, onState: (s) => states.push(s.phase) });
  assert.strictEqual(st.active, true);
  const fin = await T.whenDone();
  assert.strictEqual(fin.phase, "done");
  // init: 어휘 크기가 실제 글자 수로 맞춰진다
  assert.strictEqual(w.init.arch.vocab_size, w.init.itos.length);
  assert.deepStrictEqual(w.init.itos.slice(0, 4), ["<pad>", "<unk>", "<bos>", "<eos>"]);
  assert.ok(w.init.corpus instanceof Int32Array && w.init.evalIds instanceof Int32Array && w.init.weights === null);
  assert.strictEqual(w.train.totalPlanned, 45);
  assert.deepStrictEqual(states.filter((s, i) => states.indexOf(s) === i), ["preparing", "evaluating", "training", "checkpoint", "finishing", "done"]);
  // 완료 후 train_session 은 지워지고 train_result 가 생긴다
  assert.ok(!mem.has("train_session"));
  const res = mem.get("train_result");
  assert.ok(res && res.header && res.tensors);
  assert.strictEqual(res.evalBefore, 4.5); assert.strictEqual(res.evalAfter, 3.2);
  assert.strictEqual(res.sample.length, 3);
  assert.deepStrictEqual(Object.keys(res).sort(), ["evalAfter", "evalBefore", "header", "sample", "seconds", "tensors", "trainLoss"]);
  const h = res.header;
  assert.strictEqual(h.trained_on, "device"); assert.strictEqual(h.steps, 45); assert.strictEqual(h.total_steps, 45);
  assert.strictEqual(h.parent, null); assert.strictEqual(h.parent_model_id, null);
  assert.ok(/Pixel|휴대폰|고성능|중급/.test(h.device.label) && h.device.ua.includes("Pixel"));
  assert.strictEqual(h.eval_loss_before, 4.5); assert.strictEqual(h.eval_loss_after, 3.2);
  assert.ok(h.train_chars > 0 && h.created_at.endsWith("Z"));
  assert.strictEqual(h.arch.vocab_size, h.itos.length);
  assert.strictEqual(h.tensors.length, T._tensorSpecs(h.arch).length);
  assert.strictEqual(navStub._released, true); // wake lock 해제
  // .lbw 내보내기 -> 다시 읽기
  const buf = await T.exportLbw();
  const back = Onboard.lbw.parse(buf);
  assert.strictEqual(back.header.trained_on, "device");
  assert.strictEqual(back.tensors["tok.weight"][0], 0.5);
  assert.ok(T.result() && T.result().header.steps === 45);
});

test("체크포인트 저장 형식과 이어서 학습(continueSaved) + 소스 모델 메타", async () => {
  mem.clear(); T._reset();
  await putSnapshot(bigRows());
  const itos = ["<pad>", "<unk>", "<bos>", "<eos>", ..."가나다라마바사 아자차카타파하abcdefg"];
  const arch = { d_model: 32, n_layer: 1, n_head: 4, seq_len: 16, vocab_size: itos.length };
  const specs = T._tensorSpecs(arch);
  const tens = Object.fromEntries(specs.map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1)).fill(0.25)]));
  const lbw = { header: { arch, itos, total_steps: 140, steps: 100 }, tensors: tens };
  const log = [];
  let w;
  const mk = () => (w = scriptedWorker((m, reply) => {
    log.push(m);
    if (m.type === "init") reply({ type: "ready", params: 1 });
    if (m.type === "eval") reply({ type: "eval", loss: 3.0 });
    if (m.type === "train") { // 20단계에서 체크포인트 한 번 보내고 멈춘 척 대기 (stop 을 기다림)
      const f = Object.fromEntries(specs.map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1)).fill(1)]));
      const z = Object.fromEntries(specs.map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1))]));
      reply({ type: "progress", step: m.startStep + 20, loss: 3.5, msPerStep: 5, eta: 1 });
      reply({ type: "checkpoint", step: m.startStep + 20, state: { weights: f, opt: { m: z, v: z, t: 20 }, rng: 4242 } });
      w.f = f; w.z = z;
    }
    if (m.type === "stop") reply({ type: "stopped", step: 20, state: { weights: w.f, opt: { m: w.z, v: w.z, t: 20 }, rng: 4243 } });
  }));
  T._setWorkerFactory(mk);
  await T.start({ mode: "resume", source: { modelId: 3, name: "tiny-20261115-041044", lbw }, steps: 60, batch: 2, seqLen: 8, checkpointEvery: 20,
    thermalPause: false, minBattery: 20, maxMinutes: 30 });
  assert.deepStrictEqual(w.sent[0].itos, itos); // 이어서: 원 모델의 어휘 사용
  assert.strictEqual(w.sent[0].weights["tok.weight"][0], 0.25);
  await until(() => mem.has("train_session"));
  const rec = mem.get("train_session");
  for (const k of ["config", "mode", "arch", "itos", "step", "totalSteps", "state", "evalBefore", "startedAt", "savedAt"]) assert.ok(k in rec, "키 없음: " + k);
  assert.strictEqual(rec.mode, "resume"); assert.strictEqual(rec.step, 20); assert.strictEqual(rec.totalSteps, 60);
  assert.strictEqual(rec.config.batch, 2); assert.strictEqual(rec.config.seqLen, 8); assert.strictEqual(rec.config.minBattery, 20);
  assert.strictEqual(rec.evalBefore, 3.0);
  assert.ok(rec.state.weights["tok.weight"] instanceof Float32Array && rec.state.opt.t === 20 && rec.state.rng === 4242);
  assert.strictEqual(rec.source.modelId, 3); assert.strictEqual(rec.source.totalSteps, 140);
  const hr = await T.hasResumable();
  assert.deepStrictEqual(Object.keys(hr).sort(), ["arch", "mode", "savedAt", "step", "totalSteps"]);
  assert.strictEqual(hr.step, 20);
  // 중지 -> 이어서
  T.stop();
  await T.whenDone();
  assert.strictEqual(T.status().phase, "idle");
  assert.strictEqual(T.status().canResume, true);
  assert.ok(w.terminated);
  // 새로 이어서 시작 (새로고침 흉내: 상태 초기화)
  const savedStep = mem.get("train_session").step;
  T._reset();
  const resumed = [];
  T._setWorkerFactory(() => (w = scriptedWorker((m, reply) => {
    resumed.push(m);
    if (m.type === "init") reply({ type: "ready", params: 1 });
    if (m.type === "train") reply({ type: "done", step: 60, trainLoss: 2.0 });
    if (m.type === "export") reply({ type: "weights", weights: Object.fromEntries(specs.map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1)).fill(2)])) });
    if (m.type === "eval") reply({ type: "eval", loss: 2.5 });
    if (m.type === "generate") reply({ type: "generated", ids: [] });
  })));
  await T.continueSaved();
  const fin = await T.whenDone();
  assert.strictEqual(fin.phase, "done");
  const init = resumed.find((m) => m.type === "init"), train = resumed.find((m) => m.type === "train");
  assert.strictEqual(init.weights["tok.weight"][0], 1); assert.strictEqual(init.opt.t, 20); assert.strictEqual(init.rng, 4243);
  assert.strictEqual(train.startStep, savedStep); assert.strictEqual(train.steps, 60 - savedStep); assert.strictEqual(train.totalPlanned, 60);
  assert.ok(!resumed.some((m) => m.type === "eval" && false));
  assert.strictEqual(resumed.filter((m) => m.type === "eval").length, 1); // 학습 전 평가는 건너뛰고 완료 후 평가 1회
  const h = mem.get("train_result").header;
  assert.strictEqual(h.parent, "tiny-20261115-041044"); assert.strictEqual(h.parent_model_id, 3);
  assert.strictEqual(h.total_steps, 140 + 60); assert.strictEqual(h.steps, 60);
  assert.strictEqual(mem.get("train_result").evalBefore, 3.0);
  assert.ok(!mem.has("train_session"));
});

test("discard 는 세션과 체크포인트를 지운다", async () => {
  mem.set("train_session", { step: 5, totalSteps: 10, state: {}, arch: {}, mode: "scratch", savedAt: "x" });
  assert.ok(await T.hasResumable());
  await T.discard();
  assert.strictEqual(await T.hasResumable(), null);
  assert.ok(!mem.has("train_session") && !mem.has("train_session_meta"));
});

test("배터리: 충전 중에만 옵션이면 충전 아닐 때 시작 거부, 체크포인트 때 중지", async () => {
  mem.clear(); T._reset(); await putSnapshot(bigRows());
  battery = { charging: false, level: 0.8 };
  T._setWorkerFactory(() => scriptedWorker(() => {}));
  await assert.rejects(T.start({ mode: "scratch", arch: smallArch, steps: 10, batch: 2, seqLen: 8, chargingOnly: true }), /충전/);
  assert.strictEqual(T.status().phase, "error");
  // 체크포인트 때 충전기 분리 -> stop 요청
  T._reset();
  battery = { charging: true, level: 0.8 };
  let w;
  T._setWorkerFactory(() => (w = scriptedWorker((m, reply) => {
    if (m.type === "init") reply({ type: "ready" });
    if (m.type === "eval") reply({ type: "eval", loss: 3 });
    if (m.type === "train") {
      setTimeout(() => {
        battery = { charging: false, level: 0.1 };
        const z = Object.fromEntries(T._tensorSpecs(w.arch).map((s) => [s.name, new Float32Array(s.shape.reduce((a, b) => a * b, 1))]));
        reply({ type: "progress", step: 3, loss: 3, msPerStep: 1, eta: 1 });
        reply({ type: "checkpoint", step: 3, state: { weights: z, opt: { m: z, v: z, t: 3 }, rng: 1 } });
      }, 5);
    }
    if (m.type === "stop") reply({ type: "stopped", step: 3, state: { weights: w.z, opt: null, rng: 1 } });
    if (m.type === "init") w.arch = m.arch;
  })));
  await T.start({ mode: "scratch", arch: smallArch, steps: 10, batch: 2, seqLen: 8, minBattery: 20, checkpointEvery: 3 });
  const fin = await T.whenDone();
  assert.strictEqual(fin.phase, "idle");
  assert.ok(/배터리/.test(fin.message), fin.message);
  assert.ok(w.sent.some((m) => m.type === "stop"));
  battery = { charging: true, level: 0.9 };
});

test("일시정지/재개, 백그라운드 자동 일시정지, 발열 휴식(cooling)", async () => {
  mem.clear(); T._reset(); await putSnapshot(bigRows());
  T._setCoolMs(120);
  let w, mult = 1, n = 0;
  T._setWorkerFactory(() => (w = scriptedWorker((m, reply) => {
    if (m.type === "init") reply({ type: "ready" });
    if (m.type === "eval") reply({ type: "eval", loss: 3 });
    if (m.type === "pause") w.paused = true;
    if (m.type === "resume") w.paused = false;
    if (m.type === "stop") { w.stop = true; reply({ type: "stopped", step: n, state: null }); }
    if (m.type === "train") {
      const tick = () => {
        if (w.terminated || w.stop) return;
        if (w.paused) return setTimeout(tick, 2);
        n++;
        if (n === 30) mult = 8; // 30단계부터 8배 느려짐
        reply({ type: "progress", step: n, loss: 3, msPerStep: 1, eta: 1 });
        setTimeout(tick, 3 * mult);
      };
      tick();
    }
  })));
  await T.start({ mode: "scratch", arch: smallArch, steps: 1000, batch: 2, seqLen: 8, checkpointEvery: 500 });
  await until(() => T.status().step >= 3);
  T.pause();
  assert.strictEqual(T.status().phase, "paused");
  assert.ok(w.sent.some((m) => m.type === "pause"));
  T.resume();
  assert.strictEqual(T.status().phase, "training");
  // 화면 숨김 -> 자동 일시정지, 보임 -> 재개
  doc.visibilityState = "hidden"; (listeners.visibilitychange || []).forEach((f) => f());
  assert.strictEqual(T.status().phase, "paused");
  doc.visibilityState = "visible"; (listeners.visibilitychange || []).forEach((f) => f());
  assert.strictEqual(T.status().phase, "training");
  // 발열: 30단계 이후 느려지면 cooling
  await until(() => T.status().phase === "cooling", 15000);
  assert.ok(/뜨거/.test(T.status().message));
  await until(() => T.status().phase === "training", 5000);
  T.stop();
  await T.whenDone();
  T._setCoolMs(60000);
});

test("실제 trainer.js(인프로세스)로 30단계 처음부터 학습 -> 결과 -> 기기 모델 등록 -> 생성", async () => {
  mem.clear(); T._reset(); await putSnapshot(bigRows(60));
  T._setWorkerFactory(() => inProcWorker());
  const prog = [];
  await T.start({ mode: "scratch", arch: { d_model: 32, n_layer: 1, n_head: 4, seq_len: 16, vocab_size: 100 }, steps: 30, batch: 4, seqLen: 16, lr: 3e-3,
    checkpointEvery: 10, onProgress: (s) => prog.push(s.step) });
  const fin = await T.whenDone();
  assert.strictEqual(fin.phase, "done", fin.message);
  assert.strictEqual(prog[prog.length - 1], 30);
  const r = T.result();
  assert.ok(r.evalAfter < r.evalBefore, "평가 손실이 줄어야 한다: " + r.evalBefore + " -> " + r.evalAfter);
  assert.ok(r.header.eval_loss_after < r.header.eval_loss_before);
  const specs = T._tensorSpecs(r.header.arch);
  for (const s of specs) assert.strictEqual(r.tensors[s.name].length, s.shape.reduce((a, b) => a * b, 1));
  // 가중치는 LBTrain 으로 다시 불러와도 유한
  assert.ok(Object.values(r.tensors).every((t) => t.every(Number.isFinite)));
  // 로컬 모델로 채택 -> modelInfo / generate (Worker)
  const meta = await T.adoptAsLocalModel();
  assert.strictEqual(meta.source, "device");
  assert.ok(mem.has("model_weights") && mem.has("model_weights_meta"));
  const info = await Onboard.modelInfo();
  assert.strictEqual(info.source, "device"); assert.ok(/기기 학습/.test(info.name));
  const toks = [];
  const text = await Onboard.generate("가나", { maxNew: 12, temperature: 0.8, topK: 5, seed: 3, onToken: (p) => toks.push(p) });
  assert.ok(typeof text === "string" && text.length > 0 && text === toks.join(""), "생성: " + JSON.stringify(text));
  await Onboard.deleteModel();
  assert.ok(!mem.has("model_weights") && !mem.has("model_weights_meta"));
  assert.strictEqual(await Onboard.modelInfo(), null);
});

test("파라미터 상한·자료 부족·seq_len 초과는 한국어 오류", async () => {
  mem.clear(); T._reset();
  await assert.rejects(T.start({ mode: "scratch", arch: smallArch, steps: 5, batch: 2, seqLen: 8 }), /저장된 자료가 없습니다/);
  await putSnapshot(bigRows());
  T._setWorkerFactory(() => scriptedWorker(() => {}));
  await assert.rejects(T.start({ mode: "scratch", arch: smallArch, steps: 5, batch: 2, seqLen: 64 }), /seq_len/);
  await assert.rejects(T.start({ mode: "scratch", arch: { d_model: 512, n_layer: 8, n_head: 8, seq_len: 64, vocab_size: 2000 }, steps: 5, batch: 2, seqLen: 8 }), /파라미터가 너무 많습니다/);
  await assert.rejects(T.start({ mode: "resume", steps: 5 }), /가중치/);
  await assert.rejects(T.start({ mode: "scratch", arch: { ...smallArch, n_head: 5 }, steps: 5 }), /n_head/);
  assert.strictEqual(T.status().active, false);
});

test("기존 API 유지: Onboard 공개 함수들", () => {
  for (const k of ["detectDevice", "saveSnapshot", "snapshotInfo", "deleteSnapshot", "search", "extract", "saveModel", "modelInfo", "deleteModel", "generate", "answer", "supported", "requestPersist"]) {
    assert.strictEqual(typeof Onboard[k], "function", k);
  }
});

let finished = false;
process.on("exit", (c) => { if (!finished) { console.log("FAIL - 테스트가 끝나기 전에 이벤트 루프가 비었습니다 (끝나지 않는 약속)"); process.exitCode = 1; } });
(async () => {
  let fail = 0;
  for (const [name, fn] of tests) {
    try { await fn(); console.log("ok   - " + name); }
    catch (e) { fail++; console.log("FAIL - " + name + "\n  " + (e && e.stack || e)); }
  }
  console.log((tests.length - fail) + "/" + tests.length + " passed");
  finished = true;
  process.exit(fail ? 1 : 0);
})();
