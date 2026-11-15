// 서버 통합 시험용 node 도구 (Python 테스트와 스모크가 호출한다).
//  node integration_driver.js build <out.lbw>          : Onboard.lbw.build 로 무작위 가중치 .lbw 생성
//  node integration_driver.js parse <in.lbw>           : Onboard.lbw.parse 후 요약을 JSON 으로 출력
//  node integration_driver.js train <resume|scratch> <base> <token> <modelId> <steps> <out.lbw>
//      서버 스냅샷으로 말뭉치를 만들고(+resume 이면 서버 model.lbw 를 받아) 실제 trainer.js 핸들러로 학습, 결과 .lbw 저장
const path = require("path");
const fs = require("fs");
const WEB = path.join(__dirname, "..", "..", "web");
globalThis.document = { visibilityState: "visible", addEventListener() {}, removeEventListener() {} };
Object.defineProperty(globalThis, "navigator", { value: { userAgent: "Mozilla/5.0 (Linux; Android 14) Mobile", deviceMemory: 8,
  hardwareConcurrency: 8, maxTouchPoints: 5 }, configurable: true, writable: true });
const Onboard = require(path.join(WEB, "onboard.js"));
const LBTrain = require(path.join(WEB, "trainer.js"));
const T = Onboard.training;
const mem = new Map();
Onboard.db.get = async (k) => mem.get(k);
Onboard.db.set = async (k, v) => { mem.set(k, v); };
Onboard.db.del = async (k) => { mem.delete(k); };
Onboard.db.setMany = async (o) => { for (const k of Object.keys(o)) { if (o[k] === undefined) mem.delete(k); else mem.set(k, o[k]); } };

const toAB = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
function summary(p) {
  let sum = 0;
  for (const k of Object.keys(p.tensors)) for (const v of p.tensors[k]) sum += v;
  return { arch: p.header.arch, itos_len: p.header.itos.length, n_tensors: Object.keys(p.tensors).length, sum, header: p.header };
}
function inProcWorker() {
  const w = { onmessage: null, terminated: false };
  const handler = LBTrain.makeHandler((msg) => { if (!w.terminated) setTimeout(() => w.onmessage && w.onmessage({ data: msg }), 0); });
  w.postMessage = (msg) => { if (!w.terminated) handler({ data: msg }); };
  w.terminate = () => { w.terminated = true; };
  return w;
}
async function main() {
  const [mode, ...a] = process.argv.slice(2);
  if (mode === "build") {
    const arch = { d_model: 32, n_layer: 2, n_head: 4, seq_len: 16, vocab_size: 12 };
    const specs = T._tensorSpecs(arch);
    const tensors = {};
    let k = 0;
    for (const s of specs) { const n = s.shape.reduce((x, y) => x * y, 1); tensors[s.name] = Float32Array.from({ length: n }, () => ((k++ % 53) - 26) / 400); }
    const itos = ["<pad>", "<unk>", "<bos>", "<eos>", ..."가나다라마바사아"];
    const header = { format: "lbw", arch, itos, tensors: specs, parent: null, parent_model_id: null, total_steps: 7, steps: 7,
      trained_on: "device", device: { label: "시험 기기", ua: "x" }, train_loss: 3.1, eval_loss_before: 4, eval_loss_after: 3.2,
      train_chars: 100, seconds: 1.5, train_seq_len: 16, created_at: "2026-11-15T00:00:00Z" };
    fs.writeFileSync(a[0], Buffer.from(Onboard.lbw.build(header, tensors)));
    console.log(JSON.stringify({ ok: true }));
  } else if (mode === "parse") {
    console.log(JSON.stringify(summary(Onboard.lbw.parse(toAB(fs.readFileSync(a[0]))))));
  } else if (mode === "train") {
    const [kind, base, token, modelId, steps, out] = a;
    const H = { Authorization: "Bearer " + token };
    const snap = await (await fetch(base + "/api/snapshot", { headers: H })).json();
    mem.set("snapshot", { v: 2, chunks: snap.chunks, index: null, saved_at: snap.created_at, count: snap.chunks.length });
    T._setWorkerFactory(() => inProcWorker());
    const opts = { steps: +steps, batch: 4, seqLen: 32, lr: 3e-3, checkpointEvery: 10 };
    if (kind === "resume") {
      const r = await fetch(base + `/api/models/${modelId}/files/model.lbw`, { headers: H });
      if (r.headers.get("content-type") !== "application/octet-stream") throw new Error("content-type: " + r.headers.get("content-type"));
      const parsed = Onboard.lbw.parse(await r.arrayBuffer());
      Object.assign(opts, { mode: "resume", source: { modelId: +modelId, name: "tiny-src", lbw: parsed }, arch: parsed.header.arch });
      opts.seqLen = Math.min(opts.seqLen, parsed.header.arch.seq_len);
    } else {
      Object.assign(opts, { mode: "scratch", arch: { d_model: 32, n_layer: 1, n_head: 4, seq_len: 32, vocab_size: 300 } });
    }
    await T.start(opts);
    const fin = await T.whenDone();
    if (fin.phase !== "done") throw new Error("학습 실패: " + fin.message);
    const buf = Buffer.from(await T.exportLbw());
    fs.writeFileSync(out, buf);
    const r = T.result();
    console.log(JSON.stringify({ ok: true, total_steps: r.header.total_steps, steps: r.header.steps, eval_before: r.evalBefore, eval_after: r.evalAfter,
      vocab: r.header.itos.length, parent_model_id: r.header.parent_model_id }));
  } else throw new Error("mode?");
}
main().then(() => process.exit(0), (e) => { console.error(e.stack || e); process.exit(1); });
