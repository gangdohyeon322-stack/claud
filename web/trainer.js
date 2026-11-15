/* LearnBox 온디바이스 학습 엔진 (Web Worker 스크립트, Node에서도 로드 가능).
 * PC learnbox/tinylm.py 의 TinyGPT 와 같은 수식을 순수 JS(Float32Array)로 구현한다.
 * 가중치 배치는 PyTorch state_dict 와 동일([out,in] 행우선) — 전치 없이 왕복된다.
 * 메시지 프로토콜은 docs/SPEC-mobile-training.md §4.1 참고. */
"use strict";
var LBTrain = (function () {
  const PAD = 0, UNK = 1, BOS = 2, EOS = 3, N_SPECIAL = 4;
  const LN_EPS = 1e-5;
  const SQRT1_2 = Math.SQRT1_2, INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

  // ------------------------------------------------------------------ 난수
  // mulberry32. 상태는 uint32 하나 -> 체크포인트에 그대로 저장 가능.
  function makeRng(seedOrState) {
    let s = (seedOrState >>> 0) || 0;
    const f = function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    f.getState = () => s;
    f.setState = (v) => { s = v >>> 0; };
    return f;
  }
  function gauss(rng) { // Box-Muller (한 값만 써서 상태를 단순하게 유지)
    let u = rng();
    if (u < 1e-12) u = 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  // ------------------------------------------------------------------ erf (절대오차 < 1e-9)
  // 시작 시 이중정밀 급수로 격자(h=1/64, [0,6])의 erf·도함수 표를 만들고, 런타임엔 3차 Hermite 보간.
  const ERF_H = 1 / 64, ERF_N = 6 * 64;
  const ERF_V = new Float64Array(ERF_N + 2), ERF_D = new Float64Array(ERF_N + 2);
  (function () {
    for (let i = 0; i <= ERF_N + 1; i++) {
      const x = i * ERF_H;
      // erf(x) = 2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (1*3*...*(2n+1))  (항이 모두 양수 -> 상쇄 없음)
      let term = x, sum = x;
      for (let n = 1; n < 400; n++) {
        term *= 2 * x * x / (2 * n + 1);
        sum += term;
        if (term < 1e-18 * sum) break;
      }
      ERF_V[i] = 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * sum;
      ERF_D[i] = 2 / Math.sqrt(Math.PI) * Math.exp(-x * x);
    }
    ERF_V[0] = 0;
  })();
  function erf(x) {
    const a = x < 0 ? -x : x;
    if (a >= 6) return x < 0 ? -1 : 1;
    const u = a * 64, i = u | 0, t = u - i;
    const t2 = t * t, t3 = t2 * t;
    const m0 = ERF_D[i] * ERF_H, m1 = ERF_D[i + 1] * ERF_H;
    const r = (2 * t3 - 3 * t2 + 1) * ERF_V[i] + (t3 - 2 * t2 + t) * m0 +
      (-2 * t3 + 3 * t2) * ERF_V[i + 1] + (t3 - t2) * m1;
    return x < 0 ? -r : r;
  }

  // ------------------------------------------------------------------ 구조
  function normArch(a) {
    const o = {
      d_model: a.d_model | 0, n_layer: a.n_layer | 0, n_head: a.n_head | 0,
      seq_len: a.seq_len | 0, vocab_size: a.vocab_size | 0,
    };
    if (o.d_model < 1 || o.n_layer < 1 || o.n_head < 1 || o.seq_len < 1 || o.vocab_size < 1) {
      throw new Error("모델 구조 값이 올바르지 않습니다.");
    }
    if (o.d_model % o.n_head) throw new Error("d_model은 n_head로 나누어떨어져야 합니다.");
    return o;
  }
  function paramCount(arch) {
    const V = arch.vocab_size, d = arch.d_model, T = arch.seq_len, L = arch.n_layer;
    return V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d;
  }
  function tensorSpecs(arch) {
    const V = arch.vocab_size, d = arch.d_model, T = arch.seq_len, L = arch.n_layer;
    const s = [{ name: "tok.weight", shape: [V, d] }, { name: "pos.weight", shape: [T, d] }];
    for (let i = 0; i < L; i++) {
      const p = "blocks." + i + ".";
      s.push({ name: p + "ln1.weight", shape: [d] }, { name: p + "ln1.bias", shape: [d] },
        { name: p + "qkv.weight", shape: [3 * d, d] }, { name: p + "qkv.bias", shape: [3 * d] },
        { name: p + "proj.weight", shape: [d, d] }, { name: p + "proj.bias", shape: [d] },
        { name: p + "ln2.weight", shape: [d] }, { name: p + "ln2.bias", shape: [d] },
        { name: p + "mlp.0.weight", shape: [4 * d, d] }, { name: p + "mlp.0.bias", shape: [4 * d] },
        { name: p + "mlp.2.weight", shape: [d, 4 * d] }, { name: p + "mlp.2.bias", shape: [d] });
    }
    s.push({ name: "ln.weight", shape: [d] }, { name: "ln.bias", shape: [d] });
    for (const t of s) t.size = t.shape.reduce((a, b) => a * b, 1);
    return s;
  }

  function allocModel(arch) {
    arch = normArch(arch);
    const specs = tensorSpecs(arch);
    const params = {}, grads = {};
    for (const t of specs) { params[t.name] = new Float32Array(t.size); grads[t.name] = new Float32Array(t.size); }
    return { arch, specs, params, grads, buf: null, genBuf: null };
  }
  const isLnWeight = (n) => /(^|\.)ln\d?\.weight$/.test(n);

  function createModel(arch, seed) {
    const m = allocModel(arch);
    const rng = makeRng((seed == null ? 1 : seed) >>> 0);
    for (const t of m.specs) {
      const p = m.params[t.name];
      if (isLnWeight(t.name)) p.fill(1);
      else if (t.name.endsWith(".bias") || t.name === "ln.bias") p.fill(0);
      else for (let i = 0; i < p.length; i++) p[i] = 0.02 * gauss(rng);
    }
    return m;
  }
  function loadWeights(arch, weights) {
    const m = allocModel(arch);
    for (const t of m.specs) {
      const w = weights && weights[t.name];
      if (!w) throw new Error("가중치에 '" + t.name + "' 텐서가 없습니다.");
      if (w.length !== t.size) throw new Error("'" + t.name + "' 텐서 크기가 맞지 않습니다.");
      m.params[t.name].set(w);
    }
    return m;
  }

  // ------------------------------------------------------------------ 버퍼
  function mk(n) { return new Float32Array(n); }
  function ensureBuffers(model, B, T, isGen) {
    const a = model.arch, d = a.d_model, L = a.n_layer, h = a.n_head, V = a.vocab_size;
    const N = B * T, attN = B * h * T * T;
    const key = isGen ? "genBuf" : "buf";
    let b = model[key];
    if (b && b.capN >= N && b.capAtt >= attN) return b;
    b = { capN: N, capAtt: attN, xs: [], ln1x: [], ln1r: [], ln1o: [], qkv: [], att: [], ay: [], xmid: [],
      ln2x: [], ln2r: [], ln2o: [], fc1: [], gl: [], gcdf: [] };
    for (let l = 0; l < L; l++) {
      b.xs.push(mk(N * d)); b.ln1x.push(mk(N * d)); b.ln1r.push(mk(N)); b.ln1o.push(mk(N * d));
      b.qkv.push(mk(N * 3 * d)); b.att.push(mk(attN)); b.ay.push(mk(N * d)); b.xmid.push(mk(N * d));
      b.ln2x.push(mk(N * d)); b.ln2r.push(mk(N)); b.ln2o.push(mk(N * d));
      b.fc1.push(mk(N * 4 * d)); b.gl.push(mk(N * 4 * d)); b.gcdf.push(mk(N * 4 * d));
    }
    b.xs.push(mk(N * d)); // 마지막 블록 출력
    b.lnfx = mk(N * d); b.lnfr = mk(N); b.lnfo = mk(N * d);
    if (!isGen) {
      b.logits = mk(N * V);
      b.dx = mk(N * d); b.dy = mk(N * d); b.dln = mk(N * d);
      b.dqkv = mk(N * 3 * d); b.dfc = mk(N * 4 * d); b.dgl = mk(N * 4 * d);
    }
    b.dp = mk(T);
    b.lastLogits = mk(V);
    model[key] = b;
    return b;
  }

  // ------------------------------------------------------------------ 커널
  // out[N,O] = x[N,I] · W[O,I]^T + b   (2행 x 4열 레지스터 블로킹: 로드 재사용)
  function linFwd(out, x, W, bias, N, I, O) {
    const O4 = O - (O % 4);
    let n = 0;
    for (; n + 1 < N; n += 2) {
      const xa = n * I, xb = xa + I, oa = n * O, ob = oa + O;
      let o = 0;
      for (; o < O4; o += 4) {
        const w0 = o * I, w1 = w0 + I, w2 = w1 + I, w3 = w2 + I;
        const b0 = bias ? bias[o] : 0, b1 = bias ? bias[o + 1] : 0, b2 = bias ? bias[o + 2] : 0, b3 = bias ? bias[o + 3] : 0;
        let a0 = b0, a1 = b1, a2 = b2, a3 = b3, c0 = b0, c1 = b1, c2 = b2, c3 = b3;
        for (let i = 0; i < I; i++) {
          const xv = x[xa + i], yv = x[xb + i];
          const q0 = W[w0 + i], q1 = W[w1 + i], q2 = W[w2 + i], q3 = W[w3 + i];
          a0 += xv * q0; a1 += xv * q1; a2 += xv * q2; a3 += xv * q3;
          c0 += yv * q0; c1 += yv * q1; c2 += yv * q2; c3 += yv * q3;
        }
        out[oa + o] = a0; out[oa + o + 1] = a1; out[oa + o + 2] = a2; out[oa + o + 3] = a3;
        out[ob + o] = c0; out[ob + o + 1] = c1; out[ob + o + 2] = c2; out[ob + o + 3] = c3;
      }
      for (; o < O; o++) {
        const w0 = o * I;
        let sa = bias ? bias[o] : 0, sb = sa;
        for (let i = 0; i < I; i++) { const q = W[w0 + i]; sa += x[xa + i] * q; sb += x[xb + i] * q; }
        out[oa + o] = sa; out[ob + o] = sb;
      }
    }
    for (; n < N; n++) {
      const xo = n * I, oo = n * O;
      for (let o = 0; o < O; o++) {
        const w0 = o * I;
        let sum = bias ? bias[o] : 0;
        for (let i = 0; i < I; i++) sum += x[xo + i] * W[w0 + i];
        out[oo + o] = sum;
      }
    }
  }
  // dx[N,I] = dout[N,O] · W[O,I]   (덮어쓰기, 2행 x 4출력 블로킹)
  function linBwdX(dx, dout, W, N, O, I) {
    const O4 = O - (O % 4);
    let n = 0;
    for (; n < N; n += 2) {
      const two = n + 1 < N;
      const xa = n * I, xb = xa + I, oa = n * O, ob = oa + O;
      for (let i = 0; i < I; i++) { dx[xa + i] = 0; if (two) dx[xb + i] = 0; }
      let o = 0;
      if (two) {
        for (; o < O4; o += 4) {
          const d0 = dout[oa + o], d1 = dout[oa + o + 1], d2 = dout[oa + o + 2], d3 = dout[oa + o + 3];
          const e0 = dout[ob + o], e1 = dout[ob + o + 1], e2 = dout[ob + o + 2], e3 = dout[ob + o + 3];
          const w0 = o * I, w1 = w0 + I, w2 = w1 + I, w3 = w2 + I;
          for (let i = 0; i < I; i++) {
            const q0 = W[w0 + i], q1 = W[w1 + i], q2 = W[w2 + i], q3 = W[w3 + i];
            dx[xa + i] += d0 * q0 + d1 * q1 + d2 * q2 + d3 * q3;
            dx[xb + i] += e0 * q0 + e1 * q1 + e2 * q2 + e3 * q3;
          }
        }
        for (; o < O; o++) {
          const d0 = dout[oa + o], e0 = dout[ob + o], w0 = o * I;
          for (let i = 0; i < I; i++) { const q = W[w0 + i]; dx[xa + i] += d0 * q; dx[xb + i] += e0 * q; }
        }
      } else {
        for (; o < O; o++) {
          const d0 = dout[oa + o], w0 = o * I;
          for (let i = 0; i < I; i++) dx[xa + i] += d0 * W[w0 + i];
        }
      }
    }
  }
  // dW[O,I] += dout[N,O]^T · x[N,I];  db[O] += sum_n dout   (2출력 x 4행 블로킹)
  function linBwdW(dW, db, dout, x, N, O, I) {
    const N4 = N - (N % 4), O2 = O - (O % 2);
    let o = 0;
    for (; o < O2; o += 2) {
      const wa = o * I, wb = wa + I;
      let n = 0;
      for (; n < N4; n += 4) {
        const n1 = n + 1, n2 = n + 2, n3 = n + 3;
        const d0 = dout[n * O + o], d1 = dout[n1 * O + o], d2 = dout[n2 * O + o], d3 = dout[n3 * O + o];
        const e0 = dout[n * O + o + 1], e1 = dout[n1 * O + o + 1], e2 = dout[n2 * O + o + 1], e3 = dout[n3 * O + o + 1];
        const x0 = n * I, x1 = x0 + I, x2 = x1 + I, x3 = x2 + I;
        for (let i = 0; i < I; i++) {
          const p0 = x[x0 + i], p1 = x[x1 + i], p2 = x[x2 + i], p3 = x[x3 + i];
          dW[wa + i] += d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3;
          dW[wb + i] += e0 * p0 + e1 * p1 + e2 * p2 + e3 * p3;
        }
      }
      for (; n < N; n++) {
        const d0 = dout[n * O + o], e0 = dout[n * O + o + 1], x0 = n * I;
        for (let i = 0; i < I; i++) { const p = x[x0 + i]; dW[wa + i] += d0 * p; dW[wb + i] += e0 * p; }
      }
    }
    for (; o < O; o++) {
      const wa = o * I;
      for (let n = 0; n < N; n++) {
        const d0 = dout[n * O + o], x0 = n * I;
        for (let i = 0; i < I; i++) dW[wa + i] += d0 * x[x0 + i];
      }
    }
    if (db) {
      for (let n = 0; n < N; n++) { const oo = n * O; for (let o2 = 0; o2 < O; o2++) db[o2] += dout[oo + o2]; }
    }
  }
  function lnFwd(out, xhat, rstd, x, w, b, N, d) {
    for (let n = 0; n < N; n++) {
      const o = n * d;
      let mean = 0;
      for (let i = 0; i < d; i++) mean += x[o + i];
      mean /= d;
      let v = 0;
      for (let i = 0; i < d; i++) { const c = x[o + i] - mean; v += c * c; }
      const r = 1 / Math.sqrt(v / d + LN_EPS);
      rstd[n] = r;
      for (let i = 0; i < d; i++) {
        const xh = (x[o + i] - mean) * r;
        xhat[o + i] = xh;
        out[o + i] = xh * w[i] + b[i];
      }
    }
  }
  // dxAcc += LN 역전파
  function lnBwd(dxAcc, dout, xhat, rstd, w, dw, db, N, d) {
    for (let n = 0; n < N; n++) {
      const o = n * d;
      let m1 = 0, m2 = 0;
      for (let i = 0; i < d; i++) {
        const g = dout[o + i], xh = xhat[o + i];
        dw[i] += g * xh; db[i] += g;
        const dh = g * w[i];
        m1 += dh; m2 += dh * xh;
      }
      m1 /= d; m2 /= d;
      const r = rstd[n];
      for (let i = 0; i < d; i++) dxAcc[o + i] += r * (dout[o + i] * w[i] - m1 - xhat[o + i] * m2);
    }
  }
  function attnFwd(att, y, qkv, B, T, h, hd, d) {
    const scale = 1 / Math.sqrt(hd), d3 = 3 * d;
    for (let b = 0; b < B; b++) {
      for (let hh = 0; hh < h; hh++) {
        const ho = hh * hd;
        for (let t = 0; t < T; t++) {
          const qo = (b * T + t) * d3 + ho, ao = ((b * h + hh) * T + t) * T;
          let mx = -Infinity;
          for (let s = 0; s <= t; s++) {
            const ko = (b * T + s) * d3 + d + ho;
            let acc = 0;
            for (let j = 0; j < hd; j++) acc += qkv[qo + j] * qkv[ko + j];
            acc *= scale;
            att[ao + s] = acc;
            if (acc > mx) mx = acc;
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) { const e = Math.exp(att[ao + s] - mx); att[ao + s] = e; sum += e; }
          const inv = 1 / sum;
          const yo = (b * T + t) * d + ho;
          for (let j = 0; j < hd; j++) y[yo + j] = 0;
          for (let s = 0; s <= t; s++) {
            const p = att[ao + s] * inv;
            att[ao + s] = p;
            const vo = (b * T + s) * d3 + 2 * d + ho;
            for (let j = 0; j < hd; j++) y[yo + j] += p * qkv[vo + j];
          }
        }
      }
    }
  }
  function attnBwd(dqkv, dy, att, qkv, dp, B, T, h, hd, d) {
    const scale = 1 / Math.sqrt(hd), d3 = 3 * d;
    dqkv.fill(0, 0, B * T * d3);
    for (let b = 0; b < B; b++) {
      for (let hh = 0; hh < h; hh++) {
        const ho = hh * hd;
        for (let t = 0; t < T; t++) {
          const qo = (b * T + t) * d3 + ho, ao = ((b * h + hh) * T + t) * T, dyo = (b * T + t) * d + ho;
          let sum = 0;
          for (let s = 0; s <= t; s++) {
            const vo = (b * T + s) * d3 + 2 * d + ho;
            const p = att[ao + s];
            let acc = 0;
            for (let j = 0; j < hd; j++) { const g = dy[dyo + j]; acc += g * qkv[vo + j]; dqkv[vo + j] += p * g; }
            dp[s] = acc;
            sum += p * acc;
          }
          for (let s = 0; s <= t; s++) {
            const ds = att[ao + s] * (dp[s] - sum) * scale;
            const ko = (b * T + s) * d3 + d + ho;
            for (let j = 0; j < hd; j++) {
              dqkv[qo + j] += ds * qkv[ko + j];
              dqkv[ko + j] += ds * qkv[qo + j];
            }
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------ 순전파/역전파
  function forwardHidden(model, x, B, T, b) {
    const a = model.arch, d = a.d_model, L = a.n_layer, h = a.n_head, hd = d / h, p = model.params;
    if (T > a.seq_len) throw new Error("문맥 길이가 모델의 seq_len(" + a.seq_len + ")보다 큽니다.");
    const N = B * T, tok = p["tok.weight"], pos = p["pos.weight"], x0 = b.xs[0];
    for (let n = 0; n < N; n++) {
      const id = x[n], to = id * d, po = (n % T) * d, o = n * d;
      for (let i = 0; i < d; i++) x0[o + i] = tok[to + i] + pos[po + i];
    }
    for (let l = 0; l < L; l++) {
      const pre = "blocks." + l + ".";
      const xin = b.xs[l], xmid = b.xmid[l], xout = b.xs[l + 1];
      lnFwd(b.ln1o[l], b.ln1x[l], b.ln1r[l], xin, p[pre + "ln1.weight"], p[pre + "ln1.bias"], N, d);
      linFwd(b.qkv[l], b.ln1o[l], p[pre + "qkv.weight"], p[pre + "qkv.bias"], N, d, 3 * d);
      attnFwd(b.att[l], b.ay[l], b.qkv[l], B, T, h, hd, d);
      linFwd(xmid, b.ay[l], p[pre + "proj.weight"], p[pre + "proj.bias"], N, d, d);
      for (let i = 0, M = N * d; i < M; i++) xmid[i] += xin[i];
      lnFwd(b.ln2o[l], b.ln2x[l], b.ln2r[l], xmid, p[pre + "ln2.weight"], p[pre + "ln2.bias"], N, d);
      linFwd(b.fc1[l], b.ln2o[l], p[pre + "mlp.0.weight"], p[pre + "mlp.0.bias"], N, d, 4 * d);
      const f = b.fc1[l], g = b.gl[l], c = b.gcdf[l];
      for (let i = 0, M = N * 4 * d; i < M; i++) {
        const v = f[i], cd = 0.5 * (1 + erf(v * SQRT1_2));
        c[i] = cd; g[i] = v * cd;
      }
      linFwd(xout, g, p[pre + "mlp.2.weight"], p[pre + "mlp.2.bias"], N, 4 * d, d);
      for (let i = 0, M = N * d; i < M; i++) xout[i] += xmid[i];
    }
    lnFwd(b.lnfo, b.lnfx, b.lnfr, b.xs[L], p["ln.weight"], p["ln.bias"], N, d);
  }

  // 손실(평균 cross-entropy)과 캐시. 캐시는 모델의 재사용 버퍼를 가리킨다(다음 forward 전까지 유효).
  function forwardLoss(model, xBatch, yBatch, B, T) {
    const a = model.arch, V = a.vocab_size, d = a.d_model, N = B * T;
    const b = ensureBuffers(model, B, T, false);
    forwardHidden(model, xBatch, B, T, b);
    linFwd(b.logits, b.lnfo, model.params["tok.weight"], null, N, d, V);
    const lg = b.logits;
    let total = 0;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let mx = -Infinity;
      for (let i = 0; i < V; i++) if (lg[o + i] > mx) mx = lg[o + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(lg[o + i] - mx);
      total += Math.log(sum) + mx - lg[o + yBatch[n]];
    }
    return { loss: total / N, cache: { buf: b, x: xBatch, y: yBatch, B, T } };
  }

  function backward(model, cache) {
    const a = model.arch, d = a.d_model, L = a.n_layer, h = a.n_head, hd = d / h, V = a.vocab_size;
    const p = model.params, g = model.grads, b = cache.buf, B = cache.B, T = cache.T, N = B * T;
    for (const k in g) g[k].fill(0);
    // dlogits = (softmax - onehot) / N  (logits 버퍼를 덮어쓴다)
    const lg = b.logits, inv = 1 / N, y = cache.y;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let mx = -Infinity;
      for (let i = 0; i < V; i++) if (lg[o + i] > mx) mx = lg[o + i];
      let sum = 0;
      for (let i = 0; i < V; i++) { const e = Math.exp(lg[o + i] - mx); lg[o + i] = e; sum += e; }
      const s = inv / sum;
      for (let i = 0; i < V; i++) lg[o + i] *= s;
      lg[o + y[n]] -= inv;
    }
    // 출력층(tok 공유): dlnfo = dlogits·tok, dtok += dlogitsᵀ·lnfo
    linBwdX(b.dln, lg, p["tok.weight"], N, V, d);
    linBwdW(g["tok.weight"], null, lg, b.lnfo, N, V, d);
    b.dx.fill(0, 0, N * d);
    lnBwd(b.dx, b.dln, b.lnfx, b.lnfr, p["ln.weight"], g["ln.weight"], g["ln.bias"], N, d);
    for (let l = L - 1; l >= 0; l--) {
      const pre = "blocks." + l + ".";
      // MLP
      linBwdX(b.dgl, b.dx, p[pre + "mlp.2.weight"], N, d, 4 * d);
      linBwdW(g[pre + "mlp.2.weight"], g[pre + "mlp.2.bias"], b.dx, b.gl[l], N, d, 4 * d);
      const f = b.fc1[l], c = b.gcdf[l], dg = b.dgl, df = b.dfc;
      for (let i = 0, M = N * 4 * d; i < M; i++) {
        const v = f[i];
        df[i] = dg[i] * (c[i] + v * INV_SQRT_2PI * Math.exp(-0.5 * v * v));
      }
      linBwdX(b.dln, df, p[pre + "mlp.0.weight"], N, 4 * d, d);
      linBwdW(g[pre + "mlp.0.weight"], g[pre + "mlp.0.bias"], df, b.ln2o[l], N, 4 * d, d);
      lnBwd(b.dx, b.dln, b.ln2x[l], b.ln2r[l], p[pre + "ln2.weight"], g[pre + "ln2.weight"], g[pre + "ln2.bias"], N, d);
      // 어텐션
      linBwdX(b.dy, b.dx, p[pre + "proj.weight"], N, d, d);
      linBwdW(g[pre + "proj.weight"], g[pre + "proj.bias"], b.dx, b.ay[l], N, d, d);
      attnBwd(b.dqkv, b.dy, b.att[l], b.qkv[l], b.dp, B, T, h, hd, d);
      linBwdX(b.dln, b.dqkv, p[pre + "qkv.weight"], N, 3 * d, d);
      linBwdW(g[pre + "qkv.weight"], g[pre + "qkv.bias"], b.dqkv, b.ln1o[l], N, 3 * d, d);
      lnBwd(b.dx, b.dln, b.ln1x[l], b.ln1r[l], p[pre + "ln1.weight"], g[pre + "ln1.weight"], g[pre + "ln1.bias"], N, d);
    }
    // 임베딩
    const dtok = g["tok.weight"], dpos = g["pos.weight"], x = cache.x;
    for (let n = 0; n < N; n++) {
      const to = x[n] * d, po = (n % T) * d, o = n * d;
      for (let i = 0; i < d; i++) { const v = b.dx[o + i]; dtok[to + i] += v; dpos[po + i] += v; }
    }
    return g;
  }

  // ------------------------------------------------------------------ 옵티마이저
  function newOptState(model) {
    const m = {}, v = {};
    for (const t of model.specs) { m[t.name] = new Float32Array(t.size); v[t.name] = new Float32Array(t.size); }
    return { m, v, t: 0 };
  }
  function lrAt(step, steps, lr) {
    const warmup = Math.min(100, Math.max(1, Math.floor(steps / 10)));
    const minLr = lr * 0.1;
    if (step <= warmup) return lr * step / warmup;
    const frac = (step - warmup) / Math.max(1, steps - warmup);
    return minLr + 0.5 * (lr - minLr) * (1 + Math.cos(Math.PI * frac));
  }
  // grads 는 수정하지 않는다(클립 계수를 즉석 적용). step 은 1부터.
  function adamwStep(model, grads, opt, step, lr, hp) {
    hp = hp || {};
    const b1 = hp.beta1 == null ? 0.9 : hp.beta1, b2 = hp.beta2 == null ? 0.95 : hp.beta2;
    const eps = hp.eps == null ? 1e-8 : hp.eps, wd = hp.wd == null ? 0.01 : hp.wd;
    const clip = hp.clip == null ? 1.0 : hp.clip;
    let sq = 0;
    for (const t of model.specs) {
      const gr = grads[t.name];
      for (let i = 0; i < gr.length; i++) sq += gr[i] * gr[i];
    }
    const gradNorm = Math.sqrt(sq);
    const coef = clip > 0 ? Math.min(1, clip / (gradNorm + 1e-6)) : 1;
    const bc1 = 1 - Math.pow(b1, step), bc2 = 1 - Math.pow(b2, step);
    const stepSize = lr / bc1, sbc2 = Math.sqrt(bc2);
    for (const t of model.specs) {
      const P = model.params[t.name], G = grads[t.name], M = opt.m[t.name], Vv = opt.v[t.name];
      const decay = t.shape.length >= 2 ? 1 - lr * wd : 1;
      for (let i = 0; i < P.length; i++) {
        const gv = G[i] * coef;
        const m = b1 * M[i] + (1 - b1) * gv;
        const v = b2 * Vv[i] + (1 - b2) * gv * gv;
        M[i] = m; Vv[i] = v;
        P[i] = P[i] * decay - stepSize * m / (Math.sqrt(v) / sbc2 + eps);
      }
    }
    opt.t = step;
    return gradNorm;
  }

  // ------------------------------------------------------------------ 데이터/평가/생성
  // 창 [i, i+T], 타깃은 한 칸 밀기. 시작 i 는 [0, n-T-1] 균등.
  function getBatch(ids, B, T, rng, out) {
    const n = ids.length;
    if (n < T + 1) throw new Error("학습 자료가 문맥 길이(" + T + ")보다 짧습니다.");
    const x = out && out.x && out.x.length === B * T ? out.x : new Int32Array(B * T);
    const y = out && out.y && out.y.length === B * T ? out.y : new Int32Array(B * T);
    const span = n - T;
    for (let r = 0; r < B; r++) {
      const i = Math.floor(rng() * span);
      for (let t = 0; t < T; t++) { x[r * T + t] = ids[i + t]; y[r * T + t] = ids[i + t + 1]; }
    }
    return { x, y };
  }
  function evaluate(model, evalIds, batches, B, T, seed) {
    const rng = makeRng(seed == null ? 1234 : seed);
    T = Math.max(1, Math.min(T, model.arch.seq_len, evalIds.length - 1));
    B = Math.max(1, Math.min(B, evalIds.length - T));
    let sum = 0;
    const out = {};
    for (let k = 0; k < batches; k++) {
      const bt = getBatch(evalIds, B, T, rng, out);
      out.x = bt.x; out.y = bt.y;
      sum += forwardLoss(model, bt.x, bt.y, B, T).loss;
    }
    return sum / Math.max(1, batches);
  }
  // onboard.js sampleTopK 와 같은 규칙: id 0..2 금지, EOS(3)는 종료 신호, 그 외 상위 k 샘플링.
  function sampleTopK(logits, k, temperature, rng) {
    const V = logits.length;
    const banned = V > N_SPECIAL + 1 ? EOS : 0;
    if (!(temperature > 0)) {
      let best = -1, bv = -Infinity;
      for (let i = banned; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
      return best < 0 ? EOS : best;
    }
    k = Math.max(1, Math.min(k | 0 || V, V - banned));
    const idx = [], val = [];
    for (let i = banned; i < V; i++) {
      const v = logits[i];
      if (idx.length < k) {
        let j = idx.length;
        idx.push(i); val.push(v);
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
      } else if (v > val[k - 1]) {
        let j = k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
      }
    }
    const mx = val[0];
    let sum = 0;
    const pr = val.map((v) => { const e = Math.exp((v - mx) / temperature); sum += e; return e; });
    let r = rng() * sum;
    for (let j = 0; j < pr.length; j++) { r -= pr[j]; if (r <= 0) return idx[j]; }
    return idx[idx.length - 1];
  }
  // 마지막 위치의 logits 만 계산(KV 캐시 없이 매 토큰 전체 재계산 — 작은 모델용).
  function lastLogits(model, ids, len) {
    const a = model.arch, d = a.d_model, V = a.vocab_size;
    const T = Math.min(len, a.seq_len);
    const b = ensureBuffers(model, 1, a.seq_len, true);
    const x = Int32Array.from(ids.slice(ids.length - T));
    forwardHidden(model, x, 1, T, b);
    const tok = model.params["tok.weight"], o = (T - 1) * d, out = b.lastLogits;
    for (let v = 0; v < V; v++) {
      let s = 0;
      const w = v * d;
      for (let i = 0; i < d; i++) s += b.lnfo[o + i] * tok[w + i];
      out[v] = s;
    }
    return out;
  }
  function generate(model, promptIds, maxNew, temperature, topK, rng, onToken) {
    const ids = Array.from(promptIds || []);
    if (!ids.length) ids.push(BOS);
    const out = [];
    rng = rng || makeRng(1);
    for (let i = 0; i < maxNew; i++) {
      const lg = lastLogits(model, ids, ids.length);
      const next = sampleTopK(lg, topK, temperature, rng);
      if (next === EOS || next < N_SPECIAL) break;
      ids.push(next); out.push(next);
      if (onToken) onToken(next);
    }
    return out;
  }

  // ------------------------------------------------------------------ 직렬화
  function serializeState(model, opt, rng) {
    const weights = {}, m = {}, v = {};
    for (const t of model.specs) {
      weights[t.name] = model.params[t.name].slice();
      if (opt) { m[t.name] = opt.m[t.name].slice(); v[t.name] = opt.v[t.name].slice(); }
    }
    return { weights, opt: opt ? { m, v, t: opt.t } : null, rng: rng ? rng.getState() : null };
  }
  function transferList(state) {
    const l = [];
    const add = (o) => { if (o) for (const k in o) l.push(o[k].buffer); };
    add(state.weights);
    if (state.opt) { add(state.opt.m); add(state.opt.v); }
    return l;
  }
  // 모델 가중치를 제자리에 덮어쓰고, {opt, rng} 를 만들어 돌려준다.
  function deserializeState(model, state, rng) {
    for (const t of model.specs) {
      const w = state.weights && state.weights[t.name];
      if (!w || w.length !== t.size) throw new Error("체크포인트의 '" + t.name + "' 텐서가 맞지 않습니다.");
      model.params[t.name].set(w);
    }
    const opt = newOptState(model);
    if (state.opt && state.opt.m && state.opt.v) {
      for (const t of model.specs) { opt.m[t.name].set(state.opt.m[t.name]); opt.v[t.name].set(state.opt.v[t.name]); }
      opt.t = state.opt.t | 0;
    }
    rng = rng || makeRng(1);
    if (state.rng != null) rng.setState(state.rng);
    return { opt, rng };
  }

  // 한 단계(배치 추출 -> 순전파 -> 역전파 -> AdamW). 손실 반환.
  function trainStep(model, opt, ids, B, T, rng, step, steps, lr, io) {
    const bt = getBatch(ids, B, T, rng, io);
    io.x = bt.x; io.y = bt.y;
    const r = forwardLoss(model, bt.x, bt.y, B, T);
    const grads = backward(model, r.cache);
    adamwStep(model, grads, opt, step, lrAt(step, steps, lr));
    return r.loss;
  }

  // ------------------------------------------------------------------ Worker 메시지 처리
  const yieldNow = () => new Promise((r) => setTimeout(r, 0));
  const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  function makeHandler(post) {
    let model = null, opt = null, rng = null, itos = null, corpus = null, evalIds = null, seed = 1;
    let busy = false, paused = false, stopReq = false, resumeWaiter = null;
    const err = (msg) => post({ type: "error", message: msg });

    function syntheticIds(V, n) {
      const r = makeRng(99), a = new Int32Array(n);
      for (let i = 0; i < n; i++) a[i] = N_SPECIAL + Math.floor(r() * Math.max(1, V - N_SPECIAL));
      return a;
    }

    function doInit(m) {
      const arch = normArch(m.arch);
      seed = m.seed == null ? 1 : m.seed | 0;
      model = m.weights ? loadWeights(arch, m.weights) : createModel(arch, seed);
      rng = makeRng((seed + 1) >>> 0);
      opt = newOptState(model);
      if (m.opt && m.opt.m && m.opt.v) {
        for (const t of model.specs) { opt.m[t.name].set(m.opt.m[t.name]); opt.v[t.name].set(m.opt.v[t.name]); }
        opt.t = m.opt.t | 0;
      }
      if (m.rng != null) rng.setState(m.rng);
      itos = m.itos || null;
      corpus = m.corpus || null;
      evalIds = m.evalIds || null;
      post({ type: "ready", params: paramCount(arch) });
    }

    async function doBench(m) {
      const a = model.arch;
      const B = Math.max(1, m.batch | 0 || 8), T = Math.max(2, Math.min(m.seqLen | 0 || 64, a.seq_len));
      const steps = Math.max(1, m.steps | 0 || 10);
      const snap = serializeState(model, opt, rng);
      const ids = corpus && corpus.length >= T + 2 ? corpus : syntheticIds(a.vocab_size, Math.max(4096, T * 4));
      const r2 = makeRng(7), io = {};
      const times = [];
      try {
        for (let s = 1; s <= steps; s++) {
          const t0 = nowMs();
          trainStep(model, opt, ids, B, T, r2, s, steps, 1e-3, io);
          times.push(nowMs() - t0);
          await yieldNow();
        }
      } finally {
        opt = deserializeState(model, snap, rng).opt; // 원본 복구
      }
      const use = times.length > 2 ? times.slice(1) : times; // 첫 단계는 JIT 워밍업이라 제외
      const ms = use.reduce((x, y) => x + y, 0) / use.length;
      post({ type: "bench", msPerStep: ms, tokensPerSec: B * T * 1000 / ms });
    }

    async function doTrain(m) {
      const a = model.arch;
      const B = Math.max(1, m.batch | 0 || 8), T = Math.max(2, m.seqLen | 0 || 64);
      if (T > a.seq_len) throw new Error("문맥 길이가 모델의 seq_len(" + a.seq_len + ")보다 큽니다.");
      if (!corpus || corpus.length < T + 2) throw new Error("학습 자료가 너무 짧습니다. 먼저 [검색] 탭에서 자료를 저장하세요.");
      const steps = Math.max(1, m.steps | 0), start = m.startStep | 0, end = start + steps;
      const total = m.totalPlanned | 0 || end, lr = m.lr || 3e-3;
      const every = m.checkpointEvery | 0;
      const io = {}, recent = [];
      let tAcc = 0, nAcc = 0, last = start;
      paused = false; stopReq = false;
      for (let step = start + 1; step <= end; step++) {
        if (paused) await new Promise((r) => { resumeWaiter = r; });
        if (stopReq) break;
        const t0 = nowMs();
        const loss = trainStep(model, opt, corpus, B, T, rng, step, total, lr, io);
        const dt = nowMs() - t0;
        last = step;
        if (!isFinite(loss)) throw new Error("손실이 발산했습니다(NaN). 학습률을 낮춰 다시 시도하세요.");
        recent.push(loss); if (recent.length > 50) recent.shift();
        tAcc += dt; nAcc++;
        const msPerStep = tAcc / nAcc;
        post({ type: "progress", step, loss, msPerStep, eta: (end - step) * msPerStep / 1000 });
        if (every > 0 && step % every === 0 && step < end) {
          const st = serializeState(model, opt, rng);
          post({ type: "checkpoint", step, state: st }, transferList(st));
        }
        await yieldNow();
      }
      if (stopReq) {
        const st = serializeState(model, opt, rng);
        post({ type: "stopped", step: last, state: st }, transferList(st));
      } else {
        const tl = recent.reduce((x, y) => x + y, 0) / Math.max(1, recent.length);
        post({ type: "done", step: last, trainLoss: tl });
      }
    }

    async function doGenerate(m) {
      const r = makeRng((m.seed == null ? 1 : m.seed) >>> 0);
      const ids = Array.from(m.prompt || []);
      if (!ids.length) ids.push(BOS);
      const out = [];
      const max = m.maxNew | 0 || 100;
      for (let i = 0; i < max; i++) {
        const lg = lastLogits(model, ids, ids.length);
        const next = sampleTopK(lg, m.topK | 0 || 20, m.temperature == null ? 0.8 : m.temperature, r);
        if (next === EOS || next < N_SPECIAL) break;
        ids.push(next); out.push(next);
        post({ type: "token", id: next });
        await yieldNow();
      }
      post({ type: "generated", ids: out });
    }

    return async function handler(ev) {
      const m = ev && ev.data !== undefined ? ev.data : ev;
      if (!m || !m.type) return;
      try {
        switch (m.type) {
          case "pause": paused = true; return;
          case "resume": paused = false; if (resumeWaiter) { const r = resumeWaiter; resumeWaiter = null; r(); } return;
          case "stop": stopReq = true; paused = false; if (resumeWaiter) { const r = resumeWaiter; resumeWaiter = null; r(); } return;
          case "export": {
            if (!model) throw new Error("모델이 준비되지 않았습니다.");
            const w = {}; const tl = [];
            for (const t of model.specs) { w[t.name] = model.params[t.name].slice(); tl.push(w[t.name].buffer); }
            post({ type: "weights", weights: w }, tl);
            return;
          }
        }
        if (m.type === "init" && busy) throw new Error("학습 중에는 다시 초기화할 수 없습니다.");
        if (m.type !== "init" && !model) throw new Error("모델이 준비되지 않았습니다. 먼저 init 하세요.");
        if (m.type !== "init" && busy) throw new Error("작업이 진행 중입니다. 잠시 후 다시 시도하세요.");
        switch (m.type) {
          case "init": doInit(m); break;
          case "eval": {
            if (!evalIds) throw new Error("평가 자료가 없습니다.");
            const T = m.seqLen | 0 || Math.min(64, model.arch.seq_len);
            post({ type: "eval", loss: evaluate(model, evalIds, m.batches | 0 || 8, m.batch | 0 || 16, T, 1234) });
            break;
          }
          case "bench": busy = true; try { await doBench(m); } finally { busy = false; } break;
          case "train": busy = true; try { await doTrain(m); } finally { busy = false; paused = false; } break;
          case "generate": busy = true; try { await doGenerate(m); } finally { busy = false; } break;
          default: throw new Error("알 수 없는 메시지: " + m.type);
        }
      } catch (e) {
        err(String((e && e.message) || e));
      }
    };
  }

  return {
    PAD, UNK, BOS, EOS, N_SPECIAL,
    makeRng, gauss, erf, paramCount, tensorSpecs, createModel, loadWeights, newOptState,
    forwardLoss, backward, adamwStep, lrAt, getBatch, evaluate, generate, sampleTopK,
    serializeState, deserializeState, transferList, trainStep, makeHandler,
  };
})();

if (typeof module === "object" && module.exports) module.exports = LBTrain;
if (typeof self !== "undefined") {
  self.LBTrain = LBTrain;
  if (typeof self.postMessage === "function" && typeof importScripts === "function") {
    self.onmessage = LBTrain.makeHandler((msg, transfer) => self.postMessage(msg, transfer || []));
  }
}
