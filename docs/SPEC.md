# LearnBox v2 — implementation contract

This is the single source of truth for the v2 build. Every module, function name, return shape, job kind and HTTP route below is a contract: implement exactly these names so the pieces fit together. User-facing strings (UI, error messages, notifications, reports) are **Korean**. Code comments may be Korean or English.

Python ≥ 3.10, stdlib-first. Optional heavy deps (torch, transformers, peft, onnx, onnxruntime, anthropic, pypdf) are imported **lazily inside functions** so the core (collect/search/web UI/export) runs without them. Venv for tests: `/home/user/claud/.venv/bin/python` (has torch CPU, transformers, peft, onnx, onnxscript, onnxruntime, anthropic 2.8, pypdf). No GPU here.

## 0. Product goals (what the user asked for)

1. **Model size is user-configurable** (presets + custom numbers, with live size/memory estimates and a device recommendation).
2. Three answer engines, all grounded in the learned files:
   - **소형 AI 모델 (tiny)** — a character-level GPT trained *from scratch only on the learned files* (`tinylm.py`). Runs on the PC (PyTorch) and is exported to ONNX.
   - **LoRA 모델 (lora)** — a public small LM (Qwen2.5 0.5B–7B) fine-tuned with LoRA on the learned files (+ optional Q&A pairs) (`train_pc.py`). Answers with retrieved context (RAG).
   - **온보드 모델 (onboard)** — the tiny model's ONNX file running *inside the phone/PC browser* via onnxruntime-web, fully offline, plus on-device search over a downloaded snapshot (`web/onboard.js`).
   - **온라인 모델 (online)** — Claude API (Anthropic SDK) answering from retrieved chunks with citations (`online.py`).
3. **Search inside the learned files is the core feature**: keyword (FTS5 trigram) + semantic (embeddings) hybrid search, source filters (web/search/file), phrase and exclusion syntax, pagination, extractive key sentences, and a document viewer with highlighted matches. Users can import their own files (txt/md/csv/tsv/json/jsonl/html/xml/pdf/docx).
4. **Autonomous learning (자율 학습)** inside the allowed time windows: collect a little at a time (watchlist + auto-discovered topics), and retrain/continue-train when enough new text has arrived.
5. **"학습 종료" (end learning)**: stop the session, do a final training pass, export everything that was learned as **files** (JSONL, CSV, Markdown, TXT, SQLite index, model files incl. ONNX fp32/int8, report, manifest, ZIP), then **notify** the user (in-app notification + result card with file list and download links).
6. **Phone vs PC performance**: PC capability profile (CPU/RAM/GPU) drives training presets; browser device tier (memory/cores/WebGPU) drives onboard model variant (int8 vs fp32), snapshot size and generation length.

## 1. File ownership (who writes what)

| Owner | Files (create or rewrite) |
|---|---|
| A — tiny model + devices | `learnbox/tinylm.py` (new), `learnbox/devices.py` (new), `tests/test_tinylm.py`, `tests/test_devices.py` |
| B — LoRA | `learnbox/train_pc.py` (rewrite), `tests/test_train_pc.py` |
| C — ingest + search | `learnbox/ingest.py` (new), `learnbox/search.py` (new), `tests/test_ingest.py`, `tests/test_search.py` |
| D — online model | `learnbox/online.py` (new), `tests/test_online.py` |
| E — autonomous learning + results | `learnbox/autolearn.py` (new), `learnbox/results.py` (new), `tests/test_autolearn.py`, `tests/test_results.py` |
| F — browser on-device engine | `web/onboard.js` (new), `web/sw.js` |
| G — web UI | `web/index.html`, `web/app.js`, `web/style.css` |
| H — integration (runs after A–G) | `learnbox/app.py`, `learnbox/server.py`, `learnbox/jobs.py`, `learnbox/__main__.py`, delete `learnbox/train_mobile.py`, `tests/test_learnbox.py`, `README.md`, `requirements*.txt` |
| T — accelerators (docs/PRD-tpu-training.md) | `learnbox/accel.py` (device layer: cuda/tpu/xpu/mps/cpu, precision policy), `learnbox/setup_accel.py` (TPU-runtime-only installer), `requirements-tpu.txt`, `requirements-tpu-xla.txt`, `tests/test_accel.py`, `tests/test_tpu_tinylm.py`, `tests/test_tpu_lora.py`, `tests/test_device_plumbing.py`, `tests/test_setup_accel.py` |

Already done (do not rewrite; you may read): `learnbox/config.py` (DEFAULTS incl. `tiny_training`, `lora_training`, `online`, `embeddings`, `ingest`, `autolearn`), `learnbox/db.py` (schema + methods listed in §2), `learnbox/timegate.py`, `learnbox/fetcher.py`, `learnbox/extract.py`, `learnbox/collector.py`, `learnbox/search_providers.py`, `learnbox/export.py`, `learnbox/dataset.py`.

If you need a small helper in a file you don't own, put it in your own module instead.

## 2. Existing building blocks (read these files for exact signatures)

- `Config(data_dir)`: `cfg["section"]` dict access, `cfg.data_dir`, `cfg.db_path`, `cfg.models_dir`, `cfg.datasets_dir`, `cfg.public()`, `cfg.update(changes)`, `cfg.check_pin(pin) -> 'admin'|'user'|None`. Add more dirs by `cfg.data_dir / "files"`, `cfg.data_dir / "results"` (create with `mkdir(parents=True, exist_ok=True)`).
- `DB(path)`: documents (`add_document(url,title,text,source,query=None,file_id=None) -> 'new'|'updated'|'unchanged'`, `get_document`, `list_documents(limit,offset,source)`, `iter_documents(since=None)`, `delete_document`, `counts()` → `{documents,chars,chunks,embeddings,files}`, `source_counts()`, `chars_since(iso)`), chunks (`iter_chunks(doc_ids=None)` rows `{chunk_id,doc_id,title,content,url,source}`, `get_chunks(ids)`, `chunk_snapshot(max_chunks)` rows `{c,d,t,x,u,s}`), legacy `search(q,limit)`, jobs (`create_job/update_job/get_job/list_jobs/running_jobs(kinds)`), models (`add_model(kind,name,path,info) -> id`, `list_models`, `get_model(id)`), watchlist (`add_watch(kind,value,depth=0,origin='user') -> bool`, `next_watch(origin=None)`, `touch_watch`, `count_watch(origin)`, `list_watch`, `remove_watch`), files (`find_file(sha)`, `add_file(...)`, `set_file_docs`, `list_files`), embeddings (`missing_embedding_chunks(model,limit)`, `upsert_embeddings(rows)`, `iter_embeddings(model)`, `embedding_count(model)`), notifications (`notify(level,title,body,data) -> id`, `list_notifications`, `unread_notifications`, `mark_notifications_read(ids|None)`), sessions (`create_session(actor,config) -> id`, `update_session(id, **fields)`, `get_session`, `active_session()`, `list_sessions`), `audit(actor,action,detail)`, `backup_to(path)`. `now_iso()` returns UTC ISO seconds. `HL_START="\x02"`, `HL_END="\x03"` are snippet highlight markers.
- `dataset.build_corpus(db, out_dir) -> {"dir","train","eval","chars"}` writes `train.jsonl`/`eval.jsonl` with `{"text","title","url"}`; `dataset.load_texts(path)`.
- `collector.collect_links(ctx,cfg,db,fetcher,urls,depth,same_domain,source,query,max_pages)` / `collect_search(ctx,cfg,db,fetcher,query,limit)` → stats `{new,updated,unchanged,skipped,errors,stopped,visited}`.
- `timegate.is_allowed(windows,tz)`, `ensure_allowed(cfg)`, `OutsideWindow`, `describe(windows)`, `next_allowed(...)`.
- `export.validate_target(cfg, target) -> Path`, `export.export(...)`, `export._sha256(path)`.
- Job context protocol (`jobs.JobContext`): `ctx.progress(p: float 0..1, message: str)`, `ctx.check_cancel()` (raises `jobs.Cancelled`). Library functions take `progress=lambda p, m: None, check_cancel=lambda: None` keyword args, never a ctx.

## 3. Module contracts

### A. `learnbox/tinylm.py` — 소형 AI 모델 (from scratch, char-level GPT)

```python
PRESETS: dict[str, dict]  # name -> {"d_model","n_layer","n_head","seq_len","vocab_size","label","target","desc"}
# nano : 128/2/4/128/2000   target "phone-low"   label "나노 (저사양 휴대폰)"
# mini : 256/4/4/192/3000   target "phone-high"  label "미니 (고사양 휴대폰)"
# small: 384/6/6/256/4000   target "pc-cpu"      label "스몰 (PC CPU)"
# base : 512/8/8/384/6000   target "pc-gpu"      label "베이스 (PC GPU)"
# large: 768/12/12/512/8000 target "pc-gpu-high" label "라지 (고성능 GPU)"
LIMITS = {"d_model": (32, 2048), "n_layer": (1, 48), "n_head": (1, 32), "seq_len": (16, 2048), "vocab_size": (100, 20000)}

def resolve_arch(preset: str | None = None, overrides: dict | None = None, recommended: str = "mini") -> dict
    # preset "auto"/None -> `recommended`; "custom" -> start from "mini" and require overrides.
    # overrides: any of d_model,n_layer,n_head,seq_len,vocab_size (0/None = keep preset value).
    # Validate LIMITS and d_model % n_head == 0, else raise ValueError(Korean message).
    # Returns {"preset": name, "d_model","n_layer","n_head","seq_len","vocab_size"}.
def estimate(arch: dict, batch_size: int = 32) -> dict
    # {"params": int, "fp32_mb": float, "int8_mb": float, "train_ram_mb": float, "infer_ram_mb": float,
    #  "fits": {"phone-low": bool, "phone-high": bool, "pc-cpu": bool, "pc-gpu": bool}, "label": str}
    # params = exact count for the architecture below (tied embedding/head). train_ram ≈ params*16B + activations.
def train(train_path, eval_path, out_dir, arch: dict, steps: int = 1500, batch_size: int = 32, lr: float = 3e-3,
          resume_from: str | None = None, device: str | None = None, threads: int | None = None,
          progress=..., check_cancel=...) -> dict
    # Builds char vocab (top vocab_size-4 chars + specials ["<pad>","<unk>","<bos>","<eos>"]).
    # resume_from = existing tiny model dir: load its weights + vocab and continue training (must have same arch,
    #   else ValueError). New unseen chars map to <unk>. total_steps = parent total_steps + steps.
    # device: None/"auto" -> accel.pick(accel.requested_device(device)): LEARNBOX_DEVICE, then cuda > tpu > xpu > mps > cpu
    #   (unusable device -> cpu + note). Mixed precision via accel.tiny_amp_dtype (cuda as before, tpu bf16); no amp param.
    # threads: torch.set_num_threads (None -> max(1, cpu_count-1)).
    # Cosine LR with warmup, AdamW, grad clip 1.0. Batch auto-reduced if seq/batch too big for data.
    # Writes out_dir/: model.pt ({"config","state_dict"}), vocab.json ({"itos":[...]}), model.onnx (fp32, input
    #   "input_ids" int64 [1, seq] dynamic seq axis, output "logits" [1, seq, vocab], opset 17),
    #   model.int8.onnx (onnxruntime.quantization.quantize_dynamic, QInt8; skip with onnx_int8=False if ort missing),
    #   info.json. Must verify the ONNX file loads in onnxruntime and matches PyTorch logits (max abs diff < 1e-3)
    #   when onnxruntime is installed; record "onnx_verified".
    # Returns info: {"kind":"tiny","preset","arch":{...},"params","steps","total_steps","parent": str|None,
    #   "train_loss","eval_loss_before","eval_loss_after","perplexity_after","seconds","device","dtype","threads",
    #   "onnx": bool,"onnx_int8": bool,"onnx_verified": bool,"onnx_error": str|None,
    #   "files": {"model.onnx": bytes, "model.int8.onnx": bytes, "model.pt": bytes},
    #   "vocab_size","seq_len","train_chars","sample": str (60-char generation from a corpus prefix)}
def load(model_dir) -> tuple[model, list[str]]   # cached by (path, mtime)
def generate(model_dir, prompt: str, max_new: int = 120, temperature: float = 0.8, top_k: int = 20) -> str
```
Architecture: token emb + learned pos emb, pre-LN blocks (fused qkv Linear, causal mask buffer sliced `[:T,:T]`, GELU MLP 4×), final LN, head tied to token embedding, init normal(0.02). ONNX export with `dynamo=False` first, fall back to default. Keep compatible with existing model dirs written by the old `train_mobile.py` (same state_dict keys: `tok`, `pos`, `blocks.N.{ln1,qkv,proj,ln2,mlp.0,mlp.2}`, `ln`, `head`; config keys `vocab_size,d_model,n_layer,n_head,seq_len`).

### A. `learnbox/devices.py` — PC capability profile

```python
def pc_profile() -> dict
    # {"os": str, "cpu_count": int, "ram_gb": float|None, "gpu": {"type": "cuda"|"mps"|None, "name": str|None,
    #  "vram_gb": float|None}, "torch": bool, "transformers": bool, "peft": bool, "onnxruntime": bool,
    #  "anthropic": bool, "pypdf": bool, "torch_xla": bool, "torch_tpu": bool, "tier": "pc-cpu"|"pc-gpu"|"pc-gpu-high",
    #  "accel": accel.info(probe_tpu=False)}   # gpu.type may also be "xpu"; ROCm adds gpu.vendor/rocm. TPU is never opened here.
    # RAM without psutil: Linux /proc/meminfo, macOS `sysctl -n hw.memsize`, Windows ctypes GlobalMemoryStatusEx.
    # GPU: torch.cuda (name, total memory) if torch importable; mps flag. Cache result for 60 s.
def recommend(profile: dict | None = None) -> dict
    # {"tiny_preset": "small"|"base"|..., "tiny_steps_per_cycle": int, "tiny_batch_size": int, "threads": int,
    #  "lora_base_model": str|None, "lora_max_steps": int, "notes": [Korean strings]}
    # pc-cpu -> small (or mini if RAM < 6 GB), lora 0.5B only if RAM >= 8 GB; pc-gpu (VRAM>=6) -> base, lora 1.5B;
    # pc-gpu-high (VRAM>=16) -> large, lora 7B. tier "tpu" (no GPU, TPU usable per accel info) -> by HBM, lora 1.5B.
    # threads = max(1, cpu_count-1).
```

### B. `learnbox/train_pc.py` — LoRA

```python
BASE_MODELS = [  # shown in UI; users may also type any HF model id
  {"id": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "params_b": 0.5, "min_ram_gb": 4, "min_vram_gb": 2, "license": "Apache-2.0"},
  {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "params_b": 1.5, "min_ram_gb": 8, "min_vram_gb": 4, "license": "Apache-2.0"},
  {"id": "Qwen/Qwen2.5-3B-Instruct",   "size": "3B",   "params_b": 3.0, "min_ram_gb": 14, "min_vram_gb": 8, "license": "Qwen Research (비상업)"},
  {"id": "Qwen/Qwen2.5-7B-Instruct",   "size": "7B",   "params_b": 7.0, "min_ram_gb": 30, "min_vram_gb": 16, "license": "Apache-2.0"},
]
def estimate_lora(base_model: str, lora_r: int, seq_len: int, batch_size: int) -> dict
    # {"base_params_b": float|None, "trainable_params_est": int|None, "ram_gb_est": float, "vram_gb_est": float,
    #  "adapter_mb_est": float}
def train_lora(train_path, eval_path, out_dir, base_model, max_steps=200, lr=2e-4, seq_len=512, batch_size=2,
               grad_accum=4, lora_r=8, lora_alpha=16, qa_path: str | None = None, resume_from: str | None = None,
               threads: int | None = None, progress=..., check_cancel=..., *, device=None, amp=None,
               dtype="auto", ...) -> dict
    # device as tinylm.train (TPU only for training; answer() keeps cuda > mps > cpu). amp: auto|fp16|off|fp32
    # (None -> LEARNBOX_AMP -> auto) resolved by accel.lora_precision: cuda/xpu without native bf16 -> fp16 autocast +
    # GradScaler (default), with bf16 -> bfloat16, off -> previous behaviour, fp32 -> float32; tpu -> bfloat16.
    # Plain-text corpus packed into seq_len blocks (continued pretraining). If qa_path (JSONL
    # {"question","answer","source_url"}) exists, mix in chat-formatted examples via tokenizer.apply_chat_template
    # (loss on all tokens is fine). resume_from = existing adapter dir: PeftModel.from_pretrained(..., is_trainable=True).
    # Saves adapter + tokenizer to out_dir, info.json.
    # Returns {"kind":"lora","base_model","device","steps","total_steps","parent","lora_r","lora_alpha",
    #   "trainable_params","total_params","train_loss","eval_loss_before","eval_loss_after","seconds",
    #   "train_blocks","qa_examples","seq_len","dtype","amp","files": {name: bytes}}
def answer(question: str, contexts: list[dict], base_model: str, adapter_dir: str | None = None,
           max_new_tokens: int = 256) -> dict
    # contexts rows have title, content, url. Returns {"answer": str, "citations": [{"index": n, "title","url"}]}
    # where citations are the [n] markers found in the answer (fallback: all contexts if none found).
# Back-compat aliases: train_pc = train_lora; old answer() callers will be updated by H.
```
Use `dtype=` (transformers 5.x) not `torch_dtype=`. Pass `attention_mask` to generate. Cache loaded models (one at a time).

### C. `learnbox/ingest.py` — import the user's own files

```python
SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".pdf", ".docx", ".srt", ".log"}
def extract_file(name: str, data: bytes, max_docs: int = 5000) -> list[dict]
    # -> [{"title": str, "text": str, "part": str|None}]. One doc per file, except .jsonl/.json arrays of records
    # with a text-like field ("text","content","body","article") -> one doc per record (title from "title" field
    # or "{name} #{i}"). CSV/TSV -> one doc; each row rendered "header: value" lines. HTML via extract.html_to_text.
    # TXT/MD/SRT/LOG: decode utf-8-sig, then cp949, then utf-8 errors=replace. PDF via pypdf (raise ValueError
    # "PDF를 읽으려면 pypdf가 필요합니다: pip install pypdf" if missing; pages joined with "\n\n"). DOCX via zipfile
    # + word/document.xml (paragraphs <w:p>, text <w:t>, tabs/breaks). Unsupported ext -> ValueError(Korean).
def ingest_bytes(cfg, db, name: str, data: bytes, actor: str = "user") -> dict
    # Size limit cfg["ingest"]["max_file_mb"]. sha256 dedupe via db.find_file -> {"status":"duplicate",...}.
    # Saves original to cfg.data_dir/"files"/f"{file_id}_{safe_name}". Each extracted doc ->
    # db.add_document(url=f"file://{file_id}/{quote(name)}" (+ f"#{i}" for multi-doc), title, text, source="file",
    # query=None, file_id=file_id). Skips docs with < 20 chars. db.set_file_docs. db.audit(actor,"ingest",...).
    # Returns {"status":"ok"|"duplicate","file_id","name","docs","chars","skipped"}.
def ingest_path(cfg, db, path: str, actor: str = "cli", progress=..., check_cancel=...) -> dict
    # File or directory (recursive, SUPPORTED only, skips hidden). Returns {"files","docs","duplicates","errors":[...]}.
```

### C. `learnbox/search.py` — search inside the learned files (core feature)

```python
def parse_query(q: str) -> dict          # {"terms": [...], "phrases": [...], "exclude": [...]}; "quoted phrase", -term
def keyword_search(db, q, limit=20, offset=0, sources=None, doc_id=None) -> list[dict]
    # FTS5 trigram over chunks: terms/phrases of >=3 chars via MATCH (AND), exclusions via NOT, shorter terms via
    # LIKE. sources: list of document.source values ("link","search","file","auto"), doc_id restricts to one doc.
    # Rows: {"chunk_id","doc_id","url","title","source","snippet" (HL markers),"content","score" (higher=better)}.
    # Never raises on odd input (quotes, symbols, FTS operators) — sanitize.
class Embedder:
    def __init__(self, model_name: str): ...  # lazy-loads transformers AutoTokenizer/AutoModel on first use, CPU ok
    def encode(self, texts: list[str], is_query: bool) -> list[list[float]]  # e5 prefixes "query: "/"passage: ",
                                                                            # mean pooling, L2-normalized
def get_embedder(cfg) -> Embedder | None   # None if disabled or transformers missing; process-wide cache
def index_embeddings(cfg, db, progress=..., check_cancel=..., batch_size=None) -> dict   # {"indexed","total","model"}
def semantic_search(cfg, db, q, limit=20, sources=None, doc_id=None) -> list[dict]   # same row shape, brute-force
                                                                                    # cosine; float32 blobs (array)
def search(cfg, db, q, limit=10, offset=0, sources=None, doc_id=None, mode="hybrid") -> dict
    # mode: "hybrid" | "keyword" | "semantic". hybrid = Reciprocal Rank Fusion (k=60) of keyword + semantic
    # (semantic part silently skipped if no embedder / no embeddings). One best chunk per document.
    # Returns {"results": [{"doc_id","chunk_id","url","title","source","snippet","score","match": "keyword"|
    #   "semantic"|"both"}], "mode_used": str, "total": int (number of matching docs, may be estimate),
    #   "offset", "limit", "extract": extractive_answer(...)}
def extractive_answer(q: str, rows: list[dict], max_sentences: int = 3) -> list[dict]
    # Split candidate chunk contents into sentences (Korean/English punctuation and newlines), score by query-term
    # overlap, return [{"sentence","doc_id","title","url"}] best first, deduped.
def doc_view(db, doc_id: int, q: str | None = None) -> dict | None
    # {"id","url","title","source","query","fetched_at","chars","text", "matches": [[start,end],...] (char offsets
    #  of every query term/phrase occurrence, case-insensitive, max 500), "file_id"}
def retrieve(cfg, db, q, k=6, sources=None) -> list[dict]
    # Context chunks for answer engines: [{"doc_id","chunk_id","title","url","content","source"}] (hybrid order).
```

### D. `learnbox/online.py` — 온라인 모델 (Claude API)

Read `/tmp/claude-0/bundled-skills/2.1.325/e55c1a090e1e420c771ad4e3f4313787/claude-api/python/claude-api/README.md` and `.../python/claude-api/tool-use.md` (structured outputs section) before writing. Use the official `anthropic` SDK only (no raw HTTP). Default model `claude-opus-5-5` (from `cfg["online"]["model"]`). Opus 5.5: never send `thinking` disabled/budget_tokens or sampling params (temperature/top_p/top_k); set `output_config={"effort": cfg effort}`. When `cfg["online"]["fallbacks"]` is true, call `client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`; otherwise `client.messages.create`. Always check `stop_reason == "refusal"` before reading content. Catch typed errors most-specific-first (`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `RateLimitError`, `BadRequestError`, `APIStatusError`, `APIConnectionError`) and raise `OnlineError(Korean message)`.

```python
class OnlineError(Exception): ...
def status(cfg) -> dict      # {"available": bool, "reason": str|None, "model": str, "key_source": "settings"|"env"|"profile"|None}
                             # unavailable if disabled or anthropic not installed; with no key in settings/env, try
                             # client construction anyway (ant auth profiles) and report key_source "profile".
def make_client(cfg)         # anthropic.Anthropic(api_key=settings key) or anthropic.Anthropic(); base_url from env honored
def ask(cfg, question: str, contexts: list[dict], history: list[dict] | None = None) -> dict
    # contexts from search.retrieve. Send each context as a document block
    # {"type":"document","source":{"type":"text","media_type":"text/plain","data": content},"title": title,
    #  "citations":{"enabled": True}} followed by the question text block. System prompt (Korean): answer only from
    # the documents, say when they don't contain the answer, answer in Korean. max_tokens 4096.
    # Returns {"answer": str (concatenated text blocks), "citations": [{"index": i (1-based doc index), "title","url",
    #  "cited_text"}], "model": response.model, "stop_reason": str, "usage": {"input_tokens","output_tokens"}}
    # refusal -> OnlineError("요청이 거절되었습니다 ...").
def generate_qa(cfg, chunks: list[dict], per_chunk: int = 2, progress=..., check_cancel=...) -> list[dict]
    # For each chunk, ask Claude for `per_chunk` Korean Q&A pairs grounded in that chunk using structured outputs
    # (output_config={"format": {"type":"json_schema","schema": {...}}, "effort": "low"}). Returns
    # [{"question","answer","source_url","title"}]. Skips chunks that fail (collect error count in log).
def build_qa_dataset(cfg, db, out_path, limit_chunks: int = 50, per_chunk: int = 2, progress=..., check_cancel=...) -> dict
    # Picks chunks (longest/most recent first, >= 200 chars) not yet used (track used chunk_ids in a sidecar
    # out_path + ".done.json"), appends JSONL rows to out_path. Returns {"path","added","total","errors"}.
```
Tests must not hit the network: use a local `http.server` fake that mimics `POST /v1/messages` (and returns a valid Messages response JSON incl. a text block with a `citations` list of `char_location` items) and point the client at it with `base_url` (env `ANTHROPIC_BASE_URL` or pass via cfg key `online.base_url` — support an optional `base_url` key in cfg["online"]).

### E. `learnbox/autolearn.py` — 자율 학습

```python
class AutoLearner:
    def __init__(self, app)   # app exposes: app.cfg, app.db, app.jobs (JobManager: submit(kind, params, actor) -> id,
                              # cancel(id)), app.gate() -> {"allowed",...}
    def start(self) -> None    # background daemon thread calling tick() every 20 s (guard against exceptions)
    def start_session(self, actor: str, overrides: dict | None = None) -> dict
        # ValueError("이미 자율 학습이 진행 중입니다.") if db.active_session(). Config = cfg["autolearn"] deep-merged
        # with overrides (+ tiny preset/arch resolved now and stored). Creates session, db.notify("info", ...).
    def stop_session(self, actor: str) -> dict
        # Marks session "finishing", cancels its running collect jobs, submits job "finalize_learning"
        # {"session_id"} and returns {"session_id","job_id"}. ValueError if no running session.
    def status(self) -> dict
        # {"active": bool, "session": session|None, "phase": "idle"|"collecting"|"training"|"waiting_window"|
        #  "finishing"|"cooldown", "window": app.gate(), "next_collect_at": iso|None, "next_train_check_at": iso|None,
        #  "new_chars_since_train": int, "recent_sessions": db.list_sessions(5)}
    def tick(self, now: datetime | None = None) -> dict   # one decision step; returns {"actions": [str,...]}
```
Session `stats` keys (kept in DB, updated by tick and by the job handlers below): `cycles, pages_collected, docs_new, docs_updated, chars_new, discovered_topics, trainings, last_collect_at, last_train_at, collect_jobs[], train_jobs[], models[] (model ids), train_history[{ts, model_id, kind, eval_loss_before, eval_loss_after}], hour_pages[{ts, n}]`.

Tick logic: no running session → nothing. Window closed → phase waiting_window (training still allowed if `train_outside_window`). Collect when window open, no collect job queued/running, `collect_interval_min` elapsed and pages in the last hour < `max_pages_per_hour`: pick `db.next_watch()`; if empty and `discover_topics`, discover first; if still empty, seed from the most frequent title words of existing documents; submit `collect_search` `{"query", "limit": pages_per_cycle}` or `collect_links` `{"urls":[value], "depth", "max_pages": pages_per_cycle}` with actor `"autolearn"` and `"session_id"` param; touch_watch. Train when (window open or train_outside_window), no heavy job queued/running, and (`chars_since(last_train_at)` ≥ `train_min_new_chars`, or ≥ 1 new char and `train_interval_min` elapsed): submit `autolearn_train` `{"session_id"}`.

```python
def discover_topics(db, limit: int = 10, exclude: set[str] | None = None) -> list[str]
    # Korean/English keyword candidates from recent documents: tokenize on non-word chars, strip common Korean
    # particles/endings (은 는 이 가 을 를 의 에 에서 으로 로 와 과 도 만 까지 부터 이다 있다 하는 한 및 등),
    # length 2–12, not digits; score = term frequency in the last 20 docs × log(N/df); return top new ones.
def run_autolearn_train(app, session_id: int, progress=..., check_cancel=...) -> dict
    # Job handler body for "autolearn_train": build corpus; train tiny (targets.tiny) with arch from the session
    # config, resume from the session's latest tiny model if its arch matches, steps = steps_per_cycle;
    # lora (targets.lora, resume from latest session lora); embeddings (targets.embeddings via
    # search.index_embeddings); qa (targets.qa and online available via online.build_qa_dataset, before lora).
    # Registers models with db.add_model(kind, name, path, info); updates session stats; db.notify("success",...).
    # Returns {"models": [...ids], "steps": {...}, "eval": {...}}.
def run_finalize(app, session_id: int, progress=..., check_cancel=...) -> dict
    # Job handler body for "finalize_learning": optional final train (final_train and new chars since last train),
    # then results.export_results(...), optional copy to autolearn.results_export_target (export.validate_target),
    # session status "ended", ended_at, result; db.notify("success", "학습이 종료되었습니다", body listing the result
    # folder, zip, file count and size, data={"result": name}); db.audit. Returns the export_results dict.
def record_collect(app, session_id: int, stats: dict) -> None
    # Called by H's collect job wrapper when a job carries session_id: add pages/docs/chars into session stats.
```
All model-training imports (tinylm, train_pc, search embeddings, online) are lazy and failures of one target must not abort the others (record the error in the result and in a warning notification).

### E. `learnbox/results.py` — 학습 결과 파일 내보내기

```python
FORMATS = ("jsonl", "csv", "md", "txt")
def export_results(cfg, db, session: dict | None = None, model_ids: list[int] | None = None,
                   formats: list[str] | None = None, progress=..., check_cancel=...) -> dict
    # Out dir: cfg.data_dir/"results"/f"learnbox-results-{YYYYmmdd-HHMMSS}" (local time). Layout:
    #   documents/documents.jsonl  {id,url,title,source,query,fetched_at,chars,text}
    #   documents/documents.csv    (utf-8-sig for Excel; text column included)
    #   documents/markdown/{id:05d}-{slug}.md   (front matter: title, url, source, fetched_at)
    #   documents/all.txt          (every doc: "### title\nurl\n\ntext\n\n")
    #   corpus/train.jsonl, corpus/eval.jsonl (fresh dataset.build_corpus), corpus/qa.jsonl (if data/datasets/qa.jsonl)
    #   index/learnbox.db          (db.backup_to — includes FTS index + embeddings)
    #   models/{name}/...          copies of the model dirs (session models, or model_ids, or all if both None)
    #   REPORT.md (Korean: period, collected pages/docs/chars by source, files ingested, trainings with loss
    #     before→after, models with params/size, how to use each file), report.json, manifest.json (sha256)
    #   and a ZIP of the folder next to it: f"{dir}.zip".
    # Session-scoped documents: when session given, documents section includes all docs (the knowledge base) but the
    # report marks how many were added during the session (fetched_at >= session.started_at).
    # Returns {"name","dir","zip","zip_bytes","files": [{"path","bytes","sha256","kind"}], "total_bytes", "summary": {...}}
def list_results(cfg) -> list[dict]          # [{"name","dir","zip","created_at","total_bytes","files"}] newest first
def result_file_path(cfg, name: str, rel: str | None = None) -> Path
    # Safe resolver for downloads (rel None -> the zip). Raise ValueError on traversal/unknown.
```

### F. `web/onboard.js` — browser on-device engine (no framework, plain script, defines `window.Onboard`)

```js
Onboard.detectDevice() -> Promise<{tier: "low"|"mid"|"high", mobile: bool, memoryGB: number|null, cores: number,
   webgpu: bool, storageQuotaMB: number|null, label: string (Korean, e.g. "중급 휴대폰 · 메모리 4GB · 8코어"),
   limits: {snapshotChunks: number, maxNewTokens: number, preferInt8: bool, ortThreads: number, maxModelMB: number}}>
   // tier rules: memoryGB<=3 or cores<=4 -> low; memoryGB>=8 and cores>=8 (or desktop with webgpu) -> high; else mid.
   // low: snapshot 1500 chunks, 60 tokens, int8, 1 thread, 8 MB; mid: 6000/120/int8/2/40 MB; high: 30000/200/fp32/4/400 MB
Onboard.db  // IndexedDB key-value store "learnbox" v2 store "kv": get(k), set(k,v), del(k)
Onboard.saveSnapshot(chunks)   // chunks rows {c,d,t,x,u,s} from GET /api/snapshot; builds a BM25 index over
                               // character bigrams (+ ASCII words) and stores {chunks, index, saved_at, count}
Onboard.snapshotInfo() -> Promise<{count, saved_at}|null>
Onboard.search(q, {limit=20, sources=null}) -> Promise<[{doc_id, chunk_id, title, url, source, snippet, score}]>
   // BM25 (k1=1.2, b=0.75) on bigrams; snippet = ~200 chars around the first match with "\u0002"/"\u0003" markers;
   // one best chunk per doc. Supports "phrase" (post-filter) and -exclude.
Onboard.extract(q, rows, n=3) -> [{sentence, doc_id, title, url}]   // same idea as server extractive_answer
Onboard.saveModel(meta, onnxArrayBuffer, itos, variant) / Onboard.modelInfo() -> {id, name, variant, bytes, info}
Onboard.generate(prompt, {maxNew, temperature=0.8, topK=20, onToken}) -> Promise<string>
   // loads onnxruntime-web 1.18.0 from https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js
   // (wasmPaths = same dist dir), executionProviders ["webgpu","wasm"] when webgpu else ["wasm"], numThreads from
   // limits; int64 input_ids [1,T] (T = min(len, seq_len)); top-k sampling; stops at id 3 (<eos>).
Onboard.answer(q, {limit}) -> Promise<{rows, extract, generated|null}>  // offline Q&A: search + extract (+ generate
   // continuation from the best sentence when a model is saved)
```
`web/sw.js`: cache app shell (`/`, `/app.js`, `/onboard.js`, `/style.css`, `/manifest.webmanifest`, `/icon.svg`) plus the onnxruntime-web CDN files on first use (cache-first for the CDN URLs); never cache `/api/`. Bump cache name to `learnbox-shell-v2`.

### G. Web UI (`web/index.html`, `web/app.js`, `web/style.css`)

Same visual system as v1 (keep the CSS tokens, dark mode, mobile bottom tabs / desktop top tabs, 16px gutters, no horizontal scroll). Load `/onboard.js` before `/app.js`. Header: brand, online/offline pill, role pill, **notification bell with unread badge** (dropdown list, mark read; poll `/api/notifications?unread=1` every 15 s; when a new `success` notification about learning end arrives, show a toast and — if permitted — a browser `Notification`), logout.

Tabs: **검색** · **자료** · **학습** · **모델** · **작업** · **관리**(admin only).

- 검색: query input (+ help: `"구문"`, `-제외`), engine segmented control **검색만 / 소형 AI / LoRA / 온보드 / 온라인**, source filter chips (전체/웹/검색수집/파일), mode select (통합/키워드/의미), result list with source badge and match badge, pagination (이전/다음), "핵심 문장" box (extract), answer box with citations ([n] linked to source rows), **document viewer** panel (full text, highlighted matches, 이전/다음 일치 navigation, open original link) via `GET /api/documents/<id>?q=`. Offline or engine=온보드 → use `Onboard.search` / `Onboard.answer`. Snapshot card ("이 기기에 저장": size chosen by device tier, shows count/date).
- 자료: drag-and-drop + file picker upload (multiple; each file sent as raw body to `POST /api/files` with header `X-File-Name: encodeURIComponent(name)`; per-file progress/result), imported files list, gate banner, link collection, realtime web search collection, watchlist (shows origin user/auto), documents list with source filter and delete (admin).
- 학습: device card (PC profile from `/api/device` + this device from `Onboard.detectDevice()`), **자율 학습** card (status/phase/window/stats/next times, start form with targets checkboxes tiny/lora/embeddings/qa, tiny preset incl. auto/custom, steps per cycle, min new chars, collect interval, pages per cycle; **학습 종료** button with confirm → shows finishing progress → result card with file list, sizes and download links when done), manual **소형 AI 모델 학습** with size selector (presets with labels + "직접 설정" fields d_model/n_layer/n_head/seq_len/vocab_size, steps, batch) and **live estimate** (POST `/api/estimate`), fit badges (휴대폰 저사양/고사양/PC CPU/PC GPU), resume-from-model select; manual **LoRA 학습** (base model select from `/api/device` base_models + custom id, lora_r, seq_len, steps, use_qa, resume select, estimate via `/api/estimate` with `{"kind":"lora",...}`); buttons **의미 검색 색인 만들기** and **Q&A 학습 데이터 만들기 (온라인 모델)**; results list (학습 결과 파일) with download links.
- 모델: model cards (kind badge 소형 AI / LoRA, params, size, losses, steps/total_steps, parent, files with sizes); tiny models: **이 기기로 받기** (variant auto by tier: int8 if preferInt8 and available; manual override select), on-device generate box with streaming tokens; "PC에서 생성" (POST `/api/generate`).
- 작업: as v1 plus new kinds labels.
- 관리: export to drive (as v1), settings (v1 fields + 온라인 모델: enabled, model, API key (masked), effort; 의미 검색: enabled, model; 자율 학습 기본값: results_export_target, train_outside_window, max_pages_per_hour, formats), audit log.

Downloads that need auth: fetch with the Bearer header → `blob` → object URL → `<a download>`.

## 4. Job kinds (H wires these in `jobs.LANES` and `app.py`)

| kind | lane | params | handler |
|---|---|---|---|
| collect_links | collect | urls, depth, same_domain, max_pages?, session_id? | collector + autolearn.record_collect |
| collect_search | collect | query, limit?, session_id? | collector + autolearn.record_collect |
| ingest_path | light | path | ingest.ingest_path |
| index_embeddings | light | — | search.index_embeddings |
| build_qa | light | limit_chunks?, per_chunk? | online.build_qa_dataset → `data/datasets/qa.jsonl` |
| train_tiny | heavy | preset, d_model…, steps, batch_size, lr, resume_from_model_id?, device? | build_corpus + tinylm.train + add_model("tiny") |
| train_lora | heavy | base_model, max_steps, …, use_qa, resume_from_model_id?, device?, amp? | build_corpus + train_pc.train_lora + add_model("lora") |
| autolearn_train | heavy | session_id | autolearn.run_autolearn_train |
| finalize_learning | heavy | session_id | autolearn.run_finalize |
| export | heavy | target, include, actor | export.export |
| train_mobile / train_pc | — | legacy aliases of train_tiny / train_lora | |

## 5. HTTP API (H implements; G/F consume). JSON unless noted. Auth: `Authorization: Bearer <token>`.

Existing v1 routes stay (`/api/info`, `/api/login`, `/api/logout`, `/api/status`, `/api/collect/links`, `/api/collect/search`, `/api/watch*`, `/api/jobs*`, `/api/models`, `/api/settings`, `/api/drives`, `/api/export`, `/api/audit`, `/api/documents` list, `DELETE /api/documents/<id>`).

| Method & path | Role | Body / query | Response |
|---|---|---|---|
| GET /api/status | user | | v1 fields + `online` (online.status), `autolearn` (AutoLearner.status() compact), `unread` |
| GET /api/device | user | | `{"pc": pc_profile, "recommend": recommend, "presets": {name: {...preset, "estimate": estimate}}, "base_models": BASE_MODELS, "limits": LIMITS, "accel": {auto, devices: {id: {available, label, name}}, cuda_bf16, rocm, torch_version, torch_xla_version, torch_tpu_version, tpu}, "defaults": {training_device, amp}}` (defaults: env → config → auto) |
| POST /api/estimate | user | `{"kind":"tiny", preset, d_model…, batch_size}` or `{"kind":"lora", base_model, lora_r, seq_len, batch_size}`; optional `device` | tiny: `{"arch", "estimate"}`; lora: estimate_lora; 400 with Korean error on invalid |
| GET /api/search | user | `q, limit, offset, sources (comma list), mode, doc_id` | search.search(...) |
| GET /api/documents/<id> | user | `q` | doc_view |
| POST /api/ask | user | `{"q", "engine": "search"|"tiny"|"lora"|"online", "model_id"?, "sources"?: [...]}` | `{"engine","answer": str|null,"citations": [...],"sources": [rows like search results],"extract": [...],"model": str|null,"note": str|null}` |
| POST /api/generate | user | `{"model_id", "prompt", "max_new"?}` | `{"text"}` (tiny models on the PC) |
| POST /api/files | user | raw bytes, header `X-File-Name` (URI-encoded), max `ingest.max_file_mb` | ingest_bytes result |
| GET /api/files | user | | `{"files": db.list_files()}` |
| POST /api/ingest/path | admin | `{"path"}` | `{"job_id"}` |
| POST /api/train/tiny | admin | tiny params | `{"job_id"}` |
| POST /api/train/lora | admin | lora params | `{"job_id"}` |
| POST /api/index/embeddings | admin | | `{"job_id"}` |
| POST /api/qa/build | admin | `{"limit_chunks"?, "per_chunk"?}` | `{"job_id"}` |
| GET /api/autolearn | user | | AutoLearner.status() |
| POST /api/autolearn/start | admin | overrides (subset of cfg autolearn + `tiny_preset`, `tiny_arch`) | `{"session": ...}` |
| POST /api/autolearn/stop | admin | | `{"session_id","job_id"}` |
| GET /api/notifications | user | `unread=1`, `limit` | `{"notifications": [...], "unread": n}` |
| POST /api/notifications/read | user | `{"ids"?: [...]}` (omit = all) | `{"ok": true}` |
| GET /api/results | admin | | `{"results": list_results}` |
| GET /api/results/<name>/download | admin | | ZIP bytes (`application/zip`, Content-Disposition attachment) |
| GET /api/results/<name>/files/<path> | admin | | file bytes |
| GET /api/snapshot | user | `max_chunks` | `{"chunks": db.chunk_snapshot(max_chunks), "created_at"}` |
| GET /api/models/<id>/files/<name> | user | | tiny model files: model.onnx, model.int8.onnx, vocab.json, info.json |

## 6. Testing

Each owner adds `unittest` tests that run offline and fast (< 60 s each file; tiny training tests use `nano`-sized custom arch and ≤ 60 steps). Run: `cd /home/user/claud && .venv/bin/python -m unittest tests.test_<name> -v`. Do not delete existing tests owned by someone else.
