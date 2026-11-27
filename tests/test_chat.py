"""대화형 챗봇 (docs/PRD-chatbot.md): 페르소나, 정체성 대화 데이터, 대화 학습 형식, 대화 API·명령줄.

실제 모델 생성은 tiny-random-gpt2로 학습 경로만 확인하고, 대화 답변은 train_pc.chat을 대역으로 바꿔 확인한다.
"""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbox import chat, train_pc  # noqa: E402

TINY = "hf-internal-testing/tiny-random-gpt2"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
KO = "고구려는 기원전 37년에 주몽이 세운 나라이다. 수도는 졸본, 국내성, 평양으로 옮겼다. " * 4


def _tok(name):
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"transformers 없음: {e}") from e
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        raise unittest.SkipTest(f"토크나이저를 불러올 수 없음: {name}: {e}") from e


class PersonaTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        p = chat.persona(None)
        self.assertEqual(p["name"], "LearnBox")
        p = chat.persona({"name": " 하늘 ", "intro": "", "style": "반말"})
        self.assertEqual((p["name"], p["style"]), ("하늘", "반말"))
        self.assertEqual(p["intro"], chat.DEFAULT_PERSONA["intro"])   # 빈 값은 기본값
        self.assertEqual(len(chat.persona({"name": "가" * 100})["name"]), chat.LIMITS["name"])

    def test_check_persona(self):
        self.assertEqual(chat.check_persona({"name": " 하늘 ", "use_context": 0, "max_history": "8"}),
                         {"name": "하늘", "use_context": False, "max_history": 8})
        for bad in ({"name": 3}, {"name": "가" * 41}, {"max_history": 1}, {"max_new_tokens": "x"}, {"unknown": 1}):
            with self.assertRaises(ValueError, msg=bad):
                chat.check_persona(bad)
        with self.assertRaises(ValueError):
            chat.check_persona("하늘")

    def test_system_prompt(self):
        p = chat.persona({"name": "하늘", "extra": "항상 한 문장으로."})
        s = chat.system_prompt(p)
        self.assertIn("당신의 이름은 하늘입니다", s)
        self.assertIn("저는 하늘입니다", s)
        self.assertIn("항상 한 문장으로.", s)
        self.assertNotIn("[n]", s)
        self.assertIn("[n]", chat.system_prompt(p, with_context=True))

    def test_user_with_context(self):
        self.assertEqual(chat.user_with_context("질문", None), "질문")
        t = chat.user_with_context("질문", [{"title": "제목", "content": "내용" * 1000}], max_chars=10)
        self.assertTrue(t.startswith("질문\n\n참고 자료:\n[1] 제목\n"))
        self.assertLess(len(t), 60)


class MessagesTest(unittest.TestCase):
    def test_clean_messages(self):
        msgs = [{"role": "assistant", "content": "먼저 말함"}, {"role": "user", "content": " 안녕 "},
                {"role": "system", "content": "무시"}, {"role": "user", "content": "넌 누구야"},
                {"role": "assistant", "content": ""}, "이상한 값"]
        self.assertEqual(chat.clean_messages(msgs), [{"role": "user", "content": "안녕\n넌 누구야"}])
        long = [{"role": r, "content": f"{r}{i}"} for i in range(10) for r in ("user", "assistant")]
        out = chat.clean_messages(long + [{"role": "user", "content": "마지막"}], max_history=4)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[-1]["content"], "마지막")
        self.assertLessEqual(len(out), 4)
        for bad in ([], [{"role": "assistant", "content": "답"}], "안녕"):
            with self.assertRaises(ValueError):
                chat.clean_messages(bad)


class DatasetTest(unittest.TestCase):
    def test_identity_dialogs(self):
        p = chat.persona({"name": "하늘"})
        a, b = chat.identity_dialogs(p), chat.identity_dialogs(p)
        self.assertEqual(a, b)   # 규칙 생성, 결정적
        self.assertGreater(len(a), 60)
        with_sys = [r for r in a if r["system"]]
        self.assertTrue(0.3 < len(with_sys) / len(a) < 0.7)   # 절반쯤은 시스템 메시지 없이
        who = [r for r in a if r["messages"][0]["content"] == "안녕 넌 누구야"]
        self.assertTrue(who and all("하늘" in r["messages"][1]["content"] for r in who))
        self.assertTrue(any(len(r["messages"]) == 6 for r in a))   # 여러 턴 대화

    def test_build_and_load(self):
        d = Path(tempfile.mkdtemp(prefix="lb-chatds-"))
        self.addCleanup(shutil.rmtree, d, True)
        r = chat.build_dataset({"name": "하늘"}, d / "chat.jsonl",
                               qa_rows=[{"question": "수도는?", "answer": "서울입니다.", "source_url": ""}])
        self.assertEqual(r["qa"], 1)
        self.assertEqual(r["dialogs"], r["identity"] + 1)
        rows = chat.load_dialogs(d / "chat.jsonl")
        self.assertEqual(len(rows), r["dialogs"])
        with open(d / "chat.jsonl", "a", encoding="utf-8") as f:   # 형식이 틀린 줄은 건너뛴다
            f.write("{bad json\n" + json.dumps({"messages": [{"role": "assistant", "content": "x"}]}) + "\n")
        self.assertEqual(len(chat.load_dialogs(d / "chat.jsonl")), r["dialogs"])
        self.assertEqual(chat.load_dialogs(d / "없음.jsonl"), [])

    def test_josa(self):
        self.assertEqual(chat.josa("하늘", "과", "와"), "하늘과")
        self.assertEqual(chat.josa("나비", "과", "와"), "나비와")
        self.assertEqual(chat.josa("LearnBox", "과", "와"), "LearnBox와")
        rows = chat.identity_dialogs(chat.persona({"name": "하늘", "intro": "자료 도우미"}))
        text = " ".join(m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant")
        self.assertNotIn("하늘라고", text)
        self.assertNotIn("하늘예요", text)
        rows = chat.identity_dialogs(chat.persona({"name": "나비"}))
        text = " ".join(m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant")
        self.assertNotIn("나비이라고", text)

    def test_identity_score(self):
        self.assertEqual(chat.identity_score(["저는 하늘입니다", "저는 Qwen입니다"], "하늘"), 0.5)
        self.assertEqual(chat.identity_score([], "하늘"), 0.0)


class FormatChatTest(unittest.TestCase):
    MSGS = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요! 저는 하늘입니다."},
            {"role": "user", "content": "넌 누구야"}, {"role": "assistant", "content": "저는 하늘이에요."}]

    def test_qwen_template_labels_only_answers(self):
        tok = _tok(QWEN)
        for system in ("당신의 이름은 하늘입니다.", None):
            ids, labels = train_pc.format_chat(tok, self.MSGS, system)
            conv = ([{"role": "system", "content": system}] if system else []) + self.MSGS
            self.assertEqual(tok.decode(ids), tok.apply_chat_template(conv, tokenize=False))
            learned = tok.decode([i for i, lab in zip(ids, labels) if lab != -100])
            self.assertEqual(learned, "안녕하세요! 저는 하늘입니다.<|im_end|>\n저는 하늘이에요.<|im_end|>\n")

    def test_fallback_without_template(self):
        tok = _tok(TINY)
        if getattr(tok, "chat_template", None):
            self.skipTest("채팅 템플릿이 있는 토크나이저")
        ids, labels = train_pc.format_chat(tok, self.MSGS, "시스템")
        self.assertEqual(len(ids), len(labels))
        learned = [i for i, lab in zip(ids, labels) if lab != -100]
        eos = [tok.eos_token_id]
        want = (tok("안녕하세요! 저는 하늘입니다.", add_special_tokens=False)["input_ids"] + eos
                + tok("저는 하늘이에요.", add_special_tokens=False)["input_ids"] + eos)
        self.assertEqual(learned, want)   # 답변과 끝 표시에만 손실 (질문·시스템·줄바꿈은 제외)


@unittest.skipUnless(importlib.util.find_spec("peft"), "peft 필요")
class TrainWithChatTest(unittest.TestCase):
    """대화 데이터를 섞어 학습. 대화 데이터가 없으면 결과가 예전과 같다."""

    @classmethod
    def setUpClass(cls):
        _tok(TINY)
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-chattrain-"))
        texts = [KO, "LoRA freezes the base model and trains small matrices. " * 4] * 4
        with open(cls.tmp / "train.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                f.write(json.dumps({"text": t, "title": f"t{i}", "url": f"u{i}"}, ensure_ascii=False) + "\n")
        with open(cls.tmp / "eval.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": KO, "title": "e", "url": "e"}, ensure_ascii=False) + "\n")
        chat.build_dataset({"name": "하늘"}, cls.tmp / "chat.jsonl", repeats=1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    KW = dict(max_steps=3, lr=1e-3, seq_len=32, batch_size=2, grad_accum=1, lora_r=4, lora_alpha=8, threads=1, seed=3)

    def run_train(self, name, **kw):
        return train_pc.train_lora(str(self.tmp / "train.jsonl"), str(self.tmp / "eval.jsonl"), self.tmp / name, TINY,
                                   **self.KW, **kw)

    def test_chat_examples_used_and_no_chat_unchanged(self):
        base = self.run_train("a")
        none = self.run_train("b", chat_path=None)
        self.assertEqual(base["loss_history"], none["loss_history"])   # 대화 데이터 없음 = 예전과 같음
        self.assertEqual(base["chat_examples"], 0)
        self.assertNotIn("chat_path", base.get("ignored_params") or [])
        withc = self.run_train("c", chat_path=str(self.tmp / "chat.jsonl"))
        self.assertGreater(withc["chat_examples"], 10)
        self.assertNotIn("chat_path", withc.get("ignored_params") or [])

    def test_checkpoint_records_chat(self):
        ck = self.tmp / "ck"
        self.run_train("d", chat_path=str(self.tmp / "chat.jsonl"), checkpoint_dir=str(ck), checkpoint_every_steps=1)
        metas = list(ck.rglob("meta.json"))
        if not metas:
            self.skipTest("완료 후 체크포인트가 지워짐")
        m = json.loads(metas[0].read_text(encoding="utf-8"))
        self.assertTrue(m["chat_path"].endswith("chat.jsonl"))
        self.assertGreater(m["chat_examples"], 0)


class FakeChat:
    def __init__(self, answer="저는 하늘입니다."):
        self.calls, self.answer = [], answer

    def __call__(self, messages, base_model=None, adapter_dir=None, system=None, contexts=None, max_new_tokens=256,
                 **kw):
        self.calls.append({"messages": messages, "base_model": base_model, "adapter_dir": adapter_dir,
                           "system": system, "contexts": contexts, "max_new_tokens": max_new_tokens})
        cites = [{"index": 1, "title": contexts[0].get("title"), "url": ""}] if contexts else []
        return {"answer": self.answer + (" [1]" if contexts else ""), "citations": cites}


class AppChatTest(unittest.TestCase):
    def setUp(self):
        from learnbox.app import App

        self.dir = tempfile.mkdtemp(prefix="lb-chatapp-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.app = App(self.dir)
        self.app.cfg.data["embeddings"]["enabled"] = False
        self.app.cfg.data["chatbot"]["name"] = "하늘"
        for i in range(4):
            self.app.db.add_document(f"https://g/{i}", f"고구려 {i}", KO + f"번호 {i}", "link")
        mdir = Path(self.dir) / "lora-x"
        mdir.mkdir()
        self.model_id = self.app.db.add_model("lora", "lora-x", str(mdir), {"base_model": "base/x"})
        self.fake = FakeChat()
        p = mock.patch.object(train_pc, "chat", self.fake)
        p.start()
        self.addCleanup(p.stop)

    def test_chat_uses_persona_history_and_context(self):
        msgs = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"},
                {"role": "user", "content": "고구려 수도는 어디였어?"}]
        r = self.app.chat(msgs)
        call = self.fake.calls[-1]
        self.assertEqual(len(call["messages"]), 3)
        self.assertIn("하늘", call["system"])
        self.assertEqual((call["base_model"], Path(call["adapter_dir"]).name), ("base/x", "lora-x"))
        self.assertTrue(call["contexts"])                        # 자료 참고(기본 켬)
        self.assertIn("[n]", call["system"])
        self.assertEqual((r["persona"], r["model"], r["engine"]), ("하늘", "lora-x", "lora"))
        self.assertTrue(r["sources"])

    def test_without_context(self):
        r = self.app.chat([{"role": "user", "content": "안녕 넌 누구야"}], use_context=False)
        self.assertEqual(self.fake.calls[-1]["contexts"], [])
        self.assertNotIn("[n]", self.fake.calls[-1]["system"])
        self.assertEqual((r["answer"], r["sources"]), ("저는 하늘입니다.", []))
        self.app.cfg.data["chatbot"]["use_context"] = False
        self.app.chat([{"role": "user", "content": "고구려"}])
        self.assertEqual(self.fake.calls[-1]["contexts"], [])

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.app.chat([{"role": "user", "content": "안녕"}], engine="online")
        with self.assertRaises(ValueError):
            self.app.chat([{"role": "assistant", "content": "답"}])
        with self.assertRaises(ValueError):
            self.app.chat([{"role": "user", "content": "안녕"}], model_id=999)

    def test_build_chat_job_and_plan(self):
        j = self.app.jobs.run_inline("build_chat", {"include_qa": False})
        self.assertEqual(j["status"], "done", j["message"])
        path = self.app.cfg.datasets_dir / "chat.jsonl"
        self.assertTrue(path.is_file())
        info = self.app.chat_info()
        self.assertTrue(info["dataset"]["exists"])
        self.assertEqual(info["persona"]["name"], "하늘")
        self.assertEqual([m["id"] for m in info["models"]], [self.model_id])
        self.assertEqual(self.app.plan_lora({})["chat_path"], str(path))
        self.assertIsNone(self.app.plan_lora({"use_chat": False})["chat_path"])

    def test_identity_test_after_chat_training(self):
        def fake_train(train_path, eval_path, out_dir, base_model, *a, chat_path=None, **kw):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "adapter_config.json").write_text("{}", encoding="utf-8")
            return {"kind": "lora", "base_model": base_model, "steps": 1, "total_steps": 1, "notes": [],
                    "chat_examples": 5 if chat_path else 0, "eval_loss_before": 2.0, "eval_loss_after": 1.0}

        self.app.jobs.run_inline("build_chat", {"include_qa": False})
        with mock.patch.object(train_pc, "train_lora", fake_train), contextlib.redirect_stdout(io.StringIO()):
            j = self.app.jobs.run_inline("train_lora", {"base_model": "base/x", "max_steps": 1})
            self.assertEqual(j["status"], "done", j["message"])
            t = j["result"]["identity_test"]
            self.assertEqual((t["name"], t["score"]), ("하늘", 1.0))
            self.assertIsNone(self.fake.calls[-1]["system"])     # 시스템 메시지 없이 시험
            j = self.app.jobs.run_inline("train_lora", {"base_model": "base/x", "max_steps": 1, "use_chat": False})
            self.assertNotIn("identity_test", j["result"])


class ServerChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from learnbox.app import App
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp(prefix="lb-chatsrv-")
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        handler = make_handler(cls.app, Sessions())
        handler.log_message = lambda *a, **k: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.admin = cls.call("POST", "/api/login", {"pin": "987654"})[1]["token"]
        cls.user = cls.call("POST", "/api/login", {"pin": "1234"})[1]["token"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, token=None):
        data = json.dumps(body).encode() if body is not None else (b"{}" if method == "POST" else None)
        req = urllib.request.Request(cls.base + path, method=method, data=data)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def test_info_and_validation(self):
        st, info = self.call("GET", "/api/chat", token=self.user)
        self.assertEqual(st, 200)
        self.assertEqual(info["persona"]["name"], "LearnBox")
        st, _ = self.call("POST", "/api/chat", {"messages": "안녕"}, token=self.user)
        self.assertEqual(st, 400)
        with mock.patch.object(train_pc, "chat", FakeChat()):
            st, r = self.call("POST", "/api/chat", {"messages": [{"role": "user", "content": "안녕"}]}, token=self.user)
        self.assertIn(st, (400, 503))   # LoRA 모델이 없으면 한국어 오류 (학습 패키지가 없으면 503)

    def test_settings_and_build(self):
        st, _ = self.call("POST", "/api/settings", {"chatbot": {"name": "가" * 50}}, token=self.admin)
        self.assertEqual(st, 400)
        st, cfg = self.call("POST", "/api/settings", {"chatbot": {"name": " 하늘 ", "use_context": False},
                                                     "lora_training": {"use_chat": 0}}, token=self.admin)
        self.assertEqual(st, 200)
        self.assertEqual((cfg["chatbot"]["name"], cfg["chatbot"]["use_context"]), ("하늘", False))
        self.assertIs(cfg["lora_training"]["use_chat"], False)
        st, _ = self.call("POST", "/api/chat/build", {}, token=self.user)
        self.assertEqual(st, 403)
        st, r = self.call("POST", "/api/chat/build", {"include_qa": False}, token=self.admin)
        self.assertEqual(st, 200)
        self.assertEqual(self.app.db.get_job(r["job_id"])["kind"], "build_chat")


class CliTest(unittest.TestCase):
    def test_parser(self):
        from learnbox import __main__ as cli

        p = cli.build_parser() if hasattr(cli, "build_parser") else None
        if p is None:
            self.skipTest("parser builder 없음")
        a = p.parse_args(["chat", "--no-context", "-m", "안녕", "-m", "넌 누구야"])
        self.assertEqual((a.cmd if hasattr(a, "cmd") else "chat", a.no_context, a.message), ("chat", True,
                                                                                             ["안녕", "넌 누구야"]))
        self.assertTrue(p.parse_args(["build-chat", "--no-qa"]).no_qa)

    def test_chat_command_with_messages(self):
        from learnbox import __main__ as cli
        from learnbox.app import App

        d = tempfile.mkdtemp(prefix="lb-chatcli-")
        self.addCleanup(shutil.rmtree, d, True)
        app = App(d)
        mdir = Path(d) / "lora-y"
        mdir.mkdir()
        app.db.add_model("lora", "lora-y", str(mdir), {"base_model": "base/y"})
        fake = FakeChat("저는 LearnBox입니다.")
        args = mock.Mock(model_id=None, no_context=True, message=["안녕 넌 누구야", "고마워"])
        buf = io.StringIO()
        with mock.patch.object(train_pc, "chat", fake), contextlib.redirect_stdout(buf):
            self.assertEqual(cli.cmd_chat(app, args), 0)
        self.assertIn("LearnBox: 저는 LearnBox입니다.", buf.getvalue())
        self.assertEqual(len(fake.calls[-1]["messages"]), 3)   # 이전 대화를 함께 보낸다


if __name__ == "__main__":
    unittest.main()
