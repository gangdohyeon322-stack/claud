"""온라인 모델(Claude API) 테스트. 네트워크 없이 로컬 가짜 서버(POST /v1/messages)로 확인한다.
실행: python -m unittest tests.test_online -v"""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from learnbox import online
from learnbox.db import DB

HANGUL = "가-힣"
MODEL = "claude-opus-5-5"
ENV_DROP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
            "ANTHROPIC_BASE_URL", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
            "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_CUSTOM_HEADERS")


def message_json(content, stop_reason="end_turn", stop_details=None, model=MODEL):
    return {"id": "msg_01Test", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
            "usage": {"input_tokens": 1234, "output_tokens": 56, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0}}


def cite(doc_index, cited, title):
    return {"type": "char_location", "cited_text": cited, "document_index": doc_index, "document_title": title,
            "start_char_index": 0, "end_char_index": len(cited)}


def sent_key(req):
    """SDK 버전에 따라 x-api-key 또는 Authorization: Bearer 로 키를 보낸다."""
    h = req["headers"]
    if h.get("x-api-key"):
        return h["x-api-key"]
    auth = h.get("authorization", "")
    return auth[len("Bearer "):] if auth.startswith("Bearer ") else None


def error_json(kind, message):
    return {"type": "error", "error": {"type": kind, "message": message}, "request_id": "req_test"}


class FakeAnthropic:
    """POST /v1/messages 를 흉내 내는 로컬 서버. responder(path, headers, body) -> (status, json)."""

    def __init__(self):
        self.requests = []
        self.responder = lambda path, headers, body: (200, message_json([{"type": "text", "text": "ok"}]))
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                headers = {k.lower(): v for k, v in self.headers.items()}
                fake.requests.append({"path": self.path, "headers": headers, "body": body})
                status, payload = fake.responder(self.path, headers, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("request-id", "req_test")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class OnlineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeAnthropic()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        env = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
        env.update(HOME=self.tmp, XDG_CONFIG_HOME=self.tmp, APPDATA=self.tmp)  # ant 프로필이 없는 빈 홈
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        online._PROBE.update(t=0.0, key=None, v=None)
        self.fake.requests.clear()
        self.fake.responder = lambda p, h, b: (200, message_json([{"type": "text", "text": "ok"}]))

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **over):
        oc = {"enabled": True, "model": MODEL, "api_key": "sk-ant-test", "effort": "medium",
              "max_context_chunks": 8, "fallbacks": False, "base_url": self.fake.url, "max_retries": 0,
              "timeout_sec": 10}
        oc.update(over)
        return {"online": oc}

    def contexts(self):
        return [
            {"doc_id": 1, "chunk_id": 11, "title": "광합성", "url": "https://ko.wikipedia.org/wiki/광합성",
             "content": "광합성은 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정이다.", "source": "search"},
            {"doc_id": 2, "chunk_id": 21, "title": "엽록체", "url": "file://3/note.md",
             "content": "엽록체는 식물 세포에서 광합성이 일어나는 세포 소기관이다.", "source": "file"},
        ]


class AskTest(OnlineTestBase):
    def test_request_body_and_citation_parsing(self):
        content = [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "자료에 따르면 "},
            {"type": "text", "text": "광합성은 엽록체에서 일어납니다",
             "citations": [cite(1, "엽록체는 식물 세포에서 광합성이 일어나는 세포 소기관이다.", "엽록체")]},
            {"type": "text", "text": ". 그리고 "},
            {"type": "text", "text": "빛 에너지로 포도당을 만듭니다",
             "citations": [cite(0, "광합성은 빛 에너지를 이용해", "광합성"), cite(0, "광합성은 빛 에너지를 이용해", "광합성")]},
            {"type": "text", "text": "."},
        ]
        self.fake.responder = lambda p, h, b: (200, message_json(content))
        r = online.ask(self.cfg(), "광합성은 어디서 일어나?", self.contexts())

        self.assertEqual(len(self.fake.requests), 1)
        req = self.fake.requests[0]
        self.assertTrue(req["path"].startswith("/v1/messages"))
        self.assertNotIn("server-side-fallback", req["headers"].get("anthropic-beta", ""))
        self.assertEqual(sent_key(req), "sk-ant-test")
        body = req["body"]
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["output_config"], {"effort": "medium"})
        for banned in ("temperature", "top_p", "top_k", "thinking", "fallbacks"):
            self.assertNotIn(banned, body)
        self.assertRegex(body["system"], f"[{HANGUL}]")
        self.assertIn("한국어", body["system"])
        msgs = body["messages"]
        self.assertEqual(msgs[-1]["role"], "user")
        blocks = msgs[-1]["content"]
        docs = [b for b in blocks if b["type"] == "document"]
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["source"], {"type": "text", "media_type": "text/plain",
                                             "data": self.contexts()[0]["content"]})
        self.assertEqual(docs[0]["title"], "광합성")
        self.assertTrue(all(d["citations"] == {"enabled": True} for d in docs))
        self.assertEqual(blocks[-1]["type"], "text")
        self.assertIn("광합성은 어디서 일어나?", blocks[-1]["text"])
        self.assertEqual([b["type"] for b in blocks], ["document", "document", "text"])

        self.assertEqual(r["answer"], "자료에 따르면 광합성은 엽록체에서 일어납니다 [2]. 그리고 빛 에너지로 포도당을 만듭니다 [1].")
        self.assertEqual(r["model"], MODEL)
        self.assertEqual(r["stop_reason"], "end_turn")
        self.assertEqual(r["usage"], {"input_tokens": 1234, "output_tokens": 56})
        self.assertEqual(len(r["citations"]), 2)  # 같은 인용 중복 제거
        c2, c1 = r["citations"]
        self.assertEqual(c2, {"index": 2, "title": "엽록체", "url": "file://3/note.md",
                              "cited_text": "엽록체는 식물 세포에서 광합성이 일어나는 세포 소기관이다."})
        self.assertEqual(c1["index"], 1)
        self.assertEqual(c1["url"], "https://ko.wikipedia.org/wiki/광합성")
        self.assertEqual(c1["cited_text"], "광합성은 빛 에너지를 이용해")

    def test_fallbacks_use_beta_endpoint(self):
        self.fake.responder = lambda p, h, b: (200, message_json([{"type": "text", "text": "답"}]))
        r = online.ask(self.cfg(fallbacks=True, effort="high"), "질문", self.contexts())
        req = self.fake.requests[0]
        self.assertIn("server-side-fallback-2026-07-01", req["headers"].get("anthropic-beta", ""))
        self.assertEqual(req["body"]["fallbacks"], "default")
        self.assertEqual(req["body"]["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", req["body"])
        self.assertEqual(r["answer"], "답")
        self.assertEqual(r["citations"], [])

    def test_fallbacks_skipped_for_unsupported_model(self):
        online.ask(self.cfg(fallbacks=True, model="claude-haiku-4-5"), "질문", self.contexts())
        req = self.fake.requests[0]
        self.assertNotIn("fallbacks", req["body"])
        self.assertNotIn("server-side-fallback", req["headers"].get("anthropic-beta", ""))

    def test_invalid_effort_defaults_to_medium_and_context_limit(self):
        ctx = self.contexts() * 3
        online.ask(self.cfg(effort="turbo", max_context_chunks=3), "질문", ctx)
        body = self.fake.requests[0]["body"]
        self.assertEqual(body["output_config"], {"effort": "medium"})
        self.assertEqual(sum(1 for b in body["messages"][-1]["content"] if b["type"] == "document"), 3)

    def test_empty_context_content_keeps_index_mapping(self):
        ctx = self.contexts()
        ctx.insert(0, {"title": "빈 조각", "url": "u0", "content": "   "})
        self.fake.responder = lambda p, h, b: (200, message_json(
            [{"type": "text", "text": "엽록체", "citations": [cite(1, "엽록체는", "엽록체")]}]))
        r = online.ask(self.cfg(), "질문", ctx)
        docs = [b for b in self.fake.requests[0]["body"]["messages"][-1]["content"] if b["type"] == "document"]
        self.assertEqual(len(docs), 2)
        self.assertEqual(r["citations"][0]["index"], 3)  # contexts 기준 1부터 (빈 조각 포함)
        self.assertEqual(r["citations"][0]["title"], "엽록체")

    def test_history_goes_before_question(self):
        hist = [{"role": "assistant", "content": "먼저 시작한 assistant는 버림"},
                {"role": "user", "content": "이전 질문"}, {"role": "assistant", "content": "이전 답"},
                {"role": "system", "content": "무시"}, {"role": "user", "content": ""}]
        online.ask(self.cfg(), "새 질문", self.contexts(), history=hist)
        msgs = self.fake.requests[0]["body"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
        self.assertEqual(msgs[0]["content"], "이전 질문")
        self.assertEqual(msgs[1]["content"], "이전 답")

    def test_no_contexts_skips_api(self):
        r = online.ask(self.cfg(), "질문", [])
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(r["citations"], [])
        self.assertEqual(r["stop_reason"], "no_context")
        self.assertRegex(r["answer"], f"[{HANGUL}]")

    def test_refusal_raises(self):
        self.fake.responder = lambda p, h, b: (200, message_json([], stop_reason="refusal", stop_details=None))
        with self.assertRaises(online.OnlineError) as cm:
            online.ask(self.cfg(), "질문", self.contexts())
        self.assertTrue(str(cm.exception).startswith("요청이 거절되었습니다"))
        self.assertEqual(cm.exception.kind, "refusal")

        self.fake.responder = lambda p, h, b: (200, message_json(
            [{"type": "text", "text": "부분"}], stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "x"}))
        with self.assertRaises(online.OnlineError) as cm:
            online.ask(self.cfg(fallbacks=True), "질문", self.contexts())
        self.assertIn("cyber", str(cm.exception))

    def test_http_errors_map_to_korean(self):
        cases = [
            (401, "authentication_error", "auth", "인증"),
            (403, "permission_error", "permission", "권한"),
            (404, "not_found_error", "not_found", MODEL),
            (429, "rate_limit_error", "rate_limit", "사용량 제한"),
            (400, "invalid_request_error", "bad_request", "잘못된 요청"),
            (500, "api_error", "server", "서버 오류"),
            (529, "overloaded_error", "overloaded", "과부하"),
        ]
        for status, etype, kind, needle in cases:
            with self.subTest(status=status):
                self.fake.responder = lambda p, h, b, s=status, t=etype: (s, error_json(t, "boom"))
                with self.assertRaises(online.OnlineError) as cm:
                    online.ask(self.cfg(), "질문", self.contexts())
                e = cm.exception
                self.assertEqual(e.kind, kind)
                self.assertEqual(e.status, status)
                self.assertIn(needle, str(e))
                self.assertRegex(str(e), f"[{HANGUL}]")
        self.assertTrue(online.OnlineError("x", kind="auth", fatal=True).fatal)

    def test_connection_error(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(online.OnlineError) as cm:
            online.ask(self.cfg(base_url=f"http://127.0.0.1:{port}"), "질문", self.contexts())
        self.assertIn(cm.exception.kind, ("connection", "timeout"))
        self.assertRegex(str(cm.exception), f"[{HANGUL}]")

    def test_disabled_and_missing_key(self):
        with self.assertRaises(online.OnlineError) as cm:
            online.ask(self.cfg(enabled=False), "질문", self.contexts())
        self.assertEqual(cm.exception.kind, "disabled")
        with self.assertRaises(online.OnlineError) as cm:
            online.ask(self.cfg(api_key=""), "질문", self.contexts())
        self.assertEqual(cm.exception.kind, "no_key")
        self.assertEqual(self.fake.requests, [])

    def test_env_key_and_env_base_url(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env"
        os.environ["ANTHROPIC_BASE_URL"] = self.fake.url
        online.ask(self.cfg(api_key="", base_url=""), "질문", self.contexts())
        self.assertEqual(sent_key(self.fake.requests[0]), "sk-ant-env")

    def test_make_client(self):
        c = online.make_client(self.cfg())
        self.assertEqual(c.api_key, "sk-ant-test")
        self.assertTrue(str(c.base_url).startswith(self.fake.url))


class StatusTest(OnlineTestBase):
    def test_status_shapes(self):
        s = online.status(self.cfg())
        self.assertEqual(s, {"available": True, "reason": None, "model": MODEL, "key_source": "settings"})

        s = online.status(self.cfg(enabled=False))
        self.assertFalse(s["available"])
        self.assertRegex(s["reason"], f"[{HANGUL}]")
        self.assertEqual(s["model"], MODEL)

        s = online.status(self.cfg(api_key=""))  # 키 없음, 프로필 없음
        self.assertFalse(s["available"])
        self.assertIsNone(s["key_source"])
        self.assertIn("API 키", s["reason"])

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env"
        s = online.status(self.cfg(api_key=""))
        self.assertTrue(s["available"])
        self.assertEqual(s["key_source"], "env")

    def test_status_masked_key_is_not_a_key(self):
        s = online.status(self.cfg(api_key=online.MASK))
        self.assertNotEqual(s["key_source"], "settings")

    def test_status_profile(self):
        class FakeClient:
            api_key = None
            auth_token = None
            credentials = object()

            def close(self):
                pass

        with mock.patch.object(online, "_build_client", return_value=FakeClient()):
            s = online.status(self.cfg(api_key=""))
        self.assertTrue(s["available"])
        self.assertEqual(s["key_source"], "profile")

    def test_status_without_sdk(self):
        with mock.patch.object(online, "_has_sdk", return_value=False):
            s = online.status(self.cfg())
        self.assertFalse(s["available"])
        self.assertIn("pip install anthropic", s["reason"])

    def test_status_works_with_config_object(self):
        from learnbox.config import Config
        cfg = Config(self.tmp)
        s = online.status(cfg)
        self.assertEqual(s["model"], "claude-opus-5-5")
        self.assertIn(s["key_source"], (None, "profile"))


def qa_response(pairs):
    return message_json([{"type": "text", "text": json.dumps({"pairs": pairs}, ensure_ascii=False)}])


class QATest(OnlineTestBase):
    def chunk(self, i, n=260):
        return {"chunk_id": i, "doc_id": i, "title": f"문서 {i}", "url": f"https://example.com/{i}",
                "content": ("가나다라마바사 " * 60)[:n], "source": "link"}

    def test_generate_qa_request_and_parsing(self):
        pairs = [{"question": "질문1?", "answer": "답1."}, {"question": "질문2?", "answer": "답2."},
                 {"question": "질문3?", "answer": "답3."}]
        self.fake.responder = lambda p, h, b: (200, qa_response(pairs))
        prog = []
        rows = online.generate_qa(self.cfg(fallbacks=True), [self.chunk(1), self.chunk(2)], per_chunk=2,
                                  progress=lambda p, m: prog.append((p, m)))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], {"question": "질문1?", "answer": "답1.", "source_url": "https://example.com/1",
                                   "title": "문서 1"})
        self.assertEqual(rows[2]["source_url"], "https://example.com/2")
        self.assertEqual(len(self.fake.requests), 2)
        body = self.fake.requests[0]["body"]
        fmt = body["output_config"]["format"]
        self.assertEqual(body["output_config"]["effort"], "low")
        self.assertEqual(fmt["type"], "json_schema")
        schema = fmt["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["pairs"])
        item = schema["properties"]["pairs"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(sorted(item["required"]), ["answer", "question"])
        doc = body["messages"][0]["content"][0]
        self.assertEqual(doc["type"], "document")
        self.assertNotIn("citations", doc)  # 구조화 출력과 citations는 함께 쓸 수 없음
        self.assertIn("2개", body["messages"][0]["content"][1]["text"])
        for banned in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(banned, body)
        self.assertEqual(body["fallbacks"], "default")
        self.assertEqual(prog[-1][0], 1.0)

    def test_generate_qa_skips_failures(self):
        responses = iter([
            (200, message_json([{"type": "text", "text": "not json"}])),
            (200, message_json([], stop_reason="refusal")),
            (500, error_json("api_error", "boom")),
            (200, qa_response([{"question": "Q?", "answer": "A."}, {"question": "Q?", "answer": "dup"}])),
        ])
        self.fake.responder = lambda p, h, b: next(responses)
        stats = {}
        seen = []
        with self.assertLogs("learnbox.online", "WARNING"):
            rows = online.generate_qa(self.cfg(), [self.chunk(i) for i in range(4)], per_chunk=2, stats=stats,
                                      on_chunk=lambda c, r: seen.append((c["chunk_id"], len(r))))
        self.assertEqual(rows, [{"question": "Q?", "answer": "A.", "source_url": "https://example.com/3",
                                 "title": "문서 3"}])
        self.assertEqual(stats["errors"], 3)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(seen, [(3, 1)])

    def test_generate_qa_fatal_error_aborts(self):
        self.fake.responder = lambda p, h, b: (401, error_json("authentication_error", "invalid x-api-key"))
        with self.assertRaises(online.OnlineError) as cm:
            online.generate_qa(self.cfg(), [self.chunk(1), self.chunk(2), self.chunk(3)])
        self.assertEqual(cm.exception.kind, "auth")
        self.assertEqual(len(self.fake.requests), 1)

    def test_generate_qa_consecutive_failures_abort(self):
        self.fake.responder = lambda p, h, b: (200, message_json([{"type": "text", "text": "{bad"}]))
        with self.assertRaises(online.OnlineError), self.assertLogs("learnbox.online", "WARNING"):
            online.generate_qa(self.cfg(), [self.chunk(i) for i in range(10)])
        self.assertEqual(len(self.fake.requests), online.QA_MAX_CONSECUTIVE_FAILURES)

    def test_generate_qa_cancel(self):
        class Stop(Exception):
            pass

        calls = {"n": 0}

        def check():
            calls["n"] += 1
            if calls["n"] > 1:
                raise Stop()

        self.fake.responder = lambda p, h, b: (200, qa_response([{"question": "Q?", "answer": "A."}]))
        with self.assertRaises(Stop):
            online.generate_qa(self.cfg(), [self.chunk(1), self.chunk(2)], check_cancel=check)
        self.assertEqual(len(self.fake.requests), 1)


class BuildDatasetTest(OnlineTestBase):
    def setUp(self):
        super().setUp()
        self.db = DB(Path(self.tmp) / "t.db")
        self.db.add_document("https://a.example/1", "오래된 문서", "가" * 300, "link")
        self.db.add_document("https://a.example/2", "짧은 문서", "짧은 내용입니다. " * 5, "link")  # < 200자
        self.db.add_document("https://a.example/3", "새 문서", "나" * 500, "search", query="q")
        self.db.add_document("file://1/x.md", "파일 문서", "다" * 250, "file")
        self.out = Path(self.tmp) / "datasets" / "qa.jsonl"
        self.counter = 0

        def responder(p, h, b):
            self.counter += 1
            n = self.counter
            return 200, qa_response([{"question": f"질문 {n}-1?", "answer": "답."},
                                     {"question": f"질문 {n}-2?", "answer": "답."}])
        self.fake.responder = responder

    def test_sidecar_dedupe_and_append(self):
        r1 = online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=2, per_chunk=2)
        self.assertEqual(r1["path"], str(self.out))
        self.assertEqual((r1["added"], r1["total"], r1["errors"]), (4, 4, 0))
        sidecar = Path(str(self.out) + ".done.json")
        done1 = set(json.loads(sidecar.read_text(encoding="utf-8"))["chunk_ids"])
        self.assertEqual(len(done1), 2)
        # 최근 문서(파일 문서, 새 문서) 먼저
        urls = [json.loads(line)["source_url"] for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(urls, ["file://1/x.md"] * 2 + ["https://a.example/3"] * 2)

        r2 = online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=10, per_chunk=2)
        self.assertEqual((r2["added"], r2["total"], r2["chunks"]), (2, 6, 1))  # 짧은 문서는 제외
        done2 = set(json.loads(sidecar.read_text(encoding="utf-8"))["chunk_ids"])
        self.assertTrue(done1 < done2)
        self.assertEqual(len(done2), 3)

        n_req = len(self.fake.requests)
        r3 = online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=10, per_chunk=2)
        self.assertEqual((r3["added"], r3["total"], r3["errors"], r3["chunks"]), (0, 6, 0, 0))
        self.assertEqual(len(self.fake.requests), n_req)  # 새 요청 없음

        rows = [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertTrue({"question", "answer", "source_url", "title"} <= set(row))

    def test_failed_chunks_are_retried_later(self):
        first = {"n": 0}

        def responder(p, h, b):
            first["n"] += 1
            if first["n"] == 1:
                return 200, message_json([{"type": "text", "text": "oops"}])
            return 200, qa_response([{"question": f"Q{first['n']}?", "answer": "A."}])
        self.fake.responder = responder
        with self.assertLogs("learnbox.online", "WARNING"):
            r1 = online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=3, per_chunk=1)
        self.assertEqual((r1["added"], r1["errors"]), (2, 1))
        r2 = online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=3, per_chunk=1)
        self.assertEqual((r2["added"], r2["errors"], r2["chunks"]), (1, 0, 1))
        self.assertEqual(r2["total"], 3)

    def test_progress_and_disabled(self):
        prog = []
        online.build_qa_dataset(self.cfg(), self.db, self.out, limit_chunks=1, progress=lambda p, m: prog.append(p))
        self.assertEqual(prog[0], 0.0)
        self.assertEqual(prog[-1], 1.0)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in prog))
        with self.assertRaises(online.OnlineError):
            online.build_qa_dataset(self.cfg(enabled=False), self.db, self.out, limit_chunks=5)


if __name__ == "__main__":
    unittest.main()
