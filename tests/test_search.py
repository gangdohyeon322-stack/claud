"""search.py 테스트 (네트워크 없음, 큰 모델 다운로드 없음: 가짜 임베더 사용).

실행: cd /home/user/claud && .venv/bin/python -m unittest tests.test_search -v
"""
import importlib.util
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from learnbox import search as S
from learnbox.config import Config
from learnbox.db import DB, HL_END, HL_START

DOCS = [
    # (url, title, text, source)
    ("https://a/1", "딥 러닝", "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다. "
     "역전파 알고리즘으로 가중치를 학습한다.\n합성곱 신경망은 이미지 인식에 쓰인다.", "link"),
    ("https://a/2", "김치찌개", "김치찌개는 한국 음식이다. 요리할 때 돼지고기와 두부를 넣는다.\n"
     "오래 끓일수록 맛이 깊어진다.", "link"),
    ("https://a/3", "요리 추천 서비스", "이 서비스는 신경망으로 요리 레시피를 추천한다. "
     "인공 신경망이 사용자의 취향을 학습한다.", "search"),
    ("file://1/notes.txt", "Transformer notes", "Transformer 모델은 attention 메커니즘을 쓴다. "
     "Deep learning 분야의 대표적인 신경망 구조이다. 100% 병렬 처리가 가능하다.", "file"),
]

# 가짜 임베더: 개념 사전으로 결정적인 벡터를 만든다 (실제 모델 다운로드 없이 의미 검색 경로 테스트)
CONCEPTS = [
    ("food", ["요리", "김치", "음식", "레시피", "찌개", "두부", "cuisine", "food", "dish"]),
    ("ml", ["신경망", "러닝", "학습", "attention", "transformer", "deep", "neural"]),
    ("image", ["이미지", "합성곱", "vision"]),
    ("service", ["서비스", "추천", "사용자"]),
]


class FakeEmbedder:
    model_name = "fake-e5"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, is_query=False):
        self.calls += 1
        out = []
        for t in texts:
            low = (t or "").lower()
            v = [float(sum(low.count(w) for w in words)) for _, words in CONCEPTS] + [0.05]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


def make_db(tmp):
    cfg = Config(tmp)
    db = DB(cfg.db_path)
    ids = []
    for url, title, text, source in DOCS:
        db.add_document(url, title, text, source)
        ids.append(db.list_documents(1)[0]["id"])
    return cfg, db, ids


class ParseQueryTest(unittest.TestCase):
    def test_terms_phrases_exclusions(self):
        pq = S.parse_query('"인공 신경망" 딥 -요리 -"김치 찌개" 학습')
        self.assertEqual(pq["phrases"], ["인공 신경망"])
        self.assertEqual(pq["terms"], ["딥", "학습"])
        self.assertEqual(sorted(pq["exclude"]), sorted(["요리", "김치 찌개"]))

    def test_operators_and_symbols_are_stripped(self):
        self.assertEqual(S.parse_query("신경망 AND 딥 OR NOT NEAR")["terms"], ["신경망", "딥"])
        self.assertEqual(S.parse_query("a*")["terms"], ["a"])
        self.assertEqual(S.parse_query("(신경망)")["terms"], ["신경망"])
        self.assertEqual(S.parse_query("::"), {"terms": [], "phrases": [], "exclude": []})
        self.assertEqual(S.parse_query('"'), {"terms": [], "phrases": [], "exclude": []})
        self.assertEqual(S.parse_query("-"), {"terms": [], "phrases": [], "exclude": []})
        self.assertEqual(S.parse_query("C++ C#")["terms"], ["C++", "C#"])
        self.assertEqual(S.parse_query("신경망이란?")["terms"], ["신경망이란"])
        self.assertEqual(S.parse_query(None)["terms"], [])

    def test_relaxed_terms_strip_particles(self):
        rt = S.relaxed_terms(S.parse_query("인공 신경망이란 무엇인가요?"))
        self.assertIn("신경망", rt)
        self.assertNotIn("무엇인가요", rt)
        self.assertEqual(S.relaxed_terms(S.parse_query("국가")), ["국가"])  # 1글자만 남으면 떼지 않음


class KeywordSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg, self.db, self.ids = make_db(self.tmp)

    def tearDown(self):
        S.set_embedder(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def docs(self, rows):
        return {r["doc_id"] for r in rows}

    def test_korean_term(self):
        rows = S.keyword_search(self.db, "신경망")
        self.assertEqual(self.docs(rows), {self.ids[0], self.ids[2], self.ids[3]})
        for r in rows:
            self.assertIn(f"{HL_START}신경망{HL_END}", r["snippet"])
            self.assertEqual(set(r), {"chunk_id", "doc_id", "url", "title", "source", "snippet", "content", "score"})
        scores = [r["score"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_short_term_uses_like(self):
        rows = S.keyword_search(self.db, "딥")
        self.assertEqual(self.docs(rows), {self.ids[0]})
        self.assertIn(f"{HL_START}딥{HL_END}", rows[0]["snippet"])
        self.assertEqual(self.docs(S.keyword_search(self.db, "신경망 딥")), {self.ids[0]})

    def test_phrase(self):
        self.assertEqual(self.docs(S.keyword_search(self.db, '"인공 신경망"')), {self.ids[0], self.ids[2]})
        self.assertEqual(S.keyword_search(self.db, '"신경망 인공"'), [])
        # 따옴표 없이 쓰면 두 단어가 따로 있어도 된다
        self.assertEqual(self.docs(S.keyword_search(self.db, "인공 신경망")), {self.ids[0], self.ids[2]})

    def test_exclusion(self):
        self.assertEqual(self.docs(S.keyword_search(self.db, "신경망 -요리")), {self.ids[0], self.ids[3]})
        self.assertEqual(self.docs(S.keyword_search(self.db, "신경망 -attention")), {self.ids[0], self.ids[2]})
        self.assertEqual(self.docs(S.keyword_search(self.db, '신경망 -"인공 신경망"')), {self.ids[3]})
        # 제외어만 있으면 제외어가 없는 문서 전부
        self.assertEqual(self.docs(S.keyword_search(self.db, "-요리")), {self.ids[0], self.ids[3]})

    def test_case_insensitive_english(self):
        self.assertEqual(self.docs(S.keyword_search(self.db, "DEEP learning")), {self.ids[3]})
        self.assertEqual(self.docs(S.keyword_search(self.db, "transformer")), {self.ids[3]})

    def test_weird_input_never_raises(self):
        weird = ['"', "AND", "a*", "(", "::", ")", "NEAR(a b, 3)", '"unbalanced', "-", "--", "'", "OR OR", "^start",
                 "col:val", "*", "\\", "_", "a\x00b", "😀", "x" * 3000, '"" ""', "{}", "[a]", "NOT 신경망",
                 "신경망 NOT", "\x02\x03", "-\"\"", "' OR 1=1 --", "%' --"]
        for q in weird:
            with self.subTest(q=q):
                self.assertIsInstance(S.keyword_search(self.db, q), list)
                self.assertIsInstance(S.keyword_search(self.db, q, match_all=False), list)
                r = S.search(self.cfg, self.db, q)
                self.assertIsInstance(r["results"], list)
                self.assertIsNotNone(S.doc_view(self.db, self.ids[0], q))
        # LIKE 와일드카드(%, _)가 아니라 글자 그대로: '%'가 들어 있는 문서 하나만
        self.assertEqual(self.docs(S.keyword_search(self.db, "%")), {self.ids[3]})
        self.assertEqual(self.docs(S.keyword_search(self.db, "100%")), {self.ids[3]})
        self.assertEqual(S.keyword_search(self.db, "_"), [])
        self.assertEqual(self.docs(S.keyword_search(self.db, "NOT 신경망")), {self.ids[0], self.ids[2], self.ids[3]})

    def test_sources_filter(self):
        self.assertEqual(self.docs(S.keyword_search(self.db, "신경망", sources=["file"])), {self.ids[3]})
        self.assertEqual(self.docs(S.keyword_search(self.db, "신경망", sources="link,search")),
                         {self.ids[0], self.ids[2]})
        self.assertEqual(self.docs(S.keyword_search(self.db, "딥", sources=["file"])), set())
        r = S.search(self.cfg, self.db, "신경망", sources=["search"])
        self.assertEqual([x["doc_id"] for x in r["results"]], [self.ids[2]])
        self.assertEqual(r["total"], 1)
        r = S.search(self.cfg, self.db, "신경망", sources=["web"])  # 별칭
        self.assertEqual([x["doc_id"] for x in r["results"]], [self.ids[0]])

    def test_doc_id_filter(self):
        rows = S.keyword_search(self.db, "신경망", doc_id=self.ids[2])
        self.assertTrue(rows)
        self.assertEqual(self.docs(rows), {self.ids[2]})
        self.assertEqual(S.keyword_search(self.db, "신경망", doc_id=self.ids[1]), [])

    def test_pagination(self):
        for i in range(12):
            self.db.add_document(f"https://p/{i}", f"페이지 {i}", f"공통단어 문서 번호 {i}. " * (i + 1), "link")
        seen = []
        for off in (0, 5, 10):
            r = S.search(self.cfg, self.db, "공통단어", limit=5, offset=off)
            self.assertEqual(r["total"], 12)
            self.assertEqual((r["offset"], r["limit"]), (off, 5))
            seen += [x["doc_id"] for x in r["results"]]
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(set(seen)), 12)  # 겹치지 않음, 문서당 하나
        self.assertEqual(S.search(self.cfg, self.db, "공통단어", limit=5, offset=20)["results"], [])
        # keyword_search는 조각 단위 페이지
        a = S.keyword_search(self.db, "공통단어", limit=3)
        b = S.keyword_search(self.db, "공통단어", limit=3, offset=3)
        self.assertFalse({x["chunk_id"] for x in a} & {x["chunk_id"] for x in b})

    def test_one_row_per_doc_in_search(self):
        long_text = "\n".join(f"{i}번째 문단: 신경망 이야기를 길게 이어 간다. " + "가나다라마바사 " * 60 for i in range(6))
        self.db.add_document("https://long/1", "긴 문서", long_text, "link")
        rows = S.keyword_search(self.db, "신경망", limit=50)
        doc = self.db.list_documents(1)[0]["id"]
        self.assertGreater(sum(1 for r in rows if r["doc_id"] == doc), 1)
        res = S.search(self.cfg, self.db, "신경망", limit=50)
        ids = [r["doc_id"] for r in res["results"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(doc, ids)

    def test_search_keyword_mode_shape(self):
        r = S.search(self.cfg, self.db, "신경망", mode="keyword")
        self.assertEqual(r["mode_used"], "keyword")
        self.assertEqual(r["total"], 3)
        self.assertEqual(set(r["results"][0]),
                         {"doc_id", "chunk_id", "url", "title", "source", "snippet", "score", "match"})
        self.assertTrue(all(x["match"] == "keyword" for x in r["results"]))
        self.assertTrue(r["extract"])
        self.assertEqual(S.search(self.cfg, self.db, "")["results"], [])
        self.assertEqual(S.search(self.cfg, self.db, "존재하지않는단어")["results"], [])

    def test_hybrid_without_embeddings_falls_back_to_keyword(self):
        S.set_embedder(FakeEmbedder())  # 임베더는 있지만 색인이 없음
        r = S.search(self.cfg, self.db, "신경망", mode="hybrid")
        self.assertEqual(r["mode_used"], "keyword")
        r = S.search(self.cfg, self.db, "신경망", mode="semantic")
        self.assertEqual(r["mode_used"], "keyword")
        self.assertTrue(r["note"])
        self.assertTrue(r["results"])

    def test_relaxed_fallback_for_questions(self):
        r = S.search(self.cfg, self.db, "합성곱 신경망이란 무엇인가요?")
        self.assertTrue(r["relaxed"])
        self.assertEqual(r["results"][0]["doc_id"], self.ids[0])  # 두 단어 모두 있는 문서가 먼저
        ctx = S.retrieve(self.cfg, self.db, "김치찌개에는 무엇을 넣나요?", k=3)
        self.assertEqual(ctx[0]["doc_id"], self.ids[1])
        self.assertEqual(set(ctx[0]), {"doc_id", "chunk_id", "title", "url", "content", "source"})
        self.assertEqual(S.retrieve(self.cfg, self.db, "", k=3), [])

    def test_retrieve_fills_with_more_chunks(self):
        long_text = "\n".join(f"{i}번 문단 고유어휘 설명. " + "라마바사아자차 " * 100 for i in range(5))
        self.db.add_document("https://long/2", "고유 문서", long_text, "file")
        ctx = S.retrieve(self.cfg, self.db, "고유어휘", k=4)
        self.assertEqual(len(ctx), 4)
        self.assertEqual(len({c["chunk_id"] for c in ctx}), 4)


class SemanticSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg, self.db, self.ids = make_db(self.tmp)
        self.emb = FakeEmbedder()

    def tearDown(self):
        S.set_embedder(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_embedder_override_and_disabled(self):
        S.set_embedder(self.emb)
        self.assertIs(S.get_embedder(self.cfg), self.emb)
        self.cfg.data["embeddings"]["enabled"] = False
        self.assertIsNone(S.get_embedder(self.cfg))
        with self.assertRaises(ValueError):
            S.index_embeddings(self.cfg, self.db)

    def test_index_embeddings_float32_and_idempotent(self):
        calls = []
        res = S.index_embeddings(self.cfg, self.db, progress=lambda p, m: calls.append(p), batch_size=2,
                                 embedder=self.emb)
        n_chunks = self.db.counts()["chunks"]
        self.assertEqual(res, {"indexed": n_chunks, "total": n_chunks, "model": "fake-e5"})
        self.assertEqual(calls[-1], 1.0)
        for _cid, _did, dim, blob in self.db.iter_embeddings("fake-e5"):
            self.assertEqual(len(blob), 4 * dim)
            v = S.unpack_vector(blob)
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=5)
        self.assertEqual(S.index_embeddings(self.cfg, self.db, embedder=self.emb)["indexed"], 0)
        # 새 문서만 추가로 색인
        self.db.add_document("https://a/5", "두부 요리", "두부 조림은 간장으로 만드는 요리이다. " * 3, "link")
        self.assertEqual(S.index_embeddings(self.cfg, self.db, embedder=self.emb)["indexed"], 1)

    def test_cancel(self):
        class Stop(Exception):
            pass

        def cancel():
            raise Stop()

        with self.assertRaises(Stop):
            S.index_embeddings(self.cfg, self.db, check_cancel=cancel, embedder=self.emb)

    def test_semantic_search_and_filters(self):
        S.index_embeddings(self.cfg, self.db, embedder=self.emb)
        rows = S.semantic_search(self.cfg, self.db, "cuisine", limit=2, embedder=self.emb)
        self.assertEqual(rows[0]["doc_id"], self.ids[1])  # 음식 문서가 가장 가깝다
        self.assertEqual(set(rows[0]), {"chunk_id", "doc_id", "url", "title", "source", "snippet", "content", "score"})
        self.assertGreaterEqual(rows[0]["score"], rows[1]["score"])
        rows = S.semantic_search(self.cfg, self.db, "cuisine", sources=["file"], embedder=self.emb)
        self.assertEqual({r["doc_id"] for r in rows}, {self.ids[3]})
        rows = S.semantic_search(self.cfg, self.db, "cuisine", doc_id=self.ids[0], embedder=self.emb)
        self.assertEqual({r["doc_id"] for r in rows}, {self.ids[0]})
        rows = S.semantic_search(self.cfg, self.db, "cuisine -김치", embedder=self.emb)
        self.assertNotIn(self.ids[1], {r["doc_id"] for r in rows})
        self.assertEqual(S.semantic_search(self.cfg, self.db, "", embedder=self.emb), [])

    def test_hybrid_rrf(self):
        S.index_embeddings(self.cfg, self.db, embedder=self.emb)
        S.set_embedder(self.emb)  # 모듈 이음새로 주입 (search()가 get_embedder 사용)
        r = S.search(self.cfg, self.db, "신경망", mode="hybrid")
        self.assertEqual(r["mode_used"], "hybrid")
        self.assertIn(r["results"][0]["match"], ("both", "keyword"))
        kw_docs = {self.ids[0], self.ids[2], self.ids[3]}
        both = {x["doc_id"] for x in r["results"] if x["match"] == "both"}
        self.assertTrue(both and both <= kw_docs)
        scores = [x["score"] for x in r["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(scores[0], 2.0 / (S.RRF_K + 1) + 1e-9)
        self.assertEqual(len({x["doc_id"] for x in r["results"]}), len(r["results"]))
        # 키워드로는 없는 단어: 의미 검색만으로 찾는다
        r = S.search(self.cfg, self.db, "cuisine", mode="hybrid")
        self.assertEqual(r["mode_used"], "hybrid")
        self.assertTrue(r["results"])
        self.assertTrue(all(x["match"] == "semantic" for x in r["results"]))
        self.assertIn(self.ids[1], [x["doc_id"] for x in r["results"][:2]])
        # 의미 모드
        r = S.search(self.cfg, self.db, "dish", mode="semantic")
        self.assertEqual(r["mode_used"], "semantic")
        self.assertEqual(r["results"][0]["doc_id"], self.ids[1])
        # 키워드 모드에서는 의미 검색을 하지 않는다
        calls = self.emb.calls
        self.assertEqual(S.search(self.cfg, self.db, "cuisine", mode="keyword")["results"], [])
        self.assertEqual(self.emb.calls, calls)
        # 제외어는 의미 검색 결과에도 적용
        r = S.search(self.cfg, self.db, "cuisine -김치", mode="hybrid")
        self.assertNotIn(self.ids[1], [x["doc_id"] for x in r["results"]])
        # 문맥 고르기도 통합 순서
        ctx = S.retrieve(self.cfg, self.db, "cuisine", k=2)
        self.assertTrue(ctx)
        self.assertIn("content", ctx[0])

    def test_embedder_error_falls_back(self):
        S.index_embeddings(self.cfg, self.db, embedder=self.emb)

        class Broken(FakeEmbedder):
            def encode(self, texts, is_query=False):
                raise RuntimeError("모델 없음")

        r = S.search(self.cfg, self.db, "신경망", embedder=Broken())
        self.assertEqual(r["mode_used"], "keyword")
        self.assertIn("의미 검색", r["note"])
        self.assertTrue(r["results"])

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"),
                         "transformers/torch 없음")
    def test_real_embedder_class_with_tiny_local_model(self):
        """실제 Embedder 코드 경로 (오프라인): 임의 초기화한 아주 작은 BERT를 로컬 폴더에 만들어 읽는다."""
        from transformers import BertConfig, BertModel, BertTokenizer

        d = Path(self.tmp) / "tiny-bert"
        d.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("신경망요리김치학습abcdefghijklmnopqrstuvwxyz:")
        (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        BertTokenizer(str(d / "vocab.txt")).save_pretrained(str(d))
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=32, max_position_embeddings=64)).save_pretrained(str(d))
        emb = S.Embedder(str(d), batch_size=2, max_length=32)
        vecs = emb.encode(["신경망 학습", "김치 요리", "abc"], is_query=False)
        self.assertEqual(len(vecs), 3)
        for v in vecs:
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=4)
        q = emb.encode(["신경망"], is_query=True)
        self.assertEqual(len(q[0]), 16)
        res = S.index_embeddings(self.cfg, self.db, embedder=emb)
        self.assertEqual(res["indexed"], self.db.counts()["chunks"])
        self.assertTrue(S.semantic_search(self.cfg, self.db, "신경망", embedder=emb))
        bad = S.Embedder(str(Path(self.tmp) / "missing-model"))
        with self.assertRaises(RuntimeError):
            bad.encode(["x"])


class ExtractAndViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg, self.db, self.ids = make_db(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extractive_sentences(self):
        rows = [{"doc_id": 1, "title": "가", "url": "u1",
                 "content": "첫 문장은 관계가 없다. 인공 신경망은 뇌를 흉내 낸 계산 모델이다! 신경망 학습에는 데이터가 필요하다.\n"
                            "인공 신경망은 뇌를 흉내 낸 계산 모델이다! 마지막 문장."},
                {"doc_id": 2, "title": "나", "url": "u2", "content": "Neural nets? 요리는 즐겁다. 신경망 이야기."}]
        out = S.extractive_answer('"인공 신경망"', rows)
        self.assertEqual(out[0]["sentence"], "인공 신경망은 뇌를 흉내 낸 계산 모델이다!")
        self.assertEqual(set(out[0]), {"sentence", "doc_id", "title", "url"})
        sents = [o["sentence"] for o in out]
        self.assertEqual(len(sents), len(set(sents)))  # 중복 없음
        out = S.extractive_answer("신경망 학습", rows, max_sentences=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["sentence"], "신경망 학습에는 데이터가 필요하다.")
        self.assertTrue(all("신경망" in o["sentence"] for o in out))
        self.assertEqual(S.extractive_answer("없는말", rows), [])
        self.assertEqual(S.extractive_answer("", rows), [])
        self.assertEqual(S.extractive_answer("신경망", []), [])

    def test_search_extract_uses_results(self):
        r = S.search(self.cfg, self.db, "합성곱")
        self.assertEqual(r["extract"][0]["sentence"], "합성곱 신경망은 이미지 인식에 쓰인다.")
        self.assertEqual(r["extract"][0]["doc_id"], self.ids[0])

    def test_doc_view_offsets(self):
        v = S.doc_view(self.db, self.ids[0], "신경망 딥")
        text = v["text"]
        self.assertEqual(set(v) >= {"id", "url", "title", "source", "query", "fetched_at", "chars", "text",
                                    "matches", "file_id"}, True)
        self.assertEqual(v["chars"], len(text))
        found = [text[s:e] for s, e in v["matches"]]
        self.assertEqual(found.count("신경망"), text.count("신경망"))
        self.assertEqual(found.count("딥"), text.count("딥"))
        self.assertEqual(v["matches"], sorted(v["matches"]))
        v = S.doc_view(self.db, self.ids[3], '"deep LEARNING" transformer')
        self.assertEqual([v["text"][s:e].lower() for s, e in v["matches"]], ["transformer", "deep learning"])
        v = S.doc_view(self.db, self.ids[0], "인공 신경망이란")  # 원래 형태가 없으면 조사를 뗀 단어로
        self.assertTrue(v["matches"])
        self.assertEqual(S.doc_view(self.db, self.ids[0])["matches"], [])
        self.assertIsNone(S.doc_view(self.db, 99999, "x"))
        self.assertIsNone(S.doc_view(self.db, "abc"))

    def test_doc_view_limits_and_utf16(self):
        self.db.add_document("https://e/1", "이모지", "😀 신경망 " + "가 " * 600, "link")
        doc = self.db.list_documents(1)[0]["id"]
        v = S.doc_view(self.db, doc, "신경망")
        self.assertEqual(v["matches"], [[2, 5]])
        self.assertEqual(v["matches_utf16"], [[3, 6]])  # JS 문자열 위치 (이모지는 2칸)
        v = S.doc_view(self.db, doc, "가")
        self.assertEqual(len(v["matches"]), S.MAX_MATCHES)

    def test_snippet_window(self):
        text = "앞부분 " * 100 + "핵심 신경망 문장 " + "뒷부분 " * 100
        sn = S.make_snippet(text, ["신경망"])
        self.assertIn(f"{HL_START}신경망{HL_END}", sn)
        self.assertTrue(sn.startswith("…") and sn.endswith("…"))
        self.assertLessEqual(len(sn), S.SNIPPET_CHARS + 6)
        self.assertEqual(S.make_snippet("짧은 글", ["없음"]), "짧은 글")


if __name__ == "__main__":
    unittest.main()
