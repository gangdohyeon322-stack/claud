"""ingest.py 테스트 (네트워크 없음).

실행: cd /home/user/claud && .venv/bin/python -m unittest tests.test_ingest -v
"""
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from learnbox import ingest as I
from learnbox import search as S
from learnbox.config import Config
from learnbox.db import DB

KO = "인공 신경망은 사람의 뇌를 흉내 낸 계산 모델이다. 여러 층을 쌓으면 딥 러닝이 된다."

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(paragraphs_xml: str, title: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/'
                   '2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>')
        z.writestr("word/document.xml",
                   f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W_NS}">'
                   f"<w:body>{paragraphs_xml}</w:body></w:document>")
        if title:
            z.writestr("docProps/core.xml",
                       '<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties xmlns:cp="http://schemas.'
                       'openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/'
                       f'elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>')
    return buf.getvalue()


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    w = PdfWriter()
    font = w._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                           NameObject("/Subtype"): NameObject("/Type1"),
                                           NameObject("/BaseFont"): NameObject("/Helvetica")}))
    for text in pages:
        page = w.add_blank_page(612, 792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        s = DecodedStreamObject()
        s.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = w._add_object(s)
    if title:
        w.add_metadata({"/Title": title})
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


HAS_PYPDF = importlib.util.find_spec("pypdf") is not None


class ExtractFileTest(unittest.TestCase):
    def one(self, name, data):
        docs = I.extract_file(name, data)
        self.assertEqual(len(docs), 1)
        self.assertEqual(set(docs[0]), {"title", "text", "part"})
        return docs[0]

    def test_txt_encodings(self):
        self.assertEqual(self.one("메모.txt", KO.encode("utf-8"))["text"], KO)
        self.assertEqual(self.one("bom.txt", b"\xef\xbb\xbf" + KO.encode("utf-8"))["text"], KO)
        d = self.one("윈도우.txt", KO.encode("cp949"))
        self.assertEqual(d["text"], KO)
        self.assertEqual(d["title"], "윈도우")
        self.assertEqual(self.one("u16.txt", KO.encode("utf-16"))["text"], KO)
        # 깨진 바이트도 예외 없이
        self.assertIn("�", self.one("bad.log", b"ok \xff\xfe\xfa\x80 bytes")["text"])

    def test_nfd_normalized_and_control_chars_removed(self):
        import unicodedata

        nfd = unicodedata.normalize("NFD", "신경망 \x02강조\x03 문장\x00 입니다")
        d = self.one("mac.txt", nfd.encode("utf-8"))
        self.assertEqual(d["text"], "신경망 강조 문장 입니다")

    def test_markdown_title(self):
        d = self.one("readme.md", "# 신경망 정리\n\n본문입니다.\n".encode())
        self.assertEqual(d["title"], "신경망 정리")
        self.assertIn("본문입니다.", d["text"])
        self.assertEqual(self.one("x.markdown", b"no heading")["title"], "x")

    def test_srt(self):
        srt = "1\n00:00:01,000 --> 00:00:02,500\n<i>안녕하세요</i>\n\n2\n00:00:03,000 --> 00:00:04,000\n신경망 강의\n"
        d = self.one("lecture.srt", srt.encode())
        self.assertEqual(d["text"], "안녕하세요\n신경망 강의")

    def test_csv_and_tsv(self):
        csv_data = "이름,설명,비고\n딥러닝,\"신경망, 여러 층\",\n김치,발효 음식,반찬\n".encode("cp949")
        d = self.one("표.csv", csv_data)
        self.assertIn("이름: 딥러닝\n설명: 신경망, 여러 층", d["text"])
        self.assertIn("이름: 김치\n설명: 발효 음식\n비고: 반찬", d["text"])
        self.assertNotIn("비고: \n", d["text"])  # 빈 값은 생략
        d = self.one("t.tsv", "word\tmeaning\n신경망\tneural network\n".encode())
        self.assertEqual(d["text"], "word: 신경망\nmeaning: neural network")
        d = self.one("semi.csv", "a;b\n1;2\n".encode())
        self.assertEqual(d["text"], "a: 1\nb: 2")

    def test_jsonl_multi_doc(self):
        lines = [json.dumps({"title": "첫째", "text": "첫 번째 레코드 본문입니다."}, ensure_ascii=False),
                 "{broken json",
                 json.dumps({"content": "두 번째 레코드 본문입니다."}, ensure_ascii=False),
                 json.dumps({"question": "신경망이란?", "answer": "뇌를 흉내 낸 모델"}, ensure_ascii=False),
                 json.dumps({"id": 7, "meta": {"k": "v"}})]
        docs = I.extract_file("data.jsonl", "\n".join(lines).encode())
        self.assertEqual([d["part"] for d in docs], ["1", "3", "4", "rest"])
        self.assertEqual(docs[0]["title"], "첫째")
        self.assertEqual(docs[1]["title"], "data.jsonl #3")
        self.assertEqual(docs[2]["text"], "질문: 신경망이란?\n답변: 뇌를 흉내 낸 모델")
        self.assertIn("meta.k: v", docs[3]["text"])
        self.assertEqual(len(I.extract_file("data.jsonl", "\n".join(lines).encode(), max_docs=2)), 2)

    def test_json_variants(self):
        arr = [{"title": f"글 {i}", "body": f"본문 {i} 내용입니다."} for i in range(3)]
        docs = I.extract_file("a.json", json.dumps(arr, ensure_ascii=False).encode())
        self.assertEqual([d["title"] for d in docs], ["글 0", "글 1", "글 2"])
        self.assertEqual([d["part"] for d in docs], ["1", "2", "3"])
        d = self.one("one.json", json.dumps({"title": "단일", "article": "기사 본문"}, ensure_ascii=False).encode())
        self.assertEqual((d["title"], d["text"], d["part"]), ("단일", "기사 본문", None))
        docs = I.extract_file("wrap.json", json.dumps({"data": arr}, ensure_ascii=False).encode())
        self.assertEqual(len(docs), 3)
        d = self.one("cfg.json", json.dumps({"설정": {"이름": "신경망", "값": [1, 2]}}, ensure_ascii=False).encode())
        self.assertIn("설정.이름: 신경망", d["text"])
        d = self.one("broken.json", b'{"text": "unterminated')
        self.assertIn("unterminated", d["text"])

    def test_html_and_xml(self):
        html = ("<html><head><meta charset='euc-kr'><title>신경망 페이지</title><script>var x=1;</script></head>"
                "<body><nav>메뉴</nav><p>인공 신경망 설명 문단</p></body></html>").encode("cp949")
        d = self.one("page.html", html)
        self.assertEqual(d["title"], "신경망 페이지")
        self.assertIn("인공 신경망 설명 문단", d["text"])
        self.assertNotIn("var x", d["text"])
        d = self.one("p.htm", "<p>제목 없는 문서의 본문</p>".encode("cp949"))  # meta 없음 -> cp949 추정
        self.assertEqual((d["title"], d["text"]), ("p", "제목 없는 문서의 본문"))
        xml = '<?xml version="1.0"?><rss><channel><title>피드 제목</title><item><description>항목 내용' \
              '</description></item></channel></rss>'
        d = self.one("feed.xml", xml.encode())
        self.assertEqual(d["title"], "피드 제목")
        self.assertIn("항목 내용", d["text"])
        d = self.one("broken.xml", "<a><b>깨진 XML 본문".encode())
        self.assertIn("깨진 XML 본문", d["text"])

    def test_docx(self):
        body = (f'<w:p><w:r><w:t>첫 문단 신경망</w:t></w:r></w:p>'
                f'<w:p><w:r><w:t>둘째</w:t><w:tab/><w:t>탭 뒤</w:t><w:br/><w:t xml:space="preserve">줄바꿈 뒤</w:t>'
                f'</w:r><w:del><w:r><w:delText>지운 글</w:delText></w:r></w:del></w:p>'
                f'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>표 칸1</w:t></w:r></w:p></w:tc>'
                f'<w:tc><w:p><w:r><w:t>표 칸2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
        d = self.one("보고서.docx", make_docx(body, title="문서 제목"))
        self.assertEqual(d["title"], "문서 제목")
        self.assertEqual(d["text"], "첫 문단 신경망\n둘째\t탭 뒤\n줄바꿈 뒤\n표 칸1\n표 칸2")
        self.assertEqual(self.one("무제.docx", make_docx(body))["title"], "무제")
        with self.assertRaises(ValueError):
            I.extract_file("bad.docx", b"not a zip")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("other.xml", "<x/>")
        with self.assertRaises(ValueError):
            I.extract_file("nobody.docx", buf.getvalue())

    @unittest.skipUnless(HAS_PYPDF, "pypdf 없음")
    def test_pdf(self):
        d = self.one("paper.pdf", make_pdf(["Neural networks learn representations",
                                            "Second page about transformers"], title="Test PDF"))
        self.assertEqual(d["title"], "Test PDF")
        self.assertEqual(d["text"], "Neural networks learn representations\n\nSecond page about transformers")
        self.assertEqual(self.one("untitled.pdf", make_pdf(["Hello world text"]))["title"], "untitled")
        with self.assertRaises(ValueError) as cm:
            I.extract_file("broken.pdf", b"%PDF-1.4 garbage")
        self.assertIn("PDF", str(cm.exception))

    def test_pdf_without_pypdf(self):
        with mock.patch.dict(sys.modules, {"pypdf": None}):
            with self.assertRaises(ValueError) as cm:
                I.extract_file("x.pdf", b"%PDF-1.4")
        self.assertEqual(str(cm.exception), "PDF를 읽으려면 pypdf가 필요합니다: pip install pypdf")

    def test_unsupported(self):
        with self.assertRaises(ValueError) as cm:
            I.extract_file("virus.exe", b"MZ")
        self.assertIn("지원하지 않는 파일 형식", str(cm.exception))
        with self.assertRaises(ValueError):
            I.extract_file("noext", b"abc")
        self.assertEqual(I.SUPPORTED, {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html",
                                       ".htm", ".xml", ".pdf", ".docx", ".srt", ".log"})


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = Config(Path(self.tmp) / "data")
        self.db = DB(self.cfg.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ingest_bytes_ok_and_searchable(self):
        r = I.ingest_bytes(self.cfg, self.db, "../../신경망 노트.txt", ((KO + "\n") * 3).encode(), actor="tester")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["name"], "신경망 노트.txt")  # 경로 부분은 버린다
        self.assertEqual((r["docs"], r["skipped"]), (1, 0))
        self.assertEqual(r["chars"], len(((KO + "\n") * 3).strip()))
        f = self.db.list_files()[0]
        self.assertEqual((f["id"], f["docs"], f["actor"], f["mime"]), (r["file_id"], 1, "tester", "text/plain"))
        path = Path(f["path"])
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, self.cfg.data_dir / "files")
        self.assertTrue(path.name.startswith(f"{r['file_id']}_"))
        self.assertEqual(path.read_bytes(), ((KO + "\n") * 3).encode())
        doc = self.db.list_documents(5)[0]
        self.assertEqual((doc["source"], doc["file_id"], doc["title"]), ("file", r["file_id"], "신경망 노트"))
        self.assertEqual(doc["url"], f"file://{r['file_id']}/%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%EB%85%B8%ED%8A%B8.txt")
        self.assertEqual(self.db.list_audit(1)[0]["action"], "ingest")
        res = S.search(self.cfg, self.db, "신경망", sources=["file"])
        self.assertEqual([x["doc_id"] for x in res["results"]], [doc["id"]])

    def test_duplicate(self):
        data = (KO * 2).encode()
        a = I.ingest_bytes(self.cfg, self.db, "a.txt", data)
        b = I.ingest_bytes(self.cfg, self.db, "b.md", data)
        self.assertEqual(b["status"], "duplicate")
        self.assertEqual((b["file_id"], b["docs"]), (a["file_id"], 1))
        self.assertEqual(self.db.counts()["files"], 1)
        self.assertEqual(self.db.counts()["documents"], 1)

    def test_size_limit_empty_and_unsupported(self):
        self.cfg.data["ingest"]["max_file_mb"] = 0.001  # 약 1KB
        with self.assertRaises(ValueError) as cm:
            I.ingest_bytes(self.cfg, self.db, "big.txt", b"x" * 5000)
        self.assertIn("너무 큽니다", str(cm.exception))
        self.cfg.data["ingest"]["max_file_mb"] = 50
        with self.assertRaises(ValueError):
            I.ingest_bytes(self.cfg, self.db, "empty.txt", b"  \n ")
        with self.assertRaises(ValueError):
            I.ingest_bytes(self.cfg, self.db, "a.exe", b"MZ" * 50)
        with self.assertRaises(ValueError):  # 형식 오류면 파일 기록도 남기지 않는다
            I.ingest_bytes(self.cfg, self.db, "bad.docx", b"not a zip at all")
        self.assertEqual(self.db.counts()["files"], 0)

    def test_short_docs_skipped(self):
        r = I.ingest_bytes(self.cfg, self.db, "short.txt", "짧은 글".encode())
        self.assertEqual((r["status"], r["docs"], r["skipped"]), ("ok", 0, 1))
        self.assertIn("message", r)
        self.assertEqual(self.db.counts()["documents"], 0)

    def test_multi_doc_urls(self):
        lines = [json.dumps({"title": f"레코드 {i}", "text": f"{i}번째 레코드의 충분히 긴 본문 내용입니다."},
                            ensure_ascii=False) for i in range(3)]
        lines.append(json.dumps({"text": "짧음"}, ensure_ascii=False))
        r = I.ingest_bytes(self.cfg, self.db, "recs.jsonl", "\n".join(lines).encode())
        self.assertEqual((r["docs"], r["skipped"]), (3, 1))
        urls = sorted(d["url"] for d in self.db.list_documents(10))
        self.assertEqual(urls, [f"file://{r['file_id']}/recs.jsonl#{i}" for i in (1, 2, 3)])
        self.assertEqual(self.db.list_files()[0]["docs"], 3)

    def test_max_docs_per_file(self):
        self.cfg.data["ingest"]["max_docs_per_file"] = 2
        lines = [json.dumps({"text": f"{i}번째 레코드의 충분히 긴 본문 내용입니다."}, ensure_ascii=False) for i in range(5)]
        r = I.ingest_bytes(self.cfg, self.db, "many.jsonl", "\n".join(lines).encode())
        self.assertEqual(r["docs"], 2)
        self.assertTrue(r["truncated"])

    @unittest.skipUnless(HAS_PYPDF, "pypdf 없음")
    def test_ingest_pdf_and_docx(self):
        r = I.ingest_bytes(self.cfg, self.db, "paper.pdf", make_pdf(["Neural networks learn representations"]))
        self.assertEqual((r["status"], r["docs"]), ("ok", 1))
        self.assertEqual(self.db.list_files()[0]["mime"], "application/pdf")
        r = I.ingest_bytes(self.cfg, self.db, "보고서.docx",
                           make_docx("<w:p><w:r><w:t>문서 파일 안의 신경망 설명 문단입니다.</w:t></w:r></w:p>"))
        self.assertEqual(r["docs"], 1)
        self.assertEqual(S.search(self.cfg, self.db, "representations")["total"], 1)
        self.assertEqual(S.search(self.cfg, self.db, "신경망 설명")["total"], 1)

    def test_ingest_path_recursive(self):
        root = Path(self.tmp) / "import"
        (root / "sub" / "deeper").mkdir(parents=True)
        (root / ".hidden").mkdir()
        (root / "a.txt").write_text(KO, encoding="utf-8")
        (root / "sub" / "b.md").write_text("# 제목\n" + KO + " 마크다운", encoding="utf-8")
        (root / "sub" / "deeper" / "c.csv").write_text("열1,열2\n신경망 자료,값이 들어 있는 행\n", encoding="utf-8")
        (root / "sub" / "copy.txt").write_text(KO, encoding="utf-8")         # a.txt 와 같은 내용 -> 중복
        (root / ".hidden" / "secret.txt").write_text(KO + " 비밀", encoding="utf-8")  # 숨김 폴더 -> 건너뜀
        (root / ".dot.txt").write_text(KO + " 점", encoding="utf-8")          # 숨김 파일 -> 건너뜀
        (root / "program.exe").write_bytes(b"MZ" * 100)                      # 지원 안 함 -> 건너뜀
        (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really")             # 오류
        (root / "empty.txt").write_bytes(b"")                                # 오류 (빈 파일)
        calls = []
        r = I.ingest_path(self.cfg, self.db, str(root), actor="cli", progress=lambda p, m: calls.append((p, m)))
        self.assertEqual(r["files"], 3)
        self.assertEqual(r["docs"], 3)
        self.assertEqual(r["duplicates"], 1)
        self.assertEqual(r["total"], 6)
        self.assertEqual(sorted(Path(e["path"]).name for e in r["errors"]), ["broken.pdf", "empty.txt"])
        self.assertTrue(all(e["error"] for e in r["errors"]))
        self.assertEqual(calls[-1][0], 1.0)
        names = {f["name"] for f in self.db.list_files()}
        self.assertEqual(names, {"a.txt", "b.md", "c.csv"})
        # 단일 파일 경로
        single = Path(self.tmp) / "single.log"
        single.write_text("로그 파일 한 줄 이상의 충분히 긴 내용입니다.", encoding="utf-8")
        r = I.ingest_path(self.cfg, self.db, str(single))
        self.assertEqual((r["files"], r["docs"]), (1, 1))
        with self.assertRaises(ValueError):
            I.ingest_path(self.cfg, self.db, str(Path(self.tmp) / "nope"))

    def test_ingest_path_cancel_and_size(self):
        root = Path(self.tmp) / "imp2"
        root.mkdir()
        for i in range(3):
            (root / f"{i}.txt").write_text(f"{i} " + KO, encoding="utf-8")

        class Stop(Exception):
            pass

        seen = []

        def cancel():
            if len(seen) >= 1:
                raise Stop()
            seen.append(1)

        with self.assertRaises(Stop):
            I.ingest_path(self.cfg, self.db, str(root), check_cancel=cancel)
        self.assertEqual(self.db.counts()["files"], 1)
        self.cfg.data["ingest"]["max_file_mb"] = 0.00001
        r = I.ingest_path(self.cfg, self.db, str(root))
        self.assertEqual(len(r["errors"]), 3)
        self.assertIn("너무 큽니다", r["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
