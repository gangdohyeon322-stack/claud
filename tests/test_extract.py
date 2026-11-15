"""extract.html_to_text: 본문이 아닌 덩어리(위키백과 메뉴/언어 목록/분류/수식 원문)를 건너뛰는지."""
import unittest

from learnbox.extract import html_to_text

WIKI = """<!doctype html><html><head><title>인공 신경망 - 위키백과</title><style>.x{color:red}</style></head>
<body><a class="mw-jump-link" href="#c">본문으로 이동</a>
<div class="vector-header-container"><header class="vector-header"><div class="vector-search-box">검색</div>
<div id="p-lang-btn" class="vector-dropdown mw-portlet mw-portlet-lang"><ul><li class="interlanguage-link"><a href="https://en.wikipedia.org/wiki/ANN">English</a></li>
<li class="interlanguage-link"><a href="https://de.wikipedia.org/wiki/ANN">Deutsch</a></li></ul></div></header></div>
<main><h1>인공 신경망</h1>
<div id="bodyContent">
<table class="sidebar nomobile"><tr><th>기계 학습과 데이터 마이닝</th></tr><tr><td>지도 학습 비지도 학습</td></tr></table>
<p>인공 신경망은 뇌의 뉴런을 흉내 낸 모델이다.<sup class="reference"><a href="#r1">[1]</a></sup> 토큰은 <span class="mwe-math-element"><math><semantics><mrow><mi>n</mi></mrow><annotation encoding="application/x-tex">{\\displaystyle n}</annotation></semantics></math></span>개이다.</p>
<h2>배경<span class="mw-editsection"><span>[</span><a href="/edit">편집</a><span>]</span></span></h2>
<div class="wrapper"><div class="inner">중첩된 div 안의 본문 문장입니다.</div> 닫힌 뒤에 이어지는 본문도 남아야 합니다.</div>
<div class="navbox"><div class="navbox-inner"><div>틀 안쪽 목록 하나</div></div> 틀 안쪽 목록 둘</div>
<p>navbox 뒤의 마지막 본문 문장입니다. 이 문장은 남아야 합니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. 학습 자료를 충분히 길게 만들기 위한 덧붙임 문장입니다. </p>
<div id="catlinks" class="catlinks"><div>분류: 신경망</div><div>숨은 분류: 위키데이터 속성</div></div>
</div></main><footer>사이트 정보 푸터</footer></body></html>"""


class ExtractBoilerplateTest(unittest.TestCase):
    def setUp(self):
        self.title, self.text, self.links = html_to_text(WIKI, "https://ko.wikipedia.org/wiki/x")

    def test_title_and_body_kept(self):
        self.assertEqual(self.title, "인공 신경망 - 위키백과")
        self.assertIn("인공 신경망은 뇌의 뉴런을 흉내 낸 모델이다.", self.text)
        self.assertIn("중첩된 div 안의 본문 문장입니다.", self.text)
        self.assertIn("닫힌 뒤에 이어지는 본문도 남아야 합니다.", self.text)
        self.assertIn("navbox 뒤의 마지막 본문 문장입니다.", self.text)
        self.assertIn("인공 신경망", self.text.split("\n")[0])  # <h1> 제목은 남는다

    def test_chrome_removed(self):
        for junk in ("본문으로 이동", "검색", "English", "Deutsch", "기계 학습과 데이터 마이닝", "지도 학습 비지도 학습",
                     "편집", "[1]", "displaystyle", "틀 안쪽", "분류:", "숨은 분류", "사이트 정보"):
            self.assertNotIn(junk, self.text, junk)

    def test_math_text_kept_without_latex(self):
        self.assertIn("n개이다", self.text.replace(" ", ""))

    def test_links_still_collected(self):
        # 링크 수집(깊이 크롤링)은 그대로: 덩어리 안의 링크도 예전처럼 모은다
        self.assertTrue(any("en.wikipedia.org" in u for u in self.links))


class ExtractSafetyTest(unittest.TestCase):
    def test_generic_article_header_kept(self):
        html = "<html><body><article><header><h1>글 제목</h1><p>작성자 홍길동</p></header><p>" + "본문 " * 80 + "</p></article></body></html>"
        _, text, _ = html_to_text(html)
        self.assertIn("글 제목", text)
        self.assertIn("작성자 홍길동", text)

    def test_unclosed_boilerplate_falls_back(self):
        # 닫히지 않은 navbox가 본문 전체를 삼키면 예전 방식(건너뛰지 않음)으로 되돌아간다
        html = '<html><body><div class="navbox"><p>' + "짧지 않은 본문 문장입니다. " * 20 + "</p></body></html>"
        _, text, _ = html_to_text(html)
        self.assertIn("짧지 않은 본문 문장입니다.", text)

    def test_plain_page_unchanged(self):
        html = "<html><head><title>T</title></head><body><p>첫 문단입니다.</p><p>둘째 문단입니다.</p><a href='/a'>링크</a></body></html>"
        t, text, links = html_to_text(html, "https://example.org/x")
        self.assertEqual(t, "T")
        self.assertEqual(text, "첫 문단입니다.\n둘째 문단입니다.\n링크")
        self.assertEqual(links, ["https://example.org/a"])

    def test_role_navigation_skipped(self):
        html = '<html><body><div role="navigation">메뉴 항목</div><p>' + "본문 문장입니다. " * 30 + "</p></body></html>"
        _, text, _ = html_to_text(html)
        self.assertNotIn("메뉴 항목", text)


if __name__ == "__main__":
    unittest.main()
