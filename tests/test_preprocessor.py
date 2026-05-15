"""
tests/test_preprocessor.py
─────────────────────────────────────
services/preprocessor.py 의 텍스트 정제 / 도메인 파싱 / 언론사명 매핑
함수에 대한 유닛 테스트.

[왜 이 모듈을 따로 테스트하는가]
  preprocessor 는 파이프라인의 가장 앞 단계라서 여기서 오염된 데이터가 흘러가면
  뒤따르는 모든 모듈(feature_extractor, classifier, rule_based_scorer ...)의
  점수가 동시에 무너진다. 단일 함수 단위로 회귀를 잡아두는 것이 가장 비용이 싸다.

[테스트 전략]
  1. 각 정제 단계(HTML 제거 / 엔티티 변환 / 공백 정규화)를 독립적으로 검증
  2. 정제 파이프라인 통합(clean_text) — 단계 합성 후 결과 검증
  3. 도메인 → 언론사 매핑(extract_source_name) — 등록/미등록 분기 모두
  4. 기사 단위/배치 단위 entry point(preprocess_article/articles)는 필드 누락,
     빈 입력 같은 실제 운영에서 발생할 수 있는 가장자리 케이스 위주
"""

import os
import sys
import unittest

# 프로젝트 루트를 import 경로에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.preprocessor import (
    clean_text,
    clean_special_characters,
    remove_html_tags,
    normalize_whitespace,
    extract_domain,
    extract_source_name,
    preprocess_article,
    preprocess_articles,
)


class TestRemoveHtmlTags(unittest.TestCase):
    # 네이버 뉴스 API 응답의 title 에는 검색어 강조를 위한 <b> 태그가 끼어든다.
    # 이 태그가 제거되지 않으면 분류기 입력에 잡음으로 들어가 정확도가 떨어진다.

    def test_strips_bold_tag(self):
        # 가장 흔한 케이스(<b>): 네이버 API 응답에 항상 등장
        self.assertEqual(remove_html_tags("<b>금리</b> 인상"), "금리 인상")

    def test_strips_nested_tags(self):
        # 일부 본문 크롤링 결과에 중첩 태그가 포함될 수 있음
        self.assertEqual(
            remove_html_tags("<p>한국<span>은행</span></p>"), "한국은행"
        )

    def test_no_tag_unchanged(self):
        # 태그가 없는 텍스트는 손대지 않아야 함(정상 텍스트 손상 방지)
        self.assertEqual(remove_html_tags("순수 텍스트"), "순수 텍스트")


class TestCleanSpecialCharacters(unittest.TestCase):
    # HTML 엔티티(&amp; 등)가 그대로 남으면 사람·모델 모두 가독성이 떨어진다.
    # 동시에 \r\n\t 도 한 칸 공백으로 통일해서 줄바꿈으로 인한 단어 분리를 막는다.

    def test_html_entities(self):
        # 네이버 API description 에서 자주 등장하는 3종 엔티티 (&amp; &lt; &quot;)
        self.assertEqual(
            clean_special_characters("Tom &amp; Jerry"), "Tom & Jerry"
        )
        self.assertEqual(clean_special_characters("&lt;tag&gt;"), "<tag>")
        self.assertEqual(clean_special_characters("&quot;공식&quot;"), '"공식"')

    def test_newlines_become_space(self):
        # 본문 크롤링 결과에는 \n \t \r\n 이 섞이는 경우가 흔함
        self.assertEqual(
            clean_special_characters("a\nb\tc\r\nd"), "a b c d"
        )


class TestNormalizeWhitespace(unittest.TestCase):
    def test_collapses_spaces(self):
        self.assertEqual(normalize_whitespace("a   b   c"), "a b c")

    def test_strips_edges(self):
        self.assertEqual(normalize_whitespace("   hello   "), "hello")


class TestCleanText(unittest.TestCase):
    def test_full_pipeline(self):
        raw = "<b>한국은행</b>은 &quot;기준금리&quot;를\n인상했다."
        expected = '한국은행은 "기준금리"를 인상했다.'
        self.assertEqual(clean_text(raw), expected)

    def test_empty_input(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_non_string_input(self):
        self.assertEqual(clean_text(123), "")


class TestExtractDomain(unittest.TestCase):
    def test_basic_url(self):
        self.assertEqual(
            extract_domain("https://yna.co.kr/news/123"), "yna.co.kr"
        )

    def test_subdomain_preserved(self):
        self.assertEqual(
            extract_domain("https://m.chosun.com/site/data"), "m.chosun.com"
        )

    def test_uppercase_normalized(self):
        self.assertEqual(
            extract_domain("https://YNA.co.kr/news"), "yna.co.kr"
        )

    def test_empty_url(self):
        self.assertEqual(extract_domain(""), "")


class TestExtractSourceName(unittest.TestCase):
    def test_known_domain_maps_to_korean(self):
        self.assertEqual(extract_source_name("yna.co.kr"), "연합뉴스")
        self.assertEqual(extract_source_name("chosun.com"), "조선일보")
        self.assertEqual(extract_source_name("hankyung.com"), "한국경제")

    def test_strips_www_prefix(self):
        self.assertEqual(extract_source_name("www.donga.com"), "동아일보")

    def test_unknown_domain_returns_itself(self):
        self.assertEqual(
            extract_source_name("unknown-news.example.com"),
            "unknown-news.example.com",
        )

    def test_empty_domain(self):
        self.assertEqual(extract_source_name(""), "")


class TestPreprocessArticle(unittest.TestCase):
    def test_full_article_cleaned(self):
        article = {
            "title": "<b>한국은행</b>, 기준금리 동결",
            "description": "한국은행은 &quot;금리 동결&quot;을\n발표했다.",
            "originallink": "https://yna.co.kr/news/123",
            "link": "https://news.naver.com/abc",
            "pubDate": "Mon, 01 Apr 2026 09:00:00 +0900",
        }
        result = preprocess_article(article)
        self.assertEqual(result["title"], "한국은행, 기준금리 동결")
        self.assertEqual(result["content"], '한국은행은 "금리 동결"을 발표했다.')
        self.assertEqual(result["domain"], "yna.co.kr")
        self.assertEqual(result["source"], "연합뉴스")

    def test_missing_fields_handled(self):
        result = preprocess_article({"title": "제목만 있음"})
        self.assertEqual(result["title"], "제목만 있음")
        self.assertEqual(result["domain"], "")
        self.assertEqual(result["source"], "")


class TestPreprocessArticles(unittest.TestCase):
    def test_filters_empty_titles(self):
        articles = [
            {"title": "유효 기사", "description": "내용", "originallink": "https://yna.co.kr/a"},
            {"title": "", "description": "제목 없음", "originallink": "https://chosun.com/b"},
            {"title": "또 다른 기사", "description": "내용2", "originallink": "https://donga.com/c"},
        ]
        results = preprocess_articles(articles)
        self.assertEqual(len(results), 2)
        titles = [r["title"] for r in results]
        self.assertIn("유효 기사", titles)
        self.assertIn("또 다른 기사", titles)

    def test_empty_input(self):
        self.assertEqual(preprocess_articles([]), [])


if __name__ == "__main__":
    unittest.main()
