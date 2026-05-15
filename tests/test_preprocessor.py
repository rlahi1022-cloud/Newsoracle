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
    # 본문 크롤링 시 발생하는 다중 공백/탭/줄바꿈을 한 칸 공백으로 통일하지 않으면
    # 토크나이저가 "기준금리"와 "기준  금리"를 다른 토큰으로 처리하는 부작용이 생긴다.

    def test_collapses_spaces(self):
        # 다중 공백 → 단일 공백 (가장 흔한 케이스)
        self.assertEqual(normalize_whitespace("a   b   c"), "a b c")

    def test_strips_edges(self):
        # 좌우 공백 제거(strip) — 분류기 입력의 일관성 보장
        self.assertEqual(normalize_whitespace("   hello   "), "hello")


class TestCleanText(unittest.TestCase):
    # 위의 3단계(HTML 제거 / 엔티티 / 공백)가 올바르게 합성되는지 통합 검증.
    # 잘못된 입력(None, 비문자열)에도 빈 문자열을 반환해야 파이프라인이 죽지 않는다.

    def test_full_pipeline(self):
        # HTML + 엔티티 + 줄바꿈 — 실제 네이버 API description 의 전형적 형태
        raw = "<b>한국은행</b>은 &quot;기준금리&quot;를\n인상했다."
        expected = '한국은행은 "기준금리"를 인상했다.'
        self.assertEqual(clean_text(raw), expected)

    def test_empty_input(self):
        # 빈 입력은 빈 출력 (KeyError/AttributeError 가 아니라)
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_non_string_input(self):
        # int 같은 비문자열도 안전하게 처리 — 잘못 들어온 description 방어
        self.assertEqual(clean_text(123), "")


class TestExtractDomain(unittest.TestCase):
    # 도메인 추출 실수는 곧 도메인 등급 점수 오류로 이어진다(grade_1~5 매핑이 다 깨짐).
    # 그래서 다양한 URL 형태를 미리 잡아둔다.

    def test_basic_url(self):
        # 가장 기본적인 케이스
        self.assertEqual(
            extract_domain("https://yna.co.kr/news/123"), "yna.co.kr"
        )

    def test_subdomain_preserved(self):
        # m.chosun.com 같은 모바일 서브도메인은 유지되어야 함
        # (extract_source_name 단계에서 m. 을 떼어내 매핑)
        self.assertEqual(
            extract_domain("https://m.chosun.com/site/data"), "m.chosun.com"
        )

    def test_uppercase_normalized(self):
        # 대소문자 차이로 매핑이 빗나가지 않도록 소문자 정규화
        self.assertEqual(
            extract_domain("https://YNA.co.kr/news"), "yna.co.kr"
        )

    def test_empty_url(self):
        # originallink 가 비어 있는 기사도 있음 — 예외 대신 빈 문자열
        self.assertEqual(extract_domain(""), "")


class TestExtractSourceName(unittest.TestCase):
    # 네이버 API 응답에는 source 필드가 없으므로 도메인 → 언론사명 역매핑이 필수.
    # 매핑 실패가 누적되면 출처 다양성/도메인 등급 점수가 다 흔들린다.

    def test_known_domain_maps_to_korean(self):
        # 매핑 테이블에 등록된 주요 통신사/일간지/경제지가 한글명으로 변환되는지
        self.assertEqual(extract_source_name("yna.co.kr"), "연합뉴스")
        self.assertEqual(extract_source_name("chosun.com"), "조선일보")
        self.assertEqual(extract_source_name("hankyung.com"), "한국경제")

    def test_strips_www_prefix(self):
        # www. 접두사가 있어도 매핑이 동작해야 함
        self.assertEqual(extract_source_name("www.donga.com"), "동아일보")

    def test_unknown_domain_returns_itself(self):
        # 매핑에 없으면 도메인을 그대로 반환 — None 이나 예외가 아니라
        self.assertEqual(
            extract_source_name("unknown-news.example.com"),
            "unknown-news.example.com",
        )

    def test_empty_domain(self):
        # 빈 도메인은 빈 문자열 (downstream 에서 .lower() 등 호출 안전성)
        self.assertEqual(extract_source_name(""), "")


class TestPreprocessArticle(unittest.TestCase):
    # 단일 기사 전체 전처리 흐름. 위 함수들이 올바르게 조립되어
    # 최종 출력 딕셔너리에 모든 필드가 들어가는지 검증한다.

    def test_full_article_cleaned(self):
        # 운영 환경에서 들어오는 형태와 가까운 입력으로 통합 검증
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
        # 필드가 부분적으로만 있는 기사가 들어와도 KeyError 없이 처리되어야 함
        # (예: 일부 API 응답이 originallink 를 누락하는 경우)
        result = preprocess_article({"title": "제목만 있음"})
        self.assertEqual(result["title"], "제목만 있음")
        self.assertEqual(result["domain"], "")
        self.assertEqual(result["source"], "")


class TestPreprocessArticles(unittest.TestCase):
    # 배치 입력 처리. title 이 비어 있는 기사를 제외하지 않으면
    # 뒤따르는 feature_extractor 에서 임베딩 입력이 빈 문자열이 되어 점수가 망가진다.

    def test_filters_empty_titles(self):
        # title 빈 항목 1건이 빠지고 2건만 남는지
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
        # 수집 결과 0건일 때도 빈 리스트 반환 (예외 대신)
        self.assertEqual(preprocess_articles([]), [])


if __name__ == "__main__":
    unittest.main()
