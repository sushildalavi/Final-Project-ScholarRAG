import unittest
from types import SimpleNamespace
from unittest.mock import patch

import backend.app as app_module


class _DummyCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


def _dummy_client(content: str):
    return SimpleNamespace(chat=SimpleNamespace(completions=_DummyCompletions(content)))


class AssistantAnswerRoutingTests(unittest.TestCase):
    def test_public_scope_short_ambiguous_query_uses_retrieval_not_chat_bypass(self):
        uploaded_results = {
            "results": [
                {
                    "id": 101,
                    "document_id": 2,
                    "title": "02_ColBERT.pdf",
                    "doc_type": "research_paper",
                    "text": (
                        "ColBERT is a late interaction retrieval model over BERT for "
                        "passage ranking in information retrieval."
                    ),
                    "page_no": 1,
                    "distance": 0.08,
                }
            ]
        }
        public_results = {
            "results": [
                {
                    "title": "Stephen Colbert and political satire",
                    "source": "semanticscholar",
                    "abstract": "Late-night television comedy and satire.",
                    "_sim": 0.92,
                    "url": "https://example.com/colbert-tv",
                }
            ],
            "provider_status": {},
        }

        with (
            patch.object(app_module, "_chat_answer", side_effect=AssertionError("chat bypass should not run")),
            patch.object(app_module, "search_uploaded_chunks", return_value=uploaded_results),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(app_module, "_rank_and_trim_citations", side_effect=lambda query, citations, k, **kwargs: citations[:k]),
            patch.object(app_module, "_build_generation_prompt", return_value="prompt"),
            patch.object(app_module, "_compute_citation_msa", return_value=({}, 0)),
            patch.object(app_module, "_has_official_company_docs", return_value=True),
            patch.object(app_module, "client", _dummy_client("ColBERT is a retrieval model [S1].")),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer(
                {"query": "tell me about Colbert", "scope": "public", "k": 4}
            )

        self.assertTrue(resp["citations"])
        self.assertEqual(resp["citations"][0]["source"], "uploaded")
        self.assertIn("colbert", resp["citations"][0]["title"].lower())
        self.assertNotIn("stephen colbert", " ".join(c["title"] for c in resp["citations"]).lower())

    def test_public_scope_offtopic_ambiguous_query_abstains_after_filtering(self):
        public_results = {
            "results": [
                {
                    "title": "Stephen Colbert and the Late Show",
                    "source": "semanticscholar",
                    "abstract": "Political satire and television studies.",
                    "_sim": 0.94,
                    "url": "https://example.com/late-show",
                }
            ],
            "provider_status": {},
        }

        with (
            patch.object(app_module, "_chat_answer", side_effect=AssertionError("chat bypass should not run")),
            patch.object(app_module, "search_uploaded_chunks", return_value={"results": []}),
            patch.object(app_module, "public_live_search", return_value=public_results),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer(
                {"query": "tell me about Colbert", "scope": "public", "k": 4}
            )

        self.assertEqual(resp["citations"], [])
        self.assertEqual(resp["retrieval_policy"]["mode"], "abstention")
        self.assertIn("Abstained", resp["confidence"]["label"])

    def test_public_scope_plain_small_talk_still_uses_chat_bypass(self):
        with (
            patch.object(app_module, "_chat_answer", return_value="hello there"),
            patch.object(app_module, "log_json", return_value=None),
        ):
            resp = app_module.assistant_answer({"query": "hello", "scope": "public"})

        self.assertEqual(resp["answer"], "hello there")
        self.assertEqual(resp["citations"], [])


if __name__ == "__main__":
    unittest.main()
