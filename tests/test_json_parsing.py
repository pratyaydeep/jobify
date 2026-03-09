"""Tests for LLM JSON response parsing helpers."""

from jobify.llm import _parse_json_array, _parse_json_object


class TestParseJsonArray:
    def test_clean_array(self):
        assert _parse_json_array("[1, 2, 3]") == [1, 2, 3]

    def test_array_with_surrounding_text(self):
        text = "Here are the indices:\n[3, 7, 12]\nThose are the job links."
        assert _parse_json_array(text) == [3, 7, 12]

    def test_empty_array(self):
        assert _parse_json_array("[]") == []

    def test_no_brackets(self):
        assert _parse_json_array("no json here") == []

    def test_malformed_json(self):
        assert _parse_json_array("[1, 2, ") == []

    def test_nested_array(self):
        assert _parse_json_array("[[1, 2], [3]]") == [[1, 2], [3]]

    def test_whitespace_padding(self):
        assert _parse_json_array("  \n  [5, 10]  \n  ") == [5, 10]

    def test_string_array(self):
        assert _parse_json_array('["a", "b"]') == ["a", "b"]


class TestParseJsonObject:
    def test_clean_object(self):
        result = _parse_json_object('{"score": 8, "reasoning": "Good match"}')
        assert result == {"score": 8, "reasoning": "Good match"}

    def test_object_with_surrounding_text(self):
        text = 'Here is the result:\n{"title": "SDE", "company": "Acme"}\nDone.'
        result = _parse_json_object(text)
        assert result["title"] == "SDE"
        assert result["company"] == "Acme"

    def test_empty_object(self):
        assert _parse_json_object("{}") == {}

    def test_no_braces(self):
        assert _parse_json_object("no json here") == {}

    def test_malformed_json(self):
        assert _parse_json_object('{"key": ') == {}

    def test_nested_object(self):
        text = '{"job": {"title": "Engineer"}, "score": 7}'
        result = _parse_json_object(text)
        assert result["job"]["title"] == "Engineer"
        assert result["score"] == 7

    def test_null_values(self):
        text = '{"min_experience_years": null, "max_experience_years": null}'
        result = _parse_json_object(text)
        assert result["min_experience_years"] is None
        assert result["max_experience_years"] is None
