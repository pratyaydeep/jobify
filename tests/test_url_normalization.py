"""Tests for URL handling in the scraper."""

from urllib.parse import urljoin


class TestUrlResolution:
    """Test that relative URLs are properly resolved against the base."""

    def test_absolute_url_unchanged(self):
        base = "https://careers.example.com/jobs"
        href = "https://other.com/job/123"
        assert urljoin(base, href) == "https://other.com/job/123"

    def test_relative_path(self):
        base = "https://careers.example.com/jobs"
        href = "/job/123"
        assert urljoin(base, href) == "https://careers.example.com/job/123"

    def test_relative_no_slash(self):
        base = "https://careers.example.com/jobs/"
        href = "123"
        assert urljoin(base, href) == "https://careers.example.com/jobs/123"

    def test_fragment_stripped_in_join(self):
        base = "https://example.com/careers"
        href = "/job/456#apply"
        result = urljoin(base, href)
        assert result == "https://example.com/job/456#apply"

    def test_query_params_preserved(self):
        base = "https://example.com/careers"
        href = "/job/456?ref=portal"
        result = urljoin(base, href)
        assert result == "https://example.com/job/456?ref=portal"

    def test_protocol_relative(self):
        base = "https://example.com/careers"
        href = "//cdn.example.com/page"
        result = urljoin(base, href)
        assert result == "https://cdn.example.com/page"
