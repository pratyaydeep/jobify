"""Tests for experience range filtering logic in the pipeline."""

from jobify.models import Job


def _make_job(
    min_exp: float | None = None,
    max_exp: float | None = None,
    title: str = "Engineer",
) -> Job:
    return Job(
        title=title,
        company="TestCo",
        location="Remote",
        url="https://example.com/job",
        description="A test job",
        source_portal="TestPortal",
        min_experience_years=min_exp,
        max_experience_years=max_exp,
    )


def is_eligible(job: Job, user_exp: float) -> bool:
    """Replicate the eligibility logic from matcher.py."""
    if job.min_experience_years is not None and user_exp < job.min_experience_years:
        return False
    if job.max_experience_years is not None and user_exp > job.max_experience_years:
        return False
    return True


class TestExperienceFilter:
    def test_no_requirements(self):
        """Jobs with no experience requirements are always eligible."""
        job = _make_job()
        assert is_eligible(job, 0)
        assert is_eligible(job, 10)

    def test_min_only_eligible(self):
        """'3+ years' — user with 5 years is eligible."""
        job = _make_job(min_exp=3)
        assert is_eligible(job, 5)

    def test_min_only_not_eligible(self):
        """'5+ years' — user with 3 years is not eligible."""
        job = _make_job(min_exp=5)
        assert not is_eligible(job, 3)

    def test_min_only_exact(self):
        """Exactly at the minimum is eligible."""
        job = _make_job(min_exp=3)
        assert is_eligible(job, 3)

    def test_range_within(self):
        """'3-7 years' — user with 5 years is eligible."""
        job = _make_job(min_exp=3, max_exp=7)
        assert is_eligible(job, 5)

    def test_range_below(self):
        """'3-7 years' — user with 2 years is not eligible."""
        job = _make_job(min_exp=3, max_exp=7)
        assert not is_eligible(job, 2)

    def test_range_above(self):
        """'3-7 years' — user with 10 years is not eligible."""
        job = _make_job(min_exp=3, max_exp=7)
        assert not is_eligible(job, 10)

    def test_range_at_min(self):
        """Exactly at min boundary is eligible."""
        job = _make_job(min_exp=3, max_exp=7)
        assert is_eligible(job, 3)

    def test_range_at_max(self):
        """Exactly at max boundary is eligible."""
        job = _make_job(min_exp=3, max_exp=7)
        assert is_eligible(job, 7)

    def test_max_only(self):
        """Max only, no min — anyone below is eligible."""
        job = _make_job(max_exp=5)
        assert is_eligible(job, 3)
        assert not is_eligible(job, 6)

    def test_fractional_experience(self):
        """3.5 years against 3-7 range."""
        job = _make_job(min_exp=3, max_exp=7)
        assert is_eligible(job, 3.5)

    def test_zero_experience(self):
        """Fresh grad with 0 years against 0+ requirement."""
        job = _make_job(min_exp=0)
        assert is_eligible(job, 0)
