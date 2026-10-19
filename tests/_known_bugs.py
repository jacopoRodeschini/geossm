"""Marker for confirmed, not-yet-fixed defects (docs/design_audit/design_findings.md)."""

import pytest


def known_bug(finding_id, reason, raises=AssertionError):
    """Strict xfail tied to an audit finding.

    The test asserts the *correct* behaviour. While the defect exists it fails
    (reported as xfailed); when the defect is fixed it passes, which `strict`
    turns into a failure so the marker is removed together with the fix.

    `raises` restricts the expected failure to the exception the defect
    actually produces, so an unrelated error in the test is still reported
    as a failure instead of being hidden behind the xfail.
    """
    return pytest.mark.xfail(strict=True, raises=raises, reason=f"{finding_id}: {reason}")
