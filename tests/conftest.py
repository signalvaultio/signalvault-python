"""Shared test setup."""

import pytest

import signalvault.client


@pytest.fixture(autouse=True)
def _fresh_refused_event_warnings(monkeypatch):
    """The warning for an audit event refused with 401, 402 or 403 is once per process; each test starts afresh."""
    monkeypatch.setattr(signalvault.client, "_REFUSED_WARNED", set(), raising=False)
