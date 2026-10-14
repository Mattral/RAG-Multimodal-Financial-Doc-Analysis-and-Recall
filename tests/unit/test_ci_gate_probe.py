"""Deliberately failing test: proves CI blocks on a red test. Reverted in the next commit."""


def test_ci_gate_probe_must_fail():
    assert 1 == 2, "deliberate failure to verify the CI gate"
