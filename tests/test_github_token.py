"""get_token()'s ladder: env -> PAT file -> `gh auth token` (issue #148).

The file rung is the headless-safe source: under scron/cron `gh auth token` can hand
back an INVALID token, which is why scripts/refresh_all_issues.sh has read the PAT file
since it was written.  These tests pin the ORDER (the file is tried BEFORE `gh`, never
after) and that `gh` is not consulted at all once the file supplies a token.

No test reads or prints the real ~/.config/data-qa/github_token: every case points
_github.TOKEN_FILE at a tmp_path.
"""
import subprocess

import pytest

from data_qa import _github

# captured at import, before the autouse fixtures repoint them: TOKEN_FILE at a
# tmp_path (this file's fixture) and check_auth at an offline stub (tests/conftest.py --
# the preflight tests below need the REAL one)
_DEFAULT_TOKEN_FILE = _github.TOKEN_FILE
_REAL_CHECK_AUTH = _github.check_auth


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch, tmp_path):
    """Clear the env and point the file rung at an empty tmp dir; ban real `gh`."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tmp_path / "absent_token"))

    def _no_subprocess(*a, **kw):
        raise AssertionError(f"`gh` must not be invoked here: {a!r}")

    monkeypatch.setattr(_github.subprocess, "run", _no_subprocess)


def _fake_gh(monkeypatch, stdout, returncode=0):
    def run(cmd, **kw):
        assert cmd[:3] == ["gh", "auth", "token"]
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(_github.subprocess, "run", run)


def test_file_rung_supplies_the_token_and_gh_is_not_consulted(monkeypatch, tmp_path):
    """The regression: no env token, a PAT file on disk -> the file's token."""
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile\n")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    # the autouse fixture makes any subprocess.run call an AssertionError, so this
    # also pins that the `gh` rung is never reached when the file has a token
    assert _github.get_token() == "github_pat_fromfile"


def test_env_still_wins_over_the_file(monkeypatch, tmp_path):
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    monkeypatch.setenv("GITHUB_TOKEN", "env_tok")
    assert _github.get_token() == "env_tok"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "gh_env_tok")
    assert _github.get_token() == "gh_env_tok"


def test_file_rung_comes_before_the_gh_fallback(monkeypatch, tmp_path):
    """`gh` can hand back an invalid token headless; the file must win over it."""
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    _fake_gh(monkeypatch, "stale_gh_token\n")
    assert _github.get_token() == "github_pat_fromfile"


def test_missing_or_empty_file_falls_through_to_gh(monkeypatch, tmp_path):
    _fake_gh(monkeypatch, "gh_tok\n")
    assert _github.get_token() == "gh_tok"          # TOKEN_FILE does not exist
    empty = tmp_path / "empty_token"
    empty.write_text("   \n")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(empty))
    assert _github.get_token() == "gh_tok"


def test_unreadable_file_falls_through_rather_than_raising(monkeypatch, tmp_path):
    """A directory (or a 000-perm file) at the path must not crash the report path."""
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tmp_path))   # a directory
    _fake_gh(monkeypatch, "gh_tok\n")
    assert _github.get_token() == "gh_tok"


def test_token_file_default_is_the_documented_path():
    """scripts/refresh_all_issues.sh and the scrontab docs name this exact path."""
    assert _DEFAULT_TOKEN_FILE == "~/.config/data-qa/github_token"


def test_token_from_file_strips_and_expands(monkeypatch, tmp_path):
    tok_file = tmp_path / "github_token"
    tok_file.write_text("  github_pat_padded\n\n")
    assert _github.token_from_file(str(tok_file)) == "github_pat_padded"
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _github.token_from_file("~/github_token") == "github_pat_padded"
    assert _github.token_from_file(str(tmp_path / "nope")) is None


# ------------------------------------------------- malformed content (review B1)
# A readable file is not a usable token: `.strip()` only touches the ends, so a PAT
# file saved with a second line yields a token with an INTERIOR newline.  Sending it
# raises `ValueError: Invalid header value b'token <the token>\nsomethingelse'` out of
# http.client -- which puts the token in the log AND, from mast_monitor's report path,
# aborts main() three lines before its save_state().

MULTILINE = "github_pat_AAAA\nsomethingelse\n"


@pytest.mark.parametrize("content, why", [
    (MULTILINE, "second line"),
    ("github_pat_AAAA somethingelse\n", "interior space"),
    ("github_pat_AAAA\rsomethingelse\n", "interior CR"),
    ("github_pat_AAAA\tsomethingelse\n", "interior tab"),
    ("github_pat_\u00e9AAA\n", "non-ASCII"),
])
def test_malformed_file_is_skipped_not_returned(monkeypatch, tmp_path, capsys,
                                                content, why):
    tok_file = tmp_path / "github_token"
    tok_file.write_text(content)
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    _fake_gh(monkeypatch, "gh_tok\n")
    assert _github.get_token() == "gh_tok", why
    err = capsys.readouterr().err
    assert str(tok_file) in err                      # names the file...
    assert "github_pat_AAAA" not in err              # ...never its contents


def test_malformed_env_token_is_skipped_and_the_file_rung_still_runs(
        monkeypatch, tmp_path):
    """The same defect via $GITHUB_TOKEN: skip the rung, do not hand it on."""
    tok_file = tmp_path / "github_token"
    tok_file.write_text("github_pat_fromfile\n")
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    monkeypatch.setenv("GITHUB_TOKEN", MULTILINE)
    assert _github.get_token() == "github_pat_fromfile"


def test_a_malformed_token_never_reaches_a_request_header(monkeypatch):
    """Defence in depth for the callers that read the env directly (make_issues.py):
    request() refuses to build the header instead of raising."""
    sent = []
    monkeypatch.setattr(_github.urllib.request, "urlopen",
                        lambda req, **kw: sent.append(req) or None)
    status, data = _github.request("GET", _github.API + "/user", MULTILINE.strip())
    assert status == _github.BAD_TOKEN_STATUS and status >= 300
    assert sent == []
    assert "github_pat_AAAA" not in data["message"]


def test_report_path_returns_an_rc_instead_of_raising_on_a_malformed_file(
        monkeypatch, tmp_path, capsys):
    """The end-to-end contract: post_status returns, so mast_monitor.main() reaches
    save_state()."""
    from data_qa import status_report

    tok_file = tmp_path / "github_token"
    tok_file.write_text(MULTILINE)
    monkeypatch.setattr(_github, "TOKEN_FILE", str(tok_file))
    monkeypatch.setattr(_github.subprocess, "run",      # no `gh` on PATH (scron)
                        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
    monkeypatch.setattr(_github.urllib.request, "urlopen",
                        lambda *a, **kw: pytest.fail("no request may be attempted"))
    assert status_report.post_status("T", "BODY", dry_run=False) == 2
    assert "no GitHub token" in capsys.readouterr().err


# ------------------------------------------------------ auth preflight (review B2)
# scripts/refresh_all_issues.sh pairs this same ladder with a loud `gh api user`
# preflight (:43-51) because a 401 is otherwise SWALLOWED: _paginate breaks on a
# non-200, existing_issues returns {}, and post_status reports rc=3 "no issue titled".

def _fake_request(monkeypatch, status, data):
    calls = []

    def req(method, url, token, data_=None):
        calls.append((method, url))
        return status, data

    monkeypatch.setattr(_github, "request", req)
    return calls


def test_check_auth_accepts_a_good_token_and_memoizes(monkeypatch):
    _github._AUTH_CHECKED.clear()
    calls = _fake_request(monkeypatch, 200, {"login": "keflavich"})
    assert _REAL_CHECK_AUTH("tok") == (True, "keflavich")
    assert _REAL_CHECK_AUTH("tok") == (True, "keflavich")
    assert len(calls) == 1                    # one extra request per RUN, not per issue
    assert calls[0][1].endswith("/user")


def test_check_auth_rejects_an_expired_token(monkeypatch):
    _github._AUTH_CHECKED.clear()
    _fake_request(monkeypatch, 401, {"message": "Bad credentials"})
    ok, detail = _REAL_CHECK_AUTH("expired")
    assert ok is False
    assert "401" in detail and "Bad credentials" in detail


def test_check_auth_does_not_block_an_offline_run(monkeypatch):
    """A transport failure is not an auth verdict: the preflight must not turn an
    unreachable api.github.com into 'your token is bad'."""
    _github._AUTH_CHECKED.clear()
    _fake_request(monkeypatch, _github.NETWORK_ERROR_STATUS,
                  {"message": "network error: [Errno -2] Name or service not known"})
    ok, detail = _REAL_CHECK_AUTH("tok")
    assert ok is True and "unchecked" in detail


def test_network_failure_is_an_rc_not_an_exception(monkeypatch):
    """urlopen raising URLError used to propagate out of existing_issues ->
    post_status -> act_report and abort the poll before save_state()."""
    import urllib.error

    def boom(*a, **kw):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(_github.urllib.request, "urlopen", boom)
    status, data = _github.request("GET", _github.API + "/user", "ghp_tok")
    assert status == _github.NETWORK_ERROR_STATUS and status >= 300
    assert "network error" in data["message"]


def test_post_status_names_auth_on_a_rejected_token(monkeypatch, capsys):
    """Without the preflight this run reports rc=3 'no issue titled ...' -- the
    failure the reviewer measured, which names nothing about auth."""
    from data_qa import status_report

    monkeypatch.setattr(_github, "get_token", lambda: "expired_pat")
    monkeypatch.setattr(_github, "check_auth", lambda token, force=False:
                        (False, "HTTP 401: Bad credentials"))
    monkeypatch.setattr(_github, "existing_issues",   # what a 401 listing looks like
                        lambda token, repo: {})
    rc = status_report.post_status("Some QA issue", "BODY", dry_run=False)
    err = capsys.readouterr().err
    assert rc == 5                                    # a distinct rc, not 3
    assert "auth failed" in err and "401" in err
    assert _github.TOKEN_FILE in err                  # says how to fix it
    assert "no issue titled" not in err


def test_dry_run_still_needs_no_token_or_network(monkeypatch, capsys):
    """The preflight must not fire on a dry run (the default everywhere)."""
    from data_qa import status_report

    monkeypatch.setattr(_github, "get_token",
                        lambda: pytest.fail("dry run must not resolve a token"))
    assert status_report.post_status("T", "BODY", dry_run=True) == 0
    assert "DRY-RUN" in capsys.readouterr().out
