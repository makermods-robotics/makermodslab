"""client.describe() (the one-call orientation snapshot) and the
progressive-disclosure docs engine (index -> namespace card -> method detail,
plus search and the --all dump)."""

from __future__ import annotations

import inspect

import httpx
import pytest
from helpers import mock_client
from makermodslab_sdk import docs
from makermodslab_sdk.client import RESOURCE_CLASSES
from makermodslab_sdk.describe import ServerSnapshot
from makermodslab_sdk.docs import cheatsheet, index, main, method_detail, namespace_card, search

HEALTH = {
    "status": "ok",
    "message": "up",
    "version": "0.1.0",
    "instance_id": "ab" * 16,
    "capabilities": {"serves_ui": True, "accepts_jobs": True},
}
JOB = {
    "id": "job-1",
    "job_number": 4,
    "name": "act_run",
    "state": "running",
    "config": {"dataset_repo_id": "u/d", "policy_type": "act", "steps": 100, "batch_size": 8},
    "output_dir": "outputs/train/job-1",
    "started_at": 1756200000.0,
    "metrics": {"current_step": 10, "total_steps": 100},
}


def scripted(responses):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in responses:
            return responses[path]
        return httpx.Response(500, json={"detail": f"unexpected {path}"})

    return handler


def test_describe_composes_all_sections():
    handler = scripted(
        {
            "/api/v1/health": httpx.Response(200, json=HEALTH),
            "/api/v1/sessions/current": httpx.Response(200, json={"session": None, "last_ended": None}),
            "/api/v1/jobs": httpx.Response(200, json={"jobs": [JOB]}),
            "/api/v1/nodes": httpx.Response(200, json={"nodes": []}),
        }
    )
    with mock_client(handler) as client:
        snap = client.describe()
    assert isinstance(snap, ServerSnapshot)
    assert snap.health is not None and snap.health.version == "0.1.0"
    assert snap.session is None
    assert [job.id for job in snap.running_jobs] == ["job-1"]
    assert snap.errors == {}
    text = snap.summary()
    assert "v0.1.0" in text
    assert "robot is free" in text
    assert "step 10/100" in text


def test_describe_survives_a_failing_section():
    handler = scripted(
        {
            "/api/v1/health": httpx.Response(200, json=HEALTH),
            "/api/v1/sessions/current": httpx.Response(200, json={"session": None, "last_ended": None}),
            "/api/v1/jobs": httpx.Response(500, json={"detail": "registry exploded"}),
            "/api/v1/nodes": httpx.Response(200, json={"nodes": []}),
        }
    )
    with mock_client(handler) as client:
        snap = client.describe()
    assert snap.health is not None
    assert "jobs" in snap.errors and "registry exploded" in snap.errors["jobs"]
    assert "jobs unavailable" in snap.summary()


def test_describe_end_to_end(sdk_client):
    snap = sdk_client.describe()
    assert snap.health is not None and snap.health.status == "ok"
    assert snap.errors == {}
    assert isinstance(snap.summary(), str) and snap.summary()


# --- Tier 0: index() ---------------------------------------------------------

# The index is the tier that sits in an agent's context on every task, so it
# gets a hard budget: < 4000 chars is roughly 1k tokens. Growth belongs in the
# cards (tier 1) and method details (tier 2), never here.
INDEX_BUDGET_CHARS = 4_000


def test_index_stays_under_the_tier0_budget():
    assert len(index()) < INDEX_BUDGET_CHARS


def test_index_names_every_namespace_and_the_drilldowns():
    text = index()
    for tag in RESOURCE_CLASSES:
        assert f"client.{tag} — " in text, f"index missing namespace {tag}"
    assert "client.flows — " in text
    assert "- client — top level" in text
    # The drill-down affordances an agent needs to go deeper:
    assert 'client.docs("jobs")' in text
    assert 'client.docs("jobs.create_training")' in text
    assert "client.docs(search=" in text
    assert "--all" in text
    assert "python -m makermodslab_sdk.docs" in text
    # The load-bearing rules survive the compression:
    assert "ERRORS ARE THE MANUAL" in text
    assert "with client.sessions.teleoperate" in text
    assert "POLLING" in text
    assert "--sfu" in text
    assert 'extra="allow"' in text


# --- Tier 1: namespace_card() ------------------------------------------------

# Per-NAMESPACE budgets (measured 2026-09-22, cap = measured * ~1.2 rounded up
# to the nearest 100) so a single namespace ballooning is flagged BY NAME and
# a small card can't grow 10x under a shared cap. Re-derive an entry when its
# card legitimately outgrows it:
#   .venv/bin/python -c "from makermodslab_sdk import docs;
#     print({t: len(docs.namespace_card(t)) for t in docs._namespaces()})"
CARD_BUDGETS_CHARS = {
    "client": 4300,  # measured 3561 after flow exceptions
    "datasets": 5100,  # measured 4196
    "flows": 2900,  # measured 2375 after hardware inspection flow
    "inference": 2300,  # measured 1858
    "jobs": 3900,  # measured 3247
    "models": 3200,  # measured 2587
    "nodes": 2400,  # measured 1991
    "realtime": 3700,  # measured 3059
    "recording": 1000,  # measured 781
    "remote": 2800,  # measured 2263
    "robots": 2400,  # measured 1956
    "sessions": 8700,  # measured 7204
    "sfu": 1000,  # measured 754
    "system": 4100,  # measured 3364
}


def test_budget_table_matches_the_namespace_set():
    from makermodslab_sdk import docs as docs_module

    assert set(CARD_BUDGETS_CHARS) == set(docs_module._namespaces()), (
        "a namespace was added or removed — measure its card and update CARD_BUDGETS_CHARS"
    )


@pytest.mark.parametrize("tag", sorted([*RESOURCE_CLASSES, "client", "flows", "realtime"]))
def test_every_namespace_card_stays_under_its_budget(tag):
    assert len(namespace_card(tag)) < CARD_BUDGETS_CHARS[tag], f"card {tag} ballooned"


def test_cards_carry_the_module_docstring_as_pattern_intro():
    # Module docstrings are the single source of pattern prose; spot-check the
    # blocks that moved out of the old hand-written header.
    assert "with client.sessions.teleoperate" in namespace_card("sessions")
    assert "SessionLostError" in namespace_card("sessions")
    assert "FULL BACKEND POWER" in namespace_card("jobs")
    assert "TrainingOptions" in namespace_card("jobs")
    assert "coaching" in namespace_card("inference")
    assert "gpu_start" in namespace_card("remote")
    assert "--sfu" in namespace_card("remote")
    assert "refetch hints" in namespace_card("realtime")
    assert "empty list means no hardware flow" in namespace_card("realtime")
    assert "client.sessions" in namespace_card("flows")
    assert "inspect_hardware(" in namespace_card("flows")
    assert "record_episodes(" in namespace_card("flows")
    assert "remote_inference(" in namespace_card("flows")
    assert "train_and_publish(" in namespace_card("flows")
    # The client card carries the exception taxonomy:
    client_card = namespace_card("client")
    assert "SessionHeldError" in client_card and "RobotBusyError" in client_card


def test_every_operation_method_is_in_its_card_and_searchable_by_name():
    for tag, cls in RESOURCE_CLASSES.items():
        card = namespace_card(tag)
        for name, member in vars(cls).items():
            if hasattr(member, "_operation_id"):
                assert f"- {name}(" in card, f"card {tag} missing {name}"
                assert f"{tag}.{name}(" in search(name), f"search cannot find {tag}.{name}"


# --- Tier 2: method_detail() -------------------------------------------------


def test_method_detail_returns_the_full_docstring():
    from makermodslab_sdk.resources.jobs import JobsResource

    text = method_detail("jobs.wait")
    assert text.startswith("jobs.wait(")
    assert inspect.getdoc(JobsResource.wait) in text  # FULL docstring, not the one-liner
    assert "sessions.record" in method_detail("flows.record_episodes")
    assert "sessions.remote_inference_transport" in method_detail("flows.remote_inference")
    assert "gpu_stop(launch_id=...)" in method_detail("flows.remote_inference")
    assert "models.publish_status" in method_detail("flows.train_and_publish")
    assert "system.available_ports" in method_detail("flows.inspect_hardware")


def test_method_detail_unknown_namespace_is_helpful_not_raised():
    text = method_detail("bogus.thing")
    assert "Unknown namespace" in text
    for tag in RESOURCE_CLASSES:
        assert tag in text  # names the valid namespaces


def test_method_detail_misspelled_method_suggests_close_matches():
    text = method_detail("sessions.teleoperatee")
    assert "sessions.teleoperate" in text  # difflib close match
    text = method_detail("jobs.creat_training")
    assert "jobs.create_training" in text


# --- search() ------------------------------------------------------------------


def test_search_hits_carry_the_drill_affordance():
    text = search("publish")
    assert "models.publish(" in text
    assert 'client.docs("models.publish")' in text
    assert "flows.train_and_publish(" in text


def test_evaluation_is_discoverable_by_task_word():
    assert 'client.docs("sessions.infer")' in search("eval")


def test_search_zero_hits_points_back_at_the_index():
    text = search("zzz-no-such-thing")
    assert "client.docs()" in text and "makermodslab_sdk.docs" in text


def test_search_scope_is_names_and_one_liners_only():
    # USER DECISION made durable: search never looks at full docstring bodies.
    # "monotonic" occurs only deep in a docstring body (client.sample_joints's
    # clock parameter note) — first the guard that it is really there…
    spaces = docs._namespaces()
    bodies = "\n".join(
        "\n".join((inspect.getdoc(member) or "").splitlines()[1:])
        for tag, (_intro, members) in spaces.items()
        if tag != "realtime"
        for member in members.values()
    )
    assert "monotonic" in bodies
    # …then the pin: a body-only term does NOT match.
    assert "No method name or one-liner contains" in search("monotonic")


# --- cheatsheet() (--all) -------------------------------------------------------


def test_cheatsheet_covers_every_operation_method():
    # The full dump is deliberately UNBUDGETED — it lives behind --all only.
    text = cheatsheet()
    assert text.startswith(index())
    for tag, cls in RESOURCE_CLASSES.items():
        assert f"client.{tag}" in text
        for name, member in vars(cls).items():
            if hasattr(member, "_operation_id"):
                assert f"- {name}(" in text, f"cheatsheet missing {tag}.{name}"
    # The load-bearing patterns are present…
    assert "with client.sessions.teleoperate" in text
    assert "Next step" in text
    assert "refetch hints" in text


# --- Client.docs() wiring --------------------------------------------------------


def test_client_docs_makes_zero_http_requests():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={})

    with mock_client(handler) as client:
        assert client.docs() == index()
        assert client.docs("jobs") == namespace_card("jobs")
        assert client.docs("jobs.create_training") == method_detail("jobs.create_training")
        assert client.docs(search="publish") == search("publish")
        assert "Unknown namespace" in client.docs("bogus")
    assert calls["count"] == 0


# --- CLI (python -m makermodslab_sdk.docs) ---------------------------------------


def test_cli_no_args_prints_the_index(capsys):
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == index().strip()


def test_cli_topic_prints_the_card(capsys):
    assert main(["jobs"]) == 0
    assert capsys.readouterr().out.strip() == namespace_card("jobs").strip()


def test_cli_search_prints_hits(capsys):
    assert main(["--search", "publish"]) == 0
    assert "models.publish(" in capsys.readouterr().out


def test_cli_all_prints_the_dump(capsys):
    assert main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "client.datasets" in out and len(out) > len(index())


def test_cli_unknown_topic_exits_zero_with_the_helpful_string(capsys):
    # Agents read output, not exit codes — an unknown topic is an answer.
    assert main(["bogus"]) == 0
    assert "Unknown namespace" in capsys.readouterr().out


# --- review follow-ups (2026-09-22) -------------------------------------------


def test_client_card_teaches_exception_attributes():
    """An agent planning a try/except from docs alone must discover the
    attribute affordances — the errors MODULE docstring is the source."""
    card = namespace_card("client")
    for affordance in ("suggestion", ".busy_with", ".holder", "isinstance"):
        assert affordance in card, f"exceptions section lost {affordance}"
    # …and the ergonomics-layer errors the module docstring doesn't name.
    for name in ("JobWaitTimeout", "SessionLostError", "WaitTimeoutError"):
        assert name in card


def test_index_advertises_realtime_and_its_extra():
    text = index()
    assert "realtime" in text
    assert "[realtime]" in text


def test_search_reaches_realtime_only_helpers_without_twin_duplicates():
    hits = search("parse_message")
    assert "realtime.parse_message" in hits
    # The socket-facing twins surface once, via the client card, not twice.
    sample_hits = [line for line in search("sample_joints").splitlines() if line.strip()]
    assert any(line.startswith("client.sample_joints") for line in sample_hits)
    assert not any(line.startswith("realtime.sample_joints") for line in sample_hits)


def test_map_lines_pluralize():
    assert "(1 method)" in index()  # sfu
    assert "(1 methods)" not in index()
