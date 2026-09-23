"""Progressive-disclosure docs for the SDK — three tiers plus search.

A flat catalog degrades agent accuracy, so the docs disclose progressively:

* Tier 0 — :func:`index`: identity, the core rules, a one-line-per-namespace
  map, and how to drill down. Under ~1k tokens; the default everywhere.
* Tier 1 — :func:`namespace_card`: one namespace's pattern intro (the
  resource MODULE docstring — the single source of pattern prose) plus its
  introspected method reference.
* Tier 2 — :func:`method_detail`: one method's full signature and full
  docstring.
* :func:`search`: case-insensitive substring match over method names and
  their docstring one-liners ONLY (never full bodies — deliberate).
* :func:`cheatsheet`: the old full dump (index + every card), kept only
  behind the CLI's ``--all``.

``python -m makermodslab_sdk.docs [topic|--search q|--all]`` prints any of
them; ``client.docs(...)`` is the in-process door. Everything but the index's
rule list is introspected from the live classes, so it can never drift from
the code.
"""

from __future__ import annotations

import difflib
import inspect
import sys
from collections.abc import Callable

# The two pseudo-namespaces that exist beside RESOURCE_CLASSES: the client's
# own top-level methods, and the realtime module's pure helpers.
CLIENT_TAG = "client"
REALTIME_TAG = "realtime"

# Client methods worth documenting (the rest are dunder/context-manager glue).
_CLIENT_METHODS = ("describe", "docs", "events", "sample_joints", "stream_joints", "close")


def _signature(member: Callable) -> str:
    try:
        text = str(inspect.signature(member))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "(...)"
    return text.replace("(self, ", "(").replace("(self)", "()")


# Parameters that exist only so tests can drive a loop without sleeping. They
# are noise on a browsing card (and render as "<built-in function monotonic>"),
# so tier 1 elides them; tier 2 still shows the true, complete signature.
_TEST_SEAM_PARAMS = frozenset({"sleep_fn", "clock", "poll_interval"})

# A card is for BROWSING: past this many chars a signature stops informing and
# starts crowding out the other methods, so the card points at tier 2 instead.
_CARD_SIGNATURE_LIMIT = 160


def _card_signature(member: Callable) -> str:
    """The tier-1 signature: what you MUST pass, names only.

    A browsing agent needs the required arguments to choose a method; the
    optional tail, the annotations and the test seams (sleep_fn/clock) are
    noise at this tier and are replaced by a trailing "…" so the elision is
    visible. Tier 2 prints the true, complete, annotated signature.
    """
    try:
        signature = inspect.signature(member)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "(...)"
    required: list[str] = []
    optional = False
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            optional = True
            continue
        if parameter.default is not parameter.empty or name in _TEST_SEAM_PARAMS:
            optional = True
            continue
        if parameter.kind is parameter.KEYWORD_ONLY and "*" not in required:
            required.append("*")
        required.append(name)
    shown = [*required, "…"] if optional else required
    text = f"({', '.join(shown)})"
    return text if len(text) <= _CARD_SIGNATURE_LIMIT else "(…)"


def _first_line(member: object) -> str:
    doc = inspect.getdoc(member)
    return doc.splitlines()[0] if doc else ""


def _resource_members(cls: type) -> dict[str, Callable]:
    return {
        name: member
        for name, member in sorted(vars(cls).items())
        if not name.startswith("_") and callable(member)
    }


def _flow_members(cls: type) -> dict[str, Callable]:
    """The flows facade inherits its public methods from implementation modules."""
    return {
        name: member
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_") and callable(member)
    }


def _client_members() -> dict[str, Callable]:
    from makermodslab_sdk.client import Client

    return {name: vars(Client)[name] for name in _CLIENT_METHODS if name in vars(Client)}


def _realtime_members() -> dict[str, Callable]:
    from makermodslab_sdk import realtime

    return {
        name: member
        for name, member in sorted(vars(realtime).items())
        if not name.startswith("_") and inspect.isfunction(member) and member.__module__ == realtime.__name__
    }


def _exceptions_section() -> list[str]:
    """The error taxonomy for the client card.

    The errors MODULE docstring is the single source for the base hierarchy —
    it is what documents the attribute affordances (.status/.code/.detail/
    .details/.suggestion, RobotBusyError.busy_with, SessionHeldError.holder)
    an agent needs to plan a try/except. Ergonomics-layer exceptions it does
    not name (JobWaitTimeout, SessionLostError, …) are appended as
    introspected one-liners.
    """
    import makermodslab_sdk
    from makermodslab_sdk import errors

    module_doc = inspect.getdoc(errors) or ""
    lines = [module_doc, ""]
    for name in sorted(makermodslab_sdk.__all__):
        member = getattr(makermodslab_sdk, name)
        if isinstance(member, type) and issubclass(member, BaseException) and name not in module_doc:
            lines.append(f"- {name} — {_first_line(member)}")
    return lines


def _namespaces() -> dict[str, tuple[str, dict[str, Callable]]]:
    """tag -> (pattern intro, {method name: callable}) for every card."""
    from makermodslab_sdk.client import RESOURCE_CLASSES, Client

    spaces: dict[str, tuple[str, dict[str, Callable]]] = {}
    for tag in sorted(RESOURCE_CLASSES):
        cls = RESOURCE_CLASSES[tag]
        module = inspect.getmodule(cls)
        intro = inspect.getdoc(module) if module else ""
        spaces[tag] = (intro or "", _resource_members(cls))
    spaces[CLIENT_TAG] = (inspect.getdoc(Client) or "", _client_members())

    from makermodslab_sdk import flows

    spaces["flows"] = (inspect.getdoc(flows) or "", _flow_members(flows.Flows))

    from makermodslab_sdk import realtime

    spaces[REALTIME_TAG] = (inspect.getdoc(realtime) or "", _realtime_members())
    return spaces


def _map_line(tag: str, members: dict[str, Callable], cls: type) -> str:
    one_liner = _first_line(cls)
    prefix = f"``client.{tag}`` — "
    if one_liner.startswith(prefix):
        one_liner = one_liner[len(prefix) :]
    plural = "method" if len(members) == 1 else "methods"
    return f"- client.{tag} — {one_liner} ({len(members)} {plural})"


def index() -> str:
    """Tier 0: identity, the core rules, the namespace map, how to drill down."""
    from makermodslab_sdk.client import RESOURCE_CLASSES
    from makermodslab_sdk.flows import Flows

    lines = [
        "# makermodslab-sdk — start here (tier 0 of 3)",
        "",
        "Agent-first Python SDK for a MakerMods Lab robot server (SO-101 /",
        "Maker / Metal arms: teleoperation, recording, training, inference,",
        "replay, calibration).",
        "",
        "    from makermodslab_sdk import Client",
        '    client = Client("http://localhost:8000")   # the app\'s single port',
        "    print(client.describe().summary())          # ALWAYS a good first call",
        "",
        "## The rules that matter",
        "",
        "- ERRORS ARE THE MANUAL. Every failure's text ends \"Next step: <the",
        '  literal call to make>". Branch on err.code (e.g. "session.held") or',
        "  exception type — never on the prose.",
        "- SESSIONS HOLD THE ARM. One robot flow at a time; start flows through",
        '  the with-form (with client.sessions.teleoperate("bench") as s: ...)',
        "  so the lease heartbeat + stop are automatic. Busy? SessionHeldError",
        "  names the holder; sessions.stop_current() is the hammer.",
        "- NEVER WRITE POLLING LOOPS. Long-running work has blocking waiters:",
        "  jobs.wait(job_id), datasets.wait_for_download(repo_id), ... All take",
        "  timeout=; on timeout the error says how to keep waiting.",
        "- STREAMS ARE BOUNDED OR URL-ONLY. sample_joints() returns a LIST",
        "  (empty = nothing moving, that's an answer); cameras come as URLs.",
        "- FULL BACKEND POWER, wider than the web UI: jobs.create_training(...)",
        '  accepts EVERY server training knob — see client.docs("jobs").',
        "- REMOTE needs the server started with --sfu plus the [remote] extra —",
        '  see client.docs("remote").',
        '- Responses are pydantic models with extra="allow" — unknown server',
        "  fields stay readable, never crash.",
        "",
        '## Namespaces — drill down with client.docs("<tag>")',
        "",
    ]
    spaces = _namespaces()
    for tag in sorted([*RESOURCE_CLASSES, "flows"]):
        _intro, members = spaces[tag]
        cls = Flows if tag == "flows" else RESOURCE_CLASSES[tag]
        lines.append(_map_line(tag, members, cls))
    lines.append(
        f"- client — top level: describe() / docs() / realtime reads ({len(spaces[CLIENT_TAG][1])} methods)"
    )
    lines.append("- realtime — WS helpers (parse_message, ws_url, …; needs the [realtime] extra)")
    lines += [
        "",
        "## Drill down (tiers 1-2 + search)",
        "",
        '- client.docs("jobs")                 -> one namespace: pattern + methods',
        '- client.docs("jobs.create_training") -> one method: full signature + docstring',
        '- client.docs(search="publish")       -> find methods by name/one-liner',
        "- CLI: python -m makermodslab_sdk.docs [topic|--search q|--all]",
        "  (--all is the full dump; the tiers above are always the better start)",
    ]
    return "\n".join(lines) + "\n"


def namespace_card(tag: str) -> str:
    """Tier 1: one namespace's pattern intro + introspected method reference."""
    spaces = _namespaces()
    if tag not in spaces:
        return _unknown_tag(tag, spaces)
    intro, members = spaces[tag]
    title = "client (top level)" if tag == CLIENT_TAG else f"client.{tag}"
    if tag == REALTIME_TAG:
        title = "realtime (module helpers; socket methods live on client)"
    lines = [f"## {title} — tier 1", ""]
    if intro:
        lines += [intro, ""]
    lines.append("### Methods")
    for name, member in members.items():
        lines.append(f"- {name}{_card_signature(member)} — {_first_line(member)}")
    if tag == CLIENT_TAG:
        lines += ["", "### Exceptions"]
        lines += _exceptions_section()
    lines += [
        "",
        f'Full signature and docstring of one method: client.docs("{tag}.<method>")',
    ]
    return "\n".join(lines) + "\n"


def method_detail(path: str) -> str:
    """Tier 2: "tag.method" -> full signature + full docstring (never raises)."""
    spaces = _namespaces()
    tag, sep, name = path.partition(".")
    if not sep:
        return _unknown_tag(path, spaces)
    if tag not in spaces:
        return _unknown_tag(tag, spaces)
    _intro, members = spaces[tag]
    if name not in members:
        close = difflib.get_close_matches(name, members, n=3)
        hint = (
            "did you mean " + " / ".join(f'"{tag}.{match}"' for match in close) + "?"
            if close
            else f'client.docs("{tag}") lists them all.'
        )
        return f'No method "{name}" in client.{tag} — {hint}\n'
    member = members[name]
    doc = inspect.getdoc(member) or "(no docstring)"
    return f"{tag}.{name}{_signature(member)}\n\n{doc}\n"


def search(query: str) -> str:
    """Substring search over method NAMES and docstring one-liners only."""
    needle = query.lower()
    spaces = _namespaces()
    client_names = set(spaces[CLIENT_TAG][1])
    hits: list[str] = []
    for tag, (_intro, members) in spaces.items():
        for name, member in members.items():
            if tag == REALTIME_TAG and name in client_names:
                continue  # socket-facing twins (events, sample_joints) hit via the client card
            one_liner = _first_line(member)
            if needle in name.lower() or needle in one_liner.lower():
                signature = _signature(member)
                if len(signature) > 60:
                    signature = signature[:57] + "...)"
                hits.append(f'{tag}.{name}{signature} — {one_liner}  [drill: client.docs("{tag}.{name}")]')
    if not hits:
        return (
            f'No method name or one-liner contains "{query}". Search covers names\n'
            "and first docstring lines only — try a shorter term, or start from\n"
            "the namespace map: client.docs() (or python -m makermodslab_sdk.docs).\n"
        )
    return "\n".join(hits) + "\n"


def cheatsheet() -> str:
    """The full dump (--all): the index plus every namespace card."""
    parts = [index()]
    parts.extend(namespace_card(tag) for tag in _namespaces())
    return "\n".join(parts)


def render_topic(topic: str) -> str:
    """Dispatch a CLI/client topic string to the right tier (never raises)."""
    return method_detail(topic) if "." in topic else namespace_card(topic)


def _unknown_tag(tag: str, spaces: dict[str, tuple[str, dict[str, Callable]]]) -> str:
    close = difflib.get_close_matches(tag, spaces, n=3)
    hint = f" Did you mean {' / '.join(close)}?" if close else ""
    return (
        f'Unknown namespace "{tag}".{hint} Valid topics: '
        + ", ".join(sorted(spaces))
        + '\n(or "tag.method" for one method; client.docs() prints the index).\n'
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: no args -> index; [topic] -> card/detail; --search q; --all."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--all" in args:
        print(cheatsheet())
    elif "--search" in args:
        position = args.index("--search")
        query = args[position + 1] if position + 1 < len(args) else ""
        print(search(query) if query else 'Usage: python -m makermodslab_sdk.docs --search "query"')
    elif args:
        print(render_topic(args[0]))
    else:
        print(index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
