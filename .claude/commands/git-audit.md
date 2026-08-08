---
description: Audit worktrees and branches for orphaned work — dirty worktrees, unparked diffs, stale branches, PR/branch drift
argument-hint: "[optional: 'clean' to also remove what the audit classifies as safely deletable]"
---

# Git audit

Multiple concurrent sessions create worktrees under session scratchpads and
park work on branches. This command finds what has been left behind before it
is lost: scratchpad worktrees sit in `/tmp` and vanish on reboot, and a dirty
worktree's diff is listed by nothing. Run it weekly, or whenever branches feel
cluttered.

The audit READS by default. Only with the explicit `clean` argument may it
remove anything, and then only items in the "safe to delete" class below.

## Steps

1. **Inventory.** In parallel:
   - `git worktree list --porcelain` — every worktree, its branch or detached
     HEAD.
   - `git branch -vv --sort=-committerdate --format='%(committerdate:short) %(refname:short) -> %(upstream:short) %(upstream:track)'`
   - `gh pr list --state open`, and closed/merged PRs
     (`gh pr list --state closed --json headRefName,number,mergedAt`).
   - `.agents/fences.json` — unexpired claims mean a session may still be
     live; do not touch its worktrees.

2. **Check every worktree for uncommitted changes.** For each worktree,
   `git -C <path> status --porcelain`. Classify dirty files:
   - *Scratch* — untracked `.patch` files, PR-body drafts, notes. Disposable.
   - *Real work* — modified tracked files. For each, check whether the change
     already exists on some branch (`git grep` for a distinctive symbol across
     `git for-each-ref --format='%(refname:short)' refs/heads`). If it exists
     nowhere: **this is an orphan — the headline finding.** Report what it
     does and which branch/PR it belongs with. Recommend parking it on a
     `wip/<topic>` branch (see CLAUDE.md "Park work on `wip/` branches").

3. **Classify every local branch.** Beware two traps: `git log origin/X..X`
   silently succeeds against the wrong ref if `origin/X` does not exist —
   verify upstreams with `%(upstream:short)`, not assumptions; and rename or
   rebase history means content can exist under different SHAs — use
   `git cherry` (patch-equivalence) and `git grep` for distinctive symbols,
   not SHA comparison. Classes:
   - **PR-backed, in sync** — fine.
   - **PR-backed, diverged** (local ahead/behind its origin) — check whether
     the local-only commits are patch-present elsewhere; if so the local copy
     can be reset, if not they need pushing.
   - **Absorbed** — all commits patch-equivalent in another branch
     (`git cherry <other> <branch>` all `-`). Safe to delete.
   - **Abandoned** — head of a PR closed without merging. Safe to delete
     unless it holds unique commits.
   - **Orphaned work** — unique commits, no PR, no upstream. NOT stale; these
     are the second headline class. Summarize what each does and whether main
     has since diverged past it.
   - **Deliberate archives** (pre-surgery tips, feature parks) — keep, note
     the condition under which they become deletable.

4. **Report.** A triage table: item, class, what it does, recommended action.
   Lead with orphaned uncommitted work (step 2) and orphaned branches
   (step 3) — those are the findings that rot. Stale-but-safe items are the
   footnote, not the headline.

5. **If and only if invoked as `/git-audit clean`:** remove items classified
   safe to delete — clean worktrees whose branch is absorbed/abandoned/gone,
   the branches themselves (`git branch -D` after re-verifying absorption),
   and scratch-only dirt. Never delete: anything with real uncommitted work,
   worktrees of sessions with unexpired fence claims, deliberate archives, or
   anything whose classification required a guess. When in doubt, report
   instead of removing.
