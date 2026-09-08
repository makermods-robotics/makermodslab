# MakerLab: `main` history rewrite — what's happening and what you need to do

**Repo:** `makermods-robotics/MakerLab` · **Branch affected:** `main`

You can hand this whole document to your coding agent. It does not assume anything
about which branches you have locally or what state they're in — every step
discovers that for itself.

---

## 1. What is happening, and why

`main` accumulated ~197 commits between **June 30 and July 22** in two dense bursts
(a hardware-safety/UX wave, and the MakerMods redesign). The log is unreadable and
looks like churn to anyone browsing the repo.

Those ~197 commits are being **squashed into 8**, one per coherent chunk of work,
with the original commit subjects preserved in each squash's message body. Everything
before June 30 and after July 22 is untouched.

**The critical property: no code changes.** The resulting file tree is byte-identical
to today's `main` — verified by tree hash. Nothing is added, removed, or reverted.
Only the *shape of the commit graph* changes: ~197 commits become 8, so every commit
from June 30 onward gets a **new SHA**.

The full pre-rewrite history is permanently preserved on the remote as
`backup/main-pre-squash` and the tag `archive/pre-squash-2026-07`. Nothing is
destroyed, and the whole thing can be rolled back with one force-push.

**Why this affects you:** your local clone's `main` shares no commits with the new
`main` from June 30 onward. Git sees them as two unrelated histories that happen to
contain identical files. Every open PR branch is being rebuilt on the new history
too — PRs keep their number, comments, and reviews.

---

## 2. The one rule that matters most

> ### After the rewrite, NEVER run `git pull` on a branch you have not reset.

`git pull` will **merge** the old history into the new one. Because both contain
identical files, this merge succeeds with **zero conflicts and no warning** — and
silently restores all ~197 old commits, undoing the rewrite and leaving `main`
in a worse state than before. This has been tested and confirmed.

Use `git fetch` + `git reset --hard`, never `git pull`. There will be a second
notification when the rewrite is complete; the rules differ before and after.

---

## 3. PHASE 1 — Before the rewrite (do this now)

Goal: get every commit you care about onto the remote, or into a local backup file.
Anything that exists only on your laptop cannot be repaired for you.

### 3a. Inventory — find work that exists nowhere but here

```bash
git status                                   # uncommitted changes?
git stash list                               # forgotten stashes?
git log --branches --not --remotes --oneline # commits that exist ONLY locally
git branch -vv                               # every local branch + ahead/behind
```

`git log --branches --not --remotes` is the important one: it lists every local
commit that is not on any remote, across all branches, without you needing to know
your own branch names. **If it prints nothing, you have nothing at risk.**

### 3b. Take a complete local backup (one file, nothing touches the remote)

```bash
git bundle create ~/makerlab-backup.bundle --all
```

This captures every branch, tag, and commit in your clone into a single file. If
anything goes wrong later, everything is recoverable from it:
`git clone ~/makerlab-backup.bundle recovered/`

### 3c. Commit or stash anything uncommitted

```bash
git add -A && git commit -m "wip: checkpoint before history rewrite"
# or, if you'd rather not commit:  git stash push -u -m "pre-rewrite wip"
```

Note: stashes are **not** included when you push, and are awkward to recover after a
reset. Committing to a branch and pushing it is strongly preferred.

### 3d. Push every branch that has unpushed commits

```bash
# Push all local branches that have a remote counterpart and are ahead
git push origin --all
```

If `--all` is too broad for you, push selectively — the goal is only that
`git log --branches --not --remotes --oneline` comes back **empty** when you're done.

### 3e. Confirm and report

```bash
git log --branches --not --remotes --oneline   # must print nothing
```

Reply to Isaac with: **"pushed, nothing local"** — or paste the output if anything
remains. Then **stop pushing** until you get the all-clear.

---

## 4. PHASE 2 — After the rewrite (only once Isaac says it's done)

### Option A (recommended, especially with several branches): fresh clone

Simplest and impossible to get wrong.

```bash
cd ..
mv MakerLab MakerLab-old        # keep the old one, don't delete it
git clone https://github.com/makermods-robotics/MakerLab.git
cd MakerLab
```

Your old clone stays at `MakerLab-old` and your bundle at `~/makerlab-backup.bundle`.
Re-create your dev environment in the new clone as usual (for MakerLab: `uv` editable
install into the repo `.venv`, per the README).

### Option B: repair the existing clone in place

```bash
git fetch origin --prune --tags
```

**Capture the old `main` from the reflog before it ages out** — the later steps need it:

```bash
OLD_MAIN=$(git rev-parse origin/main@{1})
echo "old main was: $OLD_MAIN"
```

Reset `main`:

```bash
git checkout main
git reset --hard origin/main
```

Then for every other local branch, discover and handle each one:

```bash
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
  [ "$b" = "main" ] && continue
  if git rev-parse --verify -q "origin/$b" >/dev/null; then
    echo "== $b : exists on remote, resetting to it"
    git checkout -q "$b" && git reset --hard "origin/$b"
  else
    echo "== $b : LOCAL ONLY, needs rebasing onto the new main (see below)"
  fi
done
```

**For any branch reported as LOCAL ONLY**, replay its own commits onto the new `main`.
This uses `$OLD_MAIN` from above to find where the branch originally forked:

```bash
git checkout <branch>
git rebase --onto origin/main $(git merge-base "$OLD_MAIN" <branch>) <branch>
```

Conflicts here are possible if the branch forked from inside the squashed June 30 –
July 22 range. If you hit conflicts you don't want to resolve, stop and ask Isaac —
the original commits still exist on the remote at `backup/main-pre-squash`, so nothing
is lost and there's no time pressure.

---

## 5. Verify you're on the new history

```bash
# 1. This subject belonged to a squashed-away commit. Must be 0.
git log --oneline origin/main | grep -c "enable renaming of robots"

# 2. This is one of the 8 new squash commits. Must be 1.
git log --oneline origin/main | grep -c "bimanual arms end to end"

# 3. Sanity: your files should be unchanged vs. the remote.
git status --short
```

If check 1 returns anything other than `0`, the old history has been reintroduced
somewhere — tell Isaac immediately and do not push.

---

## 6. If something looks wrong

Don't try to fix `main` yourself, and don't force-push it. Everything is recoverable:

- pre-rewrite `main` → `backup/main-pre-squash` and tag `archive/pre-squash-2026-07`
  on the remote
- each PR branch's pre-rewrite head → `backup/pr-<number>` on the remote
- your entire local clone → `~/makerlab-backup.bundle`

Report what you saw, paste the command output, and stop. A full rollback takes about
30 seconds.

---

## 7. Known rough edge

Two branches — `windows_support` and `windows_support-fixes` (~13 commits) — fork from
a point inside the squashed range, so they have no clean landing spot on the new
history. They keep working as branches, but merging them into the new `main` will need
a manual rebase with conflict resolution. If that work still matters, say so before
the rewrite and it can be merged first or the squash boundary adjusted to preserve
their fork point.
