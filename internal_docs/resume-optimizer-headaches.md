# Resume and the optimizer: every design headache we have hit

Working document for a supervisor discussion. Everything below was verified by reading source
in this working tree on 2026-08-07; each mechanism claim carries a `file:line` citation.

**Repo state these citations were re-anchored against:** `rig` at `5f350b1`, plus the uncommitted
working-tree diff (fork-aware refusal hints, resume-flag validation, persisted job numbers). Two
commits landed *after* the first draft of this document — `7915ba0` (one row per leaf, lineage
server-derived, mid-chain delete refused) and `b336138` (**resume is sticks-only**) — so H8 below
has been updated from "undecided" to "shipped, and superseded by a further proposal". Every
`makermodslab/jobs.py` and `frontend/src/components/jobs/resumeSeed.ts` line number in this
document has been re-verified against that state; the `lerobot/`, `torch/`, `makermodslab/train.py`
and `frontend/src/components/training/` citations were unaffected by those commits.

**Path conventions.** Repo-relative paths (`makermodslab/…`, `frontend/src/…`) are MakerMods Lab.
Paths written `lerobot/…` are the pinned upstream release, installed at
`.venv/lib/python3.12/site-packages/lerobot/` (lerobot is pinned to the `v0.6.0` release tag in
`pyproject.toml`). `torch/…` likewise resolves under `.venv/lib/python3.12/site-packages/`.

---

## Findings first

1. **Our app pins optimizer settings on resume; upstream does not.** MakerMods Lab disables the whole
   "Optimizer and LR" pane when a resume seed is present. Upstream lerobot documents the opposite
   policy — the checkpoint's config wins *but* "CLI `--*` flags still override"
   (`lerobot/configs/train.py:87-91`).

2. **Our lock is nevertheless honest, because a base-LR override on resume would evaporate.**
   The trainer builds the optimizer and scheduler from the (possibly overridden) config, then
   *overwrites* the LR from the checkpoint: `optimizer.load_state_dict` restores `param_groups`
   including `lr` and `initial_lr` (`lerobot/optim/optimizers.py:360-367`), and
   `LambdaLR.load_state_dict` restores `base_lrs` and `last_epoch`
   (`torch/optim/lr_scheduler.py:431-432`). Every scheduled step then recomputes
   `lr = restored base_lr × λ(restored last_epoch)` (`torch/optim/lr_scheduler.py:464-467`).
   An "editable LR on resume" control would lie.

3. **But the *shape* of the schedule is not pinned — and the two override namespaces swap
   effectiveness between fresh runs and resumes.** `LambdaLR.state_dict` refuses to serialize plain
   function lambdas (`torch/optim/lr_scheduler.py:412-417`), so the λ closures are always rebuilt
   from the *current* config. On a fresh run the preset overwrites `cfg.optimizer`/`cfg.scheduler`,
   so `--scheduler.*` is dead and `--policy.scheduler_*` is the live lever. On a resume the preset
   re-derivation is skipped (`lerobot/configs/train.py:251-253`), which reverses it exactly:
   `--policy.scheduler_*` becomes dead and `--scheduler.*` becomes live. Nothing warns about
   either. This is the sharpest upstream-issue candidate in the document.

4. **The auto-scale wrinkle is reachable in our product today, through a control we deliberately
   left editable.** SmolVLA/π0's cosine schedule compresses itself when
   `num_training_steps < num_decay_steps` (`lerobot/optim/schedulers.py:149-161`), and `--steps`
   *is* forwarded on our resume branch (`makermodslab/train.py:464`) and *is* editable in the Run
   pane (`frontend/src/components/training/config/RunPane.tsx:30-36`). Editing steps on a SmolVLA
   resume therefore rebuilds λ against a different horizon while `last_epoch` is restored — a
   silent LR discontinuity at the resume step. Worked example below: **9.07e-5 → 3.62e-5, a 2.5×
   drop, in one step, with no log line.**

5. **Done-run resume is refused on both sides**, because a finished run's schedule is spent and the
   continuation would train at the 2.5e-6 floor while its flat loss reads as convergence
   (`makermodslab/jobs.py:2816-2836`, `frontend/src/components/jobs/resumeSeed.ts:104-120`).
   Fine-tune is the sanctioned fork. **Gap found:** the same floor-LR outcome is reachable on an
   *interrupted* run by raising the steps target past `num_decay_steps`, and nothing refuses or
   warns about that.

6. **Correction to a briefed claim — resume in this pin is sample-exact, not stochastically
   divergent.** The brief assumed "data order differs after resume, so a transient failure often
   doesn't recur." In v0.6.0 the sampler order is a pure function of `(seed, epoch)` and the resume
   offset is computed from the step (`lerobot/scripts/lerobot_train.py:415-418, 449-455`), and the
   RNG state is restored too (`lerobot/common/train_utils.py:228`). A *deterministic* bad batch will
   replay identically. Details and the surviving escape hatches in H6.

7. **Cloud checkpoint attribution is genuinely missing.** A cloud resume chain shares one Hub output
   repo and a checkpoint carries nothing naming the run that wrote it, so a forked sibling's
   checkpoints appear in every relative's listing. We ship a provably-sound partial mitigation
   (`cloudSiblingStepCap`, `frontend/src/components/jobs/resumeSeed.ts:122-166`); the residual —
   an early-forked sibling inside the leaf's own step range — needs per-run attribution, which is a
   backend/identity change.

8. **Sticks-only shipped (`b336138`), enforced by refusal — and the user has since declared a
   different future direction: absorption.** Today a second resume off one parent is refused at
   creation (`JobAlreadyContinuedError`, `makermodslab/jobs.py:2347-2372`), mid-chain deletes are
   refused on both delete routes (`makermodslab/jobs.py:3788-3795`,
   `makermodslab/server.py:1917-1927`, `makermodslab/models.py:1368-1379`), and the data model stays
   forest-*capable* for legacy registries (`makermodslab/jobs.py:144-151`). The proposal in H8.1 —
   **when B resumes A, A's record is deleted and its checkpoints and history transfer to B** — makes
   sticks structural instead of refusal-enforced. H8.1 lists what it dissolves and what it costs.

9. **A second briefed claim needed correction (minor):** the in-code note at
   `frontend/src/components/training/types.ts:165-171` says the form "does not prefill" the locked
   optimizer controls from the parent's config. It does — `buildResumeSeed` carries
   `optimizerLr`/`optimizerType`/`optimizerWeightDecay`/`optimizerGradClipNorm`
   (`frontend/src/components/jobs/resumeSeed.ts:412-415`) and `useTrainingConfig` reads them into
   the form (`frontend/src/hooks/useTrainingConfig.ts:111-114`). The comment is stale; the lock
   itself is correct.

---

## Background: the four layers a resumed LR passes through

Worth twenty lines before the headaches, because every one of them lives in the seam between two of
these layers.

**Layer 1 — config assembly.** `parser.wrap` sees `--config_path` and routes the whole parse through
`TrainPipelineConfig.from_pretrained(config_path, cli_args=…)`
(`lerobot/configs/parser.py:309-311`), which ends in
`draccus.parse(cls, config_file, args=cli_args)` (`lerobot/configs/train.py:344-345`). So the
checkpoint's `train_config.json` is the base and CLI flags are merged on top of it. Then
`validate()` runs, and its optimizer/scheduler clause reads:

```python
elif self.use_policy_training_preset and not self.resume:
    self.optimizer = active_cfg.get_optimizer_preset()
    self.scheduler = active_cfg.get_scheduler_preset()
```

(`lerobot/configs/train.py:251-253`.) On a *fresh* run this discards whatever `--optimizer.*` /
`--scheduler.*` said and rebuilds both from policy fields. On a *resume* it does not run at all, so
`cfg.optimizer` / `cfg.scheduler` are whatever the checkpoint saved, plus any direct
`--optimizer.*` / `--scheduler.*` override.

**Layer 2 — construction.** `make_optimizer_and_scheduler` builds the optimizer from `cfg.optimizer`
and the scheduler from `cfg.scheduler` **with `cfg.steps` as `num_training_steps`**
(`lerobot/optim/factory.py:40-41`). This is where the resume-editable steps target enters the LR
schedule. `LambdaLR.__init__` with `last_epoch=-1` then stamps `initial_lr` onto each param group
and copies it into `base_lrs` (`torch/optim/lr_scheduler.py:130-147`).

**Layer 3 — state restore.** `load_training_state` restores RNG, step, optimizer and scheduler in
that order (`lerobot/common/train_utils.py:224-235`), called immediately after construction
(`lerobot/scripts/lerobot_train.py:366, 389-391`). The optimizer's `param_groups` — `lr`,
`initial_lr`, betas, weight decay — are deserialized from the checkpoint's JSON and loaded
(`lerobot/optim/optimizers.py:360-367`). The scheduler's `base_lrs`, `last_epoch` and `_step_count`
are restored by `self.__dict__.update(state_dict)` (`torch/optim/lr_scheduler.py:431-432`); the
λ list is *not*, because `state_dict()` writes `None` for anything that is a plain
`types.FunctionType` (`torch/optim/lr_scheduler.py:412-417`).

**Layer 4 — the loop.** `for _ in range(step, cfg.steps)` (`lerobot/scripts/lerobot_train.py:566`),
with `lr_scheduler.step()` after every optimizer step (`lerobot/scripts/lerobot_train.py:161-162`)
and the logged LR read straight off `optimizer.param_groups[0]["lr"]`
(`lerobot/scripts/lerobot_train.py:170`).

The one sentence that explains almost everything below: **numbers are restored, functions are
rebuilt.**

---

## H1 — Resume pins settings, by our choice, not upstream's

**What happens.** Open Train via *Continue* on an interrupted run and the "Optimizer and LR" segment
renders with an "inherited" chip and every control disabled
(`frontend/src/components/training/TrainingConfigurator.tsx:340-374`; `resumeLocked` is simply
`resumeSeed != null`, `frontend/src/hooks/useTrainingConfig.ts:301`). Policy, model name, batch
size, seed and AMP lock too; steps, log/save cadence, worker count, hardware and the cloud timeout
stay editable (`frontend/src/components/training/config/RunPane.tsx:7-19` documents the split).

**Why.** A deliberate product decision, explained to the user in one sentence at
`frontend/src/components/training/types.ts:172-174`: "Rebuilt from the parent run's checkpoint — a
resume continues the same experiment, so changing these here has no effect. To train with different
settings, fine-tune from this checkpoint instead."

Upstream's stated policy is the opposite. `TrainPipelineConfig.resume`'s own docstring says the
checkpoint config is used "regardless of what's provided with the training command at the time of
resumption (CLI `--*` flags still override)" (`lerobot/configs/train.py:87-91`). Read literally,
that invites a user to `--policy.optimizer_lr 3e-5` on a resume and expect it to take.

**Consequence.** We are stricter than upstream, and a user reading lerobot's docs will believe our
UI is missing a feature. Worth knowing that it is a defensible strictness, not an oversight — H2 is
the reason.

**Candidate fixes.** (a) Keep the lock, add one line of copy citing the mechanism. (b) Keep the lock
but make the *inherited values* visible-and-labelled rather than merely greyed (they are already
prefilled — see finding 9). (c) Unlock, and implement a real mid-run LR change (H2's "candidate
fixes").

---

## H2 — A base-LR override on resume would be silently ineffective anyway

**What happens.** Suppose we unlocked the LR field and passed the user's value through. The run
would train at the checkpoint's LR, not the user's, and nothing would say so — the logged
`train/lr` would report the old value, so even the chart would not betray it.

**Why.** Trace the four layers with an override in hand.

* `--optimizer.lr 3e-5` survives `validate()` on a resume (the preset clause is skipped,
  `lerobot/configs/train.py:251-253`), so the optimizer is genuinely *built* at 3e-5, and the
  scheduler's `base_lrs` is 3e-5 too (`torch/optim/lr_scheduler.py:130-147`).
* Then `load_training_state` runs (`lerobot/scripts/lerobot_train.py:389-391`).
  `_load_single_optimizer_state` deserializes the saved `param_groups` — which include both `lr` and
  the `initial_lr` the scheduler stamped in the *parent* run — and calls
  `optimizer.load_state_dict` (`lerobot/optim/optimizers.py:360-367`; the save side is
  `lerobot/optim/optimizers.py:310-318`). The user's 3e-5 is gone from the param groups.
* Then `load_scheduler_state` (`lerobot/optim/schedulers.py:190-193`) calls
  `LambdaLR.load_state_dict`, whose `self.__dict__.update(state_dict)`
  (`torch/optim/lr_scheduler.py:431-432`) replaces `base_lrs` with the parent's.
* From the first `lr_scheduler.step()` onward, `get_lr` returns
  `base_lr * λ(last_epoch)` (`torch/optim/lr_scheduler.py:464-467`) — with the parent's `base_lr`.

`--policy.optimizer_lr` is even further from working on a resume: it edits the policy config, and
the preset that would translate policy fields into `cfg.optimizer` is the very clause that resume
skips. (On *fresh* runs the situation inverts, which is why our command builder emits
`--policy.optimizer_lr` and not `--optimizer.lr`; the reasoning is written up at
`makermodslab/train.py:85-98`.)

For completeness: our resume branch emits neither. It passes only `--config_path`, `--resume`,
`--output_dir`, `--steps`, `--num_workers`, `--log_freq`, `--save_freq`, `--save_checkpoint`,
the Hub visibility flags and `--job_name`, then returns (`makermodslab/train.py:457-490`). The
optimizer values are still *stored* on the job record, so `JobRecord.config` keeps describing the
run's real shape — the lock is a display change, not a payload change
(`frontend/src/hooks/useTrainingConfig.ts:69-77`).

**Consequence.** The lock is honest. Any UI we build that appears to let the user set LR on a resume
would be a lie unless we also change the restore behaviour.

**Candidate fixes.**

* *Do nothing.* The current state is correct; it just isn't explained.
* *Post-restore re-application.* After `load_training_state`, write the requested LR into
  `param_groups[i]["lr"]` **and** `["initial_lr"]`, and into `scheduler.base_lrs`. Mechanically this
  is about six lines. The hard part is not the code: it is deciding what "change the LR mid-run"
  should mean when a schedule is also restored. Does the new LR become the peak of the *remaining*
  schedule (rescale `base_lrs`, keep `last_epoch`)? Does it become a flat floor (drop the
  scheduler)? Does it restart warmup? Those are three different experiments, and picking one
  silently is how you get the "same loss curve, different policy" class of bug.
* *Upstream issue.* See H3 — the override-looks-supported-but-evaporates story is arguably an
  upstream documentation/design bug, and item 2 is half of it.

---

## H3 — The schedule's shape is not pinned, and the live override namespace flips on resume

This is the asymmetry that makes the whole area confusing, and it has two halves.

**Half one: numbers restore, functions rebuild.** `LambdaLR.state_dict()` explicitly refuses to
serialize plain functions:

```python
state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)
for idx, fn in enumerate(self.lr_lambdas):
    if not isinstance(fn, types.FunctionType):
        state_dict["lr_lambdas"][idx] = fn.__dict__.copy()
```

(`torch/optim/lr_scheduler.py:412-417`; the docstring at :402-403 states the rule outright.) Every
lerobot scheduler builds its λ as a nested `def` closing over the config — see
`lerobot/optim/schedulers.py:163-182` for the cosine one, :119-127 and :100-105 for the others — so
every one of them is a `types.FunctionType` and every one of them is saved as `None`.
`load_state_dict` pops that list and updates nothing (`torch/optim/lr_scheduler.py:431-439`).

So on resume: `base_lrs` and `last_epoch` come from the checkpoint, and **λ comes from whatever
config the current process assembled**. Change a scheduler parameter and the curve moves under a
fixed abscissa.

**Half two: which flags reach λ inverts between fresh runs and resumes.**

| | fresh run (`use_policy_training_preset=true`) | resume |
|---|---|---|
| `--policy.scheduler_warmup_steps` etc. | **effective** — the preset rebuilds `cfg.scheduler` from policy fields (`lerobot/configs/train.py:251-253`) | **inert** — the preset clause is skipped, `cfg.scheduler` stays as saved |
| `--scheduler.num_decay_steps` etc. | **inert** — overwritten by the preset | **effective** — merged into the saved `cfg.scheduler`, which is then used verbatim |
| `--steps` | effective | **effective** (see H4) |

Both cells labelled "inert" fail silently: draccus accepts the flag, the config carries it, nothing
warns, and only the LR curve knows. Our own command builder is written entirely against the
fresh-run column (`makermodslab/train.py:129-163` documents the `--policy.scheduler_*` table and
why `--scheduler.*` is useless there) — which is correct for the code path it serves, and would be
exactly wrong if someone extended it to the resume branch by analogy.

One thing does fail loudly rather than silently: lerobot's scheduler-state loader deserializes into
a *template* taken from the live scheduler's own `state_dict()` and demands an exact key match
(`lerobot/optim/schedulers.py:190-193` → `lerobot/utils/io_utils.py:110-140`). Changing the
scheduler *type* on a resume would therefore raise rather than half-apply. Changing its
*parameters* keeps the key set identical, so it goes through silently. The loud case is the one we
would never hit; the silent case is the one a user can reach.

**Consequence.** "Resume inherits the run's settings" is true of the optimizer and false of the
schedule. Anyone reasoning from lerobot's `resume` docstring will get this backwards in whichever
direction they happen to test first.

**Candidate fixes.** Upstream issue (see the closing options) is the honest home for this. Locally:
never emit scheduler flags on the resume branch (we currently don't), and add a regression test
pinning that.

---

## H4 — The auto-scale wrinkle: reachable in our product today

**What happens.** A SmolVLA (or π0, π0.5, π0-FAST, EO1, …) run is interrupted. The user clicks
*Continue*, and — reasonably — lowers the steps target because they only want a short tail. The
continuation resumes at the right step with the right weights and the right optimizer moments, and
its learning rate silently jumps to a different point on a *differently shaped* curve.

**Why.** `CosineDecayWithWarmupSchedulerConfig.build` rescales itself when the horizon is shorter
than the configured decay:

```python
if num_training_steps < self.num_decay_steps:
    scale_factor = num_training_steps / self.num_decay_steps
    actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
    actual_decay_steps = num_training_steps
```

(`lerobot/optim/schedulers.py:149-154`; there *is* a `logging.info` at :155-161, which is the only
signal anywhere in this headache and is invisible in our UI.) `num_training_steps` is `cfg.steps`
(`lerobot/optim/factory.py:41`), `--steps` is forwarded on our resume branch
(`makermodslab/train.py:464` — with a comment explaining it must be raised above the resumed step
for the loop to do work), and the Run pane leaves it editable on resume while disabling batch size,
seed and AMP beside it (`frontend/src/components/training/config/RunPane.tsx:29-36` vs :39-54).
Combined with H3's "λ is rebuilt," a changed steps target changes the curve while `last_epoch`
stays where the checkpoint put it.

**Worked example.** SmolVLA preset: `optimizer_lr=1e-4`, `scheduler_warmup_steps=1000`,
`scheduler_decay_steps=30_000`, `scheduler_decay_lr=2.5e-6`
(`lerobot/policies/smolvla/configuration_smolvla.py:74-82, 141-147`), so `alpha = 0.025`.

* Parent run: `steps=30_000`. No auto-scale (30 000 is not `< 30 000`), so `actual_decay = 30_000`.
  It dies at step 6 000, where `λ = 0.975 · ½(1 + cos(π·0.2)) + 0.025 = 0.907` → **LR ≈ 9.07e-5**.
* Resume, steps lowered to 10 000. Now `10 000 < 30 000` → `scale = 1/3`, `actual_warmup = 333`,
  `actual_decay = 10 000`. `last_epoch` is restored as 6 000, so
  `λ = 0.975 · ½(1 + cos(π·0.6)) + 0.025 = 0.362` → **LR ≈ 3.62e-5**.

A 2.5× LR drop at the resume step, with no error, no warning in our UI, and a loss curve that will
simply look a bit flatter than the operator expected.

The discontinuity is not one-directional. Parent `steps=10_000` (auto-scaled decay to 10 000),
died at 6 000 (λ = 0.362), resumed with steps raised to 20 000 → decay becomes 20 000 and λ at
6 000 is 0.799: **an LR jump *up* by 2.2×**. And because warmup is rescaled by the same factor, a
run that died *inside* warmup can resume either at a different point in warmup or past the end of a
shrunken warmup entirely.

**Which policies are affected.** Only schedule-bearing ones. ACT and TDMPC return `None` from
`get_scheduler_preset()` (`lerobot/policies/act/configuration_act.py:159-160`), so there is no
scheduler and no discontinuity — the restored `param_groups["lr"]` simply carries on. The exposure
is SmolVLA, π0, π0-FAST (cosine-decay-with-warmup, auto-scaling), plus diffusion and VQ-BeT
(cosine schedules parameterized on `num_training_steps`, `lerobot/optim/schedulers.py:70-83` and
:56-60 — no auto-scale, but still rebuilt against the new horizon, so the same class of
discontinuity applies).

**Consequence.** The one Run-pane control we deliberately left editable on resume is the one that
silently reshapes the LR schedule for exactly the large-VLA policies where LR schedule matters most.
This is the headache most likely to have already produced a confusing experiment.

**Candidate fixes.**

* Lock the steps target on resume for schedule-bearing policies, and offer the parent's target as
  the only value. Costs the legitimate "I want to train further than originally planned" flow.
* Keep it editable, but warn on edit: "changing the target on a SmolVLA resume rescales the LR
  schedule; the LR at step 6 000 moves from 9.1e-5 to 3.6e-5." We already compute everything needed
  to render that number.
* Allow raising but not lowering (raising past `num_decay_steps` is a *different* problem — see
  H5's gap).
* Surface lerobot's own auto-scale `logging.info` in the run log view, so at minimum it is
  discoverable after the fact.

---

## H5 — Done-run resume refusal, and the spent schedule

**What happens.** A run that reached its step target cannot be resumed. The button is absent in the
UI, and a direct API call gets a 400.

**Why.** Both halves are implemented, with the same reasoning written at each.
Backend: `makermodslab/jobs.py:2816-2836` refuses `source.state == "done"` on any runner, with the
comment spelling out the mechanism — "SmolVLA's preset cosine-decays to a 2.5e-6 floor over a fixed
30k-step horizon, so continuing past the target trains at floor LR. The loss chart flattens and
reads as convergence while the run is barely learning — the failure is silent, which is why this
refuses rather than warns." Frontend: `isResumableLeaf` admits only `failed` and `interrupted`
leaves (`frontend/src/components/jobs/resumeSeed.ts:115-120`), and `resumableCheckpoints` also drops
any candidate whose *owner* is done (`…/resumeSeed.ts:226`) so the UI never offers a checkpoint the
backend will reject.

The floor is real: `cosine_decay_schedule` clamps with `step = min(current_step, actual_decay_steps)`
and bottoms out at `alpha = decay_lr / peak_lr` (`lerobot/optim/schedulers.py:170-175`), i.e.
2.5e-6 for SmolVLA's preset.

**Consequence.** Correct and well-motivated. Fine-tune (a fresh run seeded from the checkpoint's
weights, `--policy.pretrained_path`, `makermodslab/train.py:503-509`) is the sanctioned way to build
on a finished run, and fine-tune is deliberately *not* a lineage edge
(`makermodslab/jobs.py:155-158`).

**Gap found while verifying this.** The refusal is keyed on run *state*, but the failure mode is
keyed on *step vs. `num_decay_steps`*, and those come apart. Take an interrupted SmolVLA run with
`decay_steps = 30_000` that died at 25 000; the user resumes and raises the target to 60 000.
Steps 25 000–30 000 finish the cosine, and steps 30 000–60 000 run at the 2.5e-6 floor —
35 000 steps of the exact "trains at floor LR while the loss curve reads as convergence" failure
the done-run refusal exists to prevent. Nothing refuses it and nothing warns. The only steps
validation on the resume path is a UI-side lower bound (`config.steps` must exceed the checkpoint
step, `frontend/src/hooks/useTrainingConfig.ts:289-294`); I found no backend equivalent and no upper
bound anywhere.

**Candidate fixes.** Extend the H4 warning to cover "target exceeds the schedule's decay horizon —
the tail will train at `decay_lr`," and consider a backend guard mirroring the UI's lower bound so
non-UI callers cannot launch a zero-work resume (`range(step, cfg.steps)` with
`cfg.steps <= step` is an empty loop, `lerobot/scripts/lerobot_train.py:566`).

---

## H6 — What resume-from-a-non-latest-checkpoint is actually for

With settings pinned (H1/H2), choosing an older checkpoint is not an experimental knob. It is purely
a question of *which saved state to trust*. Three reasons survive scrutiny; one briefed reason does
not.

**(a) A damaged or partial newest checkpoint — real and defended.** Cloud checkpoints are uploaded
asynchronously, and a run that dies takes its upload backlog with it, leaving the newest step dirs
present-but-incomplete or absent entirely. We validate before renting a GPU:
`hub_checkpoint_missing_files` reads the repo listing without downloading bytes
(`makermodslab/jobs.py:984-1007`), and `_resolve_cloud_resume` refuses an incomplete one by name —
"incomplete on the Hub (a known uploader race) — missing …" (`makermodslab/jobs.py:1065-1070`),
with the shared remedy string "Resume an earlier checkpoint, or fine-tune from its weights instead."
(`makermodslab/jobs.py:975-977`). The completeness rule is weights plus
`training_state/training_step.json` plus `optimizer_state.safetensors`
(`makermodslab/jobs.py:946-972`) — i.e. exactly "a resume without the optimizer state is a
fine-tune wearing a resume label" (`makermodslab/jobs.py:1636`).
*Note:* the specific incident cited in the brief (a cloud run that stopped at step 80 with only
10/20/30 on the Hub) is not written down anywhere I could find in the repo or `internal_docs/`; the
mechanism is fully substantiated by the code above, the individual incident rests on session memory.

*Post-sticks change to this case:* a leaf may now only resume checkpoints owned by a *childless*
run (`frontend/src/components/jobs/resumeSeed.ts:227`), so "fall back to an earlier checkpoint" is
now restricted to the leaf's **own** earlier checkpoints. Reaching an ancestor's requires deleting
the run in front of it. This is exactly the shape H8.1's absorption proposal removes.

**(b) Transient-failure replay — CORRECTION, this mostly does not work in this pin.** The brief's
premise was that data order differs after a resume, so re-running from the same or an earlier
checkpoint dodges a stochastic failure. In lerobot v0.6.0 that is false by design:

* the sampler order is "a pure function of `(seed, epoch)` … and resume is sample-exact"
  (`lerobot/scripts/lerobot_train.py:415-418`);
* the resume offset is *computed* from the step and loaded into the sampler, logging "Resuming data
  order at epoch N, sample M" (`lerobot/scripts/lerobot_train.py:449-455`);
* `load_rng_state` restores Python/NumPy/Torch RNG before anything else
  (`lerobot/common/train_utils.py:228` → `lerobot/utils/random_utils.py:136-139`), so dropout and
  noise replay too.

So a deterministic bad batch (a corrupt sample, a NaN-producing input) **will recur at the same
global step**, from the latest checkpoint or an earlier one alike. What genuinely does vary:
GPU non-determinism (`cudnn_deterministic` defaults to `False`,
`lerobot/configs/train.py:97-98`), so numerics-driven NaNs may or may not recur; streaming datasets,
which take the `shuffle=True`/no-sampler branch and carry no order state
(`lerobot/scripts/lerobot_train.py:456-458`); and changing world size or batch size, which lerobot
warns costs sample-exactness (`lerobot/scripts/lerobot_train.py:437-448`) — though our UI locks
batch size on resume, so that lever is not available to a user anyway.
**Recommendation: do not use "different data order" as a justification in the supervisor
conversation.** The honest version is "an earlier checkpoint is how you escape a *corrupted* state,
not a *stochastic* one."

**(c) Cloud sibling contamination.** The newest checkpoint in the listing may not belong to this run
at all. That is H7.

**(d) Everything else routes to fine-tune.** Any resume-from-older-checkpoint motivated by wanting
different settings is, given H1/H2, a fine-tune wearing a resume label — which is precisely how the
product is set up to answer it.

---

## H7 — Cloud checkpoint attribution gap

**What happens.** A cloud run is resumed twice off the same parent (two children, one parent — the
data model permits it, H8). Both continuations publish into the *same* Hub output repo as the
parent. Sibling B trains on to 30 000; sibling A dies at 6 000. Opening A's resume picker offers
B's 30 000-step checkpoint as "the newest checkpoint" for A — a step A never trained, from weights A
never had.

**Why.** A cloud resume keeps pushing into the parent's repo deliberately, so the whole lineage
lives in one place (`makermodslab/train.py:469-474`), and the checkpoint tree carries nothing that
names the writing run: the ref shape is `repo@checkpoints/<zero-padded step>` and nothing else
(`makermodslab/jobs.py:1060-1064`). An ancestor walk cannot exclude a sibling the way it excludes an
unrelated run, because the sibling's bytes are *in the ancestor's repo*
(`frontend/src/components/jobs/resumeSeed.ts:72-83`).

**What we shipped.** `cloudSiblingStepCap` (`frontend/src/components/jobs/resumeSeed.ts:122-166`):
for a cloud leaf, drop every candidate strictly above the leaf's own furthest step. The argument is
a provable one — the leaf never trained past its own furthest step, so anything above it cannot lie
on the leaf's path — so the cap excludes exactly the provably-foreign steps and nothing else. It is
applied inside the single resume rule both entry points share
(`frontend/src/components/jobs/resumeSeed.ts:216-232`), with a dedicated
`"sibling-cap"` explanation for the empty case (`…/resumeSeed.ts:327`, `noResumeReason` at :330).

Two design details in that comment are worth putting in front of a supervisor because they were
argued and decided rather than defaulted:

* *Leaf-keyed vs owner-keyed.* The contaminating listing belongs to the checkpoint's **owner**, not
  to the leaf, so a *local* leaf resumed from a *cloud* parent enumerates the shared repo with no
  cap applied. Re-keying on `entry.job.runner === "hf_cloud"` would close that — but would also hide
  a superseded ancestor's above-step checkpoints, which have no other access point in the UI. It was
  reviewed and deliberately kept leaf-keyed: a real loss traded against a rarer and recoverable one,
  and the simpler rule is the predictable one (`…/resumeSeed.ts:141-153`).
* *Why `metrics.current_step` is the right reading.* It is not merely "how far it got before dying":
  a resumed run's metrics are **seeded** at its inherited checkpoint step
  (`_initial_metrics`, `makermodslab/jobs.py:435-452`, wired at `makermodslab/jobs.py:2992`), so
  even a leaf that died before its first tqdm frame reports a step at or above its inherited floor.
  Without that seeding the cap would exclude the ancestor checkpoints a leaf is most often resumed
  from. A zero means "never reported" — unknown, not step zero — so no cap is applied
  (`…/resumeSeed.ts:155-166`). The seeding was itself a bug fix: before it, a resumed run showed
  "0 / 60 · 0.0%" for the 12 s–several minutes before lerobot's first tqdm frame, and the
  monitoring chart, which treats a backwards step as a new run, wiped the inherited loss curve on
  mount (`makermodslab/jobs.py:441-448`).

**Residual.** A sibling that forked *early* wrote its checkpoints inside the leaf's own step range,
where nothing at this call site can tell them apart. Stated as a known limit in two places
(`…/resumeSeed.ts:72-83`, :136-139).

**Consequence.** The user can be handed someone else's weights, labelled with a plausible step, and
resume "successfully" — the worst kind of silent failure, because every downstream artefact looks
normal. The cap removes the common shape (the sibling that ran further) and leaves the rare one.

**Candidate fix.** Per-run attribution of Hub checkpoints. Concretely: write a run identifier into
each checkpoint upload (a small marker file under `checkpoints/<step>/`, or a per-run subtree), and
filter the listing on it. This is a backend/identity change and touches the HF Jobs wrapper as well
as the host-side uploader; it is the only thing that fully closes the gap.

---

## H8 — Forest vs sticks (decided one way, then reframed)

**What exists today, after `b336138`.** The lineage edge is `config.resume_from_job_id` and nothing
else (`makermodslab/jobs.py:2259-2266`); fine-tune is deliberately not an edge, since it starts a
fresh optimizer and schedule and is therefore a new model rather than a continuation
(`makermodslab/jobs.py:155-158`).

Sticks are now enforced at **creation**: a resume whose source already has a child raises
`JobAlreadyContinuedError` (`makermodslab/jobs.py:2858-2860`, class at :2347-2372), mapped to a 409
(`makermodslab/server.py:1357`). The frontend states the same rule at the other end — a candidate
checkpoint's owner must have no children (`frontend/src/components/jobs/resumeSeed.ts:227`, reasoned
at :185-203) — so the Resume button is absent rather than erroring on click.

Two properties of that choice are worth naming, because they were argued rather than defaulted:

* **Refusal, not abolition.** The check is at start time only, never at load or list time, so
  registries written before the rule keep their forks and keep rendering unchanged; `child_ids`
  stays a list and every reader stays forest-capable. "Treat 'several children' as legacy-only
  data, not as an impossible state" (`makermodslab/jobs.py:144-151`).
* **The remedy message is fork-aware.** A single unfinished continuation gets "delete it, then
  resume"; a legacy fork, or a continuation that ran to completion, gets "fine-tune instead" —
  because telling a user to delete a finished 30k run is advice to throw away work
  (`makermodslab/server.py:1364-1389`; the live registry hit exactly that case).

Mid-chain deletes remain refused rather than cascaded, on both routes that can reach them: the jobs
route (`makermodslab/server.py:1917-1927`) and the model-library route
(`makermodslab/models.py:1368-1379`), both 409 and both naming the continuations. The registry
raises `JobHasChildrenError` (`makermodslab/jobs.py:3788-3795`), whose docstring gives the reason:
an orphaned subtree keeps a `resume_from_job_id` pointing at nothing, its lineage walk truncates,
the inherited loss history and fallback checkpoints are gone — and for a local parent the delete
also wipes the on-disk checkpoint directory the children resumed *out of*
(`makermodslab/jobs.py:2327-2339`).

The UI half is unchanged: a run with children is superseded and hidden, one row per leaf, the rest
of the chain reached through `ancestor_ids`
(`frontend/src/components/jobs/JobsDataContext.tsx:352-361`), with checkpoint counts and the
active/leftover split computed across the whole chain (`…/JobsDataContext.tsx:372-393`).

**The cost this bought, stated plainly.** A tip that died before saving anything now has *nothing*
to resume, where it used to inherit its parent's checkpoints; the way out is to delete the empty tip,
which frees the parent (`frontend/src/components/jobs/resumeSeed.ts:198-203`, with a dedicated
`"parent-continued"` reason at :325 so the toast can say so). That delete sits directly in the retry
path of the product's most common failure. The forest's genuine use — two continuations off one
good checkpoint — is deferred, with fine-tune offered as the non-equivalent substitute (it resets
the optimizer and the schedule).

### H8.1 — Declared future direction: absorption semantics

**The user's proposal, as stated:** when run B resumes run A, **A's record is deleted and its
checkpoints *and* history transfer to B**. B owns the whole stick. A cannot be resumed twice because
A no longer exists. A B that fails before saving anything of its own is still resumable, because the
inherited checkpoints are now *B's own*. The next continuation is created off B.

Sticks stop being a rule that refuses things and become the shape of the data. This is presented
here as the declared direction, not as an implemented design.

**(a) What it dissolves.** Each of these exists today only because a parent record survives its
continuation:

* **The second-resume refusal.** `JobAlreadyContinuedError` and its 409
  (`makermodslab/jobs.py:2347-2372`, raised at :2858-2860; `makermodslab/server.py:1357`) has nothing to fire
  on — there is no continued run left to re-continue.
* **Delete-first guidance.** "Delete the continuation, then resume the parent"
  (`makermodslab/server.py:1382`) describes a manoeuvre that absorption performs automatically, at
  resume time, without asking the user to destroy anything.
* **Fork-aware message branching.** The two-shape remedy — cheap-delete vs. don't-throw-away-a-
  finished-run — and the `_is_finished_run` probe behind it (`makermodslab/server.py:1364-1389`)
  become dead code.
* **Inner-run invisibility.** The superseded/leaf split
  (`frontend/src/components/jobs/JobsDataContext.tsx:352-361`) stops being a *hiding* rule: there
  are no inner runs to hide. One record, one row, by construction.
* **The continued-run fine-tune gap.** Today `deployableModels` requires `child_ids.length === 0`
  (`frontend/src/components/jobs/JobsDataContext.tsx:432-439`), so an inner run's checkpoints cannot
  be fine-tuned from the Models library even though the backend would allow it. Under absorption
  those checkpoints belong to B, whose row is visible, so the gap closes without a special case.
* **Lineage walking and chain counting.** `ancestor_ids_of` (`makermodslab/jobs.py:2287-2312`), the
  ancestor walk in `read_metrics_history` (`makermodslab/jobs.py:3682-3690`), the client-side
  ancestor backfill, `chainCheckpointCount` and `ancestorsPending`
  (`frontend/src/components/jobs/JobsDataContext.tsx:372-393`), and the multi-job fan-out in
  `loadLineageCheckpoints` (`frontend/src/components/jobs/resumeSeed.ts:84-102`) all collapse into
  reading one record. That is a substantial simplification: chain-awareness is currently threaded
  through the jobs list, the resume rule, the metrics endpoint and the delete guards.

It would also shrink H7 further: with one record per stick, a cloud chain's shared Hub repo has one
surviving claimant, so "whose checkpoint is this?" has a trivially correct answer for everything the
new rule creates — though see cost 4 for what it cannot fix retroactively.

**(b) Costs to settle before building.**

1. **History must transfer, or the pre-resume loss curve vanishes — verified.**
   `read_metrics_history` reconstructs a continuous curve by walking `resume_from_job_id` oldest-
   first and concatenating each run's points (`makermodslab/jobs.py:3667-3699`), and it reads each
   ancestor's **own** log file, keyed by that ancestor's id:
   `_read_log_metrics(_job_log_path(self._output_root, record.id), …)`
   (`makermodslab/jobs.py:3696-3697`). Its stated termination is decisive here — "Stops at a missing
   ancestor (a deleted source) — the curve just starts later" (`makermodslab/jobs.py:3672-3673`).
   So absorbing A by deleting its record, without moving its log, deletes the entire pre-resume
   curve; and `JobRegistry.delete` also `shutil.rmtree`s the job dir the log lives in
   (`makermodslab/jobs.py:3802`). Two sub-decisions follow:
   * *What moves.* The log file itself, or a pre-parsed point series folded into B's record?
   * *Rebasing.* The per-record call passes `_resume_total_steps(record.config)`
     (`makermodslab/jobs.py:3697`), which is `config.steps` for a resumed run and `None` for a fresh
     one (`makermodslab/jobs.py:393-396`), because a resumed run's tqdm bar counts only the
     remaining window. A naive `cat A.jsonl >> B.jsonl` would apply **B's** rebasing to **A's**
     lines and corrupt the very curve absorption is meant to preserve. Either keep per-source log
     files with their own rebase parameter, or rebase A's points once at absorb time and store them
     already-global.
2. **`finetune_from_job_id` references to an absorbed run dangle — verified, and milder than it
   sounds.** The field is persisted on every job's config (`makermodslab/train.py:229`) and is read
   in exactly one place that matters: `JobRegistry.start` resolves it against the registry and
   raises `ValueError(f"Fine-tune source {...!r} not found.")` when absent
   (`makermodslab/jobs.py:2710-2717`), which is also the gate `_check_finetune_policy_type` sits
   behind (`makermodslab/jobs.py:2723`). I found **no display reader** — the frontend only plumbs it
   form→request (`frontend/src/components/training/types.ts:31`,
   `frontend/src/hooks/useTrainingConfig.ts:107, 156`, `frontend/src/lib/jobsApi.ts:54`). So the cost
   is (i) provenance loss: a fine-tuned run's record permanently names a source that no longer
   exists, and nothing can answer "what was this fine-tuned from"; (ii) a stale seed (a form left
   open, a deep link) launching into a 400 instead of a run. Neither corrupts data. The decision is
   whether absorption should *rewrite* descendants' `finetune_from_job_id` to point at B — defensible,
   since B now owns A's checkpoints — or leave the dangle and accept the provenance loss.
3. **Validate-before-absorb ordering, and crash recovery for the move window.** Today all resume
   validation is synchronous and happens *before any record exists*, precisely so a bad selection
   leaves nothing behind ("Nothing is registered yet, so a bad selection still fails with no
   orphaned record", `makermodslab/jobs.py:2700-2702`), while the multi-GB byte moves are deferred
   to a preparing thread that runs *after* the record exists
   (`makermodslab/jobs.py:2728-2738`). Absorption inserts a **destructive** step into that sequence,
   so it needs an explicit position and an explicit recovery story: absorb only after B is known
   launchable (else a failed launch has already eaten A); and make the window between "A's record
   deleted" and "A's bytes and history are B's" crash-safe, because a crash inside it loses both
   runs. The natural shape is move-then-delete with an idempotent, resumable completion step at
   registry load — which is new machinery, since nothing in the registry currently needs
   crash-recovery of a partially-applied mutation.
4. **Legacy multi-child forks cannot be absorbed retroactively.** The sticks rule was deliberately
   creation-time-only so existing forks keep working (`makermodslab/jobs.py:144-151`), and the live
   registry has one (two children off one parent, one of them a finished 30k run —
   `frontend/src/components/jobs/resumeSeed.ts:131-134`). Absorption has no defined answer for a
   parent with two children: absorbing into both duplicates the history, absorbing into one picks a
   winner and orphans the other. So the code must keep the forest-capable readers *anyway*, for old
   data — meaning the simplification in (a) is a simplification of the **new** path, not a deletion
   of the old one, unless a migration is also written.
5. **Minor, but decide it explicitly: run numbers.** `job_number` is assigned once from a persisted
   forward-only counter and deliberately never reused, so that two runs a week apart can never both
   be "#46" (`makermodslab/jobs.py:88-98`). Absorption retires A's number silently — the run a user
   has been calling "#45" stops existing and its history is now inside "#46". Does B inherit A's
   number, keep its own, or display a range? This is the user-facing identity of the whole feature,
   so it should not fall out as a side effect.

**Net.** Absorption is a genuine simplification of the model rather than another rule on top of it,
and it directly answers the cost sticks-only just introduced (a 0-checkpoint tip with nothing to
resume). The build is not free: cost 1 is a correctness requirement with a subtle rebasing trap,
cost 3 asks for crash-recovery machinery the registry has never needed, and cost 4 means the
forest-capable code stays regardless.

---

## H9 — Cross-runner resume mechanics (context for everything above)

Not a headache in itself, but the machinery H4/H6/H7 sit on top of.

A continuation may cross runners in **either** direction (F7), and what changes across the four
combinations is only where the parent's checkpoint has to end up before the trainer can read it —
"the pod cannot see this disk, and this disk does not have the pod's"
(`makermodslab/jobs.py:2861-2871`). The four branches:

* **cloud → cloud:** `_resolve_cloud_resume` names `(repo_id, step_dir)` and the HF Jobs wrapper
  downloads pod-side (`makermodslab/jobs.py:2877-2894`).
* **cloud → local:** the same download, done host-side and off-request because it is gigabytes;
  `config_path` is filled in by a preparing thread, and `resume_from_step` is pinned to the resolved
  step so the record's progress reads from the inherited step immediately rather than 0
  (`makermodslab/jobs.py:2895-2905`; the materializer is `download_hub_resume_checkpoint`,
  `makermodslab/jobs.py:1587-1598`).
* **local → local:** `_resolve_resume_config_path` returns the on-disk `train_config.json`
  (`makermodslab/jobs.py:2906-2909`, resolver at :1074-1084).
* **local → cloud:** the checkpoint must reach the Hub first. `_resolve_upload_resume`
  (`makermodslab/jobs.py:3350-3409`) delegates validation wholesale to the local→local resolver so
  the two cannot disagree about "resumable", reuses a previous upload when the Hub still confirms it
  (`:3382-3390`), and refuses without explicit consent — "an upload is a disclosure, so it is never
  a silent side effect of clicking Continue" (`:3369-3370`, enforced at `:3392-3398`). The parent
  record remembers the staging repo and steps (`makermodslab/jobs.py:125-134`) so a second
  cross-runner resume of the same step does not re-push the same gigabytes.

Every branch either points the trainer at bytes that already exist, or records the one move that has
to happen first; none of them can launch a run that quietly starts at step 0 while calling itself a
continuation (`makermodslab/jobs.py:2867-2871`). The owner's runner therefore decides only *where
the bytes must move* and seeds the form's default Compute — it is explicitly **not** a filter on
which checkpoints are offered, because filtering on it left a local leaf with a cloud parent showing
no resume row at all (`frontend/src/components/jobs/resumeSeed.ts:205-214`).

`_initial_metrics` (`makermodslab/jobs.py:435-452`) is load-bearing beyond its own bug fix: it is
what makes H7's sibling cap work for a leaf that died before its first log line.

---

## Summary table

| # | Headache | Mechanism (verified) | Consequence | Status |
|---|---|---|---|---|
| H1 | We pin optimizer settings on resume; upstream invites overrides | `TrainingConfigurator.tsx:340-374`, `useTrainingConfig.ts:301` vs `lerobot/configs/train.py:87-91` | We look stricter than the docs; users think a feature is missing | Shipped, deliberate |
| H2 | A base-LR override on resume evaporates | `optimizers.py:360-367` + `lr_scheduler.py:431-432, 464-467`; preset skipped at `train.py:251-253` | The lock is honest; an editable LR control would lie | Verified; no fix needed unless we build real mid-run LR change |
| H3 | λ is rebuilt, numbers are restored — and the live flag namespace inverts on resume | `lr_scheduler.py:412-417`; `train.py:251-253`; `makermodslab/train.py:129-163` | Scheduler params silently apply on resume while optimizer LR silently doesn't; both directions fail quietly | Open — best upstream-issue candidate |
| H4 | Auto-scale + editable steps ⇒ silent LR discontinuity | `schedulers.py:149-161`, `factory.py:41`, `makermodslab/train.py:464`, `RunPane.tsx:30-36` | 9.07e-5 → 3.62e-5 in one step on a SmolVLA resume; no warning | **Open, reachable today** |
| H5 | Done-run resume refused; schedule is spent | `jobs.py:2816-2836`, `resumeSeed.ts:115-120`, `schedulers.py:170-175` | Correct; fine-tune is the fork | Shipped — but see the raise-steps-past-decay gap |
| H6 | Non-latest checkpoint = which state to trust | `jobs.py:946-972, 984-1007, 1065-1070`; `lerobot_train.py:415-418, 449-455` | Damaged-checkpoint case real; **transient-replay rationale does not hold in this pin** | Partly corrected here |
| H7 | Cloud checkpoints carry no run identity | `jobs.py:1060-1064`, `resumeSeed.ts:72-83, 122-166` | A sibling's weights can be offered as "newest"; cap removes the common shape only | Mitigated; residual open |
| H8 | Sticks-only, enforced by refusal | `jobs.py:144-151, 2347-2372, 2858-2860, 3788-3795`; `server.py:1357, 1364-1389, 1917-1927`; `resumeSeed.ts:227` | Fork deferred; a 0-checkpoint tip must delete-then-resume | **Shipped `b336138`** |
| H8.1 | Absorption: B absorbs A's record, checkpoints and history | dissolves `jobs.py:2287-2312, 3682-3690`, `JobsDataContext.tsx:372-393, 432-439`; costs at `jobs.py:3667-3699, 2710-2717, 2700-2702, 144-151, 88-98` | Sticks become structural; removes the delete-to-retry path | **Declared future direction** |
| H9 | Cross-runner resume moves bytes, not settings | `jobs.py:2861-2909, 3350-3409`, `resumeSeed.ts:205-214` | Context; `_initial_metrics` is load-bearing for H7 | Shipped (F7) |

---

## Decisions needed

Options, not recommendations — each is a real fork with a cost.

1. **Steps on resume, for schedule-bearing policies (H4).** Lock it to the parent's target; or keep
   it editable behind a computed warning ("LR at step 6 000 moves 9.1e-5 → 3.6e-5"); or allow
   raising only. Locking costs the "train further than planned" flow, which is a legitimate use.
   Needs a decision because the current state is silent, and silence is the worst of the three.

2. **Raising steps past `num_decay_steps` (H5 gap).** Refuse, warn, or accept. The done-run refusal
   already treats floor-LR training as unacceptable-and-silent; this is the same outcome reached by
   a different route, and we currently allow it.

3. **Do we want a real "change the LR mid-run" feature (H2)?** If yes, the code is small but the
   semantics are the whole question: does the new LR rescale the remaining schedule, replace it with
   a constant, or restart warmup? Picking one silently is a "same loss curve, different policy"
   hazard. If no, the decision is just to write the reason into the UI copy.

4. **Per-run attribution of Hub checkpoints (H7).** A marker file or per-run subtree under
   `checkpoints/<step>/`, plus filtering in the listing. Touches the HF Jobs wrapper and the
   host-side uploader. It is the only thing that closes the early-fork residual, and its value
   depends partly on decision 5.

5. **Absorption, or the refusal-enforced sticks we have (H8.1) — the umbrella item.** This
   supersedes the narrower forest-vs-sticks question, which was settled by `b336138`: sticks-only
   shipped, enforced by refusing a second resume. The live decision is whether to go further and
   make sticks *structural* — B absorbs A's record, checkpoints and history — which dissolves the
   second-resume refusal, the delete-first remedy and its fork-aware branching, the superseded/leaf
   hiding rule, the continued-run fine-tune gap, and all lineage walking. What has to be settled
   first, in rough order of difficulty: (i) transferring metrics history with correct per-source
   rebasing, or the pre-resume loss curve is deleted; (ii) crash-safety and ordering for the
   destructive absorb step, which is machinery the registry has never needed; (iii) whether
   descendants' `finetune_from_job_id` gets rewritten to B or left dangling; (iv) accepting that
   legacy forks keep the forest-capable readers alive regardless; (v) what happens to the absorbed
   run's `#N`. Full treatment in H8.1.

6. **Upstream issue against lerobot?** The candidate is H3 + H2 together: the `resume` docstring
   promises "CLI `--*` flags still override" (`lerobot/configs/train.py:87-91`), but on a resume
   (a) an optimizer-LR override is silently undone by the state restore, (b)
   `--policy.scheduler_*` — the namespace that works on fresh runs — becomes inert because the
   preset clause is skipped, while (c) `--scheduler.*` — inert on fresh runs — becomes live, and
   (d) `--steps` silently reshapes an auto-scaling schedule under a restored `last_epoch`. None of
   these warn. A minimal ask would be: document the resume-time override semantics precisely, and
   log a warning when a CLI override is parsed into a field that the resume path will overwrite (or
   when `build()` auto-scales against a `num_training_steps` that differs from the checkpoint's).
   Worth deciding whether we file it, and whether we offer the warning patch.

---

## Appendix: briefed claims that did not survive verification

Recorded here rather than dropped, per the review instruction.

1. **"Data order differs after resume, so stochastic divergence often doesn't recur" — false for
   this pin.** v0.6.0's non-streaming resume is sample-exact and restores RNG state
   (`lerobot/scripts/lerobot_train.py:415-418, 449-455`; `lerobot/common/train_utils.py:228`).
   Corrected in H6(b), with the surviving escape hatches (GPU non-determinism, streaming datasets,
   changed world size/batch size) named there. This changes the *argument* for resume-from-older-
   checkpoint but not the *feature*: reason (a), a damaged newest checkpoint, is fully substantiated.

2. **"Our resume form does not prefill the locked optimizer controls."** The in-code comment at
   `frontend/src/components/training/types.ts:165-171` asserts this, and it is stale: the seed
   carries the parent's optimizer values (`resumeSeed.ts:412-415`) and the form reads them
   (`useTrainingConfig.ts:111-114`), which the *other* comment at `useTrainingConfig.ts:69-77`
   states correctly. Harmless — the lock behaves right either way — but the two comments contradict
   each other and one of them will mislead the next reader. (Not fixed here: that file is claimed by
   a concurrent session.)

3. **Line-number drift from the brief (no substantive disagreement).** The auto-scale block is
   `lerobot/optim/schedulers.py:144-161` (the `if` is at :149), not "~149 `build()`" — same code.
   `resumeLocked` is used at `TrainingConfigurator.tsx:333-382` and defined at
   `useTrainingConfig.ts:301`; the brief's "~line 354" points at the segment badge specifically,
   which is one of several uses.

4. **The step-80 cloud incident (H6a) has no written record** in the repo or `internal_docs/`. The
   *mechanism* is fully substantiated by `makermodslab/jobs.py:1065-1070` ("a known uploader race")
   and the completeness rule at :946-972; the specific run rests on session memory. Flagged so it is
   not presented to a supervisor as a documented case.
