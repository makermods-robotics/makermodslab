# Unmerged content from backup/rig-2026-08-31 (exported 2026-09-03)

Content-level comparison of the pre-rebuild rig against local `staging` found
everything load-bearing on main except the items below. Patches are
`git format-patch` output; apply with `git am -3` (expect conflicts — the tree
has moved).

- 0b70b07d — camera uniqueID re-anchoring: RE-CARVED as PR #125 (fix/camera-reanchor-at-session-start) on 2026-09-03; patch removed.
- 739a5160 — finetune_audit.py + test (post-hoc damaged-fine-tune verdict
  engine). Never wired; the enforcing guard it came with IS on main
  (jobs.py read_pretrained_policy_type / _check_pretrained_policy_type).
- e04ad2f1 — TruncateWithTitle. Superseded by useTruncationTitle + DisplayName
  on main; kept only for reference.
- W&B test pins: PORTED into PR #72 (fix/wandb-endpoints-main @ 239c03f9) on 2026-09-03.

Dropped by the user on 2026-09-03: 4bc43bbb (naming) and the HubModelCard delete dialog
(removed deliberately by 76c7de97, agreed with the PR author).
