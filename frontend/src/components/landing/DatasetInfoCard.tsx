import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  AlertTriangle,
  ChevronDown,
  Download as DownloadIcon,
  ExternalLink,
  Loader2,
  Lock,
  Pencil,
  Settings2,
  Trash2,
  Upload as UploadIcon,
  X,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { ApiError } from "@/lib/apiClient";
import {
  formatBytes,
  formatCount,
  formatDuration,
} from "@/lib/datasetFormat";
import {
  datasetNameIssue,
  formatDatasetNameIssue,
} from "@/lib/datasetName";
import UploadDatasetDialog from "@/components/landing/UploadDatasetDialog";
import VisibilityToggle from "@/components/landing/VisibilityToggle";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import { useDatasetDownload } from "@/hooks/useDatasetDownload";
import {
  DatasetInfo,
  DatasetTask,
  HubStatusValue,
  getDatasetHubSettings,
  getDatasetHubStatus,
  getDatasetInfo,
  renameDataset,
  setDatasetTags,
  setDatasetVisibility,
} from "@/lib/replayApi";

const WarningBadge: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <span className="inline-flex items-center gap-1 rounded border border-red-500/40 bg-red-500/15 px-1.5 py-0.5 text-xs font-medium text-destructive">
    <AlertTriangle className="h-3 w-3 shrink-0" />
    {children}
  </span>
);

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex items-baseline gap-2">
    <span className="w-14 shrink-0 text-muted-foreground">{label}</span>
    <span className="min-w-0 flex-1 text-foreground">{children}</span>
  </div>
);

/**
 * Tasks row content. One task renders inline (with its episode count when
 * known); several render as a collapsible disclosure — closed it reads
 * "N tasks", open it lists each task with its episode count right-aligned.
 */
const TaskList: React.FC<{ tasks: DatasetTask[] }> = ({ tasks }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  if (tasks.length === 1) {
    const { task, num_episodes } = tasks[0];
    return (
      <span className="flex items-baseline gap-1.5">
        <span className="min-w-0 truncate" title={task}>
          {task}
        </span>
        {num_episodes > 0 && (
          <span className="shrink-0 text-muted-foreground">
            · {t("landing.datasetInfo.tasks.episodeCount", { n: num_episodes })}
          </span>
        )}
      </span>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-foreground hover:text-foreground">
        {t("landing.datasetInfo.tasks.count", { count: tasks.length })}
        <ChevronDown
          className={`h-3 w-3 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="mt-1 space-y-0.5">
          {tasks.map(({ task, num_episodes }) => (
            <li key={task} className="flex items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate" title={task}>
                {task}
              </span>
              <span className="shrink-0 text-muted-foreground">
                {t("landing.datasetInfo.tasks.episodeCount", {
                  n: num_episodes,
                })}
              </span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
};

/** True when the logged-in user can write to `repoId`'s namespace, so the Hub
 * settings editor should be offered. A bare repo id (no "/") lives under the
 * user's own account, always writable. Mirrors DatasetPicker's upload gate:
 * case-insensitive, false while loading / unauthenticated. */
const useCanEditHub = (repoId: string): boolean => {
  const { auth } = useHfAuth();
  if (auth.status !== "authenticated") return false;
  const ns = repoId.includes("/") ? repoId.split("/")[0] : auth.username;
  if (ns == null) return false;
  return auth.writableNamespaces.some(
    (n) => n.toLowerCase() === ns.toLowerCase(),
  );
};

/** True when Rename should be offered for `repoId`.
 *
 * Rename is an identity mutation the backend refuses outside a namespace the
 * user can write to, so a downloaded third-party dataset (`lerobot/pusht`)
 * shouldn't show the button at all — mirroring how the delete flow offers
 * remove-local-copy for third-party content rather than an identity change.
 *
 * Unlike `useCanEditHub` this defaults to TRUE while loading or unauthenticated:
 * with no Hub identity the namespace can't be judged, and the backend does a
 * purely local rename in that case, so hiding the button would break renaming
 * a never-uploaded dataset while logged out. A bare id has no namespace to
 * disown — it lives under the user's own account. */
const useCanRename = (repoId: string): boolean => {
  const { auth } = useHfAuth();
  if (auth.status !== "authenticated") return true;
  if (!repoId.includes("/")) return true;
  const ns = repoId.split("/")[0];
  return auth.writableNamespaces.some(
    (n) => n.toLowerCase() === ns.toLowerCase(),
  );
};

/** Org/required tags the backend's `with_makermodslab_tag` always re-adds on save, so
 * they can't actually be dropped. Shown as locked, non-removable chips so the
 * UI never implies the user can remove them. Matched case-insensitively. */
const REQUIRED_TAGS = ["makermods", "openbooth", "MakerModsLab"];
const isRequiredTag = (t: string): boolean =>
  REQUIRED_TAGS.some((r) => r.toLowerCase() === t.toLowerCase());

/**
 * Post-upload Hub settings editor: a popover (labeled "Visibility & tags"
 * trigger) with a Public|Private visibility toggle and a chip-based tags editor,
 * both pre-filled from the live Hub settings (`/datasets/hub-settings`).
 * Visibility and tags save independently — each MUTATES the live repo, so each
 * has its own Save/loading state, success toast, and inline error. On success
 * the parent's status/tags refresh via `onChanged`.
 *
 * Tags render as removable pills; the org/required tags (makermods, openbooth,
 * MakerMods Lab) render as locked, non-removable pills since the backend always re-adds
 * them. A text input adds a new tag on Enter or comma.
 *
 * Only rendered for datasets whose namespace the user can write to (see
 * useCanEditHub) — the same gate DatasetPicker uses for uploads.
 */
const HubSettingsEditor: React.FC<{
  repoId: string;
  onChanged?: () => void;
}> = ({ repoId, onChanged }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [isPrivate, setIsPrivate] = useState(false);
  const [initialPrivate, setInitialPrivate] = useState(false);
  const [savingVisibility, setSavingVisibility] = useState(false);
  const [visibilityError, setVisibilityError] = useState<string | null>(null);

  const [tags, setTags] = useState<string[]>([]);
  const [initialTags, setInitialTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const [savingTags, setSavingTags] = useState(false);
  const [tagsError, setTagsError] = useState<string | null>(null);

  // (Re)load the live settings each time the popover opens, so the fields
  // always reflect what's actually on the Hub (incl. a change made elsewhere).
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setLoadError(null);
    setVisibilityError(null);
    setTagsError(null);
    getDatasetHubSettings(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => {
        setIsPrivate(data.private);
        setInitialPrivate(data.private);
        setTags(data.tags);
        setInitialTags(data.tags);
        setNewTag("");
        setLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setLoadError(
          // `e.detail` is the backend's own message — surfaced verbatim; only
          // the client-side fallback beside it is translated.
          e instanceof ApiError && e.detail
            ? e.detail
            : t("landing.datasetInfo.hubSettings.loadError"),
        );
        setLoading(false);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, baseUrl, fetchWithHeaders, repoId]);

  const errText = (e: unknown): string =>
    e instanceof ApiError && e.detail
      ? e.detail
      : e instanceof Error
        ? e.message
        : String(e);

  const saveVisibility = async () => {
    setSavingVisibility(true);
    setVisibilityError(null);
    try {
      const res = await setDatasetVisibility(
        baseUrl,
        fetchWithHeaders,
        repoId,
        isPrivate,
      );
      setInitialPrivate(res.private);
      toast({
        title: t("landing.datasetInfo.hubSettings.visibilityUpdatedTitle"),
        // Two whole sentences rather than an interpolated adjective — no
        // translation can inflect a word handed to it mid-sentence.
        description: res.private
          ? t("landing.datasetInfo.hubSettings.nowPrivate", { repoId })
          : t("landing.datasetInfo.hubSettings.nowPublic", { repoId }),
      });
      onChanged?.();
    } catch (e) {
      setVisibilityError(errText(e));
    } finally {
      setSavingVisibility(false);
    }
  };

  // Add `newTag` (or any comma-joined batch) as chip(s), de-duplicated
  // case-insensitively against what's already present. Clears the input.
  const commitNewTag = () => {
    const parsed = newTag
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);
    if (parsed.length > 0) {
      setTags((prev) => {
        const next = [...prev];
        for (const t of parsed) {
          if (!next.some((e) => e.toLowerCase() === t.toLowerCase())) {
            next.push(t);
          }
        }
        return next;
      });
    }
    setNewTag("");
  };

  const removeTag = (tag: string) => {
    setTags((prev) => prev.filter((t) => t !== tag));
  };

  const saveTags = async () => {
    setSavingTags(true);
    setTagsError(null);
    try {
      const res = await setDatasetTags(baseUrl, fetchWithHeaders, repoId, tags);
      setTags(res.tags);
      setInitialTags(res.tags);
      setNewTag("");
      toast({
        title: t("landing.datasetInfo.hubSettings.tagsUpdatedTitle"),
        description: repoId,
      });
      onChanged?.();
    } catch (e) {
      setTagsError(errText(e));
    } finally {
      setSavingTags(false);
    }
  };

  const visibilityChanged = isPrivate !== initialPrivate;
  // Order-insensitive set comparison — reordering chips isn't a real change.
  const tagsChanged =
    tags.length !== initialTags.length ||
    !tags.every((t) => initialTags.includes(t));

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={t("landing.datasetInfo.hubSettings.trigger")}
          title={t("landing.datasetInfo.hubSettings.trigger")}
          className="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-xs font-medium text-foreground hover:border-foreground/30 hover:bg-accent hover:text-foreground"
        >
          <Settings2 className="h-3 w-3 shrink-0" />
          {t("landing.datasetInfo.hubSettings.trigger")}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="w-72 text-xs"
        // Same cmdk-guard rationale as UploadDatasetDialog: stop clicks from
        // bubbling to a CommandItem row that would select/close the picker.
        onClick={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
      >
        {loading ? (
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>{t("landing.datasetInfo.hubSettings.loading")}</span>
          </div>
        ) : loadError ? (
          <p className="text-destructive">{loadError}</p>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label
                id={`hub-edit-visibility-${repoId}`}
                className="font-normal text-muted-foreground"
              >
                {t("landing.datasetInfo.hubSettings.visibility")}
              </Label>
              <VisibilityToggle
                value={isPrivate}
                onChange={setIsPrivate}
                idBase={`hub-edit-visibility-${repoId}`}
                disabled={savingVisibility}
              />
              <p className="leading-snug text-muted-foreground">
                {isPrivate
                  ? t("landing.datasetInfo.hubSettings.privateNote")
                  : t("landing.datasetInfo.hubSettings.publicNote")}
              </p>
              {visibilityError && (
                <p className="text-destructive">{visibilityError}</p>
              )}
              <Button
                size="sm"
                onClick={saveVisibility}
                disabled={savingVisibility || !visibilityChanged}
                className="h-7 w-full gap-1 text-xs"
              >
                {savingVisibility ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {t("landing.datasetInfo.hubSettings.saving")}
                  </>
                ) : (
                  t("landing.datasetInfo.hubSettings.saveVisibility")
                )}
              </Button>
            </div>

            <div className="space-y-1.5 border-t border-border pt-3">
              <Label
                htmlFor={`hub-edit-tags-${repoId}`}
                className="font-normal text-muted-foreground"
              >
                {t("landing.datasetInfo.hubSettings.tags")}
              </Label>
              {tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {tags.map((tag) => {
                    const required = isRequiredTag(tag);
                    return required ? (
                      // Locked org tag: distinct style + lock icon, no remove
                      // (the backend always re-adds it on save).
                      <span
                        key={tag}
                        title={t("landing.datasetInfo.hubSettings.lockedTag")}
                        className="inline-flex items-center gap-1 rounded-full border border-info/40 bg-info/10 px-2 py-0.5 text-xs text-info"
                      >
                        <Lock className="h-2.5 w-2.5 shrink-0" />
                        {tag}
                      </span>
                    ) : (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-xs text-foreground"
                      >
                        {tag}
                        <button
                          type="button"
                          onClick={() => removeTag(tag)}
                          aria-label={t(
                            "landing.datasetInfo.hubSettings.removeTag",
                            { tag },
                          )}
                          title={t(
                            "landing.datasetInfo.hubSettings.removeTag",
                            { tag },
                          )}
                          className="-mr-0.5 rounded-full text-muted-foreground hover:text-foreground"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    );
                  })}
                </div>
              )}
              <Input
                id={`hub-edit-tags-${repoId}`}
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === ",") {
                    e.preventDefault();
                    commitNewTag();
                  } else if (
                    e.key === "Backspace" &&
                    newTag === "" &&
                    tags.length > 0
                  ) {
                    // Backspace on an empty input removes the last removable tag.
                    const last = [...tags]
                      .reverse()
                      .find((t) => !isRequiredTag(t));
                    if (last) removeTag(last);
                  }
                }}
                onBlur={commitNewTag}
                placeholder={t(
                  "landing.datasetInfo.hubSettings.tagPlaceholder",
                )}
                className="h-7 text-xs"
              />
              <p className="leading-snug text-muted-foreground">
                {t("landing.datasetInfo.hubSettings.requiredTagsNote")}
              </p>
              {tagsError && <p className="text-destructive">{tagsError}</p>}
              <Button
                size="sm"
                onClick={saveTags}
                disabled={savingTags || !tagsChanged}
                className="h-7 w-full gap-1 text-xs"
              >
                {savingTags ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {t("landing.datasetInfo.hubSettings.saving")}
                  </>
                ) : (
                  t("landing.datasetInfo.hubSettings.saveTags")
                )}
              </Button>
            </div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
};

/**
 * Hub sync line for the info card: a muted status ("Local only" / "On Hub")
 * plus, when the dataset isn't confirmed on the Hub, an "Upload to Hub" button
 * that opens a confirm popover (private-by-default toggle + optional tags).
 *
 * Status is fetched separately/lazily so it never blocks the card render, and
 * degrades to "unknown" (nothing shown) offline/unauthenticated. The upload
 * runs in the background (see useDatasetUpload): while it's in flight this row
 * shows an "Uploading…" state (which survives navigating away and back), and
 * on completion it flips to "On Hub" and toasts the Hub URL.
 */
const HubSyncRow: React.FC<{ repoId: string }> = ({ repoId }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<HubStatusValue>("unknown");
  const [hubUrl, setHubUrl] = useState<string | null>(null);
  // A repo of this name exists on the Hub but holds no dataset — an upload
  // that never finished. It is not a backup. See HubStatus.hub_has_data.
  const [uploadIncomplete, setUploadIncomplete] = useState(false);
  // Bumped after a visibility/tags edit to re-run the status fetch (the backend
  // invalidates its hub-status cache on a change, so this re-reads fresh).
  const [refreshKey, setRefreshKey] = useState(0);
  const canEdit = useCanEditHub(repoId);

  const { uploading, start } = useDatasetUpload({
    repoId,
    onDone: (url) => {
      setStatus("on_hub");
      setHubUrl(url);
      // The push that just succeeded is what the Hub repo was missing.
      setUploadIncomplete(false);
      toast({
        title: t("landing.datasetInfo.hubSync.uploadedTitle"),
        description: (
          <span>
            <Trans
              i18nKey="landing.datasetInfo.hubSync.uploadedBody"
              values={{ repoId }}
              components={[
                <a
                  key="0"
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline font-medium"
                />,
              ]}
            />
          </span>
        ),
      });
    },
    onError: (message, docsUrl) => {
      toast({
        title: t("landing.datasetInfo.hubSync.uploadFailedTitle"),
        // `message` is the backend's own failure text — rendered verbatim;
        // only the guide link's wording around it is ours.
        description: docsUrl ? (
          <span>
            <Trans
              i18nKey="landing.datasetInfo.hubSync.uploadFailedWithGuide"
              values={{ message }}
              components={[
                <a
                  key="0"
                  href={docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline font-medium"
                />,
              ]}
            />
          </span>
        ) : (
          message
        ),
        variant: "destructive",
      });
    },
  });

  useEffect(() => {
    const controller = new AbortController();
    setStatus("unknown");
    setHubUrl(null);
    setUploadIncomplete(false);
    getDatasetHubStatus(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => {
        setStatus(data.status);
        setHubUrl(data.url);
        setUploadIncomplete(data.hub_has_data === false);
      })
      .catch(() => {
        // Degrade silently to "unknown" — no error spam on the card.
        if (!controller.signal.aborted) setStatus("unknown");
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId, refreshKey]);

  if (uploading) {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>{t("landing.datasetInfo.hubSync.uploading")}</span>
      </div>
    );
  }

  // A repo of this name is on the Hub, but there is no dataset inside it: an
  // earlier upload created the repo and then failed before sending the files.
  // Saying "Uploaded to HuggingFace" here invites deleting the only real copy,
  // so name the actual state and keep the upload one click away.
  if (status === "on_hub" && uploadIncomplete) {
    return (
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-amber-700 dark:text-amber-400">
          Upload didn't finish — nothing on the Hub yet
          {hubUrl && (
            <a
              href={hubUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-0.5 hover:text-foreground"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </span>
        <UploadDatasetDialog repoId={repoId} start={start}>
          <Button
            size="sm"
            variant="outline"
            className="h-6 gap-1 border-amber-500/50 px-2 text-xs text-amber-700 hover:bg-amber-500/10 dark:text-amber-300"
          >
            <UploadIcon className="h-3 w-3" />
            Upload to Hub
          </Button>
        </UploadDatasetDialog>
      </div>
    );
  }

  if (status === "on_hub") {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <span>{t("landing.datasetInfo.hubSync.onHub")}</span>
        {hubUrl && (
          <a
            href={hubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-0.5 text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
        {canEdit && (
          <HubSettingsEditor
            repoId={repoId}
            onChanged={() => setRefreshKey((k) => k + 1)}
          />
        )}
      </div>
    );
  }

  // "absent" = neither on the Hub nor local — there's nothing local to upload,
  // so show a plain not-found line with no upload affordance (this is the
  // signal that used to be mislabeled "local_only").
  if (status === "absent") {
    return (
      <span className="text-muted-foreground">
        {t("landing.datasetInfo.hubSync.absent")}
      </span>
    );
  }

  // local_only or unknown: offer upload. For "unknown" we still allow it —
  // the endpoint is a safe upsert and reports auth failures gracefully.
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-muted-foreground">
        {status === "local_only"
          ? t("landing.datasetInfo.hubSync.localOnly")
          : t("landing.datasetInfo.hubSync.unknown")}
      </span>
      <UploadDatasetDialog repoId={repoId} start={start}>
        <Button
          size="sm"
          variant="outline"
          className="h-6 gap-1 border-teal-500/50 px-2 text-xs text-teal-700 dark:text-teal-300 hover:bg-teal-500/10"
        >
          <UploadIcon className="h-3 w-3" />
          {t("landing.datasetInfo.hubSync.upload")}
        </Button>
      </UploadDatasetDialog>
    </div>
  );
};

/**
 * Rename dialog for a local dataset (mirrors JobCard's rename UI). The namespace
 * prefix is fixed — the user edits only the name segment, shown after a static
 * "namespace/" prefix. A dataset's repo id IS its directory path, so this moves
 * the directory; if a copy also exists on the Hub, the server renames that too
 * so the two stay in sync.
 */
const RenameDatasetDialog: React.FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repoId: string;
  onRenamed: (newRepoId: string) => void;
}> = ({ open, onOpenChange, repoId, onRenamed }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const slash = repoId.lastIndexOf("/");
  const namespace = slash >= 0 ? repoId.slice(0, slash) : null;
  const currentName = slash >= 0 ? repoId.slice(slash + 1) : repoId;

  const [value, setValue] = useState(currentName);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  // Reset the field to the current name whenever the dialog (re)opens.
  useEffect(() => {
    if (open) {
      setValue(currentName);
      setError(null);
    }
  }, [open, currentName]);

  const trimmed = value.trim();
  const validationIssue = trimmed === "" ? null : datasetNameIssue(trimmed);
  const validationError = validationIssue
    ? formatDatasetNameIssue(t, validationIssue)
    : null;
  const unchanged = trimmed === currentName;

  const doRename = async () => {
    const next = value.trim();
    const nextIssue = datasetNameIssue(next);
    const nameError = nextIssue ? formatDatasetNameIssue(t, nextIssue) : null;
    if (nameError) {
      setError(nameError);
      return;
    }
    if (next === currentName) {
      onOpenChange(false);
      return;
    }
    setRenaming(true);
    setError(null);
    try {
      const res = await renameDataset(baseUrl, fetchWithHeaders, repoId, next);
      /* The dialog has closed by the time the toast shows, so it's the only
       * place the user learns whether the Hub copy moved — never claim a Hub
       * rename on "skipped" (offline / logged out / unwritable namespace). */
      if (res.hub === "renamed") {
        toast({
          title: t("landing.datasetInfo.rename.renamedTitle"),
          description: t("landing.datasetInfo.rename.renamedHub", {
            repoId: res.repo_id,
          }),
        });
      } else if (res.hub === "skipped") {
        toast({
          title: t("landing.datasetInfo.rename.renamedLocallyTitle"),
          description: t("landing.datasetInfo.rename.renamedLocally", {
            repoId: res.repo_id,
          }),
        });
      } else {
        toast({
          title: t("landing.datasetInfo.rename.renamedTitle"),
          description: res.repo_id,
        });
      }
      onOpenChange(false);
      onRenamed(res.repo_id);
    } catch (e) {
      setError(
        e instanceof ApiError && e.detail
          ? e.detail
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setRenaming(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("landing.datasetInfo.rename.title")}</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.datasetInfo.rename.description")}
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-1">
          {namespace && (
            <span className="shrink-0 font-mono text-sm text-muted-foreground">
              {namespace}/
            </span>
          )}
          <Input
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void doRename();
              }
            }}
            autoFocus
            placeholder={t("landing.datasetInfo.rename.placeholder")}
          />
        </div>
        {(error ?? validationError) && (
          <p className="text-sm text-destructive">{error ?? validationError}</p>
        )}
        <DialogFooter className="flex gap-2 justify-end">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            {t("common.cancel")}
          </Button>
          <Button
            disabled={
              renaming || trimmed === "" || unchanged || validationError !== null
            }
            onClick={doRename}
          >
            {renaming
              ? t("landing.datasetInfo.rename.submitting")
              : t("landing.datasetInfo.rename.submit")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

/**
 * "Download to this machine" affordance for a Hub-only dataset (the info card's
 * not-downloaded-locally branch). Starts a background download into the local
 * cache and, while it runs, shows a "Downloading…" state that survives
 * navigation (useDatasetDownload re-attaches by polling on mount). On completion
 * it fires onDownloaded so the parent re-reads /datasets/info (now local) and
 * refreshes the listing (source flips to "both").
 */
const HubDownloadRow: React.FC<{
  repoId: string;
  onDownloaded: () => void;
}> = ({ repoId, onDownloaded }) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { downloading, start } = useDatasetDownload({
    repoId,
    onDone: () => {
      toast({
        title: t("landing.datasetInfo.download.doneTitle"),
        description: t("landing.datasetInfo.download.doneBody", { repoId }),
      });
      onDownloaded();
    },
    onError: (message) => {
      toast({
        // `message` comes from the backend / the hook's own fallback.
        title: t("landing.datasetInfo.download.failedTitle"),
        description: message,
        variant: "destructive",
      });
    },
  });

  if (downloading) {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>{t("landing.datasetInfo.download.inProgress")}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-muted-foreground">
        {t("landing.datasetInfo.download.notDownloaded")}
      </span>
      <Button
        size="sm"
        variant="outline"
        onClick={async () => {
          const err = await start();
          if (err) {
            toast({
              title: t("landing.datasetInfo.download.startFailedTitle"),
              description: err,
              variant: "destructive",
            });
          }
        }}
        className="h-6 gap-1 border-blue-500/50 px-2 text-xs text-blue-700 dark:text-blue-300 hover:bg-blue-500/10"
      >
        <DownloadIcon className="h-3 w-3" />
        {t("landing.datasetInfo.download.button")}
      </Button>
    </div>
  );
};

/**
 * The info card's "no local detail" branch (/datasets/info returned 404). What
 * to show depends on where the dataset actually lives, so the Hub status is
 * fetched here and the cases render coherently — crucially NEVER claiming both
 * "not downloaded locally" and "Local only" at once (the contradictory pair a
 * naive 404-branch render produced for a stale pin that is neither on the Hub
 * nor local):
 *
 *   - on_hub / unknown → a genuine Hub dataset not yet downloaded: offer download.
 *   - local_only       → a local copy exists but /datasets/info couldn't read
 *                        its details (incomplete/corrupt): say so, offer upload.
 *   - absent           → neither on the Hub nor local: a deleted/renamed/stale
 *                        selection. Say that plainly; no download/upload.
 */
const NotDownloadedView: React.FC<{
  repoId: string;
  onDownloaded: () => void;
}> = ({ repoId, onDownloaded }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<HubStatusValue | "loading">("loading");

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    getDatasetHubStatus(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => setStatus(data.status))
      .catch(() => {
        if (!controller.signal.aborted) setStatus("unknown");
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId]);

  const header = (
    <div className="font-medium text-foreground break-all">{repoId}</div>
  );

  if (status === "loading") {
    return (
      <div className="space-y-1.5">
        {header}
        <p className="text-muted-foreground">
          {t("landing.datasetInfo.notDownloaded.checking")}
        </p>
      </div>
    );
  }

  // A local copy exists (hub-status said local_only) but its details couldn't be
  // read — a corrupt/incomplete local dataset. Coherent: local, but unreadable.
  if (status === "local_only") {
    return (
      <div className="space-y-1.5">
        {header}
        <p className="text-muted-foreground">
          {t("landing.datasetInfo.notDownloaded.localUnreadable")}
        </p>
        <div className="mt-1.5 border-t border-border pt-1.5">
          <HubSyncRow repoId={repoId} />
        </div>
      </div>
    );
  }

  // Neither on the Hub nor local — a stale selection (e.g. a pin to a dataset
  // that was deleted/renamed, or a merge output that was never materialized).
  if (status === "absent") {
    return (
      <div className="space-y-1.5">
        {header}
        <p className="text-muted-foreground">
          {t("landing.datasetInfo.notDownloaded.absent")}
        </p>
      </div>
    );
  }

  // on_hub or unknown: a (probable) Hub dataset not yet downloaded.
  return (
    <div className="space-y-1.5">
      {header}
      <p className="text-muted-foreground">
        {status === "on_hub"
          ? t("landing.datasetInfo.notDownloaded.onHub")
          : t("landing.datasetInfo.notDownloaded.unknown")}
      </p>
      <div className="mt-1.5 border-t border-border pt-1.5">
        <HubDownloadRow repoId={repoId} onDownloaded={onDownloaded} />
      </div>
    </div>
  );
};

interface DatasetInfoCardProps {
  repoId: string;
  /** Called after a successful rename with the new repo id, so the parent can
   * update the selection and refresh the picker list. */
  onRenamed?: (newRepoId: string) => void;
  /** When true, show a trash affordance for the selected dataset. Mirrors the
   * old picker-row gate: only local-only datasets (deleting the sole copy of a
   * not-yet-uploaded dataset). A "both"/hub dataset gets no delete here —
   * clearing its local cache lives in the "Manage cached datasets" dialog. */
  canDelete?: boolean;
  /** Invoked when the user clicks the card's delete affordance. The parent
   * routes this through its confirm dialog (nothing is deleted inline). */
  onDelete?: () => void;
  /** Called after a Hub-only dataset finishes downloading to the local cache,
   * so the parent can refresh the picker listing (source flips to "both"). The
   * card also re-reads its own /datasets/info to flip out of the Hub-only
   * fallback into the full local detail view. */
  onDownloaded?: () => void;
}

/**
 * Compact always-visible summary of the dataset selected on the home page:
 * episodes/frames/duration, camera names (the load-bearing line for vision
 * training), robot type, task strings, and size on disk. Data comes from the
 * on-demand /datasets/info endpoint, which only covers the local cache.
 */
const DatasetInfoCard: React.FC<DatasetInfoCardProps> = ({
  repoId,
  onRenamed,
  canDelete = false,
  onDelete,
  onDownloaded,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<{ notLocal: boolean } | null>(null);
  const [renameOpen, setRenameOpen] = useState(false);
  const canRename = useCanRename(repoId);
  // Bumped when a Hub-only dataset finishes downloading, to re-run the info
  // fetch (now that the dataset is local, /datasets/info succeeds and the card
  // flips from the Hub-only fallback to the full local detail view).
  const [infoRefreshKey, setInfoRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setInfo(null);
    setError(null);
    getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((data) => {
        setInfo(data);
        setLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError({ notLocal: e instanceof ApiError && e.status === 404 });
        setLoading(false);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId, infoRefreshKey]);

  return (
    <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs">
      {loading && (
        <div
          className="animate-pulse space-y-2 py-0.5"
          aria-label={t("landing.datasetInfo.loadingAria")}
        >
          <div className="h-3 w-3/4 rounded bg-muted" />
          <div className="h-3 w-1/2 rounded bg-muted" />
          <div className="h-3 w-2/3 rounded bg-muted" />
        </div>
      )}

      {/* No local detail (404). Where the dataset actually lives decides what to
          show — NotDownloadedView fetches the Hub status and renders coherently
          (a genuine Hub dataset to download, a local-but-unreadable copy, or a
          not-found stale selection), never the contradictory "not downloaded
          locally" + "Local only" pair. A non-404 error stays a bare line below. */}
      {!loading && error && error.notLocal && (
        <NotDownloadedView
          repoId={repoId}
          onDownloaded={() => {
            setInfoRefreshKey((k) => k + 1);
            onDownloaded?.();
          }}
        />
      )}

      {!loading && error && !error.notLocal && (
        <p className="text-muted-foreground">
          {t("landing.datasetInfo.loadError")}
        </p>
      )}

      {!loading &&
        info &&
        (() => {
          // ADDITIVE /datasets/info contract: "hub" = a meta/info.json summary
          // of a not-yet-downloaded Hub dataset (no tasks/size; rename is a
          // local directory move, so not applicable). Absent = "local".
          const isHubOnly = info.source === "hub";
          return (
            <div className="space-y-1.5">
              <div className="flex items-start justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2 font-medium text-foreground">
                  <span>
                    {/* A real plural (i18next _one/_other) for the episode
                        count; the frame count arrives pre-formatted ("16.7k"),
                        so it rides in as a plain variable, never as `count`.
                        The duration string itself is produced unchanged by
                        formatDuration. */}
                    {t("landing.datasetInfo.episodes", {
                      count: info.total_episodes,
                    })}
                    {" · "}
                    {t("landing.datasetInfo.frames", {
                      frames: formatCount(info.total_frames),
                    })}
                    {(() => {
                      const d = formatDuration(info.total_frames, info.fps);
                      return d ? ` · ${d}` : "";
                    })()}
                  </span>
                  {info.total_episodes === 0 && !isHubOnly && (
                    <WarningBadge>
                      {t("landing.datasetInfo.noEpisodes")}
                    </WarningBadge>
                  )}
                </div>
                <div className="-mr-1 -mt-0.5 flex shrink-0 items-center gap-0.5">
                  {/* Rename moves the local directory — meaningless for a
                      hub-only summary, and refused by the backend outside a
                      namespace the user owns (see useCanRename). */}
                  {!isHubOnly && canRename && (
                    <button
                      type="button"
                      onClick={() => setRenameOpen(true)}
                      aria-label={t("landing.datasetInfo.renameAria")}
                      title={t("landing.datasetInfo.renameAria")}
                      className="rounded p-1 text-muted-foreground hover:text-foreground"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                  )}
                  {/* Delete/remove — semantics resolved by the parent
                      (resolveDeleteAction) and routed through its confirm
                      dialog; nothing is deleted inline. */}
                  {canDelete && onDelete && (
                    <button
                      type="button"
                      onClick={onDelete}
                      aria-label={t("landing.datasetInfo.deleteAria")}
                      title={t("landing.datasetInfo.deleteAria")}
                      className="rounded p-1 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>

              {/* Cameras: show when known. For a LOCAL dataset with none, keep
                  the warning (silence would hide that it's unusable for vision
                  training). For a hub summary with none, omit the row — a hub
                  summary may simply lack features, and "unknown" is noise. */}
              {(info.cameras.length > 0 || !isHubOnly) && (
                <Row label={t("landing.datasetInfo.rowCameras")}>
                  {info.cameras.length > 0 ? (
                    info.cameras.join(", ")
                  ) : (
                    <WarningBadge>
                      {t("landing.datasetInfo.noCameras")}
                    </WarningBadge>
                  )}
                </Row>
              )}

              {/* Robot type: omit when unknown rather than render "unknown". */}
              {info.robot_type && (
                <Row label={t("landing.datasetInfo.rowRobot")}>
                  {info.robot_type}
                </Row>
              )}

              {info.tasks.length > 0 && (
                <Row label={t("landing.datasetInfo.rowTasks")}>
                  <TaskList tasks={info.tasks} />
                </Row>
              )}

              {info.size_bytes != null && (
                <Row label={t("landing.datasetInfo.rowSize")}>
                  {formatBytes(info.size_bytes)}
                </Row>
              )}

              {isHubOnly && (
                <p className="text-muted-foreground">
                  {t("landing.datasetInfo.hubOnlyNote")}
                </p>
              )}

              <div className="mt-1.5 border-t border-border pt-1.5">
                <HubSyncRow repoId={repoId} />
              </div>

              {/* Hub summary keeps the download affordance the sparse 404 view
                  has — on completion the card re-fetches and flips to the full
                  local detail. */}
              {isHubOnly && (
                <div className="border-t border-border pt-1.5">
                  <HubDownloadRow
                    repoId={repoId}
                    onDownloaded={() => {
                      setInfoRefreshKey((k) => k + 1);
                      onDownloaded?.();
                    }}
                  />
                </div>
              )}

              {!isHubOnly && (
                <RenameDatasetDialog
                  open={renameOpen}
                  onOpenChange={setRenameOpen}
                  repoId={repoId}
                  onRenamed={(newRepoId) => onRenamed?.(newRepoId)}
                />
              )}
            </div>
          );
        })()}
    </div>
  );
};

export default DatasetInfoCard;
