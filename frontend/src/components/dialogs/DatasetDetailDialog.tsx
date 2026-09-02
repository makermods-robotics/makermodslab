import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import {
  Boxes,
  Check,
  Loader2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  VideoOff,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useStudio } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import { lastFramePosition, previewPosition } from "@/lib/episodePreview";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import JointPositionChart from "@/components/dialogs/JointPositionChart";
import EpisodeReplayPanel from "@/components/dialogs/EpisodeReplayPanel";
import {
  EpisodeJointSeries,
  EpisodeSummary,
  episodeVideoUrl,
  getDatasetInfo,
  getEpisodeJoints,
  getExcludedEpisodes,
  listEpisodes,
  setExcludedEpisodes,
} from "@/lib/replayApi";

export interface DatasetDetailDialogProps {
  repoId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called when an action navigates to the studio, so a parent surface that
   * would otherwise cover the studio (e.g. the library sheet) can close too. */
  onStudioAction?: () => void;
}

// Best (cols, tileW, tileH) for `n` tiles inside a box of `boxW` x `boxH`:
// the search over column counts that a video-gallery layout needs so the
// camera grid keeps growing/reflowing as cameras are added or the window is
// resized, instead of a fixed N-up template.
function computeCameraLayout(
  n: number,
  boxW: number,
  boxH: number,
  gap = 8,
  aspect = 16 / 10,
) {
  let best = { cols: 1, tileW: boxW, tileH: boxW / aspect };
  let bestArea = -1;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cellW = (boxW - gap * (cols - 1)) / cols;
    const cellH = (boxH - gap * (rows - 1)) / rows;
    let tileW = cellW;
    let tileH = tileW / aspect;
    if (tileH > cellH) {
      tileH = cellH;
      tileW = tileH * aspect;
    }
    const area = tileW * tileH;
    if (area > bestArea) {
      bestArea = area;
      best = { cols, tileW, tileH };
    }
  }
  return best;
}

const fmtTime = (t: number) => `${t.toFixed(1)}s`;

/**
 * Episode viewer: a growable grid of camera feeds (one per dataset camera —
 * reflows on resize and as camera count changes, never a fixed 2-up) with
 * transport controls and a joint-position chart synced to the playhead. Real
 * <video> elements pointing at /datasets/episode-video (Range-request backed,
 * so scrubbing seeks without buffering the whole file); no local data means no
 * episode list, so this renders a locked "not viewable yet" state instead.
 */
const EpisodeViewer: React.FC<{
  repoId: string;
  cameras: string[];
  episodes: EpisodeSummary[];
  selectedEpisode: number;
  onSelectEpisode: (episodeIndex: number) => void;
}> = ({ repoId, cameras, episodes, selectedEpisode, onSelectEpisode }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [joints, setJoints] = useState<EpisodeJointSeries | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({});
  // True while the viewer sits at `previewT` and the user hasn't chosen a
  // position yet — pressing play rewinds to the episode's real start rather
  // than resuming from the preview frame, so the first half-second is never
  // skipped. Cleared by an explicit scrub, and by the first play.
  const [parkedAtPreview, setParkedAtPreview] = useState(true);
  // True once playback has reached this episode's end and auto-stopped, so
  // the next play restarts the episode instead of resuming. Explicit state
  // rather than comparing currentTime against the end: the corrective seek in
  // handlePrimaryTimeUpdate fires a timeupdate carrying the frame's ACTUAL
  // presentation time, and a seek resolves to the frame at-or-before its
  // target — so that value can land just under any threshold, leaving a
  // float comparison reading "not at the end" and replay silently resuming
  // for a single frame instead of restarting.
  const [atEpisodeEnd, setAtEpisodeEnd] = useState(false);

  const episode = episodes.find((e) => e.episode_index === selectedEpisode) ?? null;
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  // The camera whose onTimeUpdate drives currentTime/the end-of-episode
  // boundary check. Falls back off cameras[0] onto the next camera that
  // hasn't errored — if the "primary" camera fails to decode, its <video>
  // element is replaced by an error placeholder and never fires
  // onTimeUpdate, so pinning this to a fixed index would silently stall
  // playback tracking (auto-pause, the loop-to-start fix, the scrubber)
  // for the rest of the session.
  const primaryCamera = cameras.find((c) => !videoErrors[c]) ?? cameras[0];
  // v3.0 packs consecutive episodes into the same physical mp4 per camera —
  // this episode's slice starts at `from` within that (possibly shared)
  // file, not at 0. Every seek/play/boundary check goes through this.
  const offsetFor = useCallback(
    (camera: string) => episode?.video_offsets?.[camera]?.from ?? 0,
    [episode],
  );
  // Positions within this episode's slice of the packed v3.0 mp4. The
  // geometry — and why neither one is the episode's exact start or end — is
  // documented in lib/episodePreview.ts.
  const previewT = episode ? previewPosition(episode.duration) : 0;

  useEffect(() => {
    const controller = new AbortController();
    setJoints(null);
    getEpisodeJoints(baseUrl, fetchWithHeaders, repoId, selectedEpisode, controller.signal)
      .then(setJoints)
      .catch(() => {
        if (!controller.signal.aborted) setJoints(null);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId, selectedEpisode]);

  // Warm the backend's cache for the NEXT episode while this one plays, so
  // advancing often lands on an already-fetched chunk instead of repeating
  // the same first-time delay (see docs/superpowers/specs/2026-07-24-episode-
  // prefetch-design.md). A 1-byte Range request drives get_episode_video_path
  // all the way through — including a Hub hf_hub_download fetch for a
  // not-yet-cached chunk — without downloading video the user hasn't asked
  // for yet; FileResponse already supports Range requests (it's how the real
  // <video> tag seeks). The joints response is discarded — its fetch alone is
  // what warms the data chunk server-side. Best-effort: any failure here has
  // zero effect on the real fetch that runs when the user actually navigates
  // there, so errors are silently swallowed.
  useEffect(() => {
    const next = episodes[episodes.findIndex((e) => e.episode_index === selectedEpisode) + 1];
    if (!next) return;
    const controller = new AbortController();
    // Debounced: don't kick off a prefetch until the user has settled on
    // this episode for a moment. Without this, rapid navigation (arrow-key
    // taps, clicking through the list) fires — and immediately abandons via
    // the abort below — a prefetch for every episode passed through on the
    // way to wherever the user actually stops. The frontend abort doesn't
    // necessarily cancel an in-flight Hub download the backend already
    // started, so an unthrottled burst can leave several abandoned
    // hf_hub_download calls running server-side.
    const timer = window.setTimeout(() => {
      for (const camera of cameras) {
        fetchWithHeaders(episodeVideoUrl(baseUrl, repoId, next.episode_index, camera), {
          headers: { Range: "bytes=0-0" },
          signal: controller.signal,
        }).catch(() => {});
      }
      getEpisodeJoints(baseUrl, fetchWithHeaders, repoId, next.episode_index, controller.signal).catch(() => {});
    }, 400);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [baseUrl, fetchWithHeaders, repoId, episodes, cameras, selectedEpisode]);

  useEffect(() => {
    setPlaying(false);
    setParkedAtPreview(true);
    setAtEpisodeEnd(false);
    setCurrentTime(0);
    setVideoErrors({});
  }, [selectedEpisode]);

  const forEachVideo = useCallback((fn: (v: HTMLVideoElement) => void) => {
    Object.values(videoRefs.current).forEach((v) => v && fn(v));
  }, []);

  const handlePlayPause = useCallback(() => {
    if (playing) {
      forEachVideo((v) => v.pause());
      setPlaying(false);
    } else {
      // Rewind to this episode's real start before playing, in two cases.
      //
      // Parked at the preview frame: that position is half a second in (see
      // previewT), and playing from there would silently skip the opening of
      // every episode.
      //
      // Parked at (or past) the episode's own end: the underlying mp4 keeps
      // going into the next episode's frames past this point (see the
      // v3.0-packing note on offsetFor above), so resuming here would spill
      // into that footage before the boundary check in
      // handlePrimaryTimeUpdate catches up. Loop back to this episode's
      // start instead.
      // `atEpisodeEnd` covers auto-stop; the position test covers arriving
      // at the end by scrubbing, which clears both flags. Without it, play
      // from a fully-right scrubber runs forward into the NEXT episode's
      // packed footage until the ~4Hz boundary check catches up.
      const atEnd =
        episode && currentTime >= lastFramePosition(episode.length, episode.duration);
      if (episode && (parkedAtPreview || atEpisodeEnd || atEnd)) {
        for (const [camera, v] of Object.entries(videoRefs.current)) {
          if (v) v.currentTime = offsetFor(camera);
        }
        setCurrentTime(0);
      }
      setParkedAtPreview(false);
      setAtEpisodeEnd(false);
      forEachVideo((v) => {
        v.play().catch(() => {});
      });
      setPlaying(true);
    }
  }, [playing, forEachVideo, episode, currentTime, offsetFor, parkedAtPreview, atEpisodeEnd]);

  // Drives currentTime/scrubber from the primary camera's raw (file-relative)
  // playback position, and is the one place that notices the episode's own
  // slice has ended — native `ended` usually doesn't fire here since the
  // underlying file usually keeps going into the next episode's frames.
  const handlePrimaryTimeUpdate = (rawTime: number) => {
    if (!episode) return;
    // Parked at the preview frame: the seek that painted it fires a
    // timeupdate at previewT, but the scrubber must keep reading 0 — that is
    // where play actually begins (handlePlayPause rewinds to the episode's
    // start). Reporting 0.5s here would misstate where playback would resume,
    // and drag the joint chart's cursor off the episode's start with it.
    // Also frozen once the episode has ended: the corrective seek below fires
    // a further timeupdate, and letting it through would drag the scrubber
    // back off the end it just settled on.
    if (parkedAtPreview || atEpisodeEnd) return;
    const offset = offsetFor(primaryCamera);
    const t = Math.max(0, Math.min(episode.duration, rawTime - offset));
    setCurrentTime(t);
    // `timeupdate` only fires at roughly 4Hz, so by the time this notices the
    // episode is over, playback has already run up to ~250ms — several
    // frames — into the NEXT episode's footage, and pausing here would freeze
    // the viewer on a frame that isn't this episode's. Pause, then snap back
    // onto this episode's final frame. The target is that frame's own
    // timestamp: a boundary, so an engine may resolve it to the last or
    // second-to-last frame, and both are inside this episode — which is the
    // only property that matters here.
    // `playing` gates the snap so it can't re-trigger itself: the seek below
    // fires another timeupdate that is still past the threshold, and without
    // this the handler would re-pause and re-seek on every one of them.
    if (playing && rawTime >= offset + lastFramePosition(episode.length, episode.duration)) {
      forEachVideo((v) => v.pause());
      for (const [camera, v] of Object.entries(videoRefs.current)) {
        if (v) v.currentTime =
          offsetFor(camera) + lastFramePosition(episode.length, episode.duration);
      }
      setCurrentTime(episode.duration);
      setPlaying(false);
      setAtEpisodeEnd(true);
    }
  };

  // Fallback for the (possible) case where this episode's slice is the
  // physical file's own end — e.g. the last episode in a chunk — and the
  // video's real duration falls short of `episode.duration` (independently
  // derived from length/fps, not from the file itself). Native `ended` fires
  // here before the timeupdate threshold above ever would, and without this
  // handler `playing` would stay stuck true forever since no further
  // timeupdate events arrive once playback has actually stopped.
  const handlePrimaryEnded = () => {
    forEachVideo((v) => v.pause());
    setPlaying(false);
    // Same contract as the boundary check above — however playback stopped,
    // the next play restarts the episode rather than resuming from the end,
    // and the transport settles at the end rather than wherever the last
    // ~4Hz timeupdate happened to land (which reads as "12.7 / 12.9" with the
    // scrubber and joint cursor stuck short of the end).
    setCurrentTime(episode?.duration ?? 0);
    setAtEpisodeEnd(true);
  };

  const handleSeek = (t: number) => {
    // Clamped to the last frame, not to `duration`: `duration` IS the
    // boundary shared with the next episode, so dragging the scrubber fully
    // right would land on the same coin-flip seek that episodePreview.ts
    // exists to keep the viewer off.
    const end = episode ? lastFramePosition(episode.length, episode.duration) : 0;
    const clamped = Math.max(0, Math.min(end, t));
    setParkedAtPreview(false);
    setAtEpisodeEnd(false);
    for (const [camera, v] of Object.entries(videoRefs.current)) {
      if (v) v.currentTime = offsetFor(camera) + clamped;
    }
    setCurrentTime(clamped);
  };

  const gridWrapRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState({ cols: 1, tileW: 100, tileH: 60 });
  useEffect(() => {
    const el = gridWrapRef.current;
    if (!el) return;
    const recompute = () => {
      const rect = el.getBoundingClientRect();
      setLayout(
        computeCameraLayout(
          Math.max(1, cameras.length),
          Math.max(80, rect.width - 16),
          Math.max(80, rect.height - 16),
        ),
      );
    };
    recompute();
    const ro = new ResizeObserver(recompute);
    ro.observe(el);
    return () => ro.disconnect();
  }, [cameras.length]);

  const scrubFrac = episode?.duration ? currentTime / episode.duration : 0;

  const scrubRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const seekFromClientX = (clientX: number) => {
    const rect = scrubRef.current?.getBoundingClientRect();
    if (!rect || !episode) return;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    handleSeek(frac * episode.duration);
  };

  const gotoEpisode = useCallback(
    (delta: number) => {
      const i = episodes.findIndex((e) => e.episode_index === selectedEpisode);
      const next = episodes[i + delta];
      if (next) onSelectEpisode(next.episode_index);
    },
    [episodes, selectedEpisode, onSelectEpisode],
  );

  const currentIndex = episodes.findIndex((e) => e.episode_index === selectedEpisode);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < episodes.length - 1;

  // Mirrors the guarded-window-listener convention in RecordingSessionDialog:
  // skip when a text field has focus (the info-card tag input in the side
  // panel), preventDefault on the keys we do handle so Space doesn't scroll
  // the page or double-fire a focused button's native click.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Ignore OS auto-repeat from a held key — holding Space shouldn't
      // rapid-toggle play/pause, and holding an arrow shouldn't blow through
      // the episode list faster than an actual keypress.
      if (e.repeat) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        gotoEpisode(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        gotoEpisode(1);
      } else if (e.key === " " || e.code === "Space") {
        // A focused button already activates on Space natively, and its
        // onClick calls the same handler we'd call here (Prev/Play/Next/
        // episode rows) — don't hijack that, or tabbing to any button in
        // this dialog would silently turn Space into a play/pause toggle
        // instead of activating the focused control.
        if (target && target.tagName === "BUTTON") return;
        e.preventDefault();
        handlePlayPause();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [gotoEpisode, handlePlayPause]);

  // Drives the chart's scrubFrac from the replay's own local elapsed-time
  // clock while a hardware replay is playing — the same prop that camera
  // playback drives via currentTime, just a different clock source. Reset
  // to null (falls back to the camera-driven currentTime) once the replay
  // leaves the "playing" phase.
  const [replayElapsedS, setReplayElapsedS] = useState<number | null>(null);
  const handleReplayElapsedChange = useCallback((elapsedS: number, phase: string) => {
    setReplayElapsedS(phase === "playing" ? elapsedS : null);
  }, []);

  const effectiveCurrentTime = replayElapsedS ?? currentTime;
  const effectiveScrubFrac = episode?.duration ? effectiveCurrentTime / episode.duration : 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {cameras.length === 0 ? (
        <div className="flex flex-1 items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground">
          {t("dialogs.datasetDetail.noCameras")}
        </div>
      ) : (
        <div
          ref={gridWrapRef}
          className="relative flex-1 overflow-auto rounded-md border border-border bg-[#0c0f14]"
        >
          <div
            className="grid justify-center content-center gap-2 p-2"
            style={{
              gridTemplateColumns: `repeat(${layout.cols}, ${layout.tileW}px)`,
              gridAutoRows: `${layout.tileH}px`,
            }}
          >
            {cameras.map((camera) => (
              <div
                key={camera}
                className="relative overflow-hidden rounded border border-white/10 bg-black"
              >
                <span className="absolute left-1.5 top-1.5 z-10 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-zinc-100">
                  {camera}
                </span>
                {videoErrors[camera] ? (
                  <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-3 text-center text-zinc-300">
                    <VideoOff className="h-4 w-4" />
                    <p className="text-[10px] leading-snug">
                      {t("dialogs.datasetDetail.videoDecodeError")}
                    </p>
                  </div>
                ) : (
                  <video
                    key={`${repoId}:${selectedEpisode}:${camera}`}
                    ref={(el) => {
                      videoRefs.current[camera] = el;
                    }}
                    src={episodeVideoUrl(baseUrl, repoId, selectedEpisode, camera)}
                    className="h-full w-full object-cover"
                    muted
                    playsInline
                    onLoadedMetadata={(e) => {
                      // Only claim the position if the viewer is still parked.
                      // These are Range-backed (sometimes Hub-fetched) videos
                      // and play is enabled as soon as `episode` resolves, so
                      // a user can press play or scrub BEFORE metadata lands.
                      // A currentTime write before metadata only sets the
                      // default start position, which this would overwrite —
                      // starting playback 0.5s in and skipping the opening,
                      // the exact bug previewT exists to avoid. With several
                      // cameras only the slow ones would jump, desyncing the
                      // grid for the rest of the episode.
                      e.currentTarget.currentTime =
                        offsetFor(camera) + (parkedAtPreview ? previewT : currentTime);
                    }}
                    onTimeUpdate={
                      camera === primaryCamera
                        ? (e) => handlePrimaryTimeUpdate(e.currentTarget.currentTime)
                        : undefined
                    }
                    onEnded={camera === primaryCamera ? handlePrimaryEnded : undefined}
                    onError={() =>
                      setVideoErrors((prev) => ({ ...prev, [camera]: true }))
                    }
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {cameras.length > 0 ? (
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => gotoEpisode(-1)}
            disabled={!episode || !hasPrev}
            aria-label={t("dialogs.datasetDetail.previousEpisode")}
          >
            <SkipBack className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={handlePlayPause}
            disabled={!episode}
            aria-label={
              playing
                ? t("dialogs.datasetDetail.pause")
                : t("dialogs.datasetDetail.play")
            }
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </Button>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => gotoEpisode(1)}
            disabled={!episode || !hasNext}
            aria-label={t("dialogs.datasetDetail.nextEpisode")}
          >
            <SkipForward className="h-3.5 w-3.5" />
          </Button>
          <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
            {episode ? `${fmtTime(currentTime)} / ${fmtTime(episode.duration)}` : "—"}
          </span>
          <div
            ref={scrubRef}
            className="relative h-5 flex-1 cursor-pointer"
            onPointerDown={(e) => {
              draggingRef.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              forEachVideo((v) => v.pause());
              setPlaying(false);
              seekFromClientX(e.clientX);
            }}
            onPointerMove={(e) => {
              if (draggingRef.current) seekFromClientX(e.clientX);
            }}
            onPointerUp={(e) => {
              draggingRef.current = false;
              e.currentTarget.releasePointerCapture(e.pointerId);
            }}
          >
            <div className="absolute inset-y-0 my-auto h-1 w-full rounded-full bg-muted" />
            <div
              className="absolute inset-y-0 my-auto h-1 rounded-full bg-foreground"
              style={{ width: `${scrubFrac * 100}%` }}
            />
            <div
              className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-foreground shadow"
              style={{ left: `calc(${scrubFrac * 100}% - 6px)` }}
            />
          </div>
        </div>
      ) : null}

      <JointPositionChart joints={joints} episode={episode} scrubFrac={effectiveScrubFrac} />
      <EpisodeReplayPanel
        repoId={repoId}
        episodeIndex={selectedEpisode}
        onElapsedChange={handleReplayElapsedChange}
      />
    </div>
  );
};

const DatasetDetailDialog: React.FC<DatasetDetailDialogProps> = ({
  repoId,
  open,
  onOpenChange,
  onStudioAction,
}) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  // `.eyebrow` bundles `uppercase` with letter-spacing; the uppercase is a
  // no-op on Chinese but the tracking is not, so both come off together.
  const eyebrow = isCaselessScript(language)
    ? "text-[11px] font-semibold text-muted-foreground"
    : "eyebrow";
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openStudio } = useStudio();
  const { setSelectedDataset } = useSelectedDataset();
  const { toast } = useToast();

  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [cameras, setCameras] = useState<string[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  // Episodes excluded from training (curation, not deletion — see
  // replayApi's getExcludedEpisodes). The last SAVED state; while
  // `selecting` is true, edits live in `draftExcluded` instead and only land
  // here once "Done" commits them.
  const [excludedEpisodes, setExcludedEpisodesState] = useState<Set<number>>(
    new Set(),
  );
  // Selection mode: checkboxes only render, and Train + the dataset-info
  // panel only unblock, outside of this. Entering it seeds the draft from
  // the last saved state; leaving without pressing Done (closing the dialog,
  // switching datasets) abandons it — see the reset in the load effect below.
  const [selecting, setSelecting] = useState(false);
  const [draftExcluded, setDraftExcluded] = useState<Set<number>>(new Set());
  const [savingSelection, setSavingSelection] = useState(false);

  useEffect(() => {
    if (!repoId || !open) return;
    const controller = new AbortController();
    setEpisodesLoading(true);
    setEpisodes(null);
    setCameras([]);
    setExcludedEpisodesState(new Set());
    setSelecting(false);
    Promise.all([
      listEpisodes(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
      getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
      getExcludedEpisodes(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => []),
    ]).then(([eps, info, excluded]) => {
      if (controller.signal.aborted) return;
      setEpisodes(eps);
      setCameras(info?.cameras ?? []);
      setExcludedEpisodesState(new Set(excluded));
      setSelectedEpisode(eps && eps.length > 0 ? eps[0].episode_index : null);
      setEpisodesLoading(false);
    });
    return () => controller.abort();
  }, [repoId, open, baseUrl, fetchWithHeaders, reloadKey]);

  const validEpisodeIndices = new Set((episodes ?? []).map((e) => e.episode_index));
  const activeExcluded = selecting ? draftExcluded : excludedEpisodes;
  const includedEpisodes = (episodes ?? [])
    .map((e) => e.episode_index)
    .filter((i) => !activeExcluded.has(i));

  const startSelecting = () => {
    setDraftExcluded(new Set(excludedEpisodes));
    setSelecting(true);
  };

  const toggleDraftEpisode = (episodeIndex: number) => {
    setDraftExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(episodeIndex)) {
        next.delete(episodeIndex);
      } else {
        next.add(episodeIndex);
      }
      return next;
    });
  };

  const finishSelecting = () => {
    if (!repoId) return;
    const persisted = [...draftExcluded].filter((i) => validEpisodeIndices.has(i));
    setSavingSelection(true);
    setExcludedEpisodes(baseUrl, fetchWithHeaders, repoId, persisted)
      .then(() => {
        setExcludedEpisodesState(new Set(persisted));
        setSelecting(false);
      })
      .catch(() => {
        toast({
          title: t("dialogs.datasetDetail.curateSaveFailedTitle"),
          description: t("dialogs.datasetDetail.curateSaveFailedBody"),
          variant: "destructive",
        });
      })
      .finally(() => setSavingSelection(false));
  };

  if (!repoId) return null;

  const handleTrain = () => {
    setSelectedDataset(repoId);
    const allIncluded = includedEpisodes.length === (episodes?.length ?? 0);
    openStudio("train", {
      train: {
        datasetRepoId: repoId,
        episodeIndices: allIncluded ? undefined : includedEpisodes,
      },
    });
    onOpenChange(false);
    onStudioAction?.();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[85vh] max-w-6xl flex-col gap-0 overflow-hidden p-0">
        <DialogHeader className="shrink-0 space-y-0 border-b border-border px-6 py-4 text-left">
          <p className={eyebrow}>{t("dialogs.datasetDetail.eyebrow")}</p>
          <DialogTitle className="break-all pt-1 font-mono text-base font-semibold">
            {repoId}
          </DialogTitle>
        </DialogHeader>

        <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_300px]">
          <div className="flex min-h-0 min-w-0 flex-col gap-3 p-4">
            {episodesLoading ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                {t("dialogs.datasetDetail.loadingEpisodes")}
              </div>
            ) : episodes && episodes.length > 0 && selectedEpisode != null ? (
              <EpisodeViewer
                repoId={repoId}
                cameras={cameras}
                episodes={episodes}
                selectedEpisode={selectedEpisode}
                onSelectEpisode={setSelectedEpisode}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-muted/30 px-6 text-center">
                <VideoOff className="h-6 w-6 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground">
                  {episodes && episodes.length === 0
                    ? t("dialogs.datasetDetail.noEpisodesTitle")
                    : t("dialogs.datasetDetail.noFootageTitle")}
                </p>
                <p className="max-w-sm text-xs text-muted-foreground">
                  {episodes && episodes.length === 0
                    ? t("dialogs.datasetDetail.noEpisodesBody")
                    : t("dialogs.datasetDetail.noFootageBody")}
                </p>
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-col divide-y divide-border overflow-y-auto border-l border-border">
            <div className="flex min-h-0 flex-1 flex-col p-3">
              <div className="mb-2">
                <div className="flex items-center justify-between gap-2">
                  <p className={eyebrow}>
                    {episodes
                      ? t("dialogs.datasetDetail.episodesHeadingWithCount", {
                          total: episodes.length,
                        })
                      : t("dialogs.datasetDetail.episodesHeading")}
                  </p>
                  {episodes && episodes.length > 0 && (
                    <Button
                      size="sm"
                      variant={selecting ? "default" : "outline"}
                      onClick={selecting ? finishSelecting : startSelecting}
                      disabled={savingSelection}
                      className="h-6 gap-1 px-2 text-[11px]"
                    >
                      {savingSelection ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : selecting ? (
                        <Check className="h-3 w-3" />
                      ) : null}
                      {selecting
                        ? t("dialogs.datasetDetail.curateDone")
                        : t("dialogs.datasetDetail.curateEpisodes")}
                    </Button>
                  )}
                </div>
                {activeExcluded.size > 0 && (
                  <p className="mt-0.5 text-[10px] text-muted-foreground">
                    {t("dialogs.datasetDetail.includedCount", {
                      included: includedEpisodes.length,
                      total: episodes?.length ?? 0,
                    })}
                  </p>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {episodes && episodes.length > 0 ? (
                  <div className="space-y-0.5">
                    {episodes.map((ep) => (
                      <div
                        key={ep.episode_index}
                        className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors hover:bg-accent ${
                          selectedEpisode === ep.episode_index
                            ? "border border-border bg-accent"
                            : "border border-transparent"
                        }`}
                      >
                        {selecting && (
                          <Checkbox
                            checked={!draftExcluded.has(ep.episode_index)}
                            onCheckedChange={() => toggleDraftEpisode(ep.episode_index)}
                            aria-label={t("dialogs.datasetDetail.includeEpisodeAria", {
                              index: ep.episode_index,
                            })}
                            className="h-3.5 w-3.5 shrink-0 border-muted-foreground/40 data-[state=checked]:border-primary"
                          />
                        )}
                        <button
                          type="button"
                          onClick={() => setSelectedEpisode(ep.episode_index)}
                          className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        >
                          <span className="w-6 shrink-0 font-mono text-[11px] text-muted-foreground">
                            {String(ep.episode_index).padStart(2, "0")}
                          </span>
                          <span className="min-w-0 flex-1 truncate">
                            {t("dialogs.datasetDetail.episodeRow", {
                              index: ep.episode_index,
                            })}
                          </span>
                          <span className="shrink-0 font-mono text-[10.5px] tabular-nums text-muted-foreground">
                            {ep.duration.toFixed(1)}s
                          </span>
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="px-1 text-xs leading-relaxed text-muted-foreground">
                    {t("dialogs.datasetDetail.episodesEmpty")}
                  </p>
                )}
              </div>
            </div>

            {/* Non-related while picking episodes to train on — rename, tags,
                visibility, upload/download, delete-whole-dataset all live
                here, none of it relevant to "which episodes." A dimmed
                pointer-events-none wrapper keeps this from needing a
                `disabled` prop threaded through DatasetInfoCard's internals. */}
            <div
              className={selecting ? "pointer-events-none p-3 opacity-40" : "p-3"}
              aria-disabled={selecting}
            >
              <DatasetInfoCard
                repoId={repoId}
                onDownloaded={() => setReloadKey((k) => k + 1)}
              />
            </div>

            <div className="p-3">
              <Button
                onClick={handleTrain}
                disabled={selecting || includedEpisodes.length === 0}
                title={
                  selecting
                    ? t("dialogs.datasetDetail.finishCuratingFirst")
                    : undefined
                }
                className="w-full gap-2"
              >
                <Boxes className="h-4 w-4" />
                {t("dialogs.datasetDetail.trainSkill")}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DatasetDetailDialog;
