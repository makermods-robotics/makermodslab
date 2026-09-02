import { useCallback, useEffect, useRef } from "react";

/**
 * Short audio cues for the coaching session.
 *
 * WHY. Every feedback channel in coaching is visual, and the operator is
 * looking at the ARM, not the screen — that is the whole job during a rollout,
 * watching for the moment the policy goes wrong. So the events that most need
 * to reach them are exactly the ones they are least positioned to see: a
 * correction saved (a number moved), a correction discarded (nothing moved at
 * all), a takeover refused (a line of text appeared somewhere).
 *
 * Sound is the standard answer and upstream lerobot already reaches for it —
 * `--play_sounds` speaks "Correction 3 saved" out loud. We do not use speech:
 * a spoken sentence takes about a second and a half, which is longer than the
 * events it describes, and it arrives too late to mean anything. Short tones
 * are pre-attentive; the operator learns three of them in a session.
 *
 * WebAudio oscillators rather than audio files: no assets to ship, no network
 * fetch, nothing to fail at exactly the wrong moment, and no decode latency.
 *
 * DESIGN OF THE CUES. Pitch direction carries the meaning, so they are
 * distinguishable without being learned and without relying on timbre:
 *
 *   granted   rising  (440→660)  you now have the arm
 *   handback  falling (660→440)  the policy has it back — the mirror image
 *   saved     single confident tone, the "kept it" sound
 *   discarded low falling thud — deliberately unpleasant, deliberately short
 *   refused   two flat low beeps — a refusal, not a failure
 *
 * Browsers refuse to start an AudioContext without a user gesture. Starting a
 * coaching session IS one (the operator clicked Start), but the context is
 * created lazily on the first cue anyway, and every failure path is swallowed:
 * a session must never break because audio was unavailable, muted or blocked.
 */

export type CoachingCue = "granted" | "handback" | "saved" | "discarded" | "refused";

// [frequency Hz, start offset s, duration s] per cue.
const TONES: Record<CoachingCue, Array<[number, number, number]>> = {
  granted: [
    [440, 0, 0.08],
    [660, 0.07, 0.12],
  ],
  handback: [
    [660, 0, 0.08],
    [440, 0.07, 0.12],
  ],
  saved: [[880, 0, 0.13]],
  discarded: [
    [320, 0, 0.1],
    [200, 0.09, 0.18],
  ],
  refused: [
    [300, 0, 0.09],
    [300, 0.15, 0.09],
  ],
};

// Low enough not to startle someone standing next to a robot, high enough to
// carry across a workshop over a servo whine.
const GAIN = 0.12;

export const useCoachingCues = (enabled: boolean) => {
  const ctxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    return () => {
      const ctx = ctxRef.current;
      ctxRef.current = null;
      if (ctx) void ctx.close().catch(() => {});
    };
  }, []);

  return useCallback(
    (cue: CoachingCue) => {
      if (!enabled) return;
      try {
        if (!ctxRef.current) {
          const Ctor =
            window.AudioContext ??
            (window as unknown as { webkitAudioContext?: typeof AudioContext })
              .webkitAudioContext;
          if (!Ctor) return;
          ctxRef.current = new Ctor();
        }
        const ctx = ctxRef.current;
        // Suspended is the normal state on a tab that has been backgrounded;
        // resume is a promise we deliberately don't await — the cue for THIS
        // event is already stale by the time it resolves, but the next one
        // will play.
        if (ctx.state === "suspended") void ctx.resume().catch(() => {});
        const now = ctx.currentTime;
        for (const [freq, at, dur] of TONES[cue]) {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = "sine";
          osc.frequency.value = freq;
          // A hard start/stop on a sine clicks audibly. Ramping the envelope
          // over a few ms is the difference between a tone and a pop.
          gain.gain.setValueAtTime(0, now + at);
          gain.gain.linearRampToValueAtTime(GAIN, now + at + 0.01);
          gain.gain.linearRampToValueAtTime(0, now + at + dur);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(now + at);
          osc.stop(now + at + dur + 0.02);
        }
      } catch {
        /* audio is a courtesy; never let it break a live session */
      }
    },
    [enabled],
  );
};
