# Localization

How to translate the MakerMods Lab UI, and how to add a new language.

This document is meant to be the only thing you need to read. If you follow it, your
change will pass CI and will not break the app for anyone.

---

## 1. The central idea: localization here is **cosmetic only**

This is the one rule everything else follows from:

> **Only pixels change. Anything a machine reads stays byte-identical.**

The UI is a skin over a robot toolchain. Almost every screen shows strings that are not
prose at all — port paths, calibration file names, codec identifiers, repo ids, enum
values from the backend. Translating one of those does not "translate the UI", it
corrupts data.

Concretely, a localization change must **never**:

- alter a request body, a header, or a query parameter (there is no `Accept-Language`);
- alter anything written to `localStorage`, `sessionStorage`, or disk;
- alter a value submitted by a form (a `<select>` may *display* Chinese while
  *submitting* the original English value);
- alter a string that any code parses, compares, `startsWith`-matches, or re-reads;
- touch the Python backend at all.

The backend is not localized. Server-generated prose (`data.message`, `status.error`,
`status.hint`, `ApiError.detail`) renders in English in every language. You translate the
*client-side fallback* next to it, never the server's text.

```tsx
// Correct: server text wins, our fallback is translated.
description: data.message || t("robot.teleop.failedFallback"),
```

If you remember nothing else, remember the test in [§6](#6-is-this-string-copy-or-data).

---

## 2. Quick start

### Translate one string

```tsx
// 1. Add the English text to the namespace catalog for your area.
//    frontend/src/i18n/locales/en/launchpad.ts
export default {
  hero: { searchLabel: "Search skills" },
} as const;

// 2. Add the same key to every other language.
//    frontend/src/i18n/locales/zh-CN/launchpad.ts
export default {
  hero: { searchLabel: "搜索技能" },
} as const;

// 3. Use it.
import { useTranslation } from "react-i18next";

const { t } = useTranslation();
<input aria-label={t("launchpad.hero.searchLabel")} />;
```

Keys are **type-checked**. `t("launchpad.hero.searchLabl")` is a compile error, not a
string that quietly renders as its own key.

### Verify

```bash
cd frontend
npx tsc --noEmit -p tsconfig.app.json   # NOTE the -p flag; see §10
npm test
```

---

## 3. How the system is wired

Everything lives under `frontend/src/i18n/`.

| File | What it is |
|---|---|
| `config.ts` | Language identity, detection, persistence. No i18next import, so it is testable on its own. |
| `index.ts` | Boots i18next. Exports the `resources` object. |
| `types.d.ts` | Makes the **English catalog the source of truth for key types**. This is what turns a typo into a compile error. |
| `locales/en/*.ts` | One file per feature area, plus `index.ts` that combines them. |
| `locales/zh-CN/*.ts` | Same tree, translated. |
| `catalogs.test.ts` | Key parity, orphans, empty values, placeholder drift. |
| `keyUsage.test.ts` | Every key referenced in source actually resolves. |
| `config.test.ts` | Detection and persistence. |

Two more files outside that directory:

- `frontend/src/contexts/LanguageContext.tsx` — holds the active language, persists it,
  and syncs `document.documentElement.lang`.
- `frontend/src/components/LanguageSwitcher.tsx` — the picker.

### Boot order

`main.tsx` does a side-effect `import "@/i18n"` **before** rendering, so i18next is
initialized before any component can call `t()`. `App.tsx` then mounts
`<LanguageProvider>` directly inside `<ThemeProvider>` — above `<BrowserRouter>`, so a
language change re-renders every route.

### Why catalogs are static TypeScript, not JSON fetched at runtime

The built bundle in `frontend/dist/` is committed and served by FastAPI as plain
`StaticFiles`. There is nowhere to serve locale JSON from, and no loading state worth
designing. Static imports also mean the catalogs are type-checkable.

### The namespaces

One per feature area, mirroring the component directories, so a migration touches exactly
one catalog file per language:

```
common        shared words (cancel, close) + dataset-name validation
launchpad     the dashboard route
onboarding    tour copy + the tour chrome
robot         robot corner, teleop toasts, arm labels, setup-gap sentences
robotConfig   the robot settings dialog
studio        Collect / Train / Deploy panels
recording     camera config + recording session
calibration   calibration library
jobs          job cards, dropdowns, libraries
library       dataset library, library sheet, delete semantics
landing       hub dialogs, pickers, auth, info cards
dialogs       dataset detail, replay, teleop, skill dialogs
inference     the inference session dialog
training      configurator, config cards, monitoring
pages         NotFound, Teleoperation
shared        app-level chrome (update notice, tab guard, log panel, cameras)
```

Key names read `namespace.component.thing`, camelCase.

---

## 4. The mechanical cases

These are the 90% you can convert without thinking hard.

**Plain JSX text**

```tsx
- <p>No skills match your search.</p>
+ <p>{t("launchpad.skills.empty")}</p>
```

**Attributes** — `aria-label`, `title`, `alt`, `placeholder`

```tsx
- <input placeholder="Clean my desk…" aria-label="Search skills" />
+ <input
+   placeholder={t("launchpad.hero.searchPlaceholder")}
+   aria-label={t("launchpad.hero.searchLabel")}
+ />
```

When `aria-label` and `title` (or the visible text) are **identical**, use **one** key at
both sites. Two keys will drift. Keep two only when the strings genuinely differ — e.g.
`"Robot settings"` on the button vs `"Robot settings for {{name}}"` in the tooltip.

**Interpolation**

```tsx
- `Started teleoperation for ${robot.name}.`
+ t("robot.teleop.startedFallback", { name: robot.name })
```

```ts
startedFallback: "Started teleoperation for {{name}}.",
```

**Toasts** — call `t()` at the call site. There are two toast systems: the shadcn
`useToast` (`hooks/use-toast.ts`, the primary; `title`/`description` accept `ReactNode`,
so `<Trans>` output works) and `sonner` (used only by `UrdfContext`). Both behave the
same for our purposes.

---

## 5. The hard cases

Each of these will silently produce a broken or untranslatable UI if you convert it
naively. They are ordered roughly by how often they come up.

### 5.1 Module-level constants — the most common trap

A constant at module scope is evaluated **once, at import time**, long before React or
i18n exist. A `t()` call there resolves against whatever language happened to load first
and then never updates.

```tsx
// WRONG — frozen at import time
const STEPS = [{ label: t("studio.collect.label") }];

// WRONG — plain English, unreachable by translation
const STEPS = [{ label: "1 · Collect" }];
```

Keep the array at module scope but store **keys**, and resolve during render:

```tsx
const STEPS = [
  { labelKey: "launchpad.newSkill.steps.collect.label",
    subKey:   "launchpad.newSkill.steps.collect.sub" },
] as const;

const { t } = useTranslation();
// ...
{STEPS.map((step) => <span key={step.labelKey}>{t(step.labelKey)}</span>)}
```

This applies to every label map in the codebase: tour steps, badge labels, phase maps,
optimizer labels, filter arrays, status presentation maps.

> **React keys:** if you were using the label as the React `key`, switch to a stable
> identifier. A translated `key` remounts every row on a language change.

### 5.2 Sentences with embedded markup — use `<Trans>`

Never concatenate translated fragments. Word order differs between languages, and a
sentence assembled from pieces cannot be reordered by a translator.

```tsx
// WRONG
<p>Powered by <a href="...">LeRobot</a></p>

// RIGHT
<Trans
  i18nKey="launchpad.footer.poweredBy"
  components={[<span key="0" />, <a key="1" href="..." className="..." />]}
/>
```

```ts
poweredBy: "Powered by <1>LeRobot</1>",
```

The numbers in the string are indices into the `components` array.

> **⚠️ `<Trans>` does not resolve *nested* slots when you pass an index array.**
> A string like `<0>text<1/></0>` silently drops the inner element. This bit the
> external-link icon in `HfAuthBanner`. If you need markup inside markup, extract a small
> local component and flatten the string to top-level slots.

Watch for `{" "}` whitespace tokens in the JSX you are replacing — a mechanical
extraction loses or duplicates the spaces around them.

### 5.3 Plurals

English needs two forms; Chinese needs one. Use i18next's suffixes, never a ternary.

```ts
// en
episodesTotal_one:   "{{count}} episode",
episodesTotal_other: "{{count}} episodes",

// zh-CN — a single plural category
episodesTotal_other: "{{count}} 个回合",
```

```tsx
t("inference.eval.episodesTotal", { count: n })
```

Replace both escape hatches you will find in old code:

```tsx
- `${n} camera stream(s)`
- `arm${labels.length > 1 ? "s" : ""}`
```

> **`count` is magic.** i18next uses `count` to pick the plural variant. If you are
> passing an **already-formatted** string (`"16.7k"`, `"2h30m"`), pass it under a
> different name — `{{steps}}`, `{{total}}`, `{{time}}` — or i18next will try to derive a
> plural from a string and pick the wrong form.

### 5.4 Ternary chains that produce text

Give **each branch its own complete key**. Do not build a stem and append.

```tsx
// WRONG — the last branch also lowercases translated text
const subtitle = isRunning ? `started ${when}` : present.label.toLowerCase();

// RIGHT
const subtitle = isRunning
  ? t("jobs.jobCard.startedAt", { when })
  : t(SUBTITLE_STATE_KEYS[state]);
```

**Never call `.toLowerCase()` / `.toUpperCase()` on translated text.** It is a no-op in
Chinese and actively wrong in several languages (Turkish dotted/dotless i).

Sometimes a singular/plural pair is really *two different sentences*, not one plural. Keep
those as separate keys — Chinese's single plural category would otherwise collapse them
into one.

### 5.5 Sentence-fragment builders — the genuinely hard case

Some helpers return a *piece* of a sentence that a caller splices into a larger one. That
shape cannot be translated. It has to be **restructured**, not extracted.

The reference example is `robotSetupGap` (`frontend/src/lib/robotSetupGap.ts`). It used to
return an English verb phrase assembled from arm lists:

```ts
// BEFORE — subject, verb, conjunction, list separator and plural are all
// English-shaped and spliced together at four different call sites.
const armList = (labels) => `${labels.join(" and ")} arm${labels.length > 1 ? "s" : ""}`;
parts.push(`is missing a calibration for the ${armList(noConfig)}`);
return parts.join(" and ");

// caller:
`${robot.name} ${robotSetupGap(robot)} — open Robot settings`
```

The fix has three parts:

1. **Return structure**, not prose:
   ```ts
   robotSetupGaps(robot, scope) → { missingCalibration: ArmKey[], missingPort: ArmKey[], staleConfig: boolean }
   ```
2. **Add a renderer** that turns the structure into a whole sentence via the catalog:
   ```ts
   formatRobotSetupGap(t, robot, scope): string
   ```
3. **Keep the English function** for non-React callers, and **freeze its output with a
   test** so the restructure provably changes nothing for English users.

`lib/robotSetupGap.test.ts` and `lib/datasetName.test.ts` are that freeze — assert the
exact current strings *before* you touch the function.

Three helpers in the codebase already went through this. Reuse them; do not re-derive:

| Module | Structured | Localized renderer | English (frozen) |
|---|---|---|---|
| `lib/robotSetupGap.ts` | `robotSetupGaps()` | `formatRobotSetupGap(t, …)` | `robotSetupGap()` |
| `lib/datasetName.ts` | `datasetNameIssue()`, `datasetRepoIdIssue()` | `formatDatasetNameIssue(t, …)` | `validateDatasetName()`, `validateDatasetRepoId()` |
| `lib/deleteSemantics.ts` | `resolveDeleteAction()` → `titleKey` / `descriptionKey` / `confirmKey` | render with `t()` at the call site | — |

`datasetName.ts` is worth reading as a cautionary tale: it used to produce its
"Namespace…" messages by running `.replace("Dataset name", "Namespace")` over its own
English output. String surgery on prose survives no translation.

### 5.6 A pure data function vs. its display twin

When a function's result is used **both** for display and for logic (searching, sorting,
comparison), do not translate it in place. Split it:

```ts
/** UNTRANSLATED. Search matches against this, so "sock" keeps working in any language. */
export function skillTitle(m: ModelItem): string { … }

/** What the user reads. */
export function skillDisplayTitle(t: TFunction, m: ModelItem): string { … }
```

`SkillSlider`'s filter then matches the English title **and** the translated one — a
strict superset, so nothing that matched before stops matching:

```tsx
skillTitle(m).toLowerCase().includes(q) ||
skillDisplayTitle(t, m).toLowerCase().includes(q) ||
m.id.toLowerCase().includes(q)
```

Same pattern: `skillAuthorLabel` / `skillDisplayAuthorLabel`.

> Translating a title that search reads is the classic way to "only change pixels" and
> still break a feature. Check every consumer before you translate a shared helper.

### 5.7 Backend enums rendered as text

The enum **value** is data and never changes. Only the label does. Always keep a fallback
so an unmapped value renders its raw string rather than a key path:

```tsx
{t(`launchpad.jobState.${job.state}` as never, { defaultValue: job.state })}
```

`as never` is required because the key is built at runtime; that also means TypeScript
cannot check it — which is exactly what `keyUsage.test.ts` exists to catch ([§9](#9-the-safety-nets)).

### 5.8 Text passed as props or into hooks

A component that receives display text from its parent should receive a **key**, or text
already resolved by `t()` in the parent's render — never a module-level English constant.

Two anti-patterns to fix when you meet them:

```tsx
// English default baked into a prop signature — invisible to translation
function AdvancedSection({ title = "Advanced parameters" }) {}

// Fix: keep the prop optional, resolve the default at render
function AdvancedSection({ title }) {
  const { t } = useTranslation();
  const heading = title ?? t("studio.common.advancedParameters");
}
```

```tsx
// Copy passed into a hook
useSessionExitGuard({ confirmMessage: t("dialogs.replay.leaveConfirm") })
```

`RobotStatus` in `studio/panel/primitives.tsx` shows the shape to prefer: it takes copy as
`children` — *"the caller owns the copy, this owns the look."*

### 5.9 Text the CSS shouts, and text the CSS renders

**`uppercase` + letter-spacing.** CSS `uppercase` is a no-op on Chinese, but the
`tracking-*` that usually accompanies it is not — it renders CJK visibly over-spaced. Drop
both together:

```tsx
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";

const { language } = useLanguage();
<span className={cn("text-[10px] font-semibold",
  isCaselessScript(language) ? "" : "uppercase tracking-wide")} />
```

For the shared `.eyebrow` utility use the hook — `.eyebrow` bundles `uppercase` *and*
`tracking-[0.08em]` and lives in Tailwind's utilities layer, where a `tracking-normal`
override beside it would be a source-order coin flip:

```tsx
import { useEyebrowClass } from "@/hooks/useEyebrowClass";
const eyebrow = useEyebrowClass();
```

Store catalog values in **sentence case** and let CSS do the shouting. Uppercase baked
into the data (`"MAKERMODS SUPPORTED"`) cannot be un-shouted.

**Text rendered from an attribute.** `.media-slot` renders `content: attr(data-label)`.
Translate it by setting the attribute from `t()` in the component — never by editing the
stylesheet:

```tsx
<div className="media-slot" data-label={t("launchpad.skills.previewPlaceholder")} />
```

### 5.10 `window.confirm` — leave it in English

The OK/Cancel buttons and the frame come from the **browser's** locale, not the app's. A
translated question with English buttons is worse than an English question.

There are 13 such call sites (job cards and dropdowns, the training job dialog, and the
shared session exit guard). They stay English and carry a comment saying why. Converting them to Radix
`AlertDialog`s is a real improvement, but it is a **UX change and belongs in its own PR** —
do not smuggle it into a translation change.

---

## 6. Is this string copy, or data?

Ask one question:

> **Does anything other than a human's eyes read this string?**

If a backend, a regex, a file path, a form submission, a URL segment, a config file, or a
comparison reads it — it is **data**. Leave it in English.

Real examples from this codebase, all deliberately untranslated:

| String | Why it is data |
|---|---|
| `CAMERA_NAME_PRESETS` — `wrist`, `top`, `front`, `side` | The label **is** the camera name written into the robot record and used as a dataset feature key. Translating stores a Chinese camera name on disk. |
| `formatDurationShort()` output — `2h30m` | Written into an input and re-parsed against a regex mirroring `_DURATION_FULL_RE` in `makermodslab/train.py`. A translated `h` makes the field reject its own suggestion. |
| `FOURCC_OPTIONS`, `BACKEND_OPTIONS` | Codec ids and lerobot `Cv2Backends` enum names, sent verbatim. |
| `DISCONTINUITY_ERROR_PREFIX` | Matched with `startsWith()` against backend error text. |
| Calibration file names, `device_type` (`teleop`/`robot`) | Build API paths and pick config fields. |
| Policy-type identifiers (`act`, `smolvla`) | Wire identifiers. Their display names are product names anyway. |
| `org/name`, `my_dataset`, `hf_…`, `/path/to/…` placeholders | Literal shapes the user must type or match. |
| `SPACE`, `ENTER`, `DEL` legends | Names of physical keys. |
| `HF_HUB_OFFLINE`, `Torque_Limit`, `job.read` | Identifiers quoted inside a sentence. Keep them literal inside the `<Trans>` slot. |
| Product names — MakerMods, LeRobot, Hugging Face, W&B, ACT, SmolVLA, GitHub, Discord | Names. |

**The label/value split.** Where a constant currently serves as both, split it — translate
the label, keep the value:

```tsx
const MODE_OPTIONS = [
  { value: "single",   labelKey: "landing.createRobot.modeSingle" },
  { value: "bimanual", labelKey: "landing.createRobot.modeBimanual" },
] as const;
```

**Filter arrays**: translate `label`, never `key`.

---

## 7. Adding a new language

Say you are adding Japanese (`ja`). Five steps.

**1. Create the catalog directory.** Copy `locales/en/` to `locales/ja/`, keeping the file
names and the key tree identical, and translate the values.

```bash
cp -r frontend/src/i18n/locales/en frontend/src/i18n/locales/ja
```

Remove English-only plural variants where the language does not need them (Japanese, like
Chinese, has one plural category: keep `_other`, drop `_one`).

**2. Register it** in `frontend/src/i18n/config.ts`:

```ts
export const SUPPORTED_LANGUAGES = [
  { code: "en",    label: "English" },
  { code: "zh-CN", label: "简体中文" },
  { code: "ja",    label: "日本語" },
] as const;
```

> Use the **endonym** — a language picker is the one place a language must not be named in
> the language the user is trying to leave.

**3. Teach detection about it** — `matchLanguage()` in the same file maps a browser
BCP-47 tag onto a shipped catalog:

```ts
if (lower === "ja" || lower.startsWith("ja-")) return "ja";
```

**4. Declare whether the script has letter case** — `isCaselessScript()`, also in
`config.ts`. Japanese does not, so add it; otherwise every `uppercase tracking-*` heading
renders over-spaced.

**5. Add it to the resources** in `frontend/src/i18n/index.ts`:

```ts
import ja from "./locales/ja";

export const resources = {
  en: { translation: en },
  "zh-CN": { translation: zhCN },
  ja: { translation: ja },
} as const;
```

Then `npm test` — `catalogs.test.ts` will tell you precisely which keys you missed.

Nothing else needs touching. The switcher renders `SUPPORTED_LANGUAGES` automatically, and
it is already mounted in both headers.

---

## 8. Adding a new namespace (a new feature area)

1. Create `locales/en/<area>.ts` and `locales/<lang>/<area>.ts` for **every** language:
   ```ts
   export default {} as const;
   ```
2. Import and add it to the `export default {…}` object in **both**
   `locales/en/index.ts` and every other language's `index.ts`.
3. Use keys as `<area>.<component>.<thing>`.

Scaffolding all languages up front is what lets several people migrate different areas in
parallel without colliding on a shared file.

---

## 9. The safety nets

Three tests. Know what each catches, because together they are why this is safe to change.

**`i18n/catalogs.test.ts`** — the most valuable one.
- every language has a catalog;
- no key in English is **missing** from another language;
- no **orphan** key that English does not define (catches a stale key after a rename);
- no empty string values;
- **interpolation placeholders match** — if English says `{{name}}` and the translation
  says `{{nombre}}`, that fails.

Plural suffixes are normalized before comparison, so `_one` in English and only `_other`
in Chinese is correct, not an error.

**`i18n/keyUsage.test.ts`** — covers the gap the type system cannot.
Most `t()` calls are type-checked, but around twenty sites index a map or build a
template literal and cast `as never`. A wrong key there compiles cleanly and renders the key path to a
user. This test scans the source for key-shaped literals and asserts each one resolves in
every language.

**`i18n/config.test.ts`** — detection order, unsupported-tag fallback, and that a throwing
`localStorage` does not take down the app.

Plus the **frozen-English** tests that guard restructured helpers:
`lib/robotSetupGap.test.ts` and `lib/datasetName.test.ts`. If you restructure a
sentence-builder, write one of these **first**.

And `contexts/LanguageContext.test.tsx` proves the tree repaints in another language with
no reload and that `document.documentElement.lang` follows.

---

## 10. Verification checklist

```bash
cd frontend
npx tsc --noEmit -p tsconfig.app.json
npm run lint
npm test
npm run build          # then: git checkout -- dist
```

> **⚠️ The `-p tsconfig.app.json` flag is not optional.** The root `tsconfig.json` has
> `"files": []` and only project references, so a bare `npx tsc --noEmit` type-checks
> **nothing** and always "passes". CI runs the project form on every PR (#87), so a
> type error you skip past here fails the build later rather than never.

Record the baseline **before** you start and compare against it — some lint and type
errors pre-date any given change, and `npm test` has pre-existing failures in
`lib/onboarding/storage.test.ts` on Node 25 (its built-in `localStorage` shadows jsdom's
and lacks `.clear()`).

Do **not** commit `frontend/dist/` — `npm run build` rewrites it and CI rebuilds it on
`main` and `staging`. Revert it if a build dirtied your tree.

> **⚠️ Do not run `makermodslab --dev` on a branch you are about to push.** It runs
> `npm install` on every start (`makermodslab/scripts/makermodslab.py:381`), which on
> macOS prunes platform-gated optional dependencies from `frontend/package-lock.json` and
> produces a lockfile that fails `npm ci` on the Linux runner. If it happens:
> `git checkout -- frontend/package-lock.json`.

### Manual pass

Automated tests cannot see layout. Before opening a PR:

1. Switch languages — the UI repaints with **no page reload**.
2. Reload — the choice persists; `<html lang>` is correct.
3. Open the studio — **the switcher is present in that header too** (the overlay is
   `fixed inset-0` and covers the viewport; a Launchpad-only mount disappears).
4. Clear `makerlab:onboarding-completed`, reload — the tour runs translated, including
   the step counter and the Skip / Back / Next / Done buttons, and the text does not
   overflow the tooltip boxes.
5. Check any `uppercase tracking-*` chips and `.eyebrow` headings for over-spacing.
6. Confirm the display fonts fall back acceptably — `font-orbitron` and `font-display`
   have **no CJK coverage**.
7. Search still finds things by their English name.
8. Untranslated areas still render correct English — no raw key paths leaking through.

### Proving it stayed cosmetic

With DevTools open, do the same action in both languages and diff:

- **Network** — request bodies and headers must be identical.
- **Application → Local Storage** — the only difference is `makerlab:language`.
- **On disk** — after recording a dataset or saving a robot config while in another
  language, files under `~/.cache/huggingface/lerobot/` must still be pure ASCII.

---

## 11. Reference

### Hooks and helpers

```ts
import { useTranslation, Trans } from "react-i18next";
const { t } = useTranslation();

import { useLanguage } from "@/contexts/LanguageContext";
const { language, setLanguage } = useLanguage();

import { isCaselessScript, SUPPORTED_LANGUAGES } from "@/i18n/config";
import { useEyebrowClass } from "@/hooks/useEyebrowClass";
```

For a `t` passed into a plain function, type it as `TFunction`:

```ts
import type { TFunction } from "i18next";
export function formatSomething(t: TFunction, x: X): string { … }
```

Do **not** type it as `ReturnType<typeof useTranslation>["t"]` — that triggers a
`TS2589: excessively deep` error.

### Storage

| Key | Value |
|---|---|
| `makerlab:language` | `"en"` \| `"zh-CN"` — always an ASCII locale tag |

### Detection order

stored choice → `navigator.languages` (first shipped match) → English.

### Common review comments

- "This is a module-level constant" → §5.1
- "Don't concatenate fragments" → §5.2, §5.5
- "Use a real plural" → §5.3
- "That string is data" → §6
- "Search reads this" → §5.6
- "Uppercase is baked into the value" → §5.9
