export default {
  header: {
    myLibrary: "My library",
    // Used for both aria-label and title on the same control — one key, two
    // attributes, rather than two keys that could drift apart.
    openStudio: "Open the policy studio",
  },
  hero: {
    // The rotating verbs in the slogan. Order matches WORDS in Hero.tsx.
    words: {
      run: "Run",
      train: "Train",
      share: "Share",
    },
    // <0/> is the rotating verb above. Kept as one phrase so the translator
    // controls word order rather than us concatenating fragments.
    slogan: "<0/> robot policies",
    searchPlaceholder: "Clean my desk…",
    searchLabel: "Search policies",
  },
  newPolicy: {
    title: "＋ New Policy",
    subtitle: "Collect, train, deploy — without leaving this page.",
    aria: "Open the policy studio — collect, train, and deploy a new policy",
    steps: {
      collect: {
        label: "1 · Collect",
        sub: "record a dataset — or pick an existing one",
      },
      train: {
        label: "2 · Train",
        sub: "datasets → policy → training job",
      },
      deploy: {
        label: "3 · Deploy",
        sub: "run a policy on your robot",
      },
    },
  },
  policies: {
    sectionLabel: "Policies",
    previous: "Previous policies",
    next: "Next policies",
    empty: "No policies match your search.",
    open: "Open policy {{title}}",
    previewAlt: "{{title}} rollout preview",
    previewPlaceholder: "rollout preview",
    comingSoon: "Coming soon",
    localCheckpoint: "local checkpoint",
    // `steps` arrives pre-formatted ("16k") — deliberately not an i18next
    // `count` plural, which would try to re-derive a plural from a string.
    steps: "{{steps}} steps",
    private: "private",
  },
  // Curated display names, keyed by the slug in POLICY_NAME_KEYS (PolicyCard).
  // The Hub repo ids they map from are data and never appear here.
  policyNames: {
    sortingSocks: "Sorting socks",
    openingBottleCaps: "Opening bottle caps",
    foldingTowels: "Folding towels",
    stackingCubes: "Stacking cubes",
  },
  badge: {
    mine: "MINE",
    makermods: "MAKERMODS SUPPORTED",
    community: "COMMUNITY",
    wip: "WIP",
  },
  // Backend job-state enum → label. The enum value itself is never translated;
  // an unmapped state falls back to the raw string.
  jobState: {
    running: "running",
  },
  footer: {
    // <1> wraps the LeRobot link. "LeRobot", "GitHub" and "Discord" are product
    // names and stay untranslated.
    poweredBy: "Powered by <1>LeRobot</1>",
    documentation: "Documentation",
    // Product names — identical in every language, keyed only so the link
    // list has one uniform shape.
    github: "GitHub",
    discord: "Discord",
  },
} as const;
