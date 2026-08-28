export default {
  tour: {
    aria: "Feature tour",
    stepCounter: "Step {{current}} of {{total}}",
    skip: "Skip",
    back: "Back",
    next: "Next",
    done: "Done",
  },
  // Tour step copy, keyed by tour id + step. tours.ts holds only these keys.
  launchpad: {
    search: {
      title: "Find a skill",
      description: "Search for a skill by name, or browse the ones below.",
    },
    skills: {
      title: "Browse skills",
      description:
        "Skills others have trained and shared — run one, or use it as a starting point for your own.",
    },
    newSkill: {
      title: "Build your own",
      description:
        "Collect a dataset, train a policy, and deploy it to your robot — all without leaving this page.",
    },
    robot: {
      title: "Set up your robot",
      description: "Add and configure the arm you'll record and run skills on.",
    },
    library: {
      title: "Your library",
      description: "Datasets, models, and training jobs you've saved live here.",
    },
  },
  studio: {
    collect: {
      title: "1 · Collect",
      description: "Record a new dataset here, or pick one you already have.",
    },
    train: {
      title: "2 · Train",
      description: "Turn a dataset into a trained policy.",
    },
    deploy: {
      title: "3 · Deploy",
      description: "Run a trained skill on your robot.",
    },
  },
} as const;
