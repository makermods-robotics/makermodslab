export default {
  header: {
    myLibrary: "我的库",
    openStudio: "打开技能工作室",
  },
  hero: {
    words: {
      run: "运行",
      train: "训练",
      share: "分享",
    },
    // Chinese keeps the verb first, so the slot placement carries over.
    slogan: "<0/>机器人技能",
    searchPlaceholder: "整理我的桌面…",
    searchLabel: "搜索技能",
  },
  newSkill: {
    title: "＋ 新建技能",
    subtitle: "采集、训练、部署 — 无需离开此页面。",
    aria: "打开技能工作室 — 采集、训练并部署新技能",
    steps: {
      collect: {
        label: "1 · 采集",
        sub: "录制数据集 — 或选择已有的数据集",
      },
      train: {
        label: "2 · 训练",
        sub: "数据集 → 策略 → 训练任务",
      },
      deploy: {
        label: "3 · 部署",
        sub: "在机器人上运行技能",
      },
    },
  },
  skills: {
    sectionLabel: "技能",
    previous: "上一批技能",
    next: "下一批技能",
    empty: "没有符合搜索条件的技能。",
    open: "打开技能 {{title}}",
    previewAlt: "{{title}} 运行预览",
    previewPlaceholder: "运行预览",
    comingSoon: "即将推出",
    localCheckpoint: "本地检查点",
    steps: "{{steps}} 步",
    private: "私有",
  },
  skillNames: {
    sortingSocks: "整理袜子",
    openingBottleCaps: "拧开瓶盖",
    foldingTowels: "折叠毛巾",
    stackingCubes: "堆叠方块",
  },
  badge: {
    mine: "我的",
    makermods: "MAKERMODS 官方支持",
    community: "社区",
    wip: "开发中",
  },
  jobState: {
    running: "运行中",
  },
  footer: {
    poweredBy: "由 <1>LeRobot</1> 提供支持",
    documentation: "文档",
    // Product names — identical in every language, keyed only so the link
    // list has one uniform shape.
    github: "GitHub",
    discord: "Discord",
  },
} as const;
