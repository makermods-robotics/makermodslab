export default {
  corner: {
    create: "机器人",
    createTooltip: "创建机器人",
    settings: "机器人设置",
    settingsFor: "{{name}} 的机器人设置",
    selectFirst: "请先选择一个机器人",
    activeLabel: "机器人：",
    selectRobot: "选择机器人",
    setUp: "设置你的机器人",
    robots: "机器人",
    mode: {
      single: "单臂",
      bimanual: "双臂",
    },
    status: {
      ready: "就绪",
      needsSetup: "需要设置",
    },
    empty: "还没有机器人。创建一个开始使用 — 接下来你将设置端口、标定和摄像头。",
    createItem: "创建机器人…",
    renameItem: "重命名机器人…",
    deleteItem: "删除机器人…",
    teleop: "遥操作",
  },
  rename: {
    title: "重命名机器人",
    description: "标定分配、端口和摄像头会随机器人一起迁移。",
    newName: "新名称",
    submit: "重命名",
    submitting: "正在重命名…",
  },
  delete: {
    title: "确定删除 {{name}} 吗？",
    fallbackName: "该机器人",
    description:
      "这会移除该机器人已保存的配置（端口、标定分配、摄像头）。标定文件本身仍保留在库中。此操作无法撤销。",
    confirm: "删除机器人",
  },
  teleop: {
    startedTitle: "遥操作已启动",
    startedFallback: "已为 {{name}} 启动遥操作。",
    startedWarningTitle: "已启动，但有警告",
    failedWithWarningTitle: "无法启动遥操作 — 请检查机械臂",
    failedTitle: "无法启动遥操作",
    failedFallback: "启动失败。",
    disabledReason: "{{name}}{{gap}} — 请打开机器人设置",
  },
  setupGap: {
    missingCalibration: "缺少{{arms}}的标定",
    noPort: "未分配{{arms}}的端口",
    stale: "引用的标定文件已不存在 — 请重新分配或重新标定",
    clauseJoin: "，",
    armJoin: "和",
    // Chinese has a single plural category, and the "臂" suffix already lives
    // in each arm label, so the list needs no wrapper of its own.
    armList_other: "{{arms}}",
  },
  arm: {
    leader: "主臂",
    follower: "从臂",
    leftLeader: "左主臂",
    leftFollower: "左从臂",
    rightLeader: "右主臂",
    rightFollower: "右从臂",
  },
} as const;
