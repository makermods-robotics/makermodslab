/**
 * "jobs" namespace — 训练任务与模型区域 (components/jobs/*) 及 lib/jobsApi.ts
 * 持有的标签。
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 产品名（MakerMods、LeRobot、Hugging Face、Hub、W&B、GitHub）不翻译；任务 id、
 * 仓库 id、数据集名称、策略类型、机型和文件路径都是数据，原样渲染。
 */
export default {
  jobState: {
    running: "运行中",
    done: "已完成",
    failed: "失败",
    // 与 Stop 按钮同一个词：这个状态就是用户按下停止后产生的。
    interrupted: "已停止",
  },
  stage: {
    running: "运行中",
    queued: "排队中",
    scheduling: "调度中",
    starting: "启动中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
    unknown: "未知",
  },
  location: {
    local: "本地",
    cloud: "云端",
    imported: "已导入",
    // 产品名，保持英文。
    hub: "Hub",
    localTitle: "在本机运行",
    cloudTitle: "在 Hugging Face 云端运行",
    fromHub: "来自 Hub",
    fromHubTitle: "从 Hugging Face Hub 仓库导入",
  },
  meta: {
    policy: "策略",
    dataset: "数据集",
    steps: "步数",
    flavor: "机型",
    created: "创建于",
    owner: "所有者",
    image: "镜像",
    updated: "更新于",
  },
  actions: {
    run: "运行",
    runInferenceCheckpoint: "用此检查点运行推理",
    runInferenceModel: "用此模型运行推理",
    runInference: "运行推理",
    fineTune: "微调",
    fineTuneHint: "基于此模型的权重开始一次微调训练",
    download: "下载此检查点",
    rename: "重命名",
    renameAria: "重命名模型",
    openHubJob: "打开 Hub 任务页面",
    viewOnHub: "在 Hub 上查看",
  },
  rename: {
    title: "重命名模型",
    description: "仅设置显示名称 —— 底层的{{target}}（<0/>）不会被移动或更改。",
    targetRun: "运行记录",
    targetHubRepo: "Hub 仓库",
    placeholder: "新名称",
    submit: "重命名",
    submitting: "正在重命名…",
    empty: "名称不能为空。",
    toastTitle: "模型已重命名",
    toastDescription: "“{{from}}” → “{{to}}”。",
  },
  errors: {
    trainingAlreadyRunning: "已有另一次训练正在进行。请先停止它。",
  },
  hubJob: {
    fallbackTitle: "任务 {{id}}…",
    removeAria: "从列表中移除任务",
    removeTitle: "从列表中移除",
  },
  progress: {
    starting: "启动中…",
  },
  checkpointDropdown: {
    placeholder: "选择检查点",
    latest: "最新",
    step: "第 {{step}} 步",
  },
  jobCard: {
    stopAria: "停止任务",
    deleteAria: "删除任务",
    trainingStarting: "训练启动中…",
    subtitle: {
      // {{when}} 是英文的相对时间（"5m ago"）——时间格式化不在本次改动范围内。
      started: "开始于 {{when}}",
      ended: "结束于 {{when}}",
    },
    subtitleState: {
      running: "运行中",
      done: "已完成",
      failed: "失败",
      interrupted: "已停止",
    },
    resumeLatest: "从最新检查点继续",
    resumeStep: "从第 {{step}} 步继续",
    resumeHint:
      "打开训练表单，从该检查点继续训练。算力默认沿用该检查点所属运行的位置，开始前可以改选。",
    install: "安装 {{target}}",
    downloadFailed: "下载失败",
  },
  hubModelCard: {
    uploaded: "已上传",
    deleteAria: "删除模型仓库",
    deletedToast: "模型仓库已删除",
    dialog: {
      title: "删除模型仓库",
      description:
        "这将从 Hugging Face Hub 永久删除该模型仓库及其文件，且无法撤销。",
      confirmPrompt: "输入 <0/> 以确认。",
      submit: "永久删除",
      submitting: "正在删除…",
    },
  },
  importModal: {
    title: "导入技能",
    description:
      "指向一个本地目录或一个 Hugging Face 仓库。它会出现在你的技能中，可直接用于推理。",
    sourceLabel: "本地路径或 Hugging Face 仓库 id",
    // 示例路径与仓库 id 是数据，保持原样；只翻译中间的连接词。
    sourcePlaceholder: "/path/to/pretrained_model  或  user/my-policy",
    nameLabel: "显示名称（可选）",
    namePlaceholder: "我导入的策略",
    submit: "导入",
    submitting: "正在导入…",
    alreadyImportedTitle: "已经导入过",
    alreadyImportedDescription: "“{{name}}”已在你的模型列表中。",
  },
  jobsData: {
    stopping: "任务正在停止",
    stopFailed: "停止失败",
    removed: "任务已删除",
    deleteFailed: "删除失败",
    dismissed: "任务已从列表中移除",
    dismissFailed: "移除失败",
  },
  jobsDropdown: {
    triggerAria: "选择一次训练运行",
    listAria: "训练运行",
    placeholder: "选择一次运行",
    stopAria: "停止此次运行",
    resumeAria: "从最新可用的检查点继续",
    openHubAria: "在 Hub 上打开此任务",
    openHubTitle: "在 Hub 上打开",
    hubJobTitleWithOwner: "Hugging Face 任务 · {{owner}}",
    hubJobTitle: "Hugging Face 任务",
    groups: {
      lab: "从 MakerMods Lab 启动",
      hub: "其他 Hub 任务",
      untrackedLab: "未跟踪 · 从 MakerMods Lab 启动",
      untrackedHub: "未跟踪 · 其他 Hub 任务",
    },
    hideUntracked: "隐藏未跟踪",
    untracked: "未跟踪（{{total}}）",
  },
  jobsLibrary: {
    title: "训练任务",
    refresh: "刷新任务列表",
    searchPlaceholder: "搜索任务",
    filters: {
      all: "全部",
      local: "本地",
      online: "在线",
    },
    empty: {
      search: "没有符合搜索条件的任务。",
      local: "没有本地任务。",
      online: "没有在线任务。",
      none: "还没有训练任务。",
    },
    firstRun: "还没有训练任务。在上方开始一次训练吧。",
    signIn: "登录 Hugging Face 以查看你的云端任务。",
    missingJobRead:
      "你的 Hugging Face 令牌缺少 job.read 权限，因此无法列出云端任务。",
    missingJobReadRich:
      "你的 Hugging Face 令牌缺少 <0/> 权限，因此无法列出云端任务。",
    localError: "无法加载本地任务：{{error}}",
    cloudError: "无法加载云端任务：{{error}}",
    checkpointsError: "无法加载检查点",
    noResumeTitle: "没有可继续的检查点",
    noResume: {
      notResumable: "此次运行当前的状态无法继续训练。",
      noCheckpoints: "此次运行及其继续自的运行都没有保存任何检查点。",
      ownerDone:
        "此次运行可继续的每个检查点都属于一次已达到目标步数的运行，其学习率调度已经用完。请改为从最终检查点做微调。",
      atTarget:
        "此次运行可继续的每个检查点都已达到其目标步数。请调高目标步数以继续，或从最终检查点做微调。",
      siblingCap:
        "此次运行谱系中剩下的检查点都保存于该运行到达的步数之后，因此它们属于共用同一云端输出的另一次续训。",
      other: "此次运行的谱系中没有可用于继续训练的检查点。",
    },
  },
  modelCard: {
    deleteAria: "删除模型",
    deleteTitle: "移除",
    checkpointPlaceholder: "没有检查点",
    trained: "训练于 {{when}}",
    created: "创建于 {{when}}",
    reason: {
      noCheckpointToRun: "目前还没有可运行的检查点。",
      finetuneWhileRunning: "此次运行仍在进行中，暂时无法微调。",
      noCheckpointToFinetune: "目前还没有可用于微调的检查点。",
      noCheckpointToDownload: "目前还没有可下载的检查点。",
      hubImportWeights: "此模型的权重保存在 Hub 上，而不在本机。",
      importedFromDisk: "从磁盘导入 —— 检查点已经位于 {{path}}",
      importedNoExport: "导入的模型不会被重新导出 —— 请打开原始文件夹。",
      cloudCheckpoints: "云端运行的检查点保存在 Hub 上，而不在本机。",
      localOnly: "只有本地训练运行才有可下载的检查点。",
      noCheckpointsToChoose: "目前还没有可供选择的检查点。",
      oneCheckpoint: "只有一个检查点 —— 没有可选项。",
    },
  },
  modelsLibrary: {
    title: "你的技能",
    importSkill: "导入技能",
    searchPlaceholder: "搜索技能",
    empty: "还没有技能。训练一个，或用“导入技能”从 Hub 或本地文件夹添加一个。",
    noMatch: "没有匹配的模型。",
    filters: {
      all: "全部",
      trained: "已训练",
      imported: "已导入",
      uploaded: "已上传",
    },
  },
} as const;
