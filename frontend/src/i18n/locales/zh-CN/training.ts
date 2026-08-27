/**
 * "training" namespace — 训练面板的配置表单、安装引导、云端上传提示与任务监控对话框。
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 产品名与算法名（Hugging Face、Hub、W&B、lerobot、ACT、SmolVLA、GPU、AMP…）
 * 保持原文；超参数键名、优化器/W&B 选项值、时长字符串（"2h"、"3h30m"）是
 * 发往后端的数据，一律不翻译。
 */
export default {
  header: {
    title: "训练",
  },

  cloudNotice: {
    uploadHint: "点击下方的“{{action}}”上传，然后启动。",
  },

  resumeInherited: {
    note: "这些设置会从上一次运行的检查点重建 —— 续训是同一次实验的延续，在这里修改不会生效。若要用不同的设置训练，请改为从该检查点做微调。",
    short: "从上一次运行的检查点重建。",
  },

  configurator: {
    checkingEnvironment: "正在检查训练环境…",
    resumeStepError: "总步数必须大于检查点所在的步数（{{step}}）。",
    resume: {
      titleFromStep: "从第 {{step}} 步继续“{{name}}”",
      lrSeam: "总步数与原运行不同（{{from}} → {{to}}）。LeRobot 会按新的总步数重建学习率调度，因此在恢复点学习率可能回升，而不是接着原来的衰减继续。保持 {{from}} 可获得连续的调度。",
      titleFromLatest: "从最新检查点继续“{{name}}”",
      bodyLocal:
        "设置已按那次运行预填，且仍可编辑。数据集、策略、批大小和优化器都会从检查点本身重建，在这里修改不会影响续训 —— 但<0>步数</0>和检查点保存频率会生效。把步数设为高于续训起点才能继续训练（已预填为 {{steps}}）。",
      bodyCloud:
        "设置已按那次运行预填，且仍可编辑。数据集、策略、批大小和优化器都会从检查点本身重建，在这里修改不会影响续训 —— 但<0>步数</0>、检查点保存频率以及任务超时都会生效。把步数设为高于续训起点才能继续训练（已预填为 {{steps}}）。",
      jobTimeout:
        "任务超时：<0>{{timeout}}</0> —— 续训至少需要留出跑完剩余部分的时间。",
      jobTimeoutDefault: "24h（默认）",
    },
    finetune: {
      title: "基于“{{name}}”微调",
      titleWithStep: "基于“{{name}}”微调（第 {{step}} 步）",
      titleLatest: "基于“{{name}}”微调（最新检查点）",
      checkpointLabel: "检查点",
      body: "<0>全新训练</0>，从第 0 步开始：优化器全新初始化，策略权重从该检查点载入。",
    },
    tooltip: {
      localBusy: "已有另一个本地训练在运行",
      needAuth: "登录 Hugging Face 后才能使用云端算力",
      needFlavor: "请选择硬件规格",
      offlineDataset: "离线模式已开启 —— 数据集无法上传到 Hub",
      offlineCheckpoint: "离线模式已开启 —— 检查点无法上传到 Hub",
    },
    button: {
      uploading: "正在上传…",
      starting: "正在启动…",
      startTraining: "开始训练",
      startFinetuning: "开始微调",
      continueTraining: "继续训练",
      uploadAndStart: "上传并开始训练",
      uploadAndContinue: "上传并继续训练",
    },
    toast: {
      startedTitle: "训练已开始",
      errorTitle: "出错",
      datasetRequired: "必须填写数据集仓库 ID",
      uploadFailedTitle: "上传失败",
    },
  },

  policyField: {
    label: "策略",
    placeholder: "选择策略类型",
    hint: "本次运行所训练的网络架构。",
    hintLocked: "由起始点决定 —— 本次运行使用与源检查点相同的架构。",
  },

  target: {
    computeLabel: "算力",
    runnerLocal: "本地 —— 你的机器",
    runnerCloud: "Hugging Face 云端",
    resumeRunnerHint: "默认沿用这次运行原先的执行位置 —— 也可以切换到别处继续。",
    deviceLabel: "设备",
    deviceAuto: "自动（有 GPU 就用 GPU）",
    deviceCpu: "CPU",
    deviceHint: "lerobot 会自动检测 GPU（CUDA/MPS）；只有 CPU 是强制指定的。",
    hardwareLabel: "硬件",
    hardwareLoading: "加载中…",
    hardwarePlaceholder: "选择硬件",
    loginToHf: "请登录 HF",
    costHint:
      "显示的价格按运行小时计费。训练完成后，最终策略会上传到你的 HF 账号。",
  },

  essentials: {
    steps: "训练步数",
    stepsTotal: "总训练步数",
    stepsTotalHint: "从第 {{from}} 步恢复，将再训练 {{remaining}} 步。",
    stepsTotalHintLatest: "这是总步数，而非额外增加的步数。",
    stepsTotalTooLow: "必须大于 {{from}}——该运行已训练到这一步，否则不会训练任何内容。",
    batchSize: "批大小",
    runName: "运行名称",
    resumedFromStep: "从第 {{step}} 步开始",
    resumedFromLatest: "从最新检查点开始",
    runNamePolicyFallback: "策略",
    runNameDatasetFallback: "数据集",
    runNameHint: "可选 —— 会显示在任务卡片上，并可被搜索。",
    wandbEnable: "记录到 Weights & Biases",
    wandbProject: "W&B 项目名称",
    wandbEntity: "W&B 实体（可选）",
    wandbNotes: "W&B 备注（可选）",
    wandbNotesPlaceholder: "本次训练的备注…",
    wandbMode: "W&B 模式",
    wandbModeOnline: "在线",
    wandbModeOffline: "离线",
    wandbModeDisabled: "禁用",
    wandbDisableArtifact: "禁用 artifacts",
  },

  advanced: {
    summary: "优化器、学习率、日志频率、检查点等",
    sectionPolicyPreset: "策略预设",
    useAmp: "启用自动混合精度（AMP）",
    sectionTraining: "训练",
    randomSeed: "随机种子",
    sectionOptimizer: "优化器",
    optimizerLabel: "优化器",
    // 算法名，与英文保持一致，不翻译。
    optimizerName: {
      adam: "Adam",
      adamw: "AdamW",
      sgd: "SGD",
      multiAdam: "Multi Adam",
    },
    optimizerUnknown: "由策略预设决定",
    optimizerFixedByPolicy: "由 {{policy}} 策略预设决定 —— 优化器类型不可调整。",
    optimizerFixedGeneric: "优化器类型由策略预设决定，不可调整。",
    optimizerNoKnobs:
      "{{policy}} 预设按参数分组分别构建优化器，因此这里没有可设置的学习率或权重衰减。",
    learningRate: "学习率",
    weightDecay: "权重衰减",
    gradientClipping: "梯度裁剪",
    noGradClip: "{{policy}} 策略没有提供梯度裁剪设置。",
    noGradClipOrWeightDecay: "{{policy}} 策略没有提供梯度裁剪设置和权重衰减。",
    policyDefaultValue: "{{value}}（策略默认值）",
    usePolicyDefault: "使用策略默认值",
    sectionDataLoading: "数据加载",
    numWorkers: "工作进程数",
    numWorkersHint: "为 GPU 供数的 DataLoader 进程数。",
    sectionLogging: "日志与检查点",
    logFreq: "日志频率",
    logFreqExceeds:
      "⚠ 每 {{logFreq}} 步记录一次日志，超过了本次 {{steps}} 步的运行长度 —— 不会记录任何指标。",
    logFreqHint:
      "两次记录 loss/lr 之间相隔的步数。数值越小，曲线分辨率越高（每个点是一个窗口的平均值），但日志量也越大。",
    saveFreq: "保存频率",
    saveFreqExceeds:
      "⚠ 每 {{saveFreq}} 步保存一次，超过了本次 {{steps}} 步的运行长度 —— 不会保存任何检查点。",
    sectionCloud: "云端",
    jobTimeout: "任务超时",
    jobTimeoutPlaceholder: "2h（默认）",
    jobTimeoutInvalid:
      "请填写形如 “2h”、“45m” 或 “3h30m” 的时长（单位：s、m、h、d）。",
    jobTimeoutHint: "超过这个时长后 HF Jobs 会终止运行。留空则使用 2h 的默认值。",
  },

  datasetNotice: {
    title: "该数据集只存在于这台机器上",
    offline:
      "Hugging Face 云端从 Hub 读取训练数据，但服务器处于离线模式（<0>HF_HUB_OFFLINE</0>），因此无法上传 <1>{{repoId}}</1>。请关闭离线模式，或改为在本地训练。",
    body: "Hugging Face 云端从 Hub 读取训练数据，因此在训练开始前，<0>{{repoId}}</0> 会作为<1>私有</1>数据集上传。",
    bodyWithSize:
      "Hugging Face 云端从 Hub 读取训练数据，因此在训练开始前，<0>{{repoId}}</0>（约 {{size}}）会作为<1>私有</1>数据集上传。",
    uploading: "正在上传到 Hub… 数据集较大时可能需要几分钟。",
  },

  checkpointNotice: {
    title: "该检查点只存在于这台机器上",
    stepLabel: "第 {{step}} 步的检查点",
    latestLabel: "最新检查点",
    offlineResume:
      "Hugging Face 云端从 Hub 续训，但服务器处于离线模式（<0>HF_HUB_OFFLINE</0>），因此 <1>{{runName}}</1> 的{{stepLabel}}无法上传。请关闭离线模式，或改为在本地继续这次运行。",
    offlineFinetune:
      "Hugging Face 云端从 Hub 加载基础权重，但服务器处于离线模式（<0>HF_HUB_OFFLINE</0>），因此 <1>{{runName}}</1> 的{{stepLabel}}无法上传。请关闭离线模式，或改为在本地做这次微调。",
    bodyResume:
      "Hugging Face 云端从 Hub 续训，因此在任务开始前，<0>{{runName}}</0> 的{{stepLabel}}（包含权重和优化器状态）会上传到你账号下的<1>私有</1>仓库。再次从同一检查点续训会复用这次上传。",
    bodyFinetune:
      "Hugging Face 云端从 Hub 加载基础权重，因此在任务开始前，<0>{{runName}}</0> 的{{stepLabel}}（只含权重，不含优化器状态）会上传到你账号下的<1>私有</1>仓库。再次从同一检查点微调会复用这次上传。",
    privacy: "仅对你的账号可见 —— 不会公开发布。",
  },

  install: {
    titleDone: "安装完成",
    titleError: "安装失败",
    titleInstalling: "正在安装…",
    copyAria: "复制安装命令",
    copiedTitle: "已复制",
    copyFailedTitle: "复制失败",
    copyFailedDescription: "请手动选中命令并复制。",
    installNow: "立即安装",
    installing: "正在安装 <0>{{packageName}}</0>。通常大约需要 10 秒。",
    failedFallback: "安装失败。",
    tryAgain: "重试",
    readyTraining:
      "安装完成 —— 训练功能已立即可用，无需重启。如果没有自动解锁，刷新页面即可。",
    readyWandb:
      "安装完成 —— W&B 日志记录已立即可用，无需重启。如果没有自动解锁，刷新页面即可。",
    readyPolicyTraining:
      "安装完成 —— {{policy}} 训练已立即可用，无需重启。如果没有自动解锁，刷新页面即可。",
    readyPolicyInference:
      "安装完成 —— {{policy}} 推理已立即可用，无需重启。如果没有自动解锁，刷新页面即可。",
  },

  extraGate: {
    title: "尚未安装训练扩展",
    description:
      "训练需要 <0>accelerate</0> 包，而当前环境中没有安装它。安装后即可启用训练页面。",
  },

  wandbDialog: {
    title: "尚未安装 Weights & Biases",
    srDescription: "安装 wandb 包以启用 W&B 日志记录。",
    description:
      "启用 W&B 日志记录需要 <0>wandb</0> 包，而当前环境中没有安装它。安装后即可把本次运行记录到 W&B。",
  },

  policyExtra: {
    title: "{{policy}} 需要额外的软件包",
    srDescriptionTraining: "安装 {{target}} 以使用 {{policy}} 进行训练。",
    srDescriptionInference: "安装 {{target}} 以使用 {{policy}} 进行推理。",
    descriptionTraining:
      "训练 <0>{{policy}}</0> 策略需要 <1>{{packageName}}</1> 包（通过 <2>{{target}}</2> 安装），而当前环境中还没有它。安装后即可训练该策略。",
    descriptionInference:
      "运行 <0>{{policy}}</0> 策略需要 <1>{{packageName}}</1> 包（通过 <2>{{target}}</2> 安装），而当前环境中还没有它。安装后即可运行该策略。",
  },

  monitoring: {
    progress: "进度",
    startingUp: "训练正在启动…",
    eta: "预计剩余",
    warmingUp: "正在预热…",
    loss: "损失",
    learningRate: "学习率",
    waitingForMetrics: "等待第一个指标数据点…",
    logsTitle: "训练日志",
    logsEmpty: "暂无训练日志。开始训练后即可看到输出。",
  },

  jobDialog: {
    srTitle: "训练任务状态",
    back: "技能工作室",
    loadFailed: "无法加载任务 {{jobId}}：{{errorText}}",
    loading: "正在加载任务…",
    runnerLocal: "本地",
    cloudFallback: "云端",
    viewOnHub: "在 Hub 上查看 ↗",
    viewOnWandb: "在 W&B 上查看 ↗",
    stop: "停止",
    delete: "删除",
    runInference: "运行推理",
    noCheckpoints: "还没有检查点 —— 请等待第一次保存。",
    runOnRobot: "在机器人上运行",
    toast: {
      stoppingTitle: "正在停止…",
      stopFailedTitle: "停止失败",
      removedTitle: "任务已移除",
      deleteFailedTitle: "删除失败",
    },
  },
} as const;
