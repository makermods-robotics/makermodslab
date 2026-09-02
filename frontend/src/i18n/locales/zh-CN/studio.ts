/**
 * "studio" namespace — 技能工作室及其三个面板。
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 产品名（MakerMods、LeRobot、Hugging Face、Hub、ACT、SmolVLA、RTC、SO-101）
 * 一律保留原文。数据（数据集/模型/任务/机器人/摄像头名称、repo id、策略类型、
 * 检查点步数）由插值原样渲染，后端返回的文案也保持原样。
 */
export default {
  overlay: {
    title: "技能工作室",
    backToMenu: "返回主菜单",
    close: "关闭工作室",
    sections: {
      collect: "采集数据集",
      train: "训练",
      deploy: "部署策略",
    },
  },

  common: {
    advancedParameters: "高级参数",
    dismiss: "关闭",
  },

  collect: {
    title: "采集",
    entry: "录制新数据集",
    start: "开始录制",
    library: {
      title: "你的数据集",
      clearSelected: "清除已选数据集",
      refresh: "刷新数据集列表",
      merge: "合并数据集",
    },
    form: {
      intro: "为数据集命名并设置采集参数，然后在所选机器人上开始录制。",
      noRobot: "录制前请先选择或创建机器人 — 使用本窗口右上角的机器人菜单。",
      robotNotReady:
        "<0>{{name}}</0>{{gap}}。请先打开机器人设置，然后再开始录制。",
      datasetName: "数据集名称 *",
      nameHint:
        "仅可使用字母、数字以及 <0>.</0> <1>_</1> <2>-</2>；开头和结尾必须是字母或数字。",
      savedAs: "将保存为 <0>{{repoId}}</0>",
      loginHint: "登录 Hugging Face 以设置仓库归属账号。",
      task: "任务描述 *",
      taskPlaceholder: "例如：拿起红色方块并放到蓝色方格上",
      numEpisodes: "片段数量",
      episodeTime: "单个片段时长（秒）",
      resetTime: "复位时长（秒）",
      camerasEmptyRobot:
        "该机器人没有摄像头。请在机器人设置中添加摄像头以录制视频。",
      camerasNoRobot: "选择一个机器人以查看其摄像头。",
      advancedSummary: "流式编码、推送到 Hub",
      streamingLabel: "流式视频编码",
      streamingHint:
        "在采集过程中实时编码每一帧，因此每个片段几乎可以立即保存。取消勾选则回退到较慢的「先存 PNG 再编码」流程。",
      pushToHubLabel: "推送到 Hugging Face Hub",
      pushToHubHint:
        "录制结束后在后台将数据集上传到你的 Hugging Face 账号。取消勾选则仅保存在本地 — 之后仍可从数据集库手动上传。",
    },
    toast: {
      noRobotTitle: "未选择机器人",
      noRobotBody: "请先选择或创建机器人 — 使用右上角的机器人菜单。",
      missingDetailsTitle: "数据集信息不完整",
      missingDetailsBody: "请填写数据集名称和任务描述。",
      invalidNameTitle: "数据集名称无效",
      preparingCamerasTitle: "正在准备摄像头资源",
      releasingStreams_other: "正在释放 {{count}} 路摄像头视频流以便录制…",
      camerasReadyTitle: "摄像头资源已就绪",
      camerasReadyBody: "摄像头视频流已成功释放，正在开始录制…",
    },
  },

  handoff: {
    emptyTitle: "未录制任何片段",
    emptyBody: "没有保存任何内容 — 空数据集已被丢弃，不会占用磁盘空间。",
    savedTitle: "数据集 <0>{{repoId}}</0> 已保存",
    episodes_other: "{{count}} 个片段",
    trainOnThis: "用该数据集训练",
    upload: {
      button: "上传到 Hub",
      uploading: "正在上传到 Hub…",
      doneTitle: "已上传到 Hub",
      doneBody: "{{repoId}} 现已发布到 Hub。<0>查看数据集</0>",
      failedTitle: "上传失败",
      setupGuide: "打开配置指南",
      autoFailedTitle: "自动上传到 Hub 未启动",
    },
    milestone: {
      recording: {
        title: "很好，你的第一批片段已录制完成！",
        description:
          "可以在「训练」面板中用该数据集训练策略，也可以上传到 Hub 以便分享或在云端训练。",
      },
      hubUpload: {
        title: "已上传到 Hub！",
        description:
          "你的数据集现已公开且可分享 — 在 MakerLab 的任何位置引用它的 repo id，或在「训练」面板中基于它微调技能。",
      },
    },
  },

  train: {
    title: "训练",
    entry: "开始新的训练",
    start: "开始训练",
    intro: {
      fresh: "选择训练所用的数据、运行位置和训练时长，然后开始。",
      resume:
        "正在继续已有的训练 — 其数据集和权重已固定。设置还要继续训练多久，然后开始。",
    },
    dataset: {
      label: "数据集 *",
      resumeLabel: "数据集",
      resumeHint: "继承自被继续的那次训练。",
      remove: "移除 {{repoId}}",
      searchPlaceholder: "搜索数据集，或输入公开的 org/name Hub id",
      episodeSubset:
        "将使用 {{used}} 个回合进行训练 — 可在「我的库」中该数据集的查看器里调整。",
      episodeSubsetOfTotal:
        "将使用 {{total}} 个回合中的 {{used}} 个进行训练 — 可在「我的库」中该数据集的查看器里调整。",
      choose: "选择数据集",
      useHub: "使用 Hub 上的 <0>{{repoId}}</0>",
      useHubHint: "公开数据集 — 训练时按需拉取。",
      noMatches:
        "没有匹配的数据集。输入完整的 <0>org/name</0> id 即可使用任意公开的 Hugging Face 数据集。",
      hint: "你自己的数据集，或任意公开的 Hugging Face 数据集。",
      row: {
        episodes: "{{episodes}} 片段",
        hub: "Hub",
        weighted: "带权重",
        weightedTitle: "该数据集带有按回合的采样权重，训练时部分回合会被更频繁地采样",
      },
    },
    startingPoint: {
      label: "起点",
      scratch: "从零开始训练",
      fromBase: "基于基础模型训练",
      loading: "正在加载检查点…",
      finetuneHint: "基于该技能的最新检查点进行微调。",
      hint: "微调已有技能，或从零开始。",
      foundationHint: "微调已有技能，或基于其公开的基础模型训练。",
    },
    toast: {
      noCheckpointsTitle: "该技能没有检查点",
      noCheckpointsBody: "它没有可用于微调的已保存检查点。",
      baseFailedTitle: "无法加载起点",
    },
    milestone: {
      title: "训练已启动！",
      description:
        "可在上方的任务列表中查看进度。训练完成后，可在「部署」面板中在机器人上运行。",
    },
  },

  deploy: {
    title: "运行",
    picker: {
      placeholder: "选择技能",
      loading: "正在加载技能…",
      empty: "还没有已训练或已导入的技能",
      error: "无法加载技能。请检查服务器后重试。",
      failedBadge: "运行失败",
      hubDegraded: "无法连接 Hub — 正在显示本地技能和上次的 Hub 列表。",
      import: "导入技能",
      hint: "选择一个已训练的检查点，或已从 Hub 导入的技能，在机器人上运行。",
    },
    source: {
      hub: "hub",
      local: "本地",
      both: "本地 · hub",
    },
    intro: "在机器人上运行该技能，然后开始推理。",
    noRobot: "选择要运行的机器人 — 使用本窗口右上角的机器人菜单。",
    robotNotReady_other:
      "<0>{{name}}</0>{{gap}}。请先打开机器人设置，然后再运行推理。（推理只使用从臂 — 无需配置主臂。）",
    checkpoint: {
      label: "检查点",
      none: "该技能暂无可用的检查点。",
    },
    armMismatch: {
      bimanualCheckpoint:
        "该检查点是在<0>双臂机器人</0>上训练的（状态维度 {{dim}}，{{arms}} 条机械臂），而 <1>{{name}}</1> 是单臂机器人。请改选单臂检查点，或在右上角的机器人菜单中选择一台双臂机器人。",
      singleCheckpoint:
        "该检查点是在<0>单臂机器人</0>上训练的（状态维度 {{dim}}），而 <1>{{name}}</1> 是双臂机器人。请改选双臂检查点，或在右上角的机器人菜单中选择一台单臂机器人。",
    },
    task: {
      label: "任务描述",
      placeholder: "例如：拿起红色方块",
      hint: "该策略以语言为条件（{{policyType}}）。",
    },
    duration: {
      label: "最长时长（秒）",
      hint: "按单个片段计。片段跑满该时长而你没有判定成功，即记为失败。",
    },
    episodes: {
      label: "片段数",
      evalHint:
        "评测运行：共 {{episodes}} 个片段，每个之间会复位，最终统计为准确率。",
      hint: "保持为 1 即单次运行。大于 1 则启动带评分的评测。",
    },
    engine: {
      label: "推理引擎",
      sync: "Sync（默认）",
      rtc: "RTC — 实验性，控制更平滑",
      syncHint:
        "每个控制步执行一次策略前向推理。机械臂在动作块之间会短暂停顿。",
      rtcHint:
        "Real-Time Chunking 让推理与运动重叠进行，消除动作块之间的停顿。它也改变了动作的生成方式 — 在采信结果之前请先与 Sync 对比。",
    },
    cameras: {
      title: "摄像头",
      loading: "正在读取策略配置…",
      configError: "无法加载策略配置：{{error}}",
      none: "该策略不使用摄像头。",
      intro:
        "为策略训练时使用的每个摄像头名称绑定一个本机器人的摄像头。使用哪个摄像头、以何种方式打开由机器人决定（在机器人设置中修改）；采集分辨率则来自检查点。",
      captures: "以 {{width}}×{{height}} 采集 — 即该策略的分辨率",
      mismatch: "（{{name}} 在机器人设置中被设为 {{width}}×{{height}}）",
      disconnected: "已断开 — 请重新连接后再开始",
      select: "选择摄像头",
      robotHasNone: "该机器人没有摄像头 — 请在机器人设置中添加",
    },
    thumbnail: {
      released: "已释放",
      noPreview: "无预览",
    },
    advanced: {
      summary: "ACT 的时序集成",
      actionSelection: "动作选择",
      temporalEnsemble: "时序集成",
      temporalEnsembleHint:
        "对策略在每一步预测出的、相互重叠的动作块取平均，而不是开环执行单个动作块 — 运动更平滑，但策略每个控制步都要推理一次，因此速度更慢。",
      coeffLabel: "集成系数",
      coeffPlaceholder: "{{value}}（ACT 论文默认值）",
      coeffInvalid: "请输入大于 0 的数字。",
      coeffHint:
        "权重为 exp(-系数 × 时间差)：系数越大越偏向最新的预测，越小则平均得越均匀。ACT 论文取 {{value}}。",
    },
    actions: {
      start: "开始推理",
      startEval: "开始评测（{{episodes}}）",
      starting: "正在启动…",
      checking: "正在检查…",
      stop: "停止推理",
      stopping: "正在停止…",
    },
    toast: {
      loadSkillFailed: "无法加载该技能",
      startFailed: "无法启动推理",
      stoppingTitle: "正在停止推理",
      stoppingBody: "该次运行正在收尾。",
      stopFailed: "停止失败",
    },
    milestone: {
      title: "首个技能已部署！",
      description:
        "你的机器人刚刚运行了一个训练好的策略。随时回到这里重新部署、切换检查点，或运行其他技能。",
    },
  },
} as const;
