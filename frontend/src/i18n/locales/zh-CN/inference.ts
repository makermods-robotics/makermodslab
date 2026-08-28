export default {
  dialogTitle: "推理会话",
  phase: {
    downloadingModel: "正在下载模型…",
    starting: "正在启动…",
    loadingPolicy: "正在加载策略…",
    connecting: "正在连接机械臂…",
    running: "运行中",
    stopping: "正在停止…",
    stopped: "已停止",
    error: "出错 — 请查看日志",
    resetting: "请重置场景",
    finished: "评估完成",
    aborted: "评估已中止",
    // 指导阶段。本表中唯一从操作者视角（而非系统视角）措辞的几项 —
    // 它们说明的是此刻由谁在控制机械臂。
    watching: "策略驾驶中 — 留意失败迹象",
    holding: "已保持 — 机械臂正维持当前姿态",
    correcting: "你在驾驶 — 正在录制",
    handingOver: "正在交接 — 机械臂正在移动",
    saving: "正在保存这次纠正…",
    attemptReset: "已归位 — 机械臂已松力，可自由摆放",
  },
  result: {
    success: "成功",
    failure: "失败",
    error: "错误",
  },
  pill: {
    failed: "已失败",
    ranWithWarning: "运行完成，但有警告",
    aborted: "已中止",
    evaluationComplete: "评估完成",
    resetTheScene: "请重置场景",
    settingUp: "正在准备",
    running: "运行中",
    finished: "已完成",
    coaching: "指导中",
    coachingComplete: "指导完成",
    coachingStopped: "指导已停止",
  },

  // 指导过程中的大横幅。标题刻意醒目：操作者盯着的是机械臂而不是屏幕，
  // 只能用余光扫到这里。
  coachBanner: {
    watching: {
      title: "观察中",
      hint: "策略正在驾驶。按空格键接管 — 主臂会先自行移动到机器人当前的姿态，请轻握并顺着它。",
    },
    held: {
      title: "已保持",
      hint: "机械臂正维持当前姿态。此时不会录制任何内容。按空格键接管，按 R 键复位以开始下一次尝试。",
    },
    handingOver: {
      title: "正在交接",
      hint: "机械臂正在移动到位 — 不要与它较劲，等它停稳。",
    },
    resetting: {
      title: "正在复位…",
      hint: "正缓缓退回起始姿态，然后松力，方便你用手重新摆放。此时不会录制任何内容。",
    },
    saving: {
      title: "正在保存…",
      hint: "正在把这次纠正写入磁盘。机械臂保持不动；写完后策略会继续。",
    },
    correcting: {
      title: "你在驾驶",
      hint: "正在挽回并纠正 — 每一帧都在录制。",
    },
    // 复位之后的停驻状态。与 “已保持” 区分开，是因为给操作者的指示不同 ——
    // 在这里提示 “按空格接管” 正是它要避免的那次误录。
    parked: {
      title: "就绪",
      hint: "机械臂已归位并松力 — 可自由挪动它和现场。按空格键开始下一次尝试。",
    },
    starting: {
      title: "正在启动…",
      hint: "正在加载策略并连接机械臂。",
    },
  },

  coach: {
    // 会话中的控制按钮。每个按钮旁还会显示对应按键，按键名不翻译 ——
    // space 和 esc 指的是键盘上的实体键。
    takeOver: "接管",
    handBack: "交还给策略",
    discard: "丢弃这次纠正",
    hold: "保持 — 冻结机械臂",
    resume: "让策略继续",
    ending: "正在结束…",
    endSession: "结束会话并保留纠正数据",
    reset: "任务完成 — 复位以开始下一次尝试",
    startAttempt: "开始第 {{attempt}} 次尝试",
    takeOverInstead: "改为接管",
    // 末尾的分隔符是特意保留的：它作为已录制时长的前缀。
    attemptPrefix: "第 {{attempt}} 次尝试 · ",
    // 实时计数。{{saved}} 与 {{target}} 都是原始数字；在运行器报告目标数之前
    // {{target}} 是 “?”，因此这里不是复数形式。
    tally: "已完成 {{saved}} / {{target}} 次纠正",
    recorded: "已录制 {{duration}}",
    savingTo: "正在保存到 {{dataset}}",
    // 总结。数据集名称已知与否会改变整句措辞，因此分成两种写法；
    // <0> 强调的是数据集名称，属于数据。
    summarySaved_other: "已保存 {{count}} 次纠正。",
    summarySavedTo_other: "已保存 {{count}} 次纠正到 <0>{{dataset}}</0>。",
    summaryNextSteps:
      "要把它们变成更好的策略：把这个数据集与该检查点<0>最近一次</0>训练所用的数据集合并 — 如果你做过微调，那就是微调用的数据集，而不是最初的示范数据 — 然后基于同一个检查点在合并结果上继续微调。训练只接受一个数据集，所以合并这一步不能省。两个步骤分别在数据集库和训练面板中。",
    summaryNone: "没有保存任何纠正 — 本次会话没有可用于训练的数据。",
    // 从总结页删掉整个数据集 —— 这是一种正常结果，而不是失败路径。
    delete: "删除这些纠正数据",
    deleteConfirm: "确定删除？此操作无法撤销",
    deleting: "正在删除…",
    deleted: "已删除",
    deleteRefused: "服务器拒绝了此次删除。",
    deleteFailed: "无法删除该数据集",
    deletedToast: {
      title: "纠正数据已删除",
      body: "{{dataset}} 已从磁盘移除。",
    },
    // 指令名称。`failed` 中的 {{action}} 取自下面这些名称，
    // 因此它们按短语而非按钮文案来措辞。
    cmd: {
      failed: "{{action}}失败",
      takeOver: "接管",
      takingOver: "正在接管…",
      handBack: "交还",
      handingBack: "正在交还…",
      hold: "保持",
      holding: "正在保持…",
      resume: "恢复策略",
      resuming: "正在恢复…",
      discard: "丢弃纠正",
      discarding: "正在丢弃…",
      reset: "复位",
      resetting: "正在归位…",
      startNextAttempt: "开始下一次尝试",
      starting: "正在启动…",
    },
  },
  toast: {
    startedWarningTitle: "已启动，但有警告",
    failedTitle: "推理失败",
    ranWithWarningTitle: "运行完成，但清理时有警告",
    seeLog: "详情请查看推理日志。",
    evalAbortedTitle: "评估已中止",
    evalCompleteTitle: "评估完成",
    evalAbortedDescription: "结果不完整 — 未记录准确率。",
    evalAccuracy: "成功率 {{percent}}%。",
    evalNoScoreable: "没有可评分的回合。",
    coachStoppedTitle: "指导会话已停止",
    coachCompleteTitle: "指导完成",
    // 这里 count 为 0 也是一种正常结果 —— 会话可能一次纠正都没保存。
    coachSaved_other: "已保存 {{count}} 次纠正。",
    finishedTitle: "推理已结束",
    finishedDescription: "运行已完成。",
    hungTitle: "推理似乎已卡住",
    hungDescription: "回放已超出预定时长 {{seconds}} 秒。",
    lostConnectionTitle: "与后端的连接已断开",
    stopFailedTitle: "停止失败",
    endEpisodeFailedTitle: "无法结束该回合",
    nextEpisodeFailedTitle: "无法开始下一回合",
  },
  eval: {
    // Chinese has a single plural category.
    episodesTotal_other: "{{count}} 个回合",
    episodeProgress: "第 {{index}} 回合，共 {{total}} 回合",
    unknownTotal: "?",
    done: "已完成 {{count}} 个",
    errorsExcluded: "出错的回合不计入准确率 — 硬件故障不等于策略失败。",
    abortedSummary: "在 {{total}} 个回合中完成 {{done}} 个后中止 — 未对部分运行记录准确率。",
    succeeded: "{{scored}} 个回合中成功 {{success}} 个",
    excludedAsErrors: "（{{count}} 个因出错被排除）",
    noScoreable: "没有可评分的回合 — 所有回合都出错了，无法给出准确率。",
    episodeCrashed: "第 {{index}} 回合崩溃",
    episodeCrashedBody: "该回合既不算成功也不算失败。可以继续运行下一回合，或中止本次评估。",
    episodeRecorded:
      "第 {{index}} 回合记录为<1>{{result}}</1>。请重新布置场景，然后开始下一回合 — 没有计时限制，可以从容准备。",
  },
  settingUp: "正在加载策略并连接硬件…",
  policyRef: "策略：{{ref}}",
  unknownPolicy: "（未知）",
  outcome: {
    ranWithWarning: "运行完成，但清理时有警告",
    runFailed: "运行失败",
  },
  button: {
    close: "关闭",
    starting: "正在开始…",
    startEpisode: "开始第 {{index}} 回合",
    aborting: "正在中止…",
    abortEvaluation: "中止评估",
    endingEpisode: "正在结束回合…",
    taskSucceeded: "任务成功 — 结束回合",
    stopping: "正在停止…",
    stop: "停止",
  },
  download: {
    progress: "{{done}} / {{total}}",
    soFar: "已下载 {{done}}",
    starting: "正在开始下载…",
  },
  log: {
    title: "推理日志",
    failedPlaceholder: "本次运行在回放进程启动前就失败了，因此没有日志 — 请查看上方的错误信息。",
    emptyPlaceholder: "本次运行尚无日志 — 还没有开始产生输出。",
  },
} as const;
