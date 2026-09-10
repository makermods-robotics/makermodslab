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
  coachBadge: {
    paused: "已暂停",
    armMoving: "机械臂移动中",
    recording: "录制中",
    limp: "已松力",
    notHome: "未回到原位",
    aligned: "已对齐并保持",
    mayBeStiff: "可能仍有力矩",
  },
  coachBanner: {
    watching: {
      title: "观察中",
      hint: "策略正在驾驶。按空格键接管 — 主臂会先自行移动到机器人当前的姿态，请轻握并顺着它。",
    },
    held: {
      title: "已保持",
      hint: "机械臂正维持当前姿态。此时不会录制任何内容。按空格键接管；任务已完成则按回车键。",
    },
    handingOver: {
      title: "正在交接",
      hint: "机械臂正在移动到位 — 不要与它较劲，等它停稳。",
    },
    resetting: {
      title: "正在归位…",
      hint: "机械臂正在退回起始姿态 — 先别去抓它。到位后它会松力并停住。此时不会录制任何内容。",
    },
    saving: {
      title: "正在保存…",
      hint: "正在把这次纠正写入磁盘。机械臂保持不动；写完后策略会继续。",
    },
    // 接管的第一次按键。两条手臂都静止、也还没有开始录制 —— 事实与 “已保持”
    // 相同，但给出的指示正好相反，所以措辞以 “要做什么” 开头。
    poised: {
      title: "请握住主臂",
      hint: "主臂已对齐到机器人的姿态并保持不动 — 此时还没有开始录制。请握住主臂，然后再次按空格键：那一按会松开主臂并同时开始驾驶和录制。如果两条手臂没有对上，先用手把它们对齐。",
    },
    correcting: {
      title: "你在驾驶",
      hint: "每一帧都在录制。按空格键交还。任务已完成时，按回车键会保存这次纠正并同时结束本次尝试。如果你需要先把机械臂倒回去，到达策略见过的状态时按 G 键 — 这会把挽回与纠正分开。",
    },
    // 复位之后的停驻状态。与 “已保持” 区分开，是因为给操作者的指示不同 ——
    // 在这里提示 “按空格接管” 正是它要避免的那次误录。
    parked: {
      title: "已复位",
      hint: "机械臂已归位且不会再动 — 它已松力，可以放心抓握。自由挪动它和现场，然后按回车键开始下一次。",
    },
    // 复位没有干净完成：从臂始终没有回到起始位。
    parkedStuck: {
      title: "请检查机械臂",
      hint: "机械臂已停下且不会再动，但它始终没有回到起始姿态 — 可能有东西挡住了它。请用手把它移回去，然后按回车键开始下一次。",
    },
    // 已归位，但无法确认机械臂是否已松力 —— 无法确认的事就不要许诺，
    // 因为操作者会照着这个许诺伸手去抓它。
    parkedRigid: {
      title: "已复位 — 机械臂可能仍有力",
      hint: "机械臂已归位且不会自行移动，但它可能仍在保持扭矩。不要硬掰 — 先整理现场，然后按回车键开始下一次。",
    },
    // 一次接管的两个阶段。在 lerobot 看来它们都是 `correcting`，
    // 区别只在于给出的指示 —— 而这正是关键：RaC 的数据效率来自操作者
    // 把这两件事当成两项不同的工作来做。
    recovering: {
      title: "正在挽回",
      hint: "先把机械臂带回策略见过的状态。到位后按 G 键 — 此后的一切都算作纠正。",
    },
    correcting2: {
      title: "正在纠正",
      hint: "现在把正确做法示范给它。干净利落地完成这一子任务，不要过度纠正。按空格键交还并保存。",
    },
    starting: {
      title: "正在启动…",
      hint: "正在加载策略并连接机械臂。",
    },
  },

  coach: {
    // 会话中的控制按钮。每个按钮旁还会显示对应按键，按键名不翻译 ——
    // space 和 esc 指的是键盘上的实体键。
    takeOver: "接管控制",
    handBack: "交还控制",
    // 接管的第二次按键：确认手已握住静止且已对齐的主臂，并开始录制。
    confirmHold: "我已握住 — 开始驾驶",
    discardAndReset: "丢弃这次纠正并复位",
    discard: "丢弃这次纠正",
    hold: "保持 — 冻结机械臂",
    resume: "让策略继续",
    ending: "正在结束…",
    endSession: "结束会话并保留纠正数据",
    // 在会话结束页提供的后续入口。{{percent}} 是评测中做错的比例，
    // 是数字，因此没有复数形式。
    offer: "策略表现不佳？来指导它",
    offerWithGap: "指导它 — 修好它做错的那 {{percent}}%",
    reset: "任务完成 — 复位以开始下一次尝试",
    recovered: "已挽回 — 纠正从这里开始",
    // {{seconds}} 由调用方预先格式化。用时长而不是编号来指代这次纠正：操作员刚
    // 刚亲眼看着它发生，脑子里没有回合编号，但知道自己大概操作了多久。
    dropLast: "删除这段 {{seconds}} 的纠正",
    // 在可删除窗口已经关闭之后按下退格键。运行器只会在内存里保留最近一次纠正，
    // 并在下一次接管时把它写入数据集；写入之后 lerobot 无法再从已打开的数据集里
    // 取出某一个回合。所以这里要把规则讲清楚，而不是保持沉默，并指向仍然可用的
    // 那条撤销路径。
    nothingToDropHint: {
      title: "那次纠正已经保存了",
      body: "只有最近一次纠正可以撤回，而现在没有待撤回的纠正。在它之前的都已经写入数据集了。会话结束后你仍然可以删除整个数据集。",
    },
    nothingToDelete: "没有可删除的纠正数据",
    // 在运行器收尾之后才送达的指令。这不算失败。
    sessionEnded: {
      title: "会话已经结束",
      body: "该指令是在会话结束之后才送达的 — 没有改变任何东西。你的纠正数据是安全的。",
    },
    // 会话结束时的交接。<0> 是训练数据集的名称。
    handoffNext:
      "下一步：把这些纠正数据与 <0>{{dataset}}</0>（该检查点最近一次训练所用的数据集）合并，然后基于合并结果微调它。训练只接受一个数据集，所以合并这一步不能省。",
    handoffAction: "合并并微调",
    handoffHint: "会打开已选好两个数据集的合并流程，接着打开以该技能为基底的训练面板。",
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
      recovered: "标记挽回结束",
      marking: "正在标记…",
      dropLast: "删除最后一次纠正",
      dropping: "正在删除…",
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
