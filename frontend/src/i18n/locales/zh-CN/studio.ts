/**
 * "studio" namespace — 策略工作室及其三个面板。
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 产品名（MakerMods、LeRobot、Hugging Face、Hub、ACT、SmolVLA、RTC、SO-101）
 * 一律保留原文。数据（数据集/模型/任务/机器人/摄像头名称、repo id、策略类型、
 * 检查点步数）由插值原样渲染，后端返回的文案也保持原样。
 */
export default {
  overlay: {
    title: "策略工作室",
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
          "你的数据集现已公开且可分享 — 在 MakerLab 的任何位置引用它的 repo id，或在「训练」面板中基于它微调策略。",
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
      finetuneHint: "基于该策略的最新检查点进行微调。",
      hint: "微调已有策略，或从零开始。",
      foundationHint: "微调已有策略，或基于其公开的基础模型训练。",
    },
    toast: {
      noCheckpointsTitle: "该策略没有检查点",
      noCheckpointsBody: "它没有可用于微调的已保存检查点。",
      baseFailedTitle: "无法加载起点",
    },
    milestone: {
      title: "训练已启动！",
      description:
        "可在上方的任务列表中查看进度。训练完成后，可在「部署」面板中在机器人上运行。",
    },
  },

  // 指导结束后的交接卡片，与 Launchpad 上的 CollectHandoff 并列：
  // 产出了数据的会话，应该把下一步放在操作者真正会看到的地方。
  coachHandoff: {
    saved_other: "已保存 {{count}} 次纠正到 <0>{{dataset}}</0>",
    next: "把它们与 <0>{{dataset}}</0>（该技能最近一次训练所用的数据集）合并，然后基于合并结果微调它。训练只接受一个数据集，所以合并这一步不能省。",
    manual:
      "要把它们变成更好的策略，请与该检查点<0>最近一次</0>训练所用的数据集合并，然后基于同一个检查点在合并结果上微调。两个步骤分别在数据集库和训练面板中。",
    action: "合并并微调",
  },

  deploy: {
    title: "运行",
    picker: {
      placeholder: "选择策略",
      loading: "正在加载策略…",
      empty: "还没有已训练或已导入的策略",
      error: "无法加载策略。请检查服务器后重试。",
      failedBadge: "运行失败",
      hubDegraded: "无法连接 Hub — 正在显示本地策略和上次的 Hub 列表。",
      import: "导入策略",
    },
    source: {
      hub: "hub",
      local: "本地",
      both: "本地 · hub",
    },
    intro: "在机器人上运行该策略，然后开始推理。",
    noRobot: "选择要运行的机器人 — 使用本窗口右上角的机器人菜单。",
    robotNotReady_other:
      "<0>{{name}}</0>{{gap}}。请先打开机器人设置，然后再运行推理。（推理只使用从臂 — 无需配置主臂。）",
    // 指导模式的变体：它要通过主臂遥操作，所以这里的 {{gap}} 是全部机械臂的
    // 缺项，括号里说明主臂为什么也必须配好。
    robotNotReadyCoach_other:
      "<0>{{name}}</0>{{gap}}。请先打开机器人设置，然后再运行推理。（指导还会用到主臂 — 接管时你要用它遥操作，因此它也需要端口和标定。）",
    // 本次运行的形态。选项的取值（"single"/"eval"/"coach"）是前端据以分支的
    // 标识符 — 只有这些标签会被翻译。
    runMode: {
      label: "你想用这个技能做什么？",
      // 每一行在被选中之前就先说明它要你付出什么：它们并不是可以随手互换的
      // 菜单项，而选错往往要等到人站在机械臂前才发现。
      single: {
        title: "运行",
        what: "尝试一次，然后停止。",
        commitment: "无需上手",
      },
      eval: {
        title: "给它打分",
        what: "反复执行该任务，由你为每次尝试评判，汇总为成功率。",
        commitment: "片段之间需要上手 — 由你复位现场并为每次尝试评分",
      },
      coach: {
        title: "人在回路",
        what: "在它快要失败时接管。每次挽救都会保存为可用于微调的训练数据。",
        commitment: "当它快要失败时，用主臂接管从臂并采集数据",
      },
    },
    // 仅在运行模式为 “指导” 时显示的参数。
    coaching: {
      correctionsLabel: "计划采集的纠正次数",
      correctionsHint:
        "保存到这个数量后会话就会结束。你也可以随时提前停止，此前录到的内容都会保留。",
      datasetLabel: "纠正数据集",
      datasetPlaceholder: "例如：fold_shirt_fixes",
      // 输入框为空时，名称中由用户输入的那一半的替代文字。
      datasetFallback: "correction",
      // <0> 包住 {{prefix}}，即磁盘上的实际名称 —— 它是标识符，
      // 无论什么语言都保持拉丁字母。
      datasetHint:
        "保存为 <0>{{prefix}}</0> 加一个时间戳。留空即使用灰显的名称，它取自该模型训练所用的数据集；你输入的内容会替换它，清空输入框则会恢复。",
      leaderLabel: "主臂",
      leaderNoRobot: "请先在上方选择一台机器人。",
      leaderMissing:
        "这台机器人没有配置主臂。请在机器人设置中补上它的端口和标定 — 没有主臂就无法进行指导。",
      // {{configs}} 是一到两个标定文件名 —— 属于数据，不做翻译。
      leaderFrom: "取自 {{name}}：{{configs}}。接管时你将用它进行遥操作。",
      bimanualWarning:
        "双臂注意：接管前请先把两条主臂停到接近机器人当前姿态的位置。双臂模式下是机器人去迎合主臂，而不是反过来，因此从工作台另一头接管会让两条机械臂横扫整个场景。移动距离过大的接管会被拒绝。",
    },
    checkpoint: {
      label: "检查点",
      none: "该策略暂无可用的检查点。",
    },
    armMismatch: {
      bimanualCheckpoint:
        "该检查点是在<0>双臂机器人</0>上训练的（状态维度 {{dim}}，{{arms}} 条机械臂），而 <1>{{name}}</1> 是单臂机器人。请改选单臂检查点，或在右上角的机器人菜单中选择一台双臂机器人。",
      singleCheckpoint:
        "该检查点是在<0>单臂机器人</0>上训练的（状态维度 {{dim}}），而 <1>{{name}}</1> 是双臂机器人。请改选双臂检查点，或在右上角的机器人菜单中选择一台单臂机器人。",
    },
    task: {
      label: "任务描述",
      hint: "该策略以语言为条件（{{policyType}}）。",
      // 当血缘里根本没有任务时使用的占位文字。绝不编造示例：把假任务灰显在
      // 真正继承来的任务所用的同一位置，二者将无法区分。
      //
      // 下面三条是这一条以前混为一谈、并因此误报的情形。只有 placeholderNone
      // 可以断言数据集没有任务；查询失败并不能证明这一点。
      placeholderNone: "训练数据集里没有找到任务 — 请手动输入一个",
      // 查询进行中。同一时刻只显示其中一个，大约每秒轮换一次，点号由调用方
      // 追加：它们是标点而非文案，不应进入词条。
      //
      // loading 排在第一个且刻意平实：先说清楚在做什么，再开始玩。其余都是
      // “翻找”类的说法，也正是它实际在做的事——在数据集的元数据里翻出它当初
      // 录制时用的那句话。按语感翻译即可，关键是每一条都读起来像“还在找”，
      // 而不是像出错了。
      loading: {
        loading: "加载中",
        rummaging: "翻箱倒柜",
        digging: "刨根问底",
        foraging: "四处搜罗",
        excavating: "掘地三尺",
        spelunking: "钻洞探穴",
        ferreting: "顺藤摸瓜",
        scrounging: "东翻西找",
      },
      // 动画结束后查询仍在进行。刻意不声称数据集没有任务：它还没有给出答案，
      // 若稍后返回，结果会直接覆盖这句提示。
      placeholderSlow: "仍在加载 — 不想等的话可以直接输入任务",
      // 找不到该数据集 — 已删除、已改名，或从未下载。
      placeholderMissing: "在本机找不到训练数据集 — 请手动输入任务",
      // 查询本身失败（离线、服务器报错），并不说明数据集有没有任务。
      placeholderUnreadable: "无法读取训练数据集 — 请手动输入任务",
      // 有多个任务，无法在它们之间做出可靠的猜测。
      placeholderChoose: "请在下方选择你要运行的任务",
      // 用于不读取任务的策略。指导仍会保存这个字符串。
      hintCoach: "它会随每次纠正一起保存，方便你日后知道这次会话教的是什么。",
      leaveEmpty: "留空即使用灰显的任务，它取自该模型训练所用的数据集。",
      multiTaskHint_other:
        "它的训练数据集里有 {{count}} 个任务，按出现次数从多到少排列 — 请选择你要执行的那个：",
    },
    duration: {
      label: "最长时长（秒）",
      hint: "按单个片段计。片段跑满该时长而你没有判定成功，即记为失败。",
      singleHint: "运行到这个时长后就停止。",
    },
    episodes: {
      label: "片段数",
      evalHint:
        "评测运行：共 {{episodes}} 个片段，每个之间会复位，最终统计为准确率。",
      hint: "保持为 1 即单次运行。大于 1 则启动带评分的评测。",
      scoreHint: "要计入准确率的片段数量。",
    },
    engine: {
      label: "推理引擎",
      sync: "Sync（默认）",
      rtc: "RTC — 实验性，控制更平滑",
      syncHint:
        "每个控制步执行一次策略前向推理。机械臂在动作块之间会短暂停顿。",
      rtcHint:
        "Real-Time Chunking 让推理与运动重叠进行，消除动作块之间的停顿。它也改变了动作的生成方式 — 在采信结果之前请先与 Sync 对比。",
      // 当所选检查点的架构无法运行 RTC（服务端会拒绝）时显示在选择器下方，
      // 同时该选项也会被禁用。
      rtcUnavailable: "该检查点的策略不支持 Real-Time Chunking。",
      // 指导模式固定使用 sync，因此显示这句话来代替引擎选择器。
      coachingNote:
        "指导始终使用 Sync 引擎。Real-Time Chunking 会让策略恢复时机械臂朝纠正前的姿态弹回，手就在旁边时这并不安全。",
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
    // 操作行：每个动词都在一次按下中同时选定模式并启动。
    runVerbs: {
      groupLabel: "开始一次运行",
      single: "运行",
      // {{count}} 是片段数 / 纠正次数目标，是数字，因此没有复数形式。
      eval: "打分 · {{count}}",
      coach: "人在回路 · {{count}}",
    },
    // 某个动词无法运行的原因；以键的形式提供，好让 deployGuards.ts 不含文案。
    blocked: {
      noRobot: "请先在上方选择一台机器人。",
      followerNotReady: "这台机器人的从臂尚未就绪。",
      noCheckpoint: "请选择一个策略和一个检查点。",
      armMismatch: "该检查点与这台机器人的机械臂数量不匹配。",
      camerasUnbound: "请为检查点所需的每个摄像头完成绑定。",
      temporalEnsemble: "请先修正时间集成设置。",
      runInProgress: "已有一次运行正在进行中。",
      taskRequired: "请先描述任务 — 该策略以语言为条件。",
      leaderMissing: "指导需要一条主臂 — 请在机器人设置中补上它的端口和标定。",
      coachTaskRequired: "请先描述任务 — 它会随每次纠正一起保存。",
      taskAmbiguous: "它的训练数据集包含多个任务 — 请选择你要运行的那个。",
    },
    actions: {
      start: "开始推理",
      startEval: "开始评测（{{episodes}}）",
      // 与 startEval 同理，用 {{corrections}} 而不是 {{count}}。
      startCoach: "开始指导（{{corrections}}）",
      starting: "正在启动…",
      checking: "正在检查…",
      stop: "停止推理",
      stopping: "正在停止…",
    },
    toast: {
      loadPolicyFailed: "无法加载该策略",
      startFailed: "无法启动推理",
      stoppingTitle: "正在停止推理",
      stoppingBody: "该次运行正在收尾。",
      stopFailed: "停止失败",
    },
    milestone: {
      title: "首个策略已部署！",
      description:
        "你的机器人刚刚运行了一个训练好的策略。随时回到这里重新部署、切换检查点，或运行其他策略。",
    },
  },
} as const;
