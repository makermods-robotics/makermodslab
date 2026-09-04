export default {
  form: {
    hubIdLabel: "Hub 策略 id",
    hubIdHint: "GPU 容器要加载的仓库。本机不会下载它 — 本机只负责驱动机械臂。",
    hubIdInherited: "留空时将使用本次运行自己的输出仓库。",
    // 引擎的标签位于 `studio.deploy.engine` —— 它现在是本地与远程共用的一个
    // 字段。这里只保留两条说明文案，因为它们讲的是两种引擎各自做什么，
    // 而不是谁是默认值。
    engine: {
      syncHint:
        "把每个动作块执行完，再刚好及时地请求下一个。适用于任何策略，也是 ACT 的唯一选择。",
      rtcHint:
        "每次请求都把尚未执行的动作一并发过去，让 GPU 据此生成能够接续的下一个动作块，从而消除块与块之间的接缝。只有流式策略（SmolVLA、π0、π0.5、diffusion）能这样被引导。",
    },
    // “高级参数”折叠标题下的摘要行。其中每个取值都是实时数据 —— 编码 id 是
    // 线上取值，数字就是实际发出的数字。写成两句完整的话而不是一句加片段：
    // s_min 只有 rtc 才会真正上线，因此只在那一句里提到它。
    // {{extra}} 是 GPU 侧的那一半（形如 " · A100 · bfloat16"），全部由标识符
    // 组成，里面没有可翻译的内容，因此整体拼接而不拆成词。
    advancedSummary:
      "传输：horizon {{horizon}} · {{fps}} fps · {{codec}}{{extra}}",
    advancedSummaryRtc:
      "传输：horizon {{horizon}} · {{fps}} fps · {{codec}} · 最小预留 {{sMin}}{{extra}}",
    transportGroupHint:
      "这几项必须与 GPU 侧完全一致。不一致不会报错：线上协议带有 schema 指纹，不匹配的数据包会被静默丢弃，运行看上去正常却收不到任何东西。",
    horizonLabel: "Horizon",
    fpsLabel: "帧率",
    codecLabel: "视频编码",
    // {{steps}} 是从检查点配置里读出的数字，不做任何翻译。
    horizonFromCheckpoint:
      "该检查点每个动作块返回 {{steps}} 步，因此 horizon 以此为起点，且不能超过它。",
    horizonOverCeiling:
      "horizon 超过了该检查点返回的 {{steps}} 步。这样两侧对动作块形状的理解就不一致，所有数据包都会被静默丢弃 — 运行看上去已连接，却收不到任何东西。",
    sMinLabel: "最小预留",
    // "--s-min" 是命令行标志名，与本面板中其他标识符一样保留拉丁文写法。
    sMinHint:
      "机械臂为一次往返预留的计划步数。它必须和上面命令里的 --s-min 完全一致：机械臂据此算出下一个动作块中还“新鲜”的部分，而 GPU 会直接采信这个结果。",
    // GPU 侧的两个参数（S3.8e）。它们不需要与机械臂一致 —— 决定的是容器加载
    // 什么、跑在什么硬件上。
    gpuGroupHint:
      "这两项只属于 GPU 侧：它们决定一个动作块多久返回，以及每小时的花费。修改任一项都需要重启 GPU。",
    precisionLabel: "精度",
    // 只有这一个选项是文案：它表示不传任何标志。其余都是 torch dtype 名称，
    // 属于线上取值，不翻译。
    precisionCheckpoint: "检查点默认值",
    precisionHint:
      "float32 是多数检查点保存时的精度，而且不会走 autocast —— 最慢的一条路，也是最先耗尽显存的一条。当一个动作块返回太慢、或容器显存不足时，bfloat16 就是可以拉的那根杠杆。保持检查点默认值时，不会覆盖任何设置。",
    gpuLabel: "GPU",
    gpuHint:
      "策略服务运行所用的 Modal GPU。越大越快、每小时也越贵；它是继精度之后的第二根杠杆，而且无论如何都在计费。",
    // 写在被禁用的下拉框旁边，因为这个原因属于“当前这个检查点”。
    precisionUnavailable:
      "该检查点没有可覆盖的精度设置 —— 它按保存时的精度加载。",
    // 流步数参数（S3.8f）。
    flowStepsLabel: "流步数",
    // 只有这一个选项是文案：它表示不传任何标志。第二种写法里的数字是该检查点
    // 实际会用的步数，属于数据 —— 由服务端算出，不在本文件里写死。
    flowStepsCheckpoint: "检查点默认值",
    flowStepsCheckpointKnown: "检查点默认值（{{steps}}）",
    flowStepsHint:
      "模型为生成一个动作块要推理多少遍。次数越少越快、动作也越粗糙 —— 这是缩短等待最便宜的办法，也是最先牺牲质量的办法。MolmoAct2 用 10 次；在 horizon 30、30 fps 下，它在 GPU 上的耗时目前约 880 毫秒，而预算是 777 毫秒。",
    flowStepsUnavailable: "该检查点不是分步生成动作的，因此没有可缩短的步数。",
  },
  // 按角色绑定摄像头。只有当检查点的某个摄像头在机器人上找不到同名摄像头时才会出现。
  cameraRoles: {
    title: "摄像头角色",
    hint: "该检查点使用的摄像头名称在这台机器人上不存在。请为本次运行选择由哪个摄像头承担每个角色。",
    nameMatched_other: "另有 {{count}} 个摄像头按名称自动匹配。",
    capturesAt: "策略的训练分辨率为 {{width}}×{{height}}。",
    unbound: "尚未选择",
    noCameras: "该机器人没有摄像头 — 请在机器人设置中添加。",
    disconnected: "当前未接入。",
    identityNote:
      "该选择会按此检查点与机器人记住，并且只随本次运行发送。不会重命名任何东西：摄像头仍沿用机器人设置中的名称，服务端也仍按该名称查找设备。",
  },
  // 后端引擎取值。用于匹配，不直接展示 — 原值只作为新版服务端引入新引擎时的兜底。
  engine: {
    sync: "自适应同步",
    rtc: "实时分块",
  },
  modalRun: {
    manualToggle: "改为自己手动启动",
    title: "MakerMods Lab 将要运行的命令",
    intro:
      "同一条命令，供手动启动使用 — 当 modal 命令缺失或尚未登录时，这是唯一的途径；当运行已连接却收不到任何东西时，也用它来做对照。",
    copy: "复制",
    copiedTitle: "命令已复制",
    copyFailedTitle: "复制失败",
    copyFailedBody: "请手动选中命令并复制。",
    noRoomYet: "尚未解析出房间 — 请先在下方重新检查传输，然后再复制命令。",
    // <0> 是字面占位符，<1> 是字面路径，二者都是标识符，保持拉丁字符。
    secretsHint:
      "请把 <0>{{placeholder}}</0> 替换为 <1>{{path}}</1> 中该 key id 对应的 secret。命令中的 key id 是真实值；MakerMods Lab 的接口从不返回 secret。",
    noTailnetUrl:
      "没有 tailnet 地址，命令中也就没有可供 GPU 端拨号的 URL。请在本机登录 Tailscale，然后重新检查传输。",
  },
  // GPU 侧，自 S3.8 起由 MakerMods Lab 自己启动。它不会作为远程按钮的前置条件 —
  // 那仍由传输探测中的 operator 检查决定。
  gpu: {
    title: "Modal 上的策略服务",
    start: "启动 GPU",
    retry: "重试",
    stop: "停止 GPU",
    cancel: "取消",
    // {{wrapper}} 是包装脚本的路径，{{gpu}} 是 Modal 的 GPU 规格，都属于数据，
    // 原样显示。GPU 改为插值而不是写死在句子里，因为它现在是可选的（S3.8e）。
    idleHint:
      "从本机在 Modal {{gpu}} 上运行 {{wrapper}}。冷启动通常需要 1-3 分钟；房间和凭据会自动填好。",
    // {{seconds}} 是普通整数，刻意不使用 i18next 的 count 机制。
    elapsed: "{{seconds}} 秒",
    // 后端阶段取值。用于匹配，不直接展示 — 原值只作为新版服务端引入新阶段时的兜底。
    phase: {
      pending: "正在启动容器",
      tailscale_up: "正在加入 tailnet",
      loading: "正在加载检查点",
      warmup: "正在预热模型",
      connecting: "正在连接房间",
      connected: "已在房间中",
      claimed: "正在驱动",
    },
    // 两个目标选择器。它们的选项永远不翻译：profile 名、workspace 名和
    // environment 名都是 CLI 用于匹配的标识符，面板按 modal 报告的原样显示。
    profileLabel: "Modal profile",
    environmentLabel: "Environment",
    running: "GPU 正在运行 — 这会产生费用。",
    // {{profile}}、{{workspace}} 和 {{environment}} 都是数据 — Modal 自己的
    // 名称，在译文句子中原样呈现。
    billingTo: "计费到 {{profile}}。",
    billingToWorkspace: "计费到 {{profile}} · {{workspace}}。",
    billingEnvironment: "环境 {{environment}}。",
    // {{minutes}} 是普通整数，刻意不使用 count 机制。
    idleStopIn: "若约 {{minutes}} 分钟内没有远程运行开始，它会自动停止。",
    idleStopPaused: "有远程运行正在使用它，因此不会自动停止。",
    // 表单与正在运行的服务之间的不一致。{{fields}} 是一组参数名
    //（engine、horizon、fps、codec、s_min、policy、task）— 属于数据。
    // 启动时的取值原样跟在这句话之后。
    driftBody:
      "GPU 启动后你改动了 {{fields}}。正在运行的服务会一直沿用启动时的取值，二者不一致时运行不会报错，只会什么都收不到。它当前的取值是：",
    restart: "用这些设置重启 GPU",
    // 任务为空、“启动 GPU” 被禁用时，在空闲状态下显示。
    taskRequired:
      "请先描述任务 — 该策略以语言为条件，GPU 上的策略服务没有任务就会拒绝启动。",
    roomLabel: "房间",
    logLabel: "日志",
  },
  transport: {
    // 传输区块已经撤掉，这里剩下的是：手动命令旁边那份需要人用眼睛读、再手动
    // 抄到别处的小抄；会话对话框策略行要用的来源标签；以及“开始”下面那句结论。
    // 这些标签背后的每个取值都是数据，一律原样显示。
    unresolved: "未设置",
    source: {
      sfu: "MakerMods Lab 自带的 SFU",
      cloud: "livekit.env（LiveKit Cloud）",
      process_env: "本进程的环境变量",
      none: "无来源 — 尚未配置任何内容",
    },
    roomLabel: "房间",
    extraMissing:
      "未安装可选的 drtc 附加依赖，因此无法进行任何检查。请在主检出目录中安装 — 在 worktree 中执行可编辑安装会让其他所有会话都指向该目录。",
    sfuModalUrlLabel: "供 GPU 使用的地址",
    sfuNoTailnet: "没有 tailnet 地址",
    sfuKeyIdLabel: "Key id",
    sfuKeyFileLabel: "Secret 位于",
    // 和下面那些结论句一样，只是没有放进 summary：它是唯一一条“解决办法是一条
    // 命令”的结论，面板会把那条命令（以及后端给出的安装提示，如果有）打印在这
    // 句话下面。“下面的参数”指的就是那段 <pre>。参见 transportSummary.ts。
    sfuNotRunning:
      "本 MakerMods Lab 未运行 LiveKit 服务器。可以用下面的参数启动它，也可以保持关闭并改用 livekit.env 里的 LiveKit Cloud 凭据。",
    // 把传输状态归纳成一句话，按“第一个出问题的环节”来选 —— 顺序就是操作者
    // 需要依次解决的顺序。它会取代“开始”下面那句笼统的“尚未就绪”，所以每一条
    // 都必须说清楚接下来该做什么。参见 transportSummary.ts。
    summary: {
      // {{error}} 是抛出的错误自身的文本，属于后端文案，原样显示。
      fetchFailed: "无法读取传输状态：{{error}}",
      checking: "正在检查房间…",
      notChecked: "尚未检查房间。",
      // {{vars}} 是一组环境变量名，属于数据，原样显示。
      missingVars:
        "没有 LiveKit 凭据：缺少 {{vars}}。请用 --sfu 启动 MakerMods Lab，或在 livekit.env 中填入 Cloud 凭据。",
      // {{url}} 就是地址本身，属于数据。
      unreachable:
        "{{url}} 上没有任何响应。请确认 LiveKit 服务器已启动，并且本机能连上它。",
      notProbed: "无法从本机检查该房间。",
      // {{room}} 是房间名，属于数据。
      ready: "已有 GPU 在 {{room}} 中，随时可以驱动机械臂。",
      operatorAbsent:
        "{{room}} 中还没有 GPU — 请在上方启动一个，或自行运行该命令。",
    },
  },
  phase: {
    idle: "未运行",
    resolving: "正在解析检查点",
    transport_check: "正在检查传输",
    preflight: "启动前检查",
    starting: "正在启动",
    connecting: "正在连接房间",
    warming_up: "等待策略接入",
    easing: "正在把机械臂移入初始位姿",
    running: "运行中",
    stopping: "正在停止",
    stopped: "已停止",
    error: "失败",
  },
  outcome: {
    ok: "已正常结束",
    failed: "运行失败",
    ran_with_warning: "已结束，但清理时有警告",
  },
  status: {
    // 会话对话框中远程运行专用的文案。状态药丸、按钮和日志标题与本地运行共用，
    // 位于 `inference.*` 之下。
    //
    // 整个启动阶段只用这一句：当前阶段会显示在按钮下方的阶段行里，若这里也跟着
    // 变，计时器下方就会成为整屏最吵的地方。
    connectingSubtitle: "正在连接 GPU 与机械臂…",
    // duration 为 0 的远程运行会一直跑到被停止。这里的 “/” 与有时限运行显示的
    // “/ 01:00” 保持一致。
    unbounded: "/ ∞ — 你不停它就不停",
    unboundedDone: "/ ∞",
    // {{ref}} 是策略 ref，{{room}} 是房间名，{{source}} 是解析出的来源标签，
    // 三者都是数据，原样显示。
    policyLine: "策略：{{ref}} · 远程 · {{room}}，来自 {{source}}",
    policyLineNoRoom: "策略：{{ref}} · 远程",
    gpuCardTitle: "远程 GPU",
    // {{profile}} 是 Modal 自己的 profile 名，属于数据。A100 与“计费”说明的是
    // 这次运行的成本，就写在操作者盯着它运行的地方。
    gpuBilling: "Modal · {{profile}} · A100 · 计费中",
    // 子进程会写一个日志文件并报告它的路径；浏览器这边没有任何流式日志，
    // 因此日志区域显示这个路径，由操作者自行打开。
    noLogYet: "尚无日志路径 — 本次运行还没有创建。",
    returningToRest: "正在把机械臂缓慢送回起始位姿，然后再释放力矩。",
    operator: "操作方",
    noOperatorYet: "等待中",
    chunks: "动作块 / 请求",
    chunkAge: "动作块时延",
    e2e: "端到端 p50 / p95",
    rtt: "往返时延",
    holdsRate: "保持次数",
    holdsPerSecond: "{{rate}}/秒",
    leadLabel: "调度余量",
    leadValue: "{{lead}} / {{margin}}",
    degradeHint: "质量正在下降",
    noSampleYet: "尚无采样 — 连接成功一秒后会收到第一条。",
  },
  toast: {
    startFailed: "无法启动远程运行",
    stopFailed: "无法停止远程运行",
    noSession: "该服务器上没有登记的远程运行。",
  },
} as const;
