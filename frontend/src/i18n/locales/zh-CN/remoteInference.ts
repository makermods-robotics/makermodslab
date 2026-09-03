export default {
  intro:
    "机械臂在本机运行，策略在远程 GPU 上运行，二者在一个 LiveKit 房间中会合。本机不会加载检查点 — 请用下面的命令自行启动 GPU 侧。",
  form: {
    hubIdLabel: "Hub 策略 id",
    hubIdHint: "GPU 容器要加载的仓库。本机不会下载它 — 本机只负责驱动机械臂。",
    hubIdInherited: "留空时将使用本次运行自己的输出仓库。",
    engineLabel: "动作块引擎",
    engine: {
      // 选项 VALUE（"sync" / "rtc"）是后端标识符，只翻译这些标签。
      sync: "自适应同步",
      rtc: "实时分块",
      syncHint:
        "把每个动作块执行完，再刚好及时地请求下一个。适用于任何策略，也是 ACT 的唯一选择。",
      rtcHint:
        "每次请求都把尚未执行的动作一并发过去，让 GPU 据此生成能够接续的下一个动作块，从而消除块与块之间的接缝。只有流式策略（SmolVLA、π0、π0.5、diffusion）能这样被引导。",
      rtcUnsupported:
        "该检查点不是流式策略，无法以这种方式引导 — 实时运行不会比自适应同步更好，启动还更慢。请切换回“自适应同步”后再启动。",
    },
    transportGroup: "传输",
    transportGroupHint:
      "这几项必须与 GPU 侧完全一致。不一致不会报错：线上协议带有 schema 指纹，不匹配的数据包会被静默丢弃，运行看上去正常却收不到任何东西。",
    horizonLabel: "Horizon",
    fpsLabel: "帧率",
    codecLabel: "视频编码",
    durationLabel: "最长时长（秒）",
    durationHint: "到时后自动停止。你随时可以提前停止。",
    durationUnbounded: "0 — 一直运行，直到你停止它。",
    sMinLabel: "最小预留",
    // "--s-min" 是命令行标志名，与本面板中其他标识符一样保留拉丁文写法。
    sMinHint:
      "机械臂为一次往返预留的计划步数。它必须和上面命令里的 --s-min 完全一致：机械臂据此算出下一个动作块中还“新鲜”的部分，而 GPU 会直接采信这个结果。",
  },
  // 后端引擎取值。用于匹配，不直接展示 — 原值只作为新版服务端引入新引擎时的兜底。
  engine: {
    sync: "自适应同步",
    rtc: "实时分块",
  },
  modalRun: {
    title: "在另一个终端里运行这条命令",
    intro:
      "MakerMods Lab 负责驱动机械臂并核验房间；它不会启动 GPU。请先运行这条命令并让它保持运行，然后再按下方的远程按钮。",
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
  transport: {
    title: "传输",
    refresh: "重新检查",
    checking: "检查中…",
    notCheckedYet: "尚未检查。",
    unresolved: "未设置",
    sourceLabel: "读取自",
    source: {
      sfu: "MakerMods Lab 自带的 SFU",
      cloud: "livekit.env（LiveKit Cloud）",
      process_env: "本进程的环境变量",
      none: "无来源 — 尚未配置任何内容",
    },
    urlLabel: "URL",
    roomLabel: "房间",
    credentialsLabel: "凭据",
    configured: "齐全",
    missingVars: "缺少 {{vars}}",
    reachableLabel: "端点",
    reachable: "有响应",
    unreachable: "无响应",
    notProbed: "未检查",
    operatorLabel: "GPU 操作方",
    operatorPresent: "已在房间中",
    operatorAbsent: "不在房间中",
    extraMissing:
      "未安装可选的 drtc 附加依赖，因此无法进行任何检查。请在主检出目录中安装 — 在 worktree 中执行可编辑安装会让其他所有会话都指向该目录。",
    sfuRunningTitle: "本机的 LiveKit 服务器",
    sfuModalUrlLabel: "供 GPU 使用的地址",
    sfuNoTailnet: "没有 tailnet 地址",
    sfuKeyIdLabel: "Key id",
    sfuKeyFileLabel: "Secret 位于",
    sfuExternalIpLabel: "对外媒体地址",
    sfuExternalIpOn: "已公布",
    sfuExternalIpOff: "未公布",
    sfuExternalIpHint:
      "不公布的话，远端 GPU 能连上本服务器打个招呼，却无法发送视频或动作。请用下面的参数重启 MakerMods Lab 来开启它。",
    sfuNotRunning:
      "本 MakerMods Lab 未运行 LiveKit 服务器。可以用下面的参数启动它，也可以保持关闭并改用 livekit.env 里的 LiveKit Cloud 凭据。",
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
    elapsed: "{{elapsed}} 秒 / {{duration}}",
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
    stop: "停止运行",
    stopping: "正在停止…",
  },
  toast: {
    startFailed: "无法启动远程运行",
    stopFailed: "无法停止远程运行",
    noSession: "该服务器上没有登记的远程运行。",
  },
} as const;
