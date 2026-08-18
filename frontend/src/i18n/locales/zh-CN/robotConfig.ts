/**
 * "robotConfig" 命名空间 — 机器人设置窗口
 * (components/dialogs/RobotConfigDialog.tsx)。
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 机器人名称、串口路径、标定文件名、摄像头名称以及后端返回的
 * message / error 文本都属于数据，一律原样呈现，不在此翻译。
 */
export default {
  // ---- 窗口标题栏与底栏 ---------------------------------------------------
  window: {
    // `.eyebrow` 的大写在中文上是空操作，字距仍按原样。
    eyebrow: "端口 · 标定 · 摄像头 · 电机力矩",
    title: "机器人设置 — {{name}}",
    srDescription: "为 {{name}} 配置端口、标定、摄像头和电机力矩。",
    unsaved: "有未保存的更改",
    savedWithGap: "已保存 — 但该机器人{{gap}}",
    allSaved: "所有更改已保存",
    quit: "退出",
    save: "保存",
    saving: "正在保存…",
    justSaved: "已保存 ✓",
    toast: {
      saved: "更改已保存",
      saveFailedTitle: "无法保存更改",
      saveFailedFallback: "保存配置失败。",
    },
    discard: {
      title: "放弃未保存的更改？",
      description:
        "你还有未保存的配置更改（端口、摄像头或电机力矩）。现在关闭会丢弃这些更改 — 它们尚未写入机器人。若要保留，请先保存。",
      cancel: "继续编辑",
      confirm: "放弃并退出",
    },
    abort: {
      title: "中止标定？",
      description:
        "手动标定正在进行中。关闭会中止它 — 不会保存任何数据，机械臂会被释放（处于失力状态，请扶稳）。",
      cancel: "继续标定",
      confirm: "中止并关闭",
    },
    leaveConfirm: "离开将中止本次标定 — 不会保存任何数据，机械臂会被释放。是否继续？",
  },

  // ---- 机械臂槽位名称 ----------------------------------------------------
  // 仅用于显示；请求中始终发送 device_type + arm，与这些名称无关。
  arm: {
    leader: "主臂",
    follower: "从臂",
    leftLeader: "左主臂",
    leftFollower: "左从臂",
    rightLeader: "右主臂",
    rightFollower: "右从臂",
  },

  // 后端 device_type 枚举在句子中的说法。提交的值仍是 "teleop" / "robot"。
  deviceValue: {
    teleop: "主臂",
    robot: "从臂",
  },

  // ---- 01 · 设备 ---------------------------------------------------------
  device: {
    step: "设备",
    label: "设备",
    groupBimanual: "设备与机械臂",
    groupSingle: "设备",
  },

  slotCard: {
    undetectedLabel: "已保存的端口当前未检测到",
    undetectedTitle: "已保存的端口当前未检测到 — 请接上机械臂并重新扫描",
    noPort: "未分配端口",
  },

  // ---- 端口选择、识别、抖动 ----------------------------------------------
  port: {
    label: "端口",
    select: "选择端口",
    none: "未检测到机械臂 — 请接上并刷新",
    otherArm: "其他机械臂",
    clear: "清除端口",
    clearTitle: "清除端口 — 释放它且不分配新端口",
    rescan: "重新扫描端口",
    detect: "识别",
    detecting: "监测中…",
    detectTitle: "手动识别：将机械臂底座大幅向左和向右摆动",
    detectHelp:
      "手动识别 — 将机械臂底座分别大幅摆向左侧和右侧（每个方向都要超过起始位置 10–15°）；检测到运动的端口会被分配。小幅晃动不会被识别。",
    detectLive:
      "请大幅摆动机械臂底座 — 向左和向右都要明显超过起始位置。小幅或单侧的晃动会被忽略（这样才能滤掉碰撞）。检测到运动的端口将被分配给这条机械臂。",
    wiggle: "抖动",
    wiggling: "抖动中…",
    wiggleTitle: "驱动该端口上的夹爪，看看是哪条机械臂",
    wiggleHelp: "确认机械臂接在该端口上 — 会短暂驱动它的夹爪，你可以看到哪条机械臂有反应。",
    toast: {
      missingPortTitle: "缺少端口",
      missingPortWiggle: "请先输入或识别端口，再用抖动确认是哪条机械臂。",
      wiggleStartedTitle: "正在抖动夹爪",
      wiggleFailedTitle: "抖动失败",
      noArmTitle: "未检测到机械臂",
      detectFailedTitle: "识别失败",
      swappedDetectedTitle: "已识别机械臂 — 端口已互换",
      swappedTitle: "端口已互换",
      swappedDescription: "{{port}} 现已分配给这条机械臂；{{released}}接管了 {{swapPort}}。",
      movedDetectedTitle: "已识别机械臂 — 端口已迁移",
      movedTitle: "端口已迁移",
      movedDescription:
        "{{port}} 原本分配给{{released}}，现已迁移到这里。{{released}}目前没有端口。",
      identifiedTitle: "已识别机械臂",
      assignedTitle: "端口已分配",
      identifiedDescription: "端口已分配给这条机械臂。",
      assignedDescription: "{{port}} 已分配给这条机械臂。",
    },
  },

  // ---- 端口分配确认 ------------------------------------------------------
  portAssign: {
    swapTitle: "互换端口？",
    detectTitle: "分配识别到的端口？",
    assignTitle: "分配端口？",
    leadDetect: "识别到 <0>{{port}}</0> — 要把它分配给<1>{{target}}</1>吗？",
    leadAssign: "要把 <0>{{port}}</0> 分配给<1>{{target}}</1>吗？",
    swapClause:
      "它目前分配给<0>{{released}}</0>；确认后两者互换 — <1>{{released}}</1>将接管这条机械臂当前的端口 <2>{{swapPort}}</2>，这样两条机械臂都不会没有端口。",
    takeClause:
      "它目前分配给<0>{{released}}</0>；这条机械臂没有端口可供交换，确认后端口会迁移到这里，而<1>{{released}}</1>将没有端口。",
    confirmSwap: "互换端口",
    confirmMove: "迁移并分配",
    confirmAssign: "分配端口",
  },

  // ---- 02 · 标定文件 -----------------------------------------------------
  files: {
    step: "标定文件",
    calibrateAll: "全部标定",
    calibrateAllTitle: "选中所有已检测到的机械臂进行自动标定",
    calibrateAllDisabledTitle: "未检测到机械臂 — 请接上机械臂并重新扫描",
    openLeaderFolder: "打开主臂标定文件夹",
    openFollowerFolder: "打开从臂标定文件夹",
    leader: "主臂",
    follower: "从臂",
    newCalibration: "新建标定",
    newCalibrationTitle: "为这条机械臂新建一份标定",
    // 括号里是该槽位对应的 LeRobot 设备类别。
    row: {
      leader: "主臂（遥操作器）",
      follower: "从臂（机器人）",
      leftLeader: "左主臂（遥操作器）",
      leftFollower: "左从臂（机器人）",
      rightLeader: "右主臂（遥操作器）",
      rightFollower: "右从臂（机器人）",
    },
    toast: {
      openFolderFailedTitle: "无法打开文件夹",
    },
  },

  // ---- “新建标定”面板 ----------------------------------------------------
  calib: {
    panelTitle: "新建标定 — {{row}}",
    status: {
      idle: "空闲",
      connecting: "连接中",
      recording: "正在记录行程",
      completed: "已完成",
      error: "出错",
      stopping: "停止中",
      unknown: "未知",
    },
    cancel: "取消标定",
    auto: "自动标定",
    autoTitle: "在 {{port}} 上自动标定{{arm}} — 机械臂会自行运动",
    autoDisabledTitle: "这条机械臂没有已检测到的端口 — 请在上方分配或重新连接",
    manual: "手动标定",
    torqueOffWarning:
      "电机力矩已关闭 — 标定期间机械臂无法保持姿态，取消或完成后也会保持失力状态。请让它保持低位并有支撑，避免掉落到桌沿。",
    connecting: "正在连接设备，请确认设备已连接。",
    liveData: "实时位置数据",
    rangeComplete: "行程已记录完整",
    save: "保存标定",
    rangeHint:
      "<0>重要：</0>请让每个关节走完整个行程 — <1>腕部旋转关节除外</1>：让它保持在中间附近。它可以连续旋转，行程会自动设定。某个关节的行程足够大时，它旁边会出现一个对勾。",
    completed: "标定成功完成！",
    discontinuityTitle: "检测到电机位置跳变",
    discontinuityBody:
      "开始标定时，请让机器人处于中间位置 — 所有关节都位于各自行程的中间。正确的起始姿态可参考旁边的标定演示视频。",
    errorLabel: "错误：",
    demoTitle: "标定演示",
    videoUnsupported: "你的浏览器不支持 video 标签。",
    videoLink: "点此查看标定视频",
    toast: {
      noRobotTitle: "未选择机器人",
      noRobotDescription: "请从机器人菜单打开机器人设置（⚙ 机器人设置）。",
      missingPortTitle: "缺少端口",
      missingPortDescription: "开始之前请先设置设备的串口。",
      startedTitle: "标定已开始",
      startedDescription: "已开始为{{device}}标定",
      startFailedTitle: "标定失败",
      startFailedFallback: "启动标定失败",
      errorTitle: "出错",
      startError: "启动标定失败",
      stoppedTitle: "标定已停止",
      stoppedDescription: "标定已停止",
      stopFailedFallback: "停止标定失败",
      stepCompletedTitle: "步骤已完成",
      stepFailedTitle: "步骤失败",
      stepFailedFallback: "无法完成该步骤",
      stepError: "无法完成标定步骤",
    },
  },

  // ---- 多臂并发自动标定 --------------------------------------------------
  batch: {
    titleSingle: "自动标定",
    titleMulti: "多臂自动标定",
    stopSingle: "停止自动标定",
    stopAll: "停止全部自动标定",
    pickerIntro:
      "选择要标定的机械臂。每条机械臂都会在各自分配的端口上<0>同时</0>运行免手动标定 — 某一条失败不会影响其他机械臂。端口取自上方每条机械臂的分配；尚未分配端口的机械臂无法选中。每条机械臂都会覆盖它自己已有的标定；之后可在上方的标定列表中重命名。",
    portUndetected: "未检测到端口",
    portMissing: "无端口 — 请在上方分配",
    // 中文只有一个复数形式，因此只提供 _other。
    start_other: "自动标定 {{count}} 条机械臂",
    progress_other: "已完成 {{total}} 条中的 {{done}} 条 — 机械臂正在运动。请保持工作区无障碍物。",
    armStatus: {
      completed: "✓ 完成",
      failed: "✗ 失败",
      stopped: "已停止",
      running: "运行中…",
    },
    summary: "{{completed}} 条完成，{{failed}} 条失败/已停止。",
    dismiss: "关闭",
    prompt: {
      titleSingle: "自动标定{{arm}} — 它会运动",
      titleFallbackArm: "这条机械臂",
      titleMulti: "自动标定多条机械臂 — 它们都会运动",
      bodySingle:
        "这条机械臂将<0>自行通电运动</0>以测定每个关节的行程。请清空工作区，双手远离机械臂。它会覆盖自己已有的标定。",
      bodyMulti:
        "{{count}} 条机械臂将同时<0>自行通电运动</0>以测定每个关节的行程。请清空工作区，双手远离所有机械臂。每条机械臂都会覆盖自己已有的标定。",
      confirm: "开始自动标定",
    },
    toast: {
      noArmsTitle: "未选择机械臂",
      noArmsDescription: "请至少勾选一条要自动标定的机械臂。",
      noPortTitle: "机械臂没有已检测到的端口",
      noPortDescription: "{{arm}}没有当前已接入的端口 — 开始之前请在上方分配或重新连接。",
      duplicatePortTitle: "端口重复",
      duplicatePortDescription: "每条机械臂都需要各自独立的串口。",
      startedTitle_other: "已在 {{count}} 条机械臂上开始自动标定",
      startedDescription: "机械臂正在运动 — 请保持工作区无障碍物。",
      startFailedTitle: "无法启动自动标定",
      finishedTitle_other: "已自动标定 {{count}} 条机械臂",
      issuesTitle: "批量自动标定完成，但存在问题",
    },
  },

  // ---- 高级参数（自动标定力矩） ------------------------------------------
  advanced: {
    title: "高级参数",
    subtitle: "自动标定力矩",
    torqueLabel: "自动标定力矩",
    // "Torque_Limit" 是舵机寄存器名，不翻译。
    torqueSliderLabel: "自动标定力矩（Torque_Limit 寄存器，0-1000 刻度）",
    torqueHint:
      "舵机原始 <0>Torque_Limit</0>（刻度标记为出厂值 {{ref}}）— 数值越低越柔和；低于 {{min}} 时机械臂无法抬起自身。",
  },

  // ---- 03 · 已连接的摄像头 -----------------------------------------------
  cameras: {
    step: "已连接的摄像头",
    on: "开",
    off: "关",
    toggleLabel: "打开或关闭摄像头",
    offTitle: "摄像头已关闭",
    offDescription:
      "打开摄像头后即可扫描已连接的设备并进行预览。浏览器可能会短暂开启摄像头以读取设备名称，已配置的摄像头在预览可见期间会保持开启；浏览器会请求摄像头权限。不会录制任何内容。",
    saved_other: "已为该机器人保存 {{count}} 个摄像头。",
    permissionHint: "系统会请求你授予摄像头访问权限。",
  },
} as const;
