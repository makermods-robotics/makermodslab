export default {
  notFound: {
    message: "糟糕！页面未找到",
    home: "返回首页",
  },
  teleop: {
    stoppedWarnTitle: "遥操作已停止 — 请检查机械臂",
    stoppedTitle: "遥操作已停止",
    releasingFallback: "机械臂将返回起始位置，然后释放力矩。",
    checkArmTitle: "请检查机械臂",
    disconnectedCleanly: "机械臂已正常断开连接。",
    endedWithWarning: "遥操作结束，但清理时出现警告",
    failed: "遥操作失败",
    armLink: {
      heading: "远程连接与舵机",
      loading: "加载中",
      remote: "远程动作通道",
      live: "实时",
      simulationOnly: "仅模拟",
      maintenance: "维护写入",
      disabled: "已禁用",
      servoHealth: "舵机健康状态",
      noOwner: "等待总线所有者",
      healthy: "{{count}} 个舵机 · 无故障",
      faults_one: "{{count}} 个已解码故障",
      faults_other: "{{count}} 个已解码故障",
    },
  },
} as const;
