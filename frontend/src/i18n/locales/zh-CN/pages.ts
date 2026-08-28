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
  },
} as const;
