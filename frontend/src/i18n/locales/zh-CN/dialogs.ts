/**
 * "dialogs" 命名空间 —— 见 en/dialogs.ts 的说明。
 *
 * 数据集 repo id、模型名称、Hub 命名空间、相机名称、关节名称、机械臂名称以及
 * 后端返回的提示/错误文本都是数据，原样渲染，不翻译。
 */
export default {
  datasetDetail: {
    eyebrow: "MakerMods Lab 数据集查看器",
    loadingEpisodes: "正在加载回合…",
    noEpisodesTitle: "尚未录制任何回合",
    noEpisodesBody: "先向该数据集录制至少一个回合，才能在这里查看相机画面。",
    noFootageTitle: "暂无可播放的画面",
    noFootageBody:
      "带视频的 Hub 数据集会在这里按需加载 — 首次查看某个回合可能需要稍等片刻。出现这条消息说明该数据集的格式早于查看器所支持的版本，或者本身就没有视频。",
    episodesHeading: "回合",
    episodesHeadingWithCount: "回合（{{total}}）",
    episodeRow: "第 {{index}} 回合",
    episodesEmpty: "将该数据集下载到本地后，回合会显示在这里。",
    noCameras: "该数据集没有相机画面 — 可以在机械臂上回放，或查看下方的关节曲线。",
    videoDecodeError: "当前浏览器无法解码该相机的视频。",
    previousEpisode: "上一回合",
    nextEpisode: "下一回合",
    play: "播放",
    pause: "暂停",
    trainSkill: "用它训练一项技能",
    curateEpisodes: "挑选回合",
    curateDone: "完成",
    includedCount: "已选 {{included}} / {{total}} 个回合",
    includeEpisodeAria: "将第 {{index}} 回合纳入训练",
    curateSaveFailedTitle: "无法保存回合选择",
    curateSaveFailedBody: "改动未保存 — 请重试。",
    finishCuratingFirst: "请先完成回合挑选",
  },

  jointChart: {
    heading: "关节位置 — 与播放头同步",
    episode: "第 {{index}} 回合",
    loading: "正在加载关节数据…",
    noEpisode: "未选择回合",
  },

  replay: {
    phase: {
      idle: "空闲",
      easingIn: "正在缓慢移动到起始位置…",
      playing: "回放中",
      stopping: "正在停止…",
      done: "已完成",
      error: "出错",
    },
    robotNotReady: "请选择一台可以回放的机械臂：这台机械臂{{gap}}。",
    noRobot: "请选择一台已连接从臂的机械臂，才能在硬件上回放该回合。",
    start: "在硬件上回放",
    movesArmWarning: "将移动 {{robot}} 的机械臂 — 请确保周围区域无障碍物。",
    stop: "停止",
    toast: {
      failedTitle: "回放失败",
      seeLog: "详情请查看服务器日志。",
      lostConnectionTitle: "与后端的连接已断开",
      startedWarningTitle: "已启动，但有警告",
      startFailedTitle: "无法开始回放",
      stopFailedTitle: "无法停止回放",
    },
  },

  skillDetail: {
    previewAlt: "{{title}} 的运行预览",
    previewPlaceholder: "运行预览",
    localAndHub: "本地 + Hub",
    byAuthor: "作者：{{author}}",
    steps: "{{steps}} 步",
    private: "私有",
    trainedOn: "<0>训练所用数据集：</0><1>{{dataset}}</1>",
    episodeSubset: "{{used}} 个回合",
    episodeSubsetOfTotal: "{{total}} 个回合中的 {{used}} 个",
    notTrained: "尚未训练 — 该技能仍在开发中。",
    run: "在 {{robot}} 上运行",
    robotFallback: "机械臂",
    fineTune: "微调该技能",
    likesUnavailable: "暂无点赞数据",
    viewOnHub: "在 HF Hub 上查看",
  },

  skillManage: {
    runOnRobot: "在机械臂上运行",
    toast: {
      removedFromList: "已从列表中移除",
      localCopyRemoved: "已删除本地副本",
      modelDeleted: "模型已删除",
      deleteFailed: "删除失败",
      removeFailed: "移除失败",
    },
  },

  teleop: {
    title: "遥操作",
    titleWithRobot: "遥操作 — {{robot}}",
    done: "完成",
    leftArm: "左臂",
    rightArm: "右臂",
    endedWithWarning: "遥操作已结束，但清理时有警告",
    failed: "遥操作失败",
    toast: {
      stoppedCheckArm: "遥操作已停止 — 请检查机械臂",
      stopped: "遥操作已停止",
      releasing: "机械臂会先回到起始位置，然后松开力矩。",
      checkArm: "请检查机械臂",
      disconnected: "机械臂已正常断开连接。",
    },
  },
} as const;
