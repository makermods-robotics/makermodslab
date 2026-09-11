/**
 * "library" 命名空间 —— 共享的库组件（`components/library/`：卡片网格、数据集
 * 库）、启动台的「我的库」侧边抽屉，以及 `lib/deleteSemantics` 解析出的删除
 * 确认文案。
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * 产品名（MakerMods、LeRobot、Hugging Face、Hub）一律保留原文。数据（数据集/
 * 模型名称、repo id、Hub 命名空间、摄像头名称、机器人类型、后端返回的报错）
 * 由插值原样渲染。数字、字节大小与时长均已在 `lib/datasetFormat` 中格式化好，
 * 此处只提供它们周围的文字。
 */
export default {
  grid: {
    showLess: "收起",
    showAll: "显示全部 {{total}}",
  },

  datasets: {
    source: {
      local: "本地",
      hub: "Hub",
      both: "本地 · Hub",
    },
    privateTitle: "在 Hub 上为私有",
    private: "私有",
    hubOnly: "在 Hub 上 —— 尚未下载到本地。",
    detailsError: "无法读取该数据集的详情。",
    loadingDetails: "正在加载数据集详情",
    loadingList: "正在加载数据集",
    empty: "还没有数据集。请在上方录制第一个。",
    noMatch: "没有匹配的数据集。",
    searchPlaceholder: "搜索数据集",
    filters: {
      all: "全部",
      local: "本地",
      hub: "Hub",
    },
    // 中文只有一种复数形态，因此只需 _other。
    episodes_other: "{{count}} 个片段",
    frames: "{{frames}} 帧",
    meta: {
      cameras: "摄像头",
      robot: "机器人",
      task_other: "任务",
      taskCount_other: "{{count}} 个任务",
      size: "大小",
    },
    select: "选择",
    selected: "已选择",
  },

  sheet: {
    title: "我的库",
    close: "关闭库",
    tabs: {
      policies: "我的策略",
      datasets: "我的数据集",
    },
    steps: "{{steps}} 步",
    private: "私有",
    policies: {
      loading: "正在加载策略…",
      empty: "你还没有自己的策略 —— 可在下方创建一个。",
      manage: "管理 {{name}}",
      run: "运行 {{name}}",
    },
    datasets: {
      loading: "正在加载数据集…",
      empty: "你还没有自己的数据集 —— 先录制一个吧。",
      source: {
        both: "本地 + Hub",
        hub: "在 Hub 上",
        local: "仅本地",
      },
    },
    actions: {
      addFromHub: "从 Hub 添加",
      importFromDisk: "从磁盘导入",
      manageCaches: "管理缓存",
      newPolicy: "新建策略",
      mergeDatasets: "合并数据集",
    },
    toast: {
      datasetSaved: "已保存数据集",
      datasetImported: "已导入数据集",
      modelSaved: "已保存模型",
      modelImported: "已导入模型",
      downloadStarted: "已开始下载",
      downloadFailed: "无法开始下载",
    },
  },

  // 每个（操作 × 类型）组合都是一句完整的话：中文里「数据集」「模型」这个名词
  // 所处的位置与英文不同，标题也不能靠拼接前缀得到，所以都各自成句。
  delete: {
    localCopy: {
      dataset: {
        title: "移除“{{label}}”的本地副本？",
        description:
          "这会从磁盘上删除本地副本 —— Hub 上的副本会保留，它仍会作为 Hub 数据集显示在列表中。",
        confirm: "移除本地副本",
      },
      model: {
        title: "移除“{{label}}”的本地副本？",
        description:
          "这会从磁盘上删除本地副本 —— Hub 上的副本会保留，它仍会作为 Hub 模型显示在列表中。",
        confirm: "移除本地副本",
      },
    },
    unpin: {
      dataset: {
        title: "移除“{{label}}”？",
        description:
          "这只会把该数据集从你的列表中移除。Hub 仓库和任何本地副本都不受影响 —— 你随时可以从「添加数据集」菜单重新添加。",
        confirm: "移除",
      },
      model: {
        title: "移除“{{label}}”？",
        description:
          "这只会把该模型从你的列表中移除。Hub 仓库和任何本地副本都不受影响 —— 你随时可以从「添加模型」菜单重新添加。",
        confirm: "移除",
      },
    },
    hide: {
      dataset: {
        title: "移除“{{label}}”？",
        description:
          "这会把该数据集从你的列表中隐藏。Hub 仓库不会被删除 —— 你随时可以从「添加数据集」菜单重新添加。",
        confirm: "移除",
      },
      model: {
        title: "移除“{{label}}”？",
        description:
          "这会把该模型从你的列表中隐藏。Hub 仓库不会被删除 —— 你随时可以从「添加模型」菜单重新添加。",
        confirm: "移除",
      },
    },
    local: {
      dataset: {
        title: "删除“{{label}}”？",
        description:
          "这会从本地磁盘上永久删除该数据集 —— 包括所有已录制的片段和视频。此操作无法撤销。",
        confirm: "删除",
      },
      model: {
        title: "删除“{{label}}”？",
        description:
          "这会从磁盘上永久删除该模型的本地文件 —— 包括它的检查点。此操作无法撤销。Hub 上的副本（如有）不受影响。",
        confirm: "删除",
      },
    },
  },
} as const;
