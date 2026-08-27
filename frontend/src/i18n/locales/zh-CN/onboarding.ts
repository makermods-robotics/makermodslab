export default {
  tour: {
    aria: "功能导览",
    stepCounter: "第 {{current}} 步，共 {{total}} 步",
    skip: "跳过",
    back: "上一步",
    next: "下一步",
    done: "完成",
  },
  launchpad: {
    search: {
      title: "查找技能",
      description: "按名称搜索技能，或浏览下方的技能。",
    },
    skills: {
      title: "浏览技能",
      description:
        "其他人训练并分享的技能 — 可以直接运行，也可以作为你自己技能的起点。",
    },
    newSkill: {
      title: "创建你自己的技能",
      description:
        "采集数据集、训练策略并部署到机器人 — 全部都在此页面完成。",
    },
    robot: {
      title: "设置你的机器人",
      description: "添加并配置用于录制和运行技能的机械臂。",
    },
    library: {
      title: "你的库",
      description: "你保存的数据集、模型和训练任务都在这里。",
    },
  },
  studio: {
    collect: {
      title: "1 · 采集",
      description: "在这里录制新的数据集，或选择已有的数据集。",
    },
    train: {
      title: "2 · 训练",
      description: "将数据集训练成策略。",
    },
    deploy: {
      title: "3 · 部署",
      description: "在你的机器人上运行已训练的技能。",
    },
  },
} as const;
