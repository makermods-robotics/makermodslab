/**
 * "calibration" namespace (Simplified Chinese). Key tree must match
 * `locales/en/calibration.ts` exactly — see i18n/catalogs.test.ts.
 *
 * Calibration file names, robot names, and the `teleop`/`robot` device
 * vocabulary are data: they arrive as interpolation values and are never
 * translated here.
 */
export default {
  library: {
    placeholderEmpty: "没有已保存的配置",
    placeholder: "选择配置",
    // Chips: CSS `uppercase` is a no-op here and the tracking that comes with
    // it is dropped at the call site, so these render as authored.
    inUse: "使用中",
    otherArm: "另一机械臂",
    renameAria: "重命名所选配置",
    renameTooltip: "重命名",
    deleteAria: "删除所选配置",
    deleteTooltip: "删除",
    moreAria: "更多操作",
    importShort: "导入",
    rename: {
      title: "重命名配置",
      description:
        "重命名该标定文件。正在使用它的机器人会自动更新。不会覆盖已存在的名称。",
      placeholder: "新名称",
      submit: "重命名",
      submitting: "正在重命名…",
      emptyName: "名称不能为空。",
      failed: "重命名失败。",
    },
    delete: {
      title: "确定删除配置 “{{name}}” 吗？",
      description:
        "这会永久删除该标定文件 — 只能重新标定机械臂才能重新生成。任何正在使用它的机器人在下次使用前都需要重新标定。",
      confirm: "删除",
    },
    toast: {
      deletedTitle: "配置已删除",
      deleted: "已删除 “{{name}}”。",
      // Chinese has a single plural category, so only _other exists.
      deletedUnassigned_other: "已删除 “{{name}}”。{{robots}} 在使用前需要重新标定。",
      // Chinese enumeration comma, not the Western comma-space.
      robotJoin: "、",
      deleteFailedTitle: "删除失败",
      assignedTitle: "配置已分配",
      assigned: "“{{name}}” 现已用于该机器人。",
      swappedTitle: "配置已互换",
      swapped: "“{{name}}” 现已用于该机械臂；另一机械臂改用 “{{previous}}”。",
      noConfig: "（无）",
      assignFailedTitle: "分配失败",
      renamedTitle: "配置已重命名",
      renamed: "“{{from}}” → “{{to}}”。",
    },
  },
  import: {
    labelLeader: "导入主臂标定",
    labelFollower: "导入从臂标定",
    descriptionLeader:
      "将上传的标定保存为新的主臂配置。不会覆盖已存在的名称 — 如果名称已被占用，请换一个。",
    descriptionFollower:
      "将上传的标定保存为新的从臂配置。不会覆盖已存在的名称 — 如果名称已被占用，请换一个。",
    placeholder: "配置名称",
    submit: "导入",
    submitting: "正在导入…",
    emptyName: "名称不能为空。",
    failed: "导入失败。",
    invalidJsonTitle: "不是有效的 JSON 文件",
    invalidJsonDescription: "所选文件无法解析为 JSON。",
    importedTitle: "标定已导入",
    imported: "已保存为 “{{name}}”。",
  },
} as const;
