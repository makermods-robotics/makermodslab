export default {
  cancel: "取消",
  close: "关闭",
  connectionError: {
    title: "连接错误",
    description: "无法连接到后端服务器。",
  },
  datasetName: {
    dataset: {
      empty: "数据集名称不能为空。",
      spaces: "数据集名称的首尾不能有空格。",
      slashes: "数据集名称不能包含斜杠。",
      dots: "数据集名称不能是 '.' 或 '..'。",
      tooLong: "数据集名称过长（最多 {{max}} 个字符）。",
    },
    namespace: {
      empty: "命名空间不能为空。",
      spaces: "命名空间的首尾不能有空格。",
      slashes: "命名空间不能包含斜杠。",
      dots: "命名空间不能是 '.' 或 '..'。",
      tooLong: "命名空间过长（最多 {{max}} 个字符）。",
    },
    charset: "只能使用字母、数字、'.'、'_' 和 '-'，且必须以字母或数字开头和结尾。",
    tooManySlashes: "数据集名称最多只能包含一个 '/'（命名空间/名称）。",
  },
} as const;
