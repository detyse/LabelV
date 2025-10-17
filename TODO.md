## Optimize

- ~~像剪辑软件一样, 在时间轴缩放的时候, 以 time cursor(时间轴上的竖线) 为中心缩放, 而不是以鼠标为中心 (SOLVED)~~

- ~~将颜色系统真正地利用起来, 自动为不同名称的标签改变颜色. 同时保留用户更改颜色的功能. SOLVED - 在重命名时自动应用模板颜色, 模板记住每种名称的颜色~~

- ~~删除标签时, 右侧标签栏没有对应删除 SOLVED~~

- ~~可以通过点击右侧标签栏的标签播放对应片段~~

- ~~new bug: the time of track not as same as the progress bar time (video time). the playhead could not follow the progress bar, progress bar could not follow the playhead either.~~

- ~~关于类别, 在大多数情况下, 类别都是 default, 删除现有类别 SOLVED - 已从 GUI 中移除类别显示~~

- ~~删除 GUI 中的 类别, 只保留名称, 但是在 json 中保留 类别 key, 设置为 default, 未来备用, 但是在标注中不涉及 类别. SOLVED~~

- ~~检查一下, 进度条 和 时间轴/time ruler 的一致性(使用中发现不一致的情况), 同时检查一下 playhead 和 进度条的同步性.~~

- ~~关于模板BUG, 在标签编辑器中, 每输入一个字母都自动保存一次模板, 导致保存很多无用模板. 删除自动保存模板的功能. 当用户手动添加时再添加模板. SOLVED~~ 

- ~~在换颜色时, 同步更新, 标签列表中的颜色, 同时加载的时候, 也更新标签列表颜色. SOLVED~~

- ~~还是有点奇怪, 新加载的标签颜色不是原来设定颜色. 使用 模板创建标签时, 一开始并没有没有使用模板颜色. i see, json 文件中不存在颜色字段, 为 json 中加入 color 字段. 同时要求兼容没有 color 的 json 的格式. 加载时如果标签名称不在模板中, 而且是新出现的名称, 就为其分配新颜色. ~~

- 兼容加载 csv 文件. 如果有 json, 让用户选一下加载哪个. csv 为 moseq 格式.