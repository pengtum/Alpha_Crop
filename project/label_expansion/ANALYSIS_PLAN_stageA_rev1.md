# 阶段 A 方案修订 v1（正式运行前）

写于 2026-09-24。原方案 `ANALYSIS_PLAN_stageA.md`（1563b358）保持冻结；本修订与原方案冲突时以本修订为准。

## 修订内容

1. **方法清单**：原方案写的是"全部登记方法"，实现完成后共登记了 62 个（含变体）。按文献调研第 3.1 节"每个家族取主要变体"，阶段 A 只跑 33 个方法，另加 4 个神谕（见 `stageA.py` 的 `METHODS_A`）：
   - **概率设计**：srs、srs_nuts3、lpm2、predstrat（√分配）、predstrat_prop、predstrat_equal、pmht、cube、clhs。
   - **不确定性与多样化**：margin、entropy、margin_prior、margin_r2（两轮，需要 oracle）、gist_margin、gist_maxmin、gist_geo、kcenter、kcenter_foreign、badge、kdpp。
   - **覆盖与代表性**：clue、uherding、maxherding、typiclust、typiclust_strat、fass_quota、fass_quota_equal、probcover、kherding、fscs。
   - **迁移**：tqs、bait、hybrid（第一批按预测类分层，第二批 BADGE）。
   - **神谕**：oracle_prop、oracle_sqrt、oracle_equal、oracle_err。

   未入选的变体只在阶段 B 需要时补跑。

2. **超参数覆盖**（按集成审查建议，在看任何阶段 A 结果之前确定）：
   - margin_prior 的训练先验取均匀分布（p_train='uniform'），因为外国模型用了按外国频率的类别平衡权重。
   - uherding 与 maxherding 的近邻截断放宽到 M_nn=1000。按规格默认值，在 32 维中会丢掉大部分核质量。
   - srs_nuts3 的每层最少样本量由 0 改为 1。否则小层会出现 π=0，使先验估计有偏。

3. **需要重训的方法**：margin_r2 和 hybrid 在选样过程中要重训模型，改用运行器注入的正式训练函数，与最终评估模型配置一致：S2 原始特征、300 棵树、只按外国频率的类别权重、目标样本 r(k) 倍、用阶段 0 的温度校准。不再用实现自带的轻量训练（PCA 特征、100 棵树）。

4. **ProbCover**：记录 coverage.DIAG 中的纯度是否达到 α，用于解读。

## 披露

- 提升空间诊断（`headroom.py`，PT、SE、SK、LV；随机 20 次与真实类别分层神谕）的结果已经看过。其中 PT 与开发国重叠。
- 冒烟测试只在 SI 的第 0 种划分、第 0 次重复上运行，用 5 棵树，只检查能否运行与耗时，不看指标。其输出放在 `verify/stageA_smoke/`，不进入分析。
