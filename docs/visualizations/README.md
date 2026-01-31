# MCM 2026 Problem C - Visualization System

**完整的可视化管道，为Q1-Q4分析生成出版质量的图表**

## 概述

本目录用于维护与论文主线一致的可视化口径与实现。当前阶段的目标是：所有写入论文正文的图表均可由 `data/processed/` 与 `outputs/` 中的主线 CSV 产物复现生成。

### 核心特性

- **全面覆盖**: 覆盖题目 Q1–Q4 的主线叙事所需图表
- **出版质量**: 输出 TIFF/EPS（按竞赛/论文常见要求）
- **专业设计**: 遵循学术论文可视化标准和最佳实践
- **中英双语**: 图表标题和标签支持中文显示

## 系统架构

```
src/mcm2026/visualizations/
├── __init__.py                 # 包初始化
├── visualizations.ini          # 可视化配置（字体/DPI等）
├── visualization.ps1           # Windows 生成入口（当前仅作为脚本入口；后续会补齐统一 Python 协调器）
├── q1_visualizations.py        # Q1 绘图
├── q2_visualizations.py        # Q2 绘图
├── q3_visualizations.py        # Q3 绘图
└── q4_visualizations.py        # Q4 绘图

docs/visualizations/
├── README.md                   # 本文档
├── Q1.md                       # Q1 图表清单（论文主线口径）
├── Q2.md                       # Q2 图表清单（论文主线口径）
├── Q3.md                       # Q3 图表清单（论文主线口径）
└── Q4.md                       # Q4 图表清单（论文主线口径）
```

## 主线 vs 附录（Showcase）边界

为对齐题目要求（见 `mcm2026c/2026_MCM_Problem_C.md`）并保证论文可复现：

- **主线图表（写进论文正文）**：只能使用
  - `data/processed/*.csv`
  - `outputs/tables/*.csv`
  - `outputs/predictions/*.csv`
- **附录/Showcase 图表**：可以使用 `outputs/tables/showcase/*.csv`，但必须在图注/正文明确标注为“附录对照/炫技”，不能替代主线结论。

## 生成的图表清单

### Q1: 粉丝投票推断与不确定性量化 (10个文件)
- `q1_uncertainty_heatmap.{tiff,eps}` - 不确定性热图
- `q1_fan_share_intervals.{tiff,eps}` - 粉丝份额后验区间图
- `q1_judge_vs_fan_scatter.{tiff,eps}` - 技术vs人气散点图
- `q1_mechanism_comparison.{tiff,eps}` - 机制对比图
- `q1_statistical_vs_ml_comparison.{tiff,eps}` - 统计vs机器学习对比

### Q2: 反事实机制对比 (12个文件)
- `q2_mechanism_difference_distribution.{tiff,eps}` - 机制差异分布图
- `q2_judge_save_impact.{tiff,eps}` - Judge Save影响对比
- `q2_controversial_seasons_heatmap.{tiff,eps}` - 争议赛季热图
- `q2_fan_judge_divergence.{tiff,eps}` - 粉丝-评委分歧散点图
- `q2_bobby_bones_case_study.{tiff,eps}` - Bobby Bones案例分析
- `q2_ml_mechanism_prediction.{tiff,eps}` - 机器学习机制预测

### Q3: 影响因素分析 (12个文件)
- `q3_judge_vs_fan_forest_plot.{tiff,eps}` - 技术vs人气线森林图
- `q3_effect_size_comparison.{tiff,eps}` - 效应大小对比图
- `q3_age_effect_curves.{tiff,eps}` - 年龄效应曲线
- `q3_industry_impact_heatmap.{tiff,eps}` - 职业类别影响热图
- `q3_mixed_effects_vs_ml.{tiff,eps}` - 混合效应vs机器学习对比
- `q3_uncertainty_propagation.{tiff,eps}` - 不确定性传播效果图

### Q4: 新投票机制设计 (12个文件)
- `q4_mechanism_tradeoff_scatter.{tiff,eps}` - 机制权衡散点图
- `q4_robustness_curves.{tiff,eps}` - 稳健性曲线
- `q4_champion_uncertainty_analysis.{tiff,eps}` - 冠军不确定性分析
- `q4_seasonal_variation_analysis.{tiff,eps}` - 季度差异分析
- `q4_pareto_frontier.{tiff,eps}` - Pareto前沿图
- `q4_mechanism_recommendation.{tiff,eps}` - 机制推荐决策树

## 使用方法

### 推荐流程（先保证数据，再生成图表）

```bash
uv run python run_all.py
```

随后按 `docs/visualizations/Q1.md`–`Q4.md` 的清单生成论文主线图。

## 技术规范

### 图表质量标准
- **分辨率**: 300 DPI (TIFF格式)
- **格式**: TIFF (位图) + EPS (矢量图)
- **字体**: Serif字体族，学术论文标准
- **颜色**: 色盲友好的ColorBrewer调色板
- **尺寸**: 适配论文单栏(12×8英寸)和双栏(16×8英寸)布局

### 依赖要求
```python
# 核心依赖
matplotlib >= 3.5.0
seaborn >= 0.11.0
pandas >= 1.3.0
numpy >= 1.21.0

# 可选依赖 (3D图表)
mpl_toolkits.mplot3d
```

### 数据输入要求
可视化系统需要以下数据文件存在：
- `outputs/tables/mcm2026c_q1_uncertainty_summary.csv`
- `outputs/predictions/mcm2026c_q1_fan_vote_posterior_summary.csv`
- `outputs/tables/mcm2026c_q2_mechanism_comparison.csv`
- `outputs/tables/mcm2026c_q3_impact_analysis_coeffs.csv`
- `outputs/tables/mcm2026c_q4_new_system_metrics.csv`
- `data/processed/dwts_weekly_panel.csv`

## 设计理念

### 1. 论文叙事支撑
每张图表都精心设计以支撑论文的核心论证：
- **Q1**: 展示贝叶斯推断的有效性和不确定性量化
- **Q2**: 证明反事实分析的价值和机制差异的重要性
- **Q3**: 揭示技术线vs人气线的差异机制
- **Q4**: 展示新机制设计的权衡和优化结果

### 2. 学术标准遵循
- 遵循国际数学建模竞赛的图表规范
- 支持高质量学术发表要求
- 提供完整的图例和标注说明

### 3. 可解释性优先
- 优先使用直观易懂的图表类型
- 提供详细的图注和解释
- 避免过度复杂的可视化设计

## 扩展和定制

### 添加新图表
1. 在相应的 `qX_visualizations.py` 中添加新函数
2. 明确该图属于：主线（仅用主线 CSV）或附录（可用 showcase）
3. 更新相应文档（`docs/visualizations/QX.md`）的图表口径

### 自定义样式
```python
# 在各可视化模块顶部修改
plt.rcParams.update({
    'font.size': 12,           # 基础字体大小
    'axes.titlesize': 14,      # 标题字体大小
    'font.family': 'serif',    # 字体族
    'figure.dpi': 300          # 图片分辨率
})
```

### 颜色主题定制
每个模块都定义了一致的颜色方案，可以统一修改：
```python
colors = {
    'percent': '#1f77b4',      # 蓝色
    'rank': '#ff7f0e',         # 橙色
    'judge_save': '#2ca02c',   # 绿色
    # ... 其他颜色定义
}
```

## 故障排除

### 常见问题

1. **PostScript透明度警告**
   ```
   The PostScript backend does not support transparency
   ```
   这是正常警告，不影响图表质量。EPS格式不支持透明度，会自动转换为不透明。

2. **字体显示问题**
   ```bash
   # 清除matplotlib字体缓存
   rm -rf ~/.matplotlib
   ```

3. **内存不足错误**
   - 减少同时生成的图表数量

4. **数据文件缺失**
   ```bash
   # 先运行主管道生成数据
   uv run python run_all.py
   ```

### 调试模式
```python
# 在可视化函数中添加调试信息
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 参考文档

- [Q1可视化规划](Q1.md) - 详细的Q1图表设计文档
- [Q2可视化规划](Q2.md) - 详细的Q2图表设计文档
- [Q3可视化规划](Q3.md) - 详细的Q3图表设计文档
- [Q4可视化规划](Q4.md) - 详细的Q4图表设计文档

## 🤝 贡献指南

1. 遵循现有的代码风格和命名约定
2. 新增图表需要同时生成TIFF和EPS格式
3. 添加适当的错误处理和日志记录
4. 更新相关文档和示例
5. 确保与主管道的集成兼容性

---

**最后更新**: 2026年1月31日  
**版本**: 1.0.0  
**维护者**: MCM 2026 Problem C Team