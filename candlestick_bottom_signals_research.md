# K线底部形态 / 周线底部信号 — 现有验证研究整理

调研笔记，汇总"用K线形态/颜色序列判断市场底部"这件事上，已经存在的、经过某种程度验证的报告、网站与学术论文，供后续在 `tech_score` 里做规则时参考。不代表本仓库已经实现或采纳了下面所有结论。

## 1. 美股 / 通用市场：Bulkowski — thepatternsite.com

来源：Thomas Bulkowski,《Encyclopedia of Candlestick Charts》+ 免费网站 [thepatternsite.com](https://thepatternsite.com)。对103种蜡烛形态逐个统计**反转率、出现频率、整体表现**三项排名，样本量通常上千支美股历史数据。

- **晨星 Morning Star**：78% 概率触发看涨反转，103种中排名第6；晨星十字星（Morning Doji Star）76%（932个样本）。
- **红三兵 Three White Soldiers**：理论反转率约82%，实盘胜率约70%，但出现频率低、常"迟到"。
- **锤子线 Hammer**：反转率60%，但反转后涨幅有限，整体表现排第65/103。

关键前提：这些反转率只有在"下跌趋势末端 + 放量 + 靠近关键支撑/年内低点"同时成立时才有效，脱离上下文单独看形态可靠性明显下降。

## 2. 美股实盘回测：QuantifiedStrategies.com

《Complete Backtest of All 75 Candlestick Patterns》：用 SPY 数据把75种形态逐一变成真实策略跑回测（不只是统计反转率）。整体 10.6% CAGR vs 买入持有7.9%，胜率72%，profit factor 2.5。

- 75种里约一半实际不 work；教科书讲的形态很多在实盘回测里没有正收益。
- 个别形态方向与定义相反：Bearish Engulfing 实测反而偏看涨。

## 3. A股专题学术研究

- **Zhu, Zhang & Xiang (2016), "The predictive power of Japanese candlestick charting in Chinese stock market", *Physica A* 457** — 数据1999–2009年沪深两市，测试8种两日形态，5种对中大市值股票有统计显著预测力；预测力随预测周期变长而衰减；中等市值股票信号效果强于大市值股票。
- **Deng, Su, Ren, Yu, Zhu & Wei (2022), "Can Japanese Candlestick Patterns be Profitable on the Component Stocks of the SSE50 Index?", *SAGE Open* 12(3)** — 数据2000–2018年，上证50成分股，测试10个经典日式蜡烛形态，结合"趋势方向 + 超买超卖状态"做条件过滤，12种（形态×持有期组合）平均收益显著>1%，Long White、Bullish Gap 等看涨形态表现最好。这篇的框架（形态 + 趋势/超买超卖上下文过滤 + 分持有期统计收益）跟"形态+当天K线参数"思路最接近，且样本就是A股。
- 对照组：Marshall/Young/Rose（美股DJIA）、Marshall/Young/Chang（东京交易所）均测出蜡烛形态"无稳定预测力/无经济价值"——说明形态有效性明显是**市场特定的**，不能把美股Bulkowski统计直接套到A股。
- 补充（严谨度低于同行评审论文，仅作参考）：东方财富《AI 赋能资产配置（二十二）：大模型如何征服K线图》(2025-11)、广发证券多因子系列16《基于情景切换的技术选股策略》。

## 4. 周线级别、A股相关的其他"底部信号"

| 信号 | 统计来源 | 数据 | 严谨程度 |
|---|---|---|---|
| 周线MACD/RSI底背离 | 东方财富财富号文章 | 近30年A股出现10次，无一例外走出修复行情；其中4次是趋势反转级，对应998/1849/2440三大历史大底 | 媒体统计口径，非学术论文，样本是全部历史周线数据 |
| 神奇九转（DeMark/TD序列） | 同花顺i问财、券商回测 | 个股逃顶抄底成功率约68.6%，大盘/行业指数约75.6%；另一券商测沪深300成分股有效触发率68.3% | 券商/数据平台回测，样本量大，方法论未公开到能完全复现 |
| Zweig Breadth Thrust（广度突破） | Kirkpatrick & Dahlquist (2016)《Technical Analysis》教科书统计 | 1945–2000年美股14次信号，之后6/12个月100%上涨，平均11个月涨24.6% | 目前查到最严谨的一个，但它是"广度指标"（涨跌家数比），不是K线形态，且是美股 |
| 低档五连阳 / 用户提出的"红绿红绿绿绿绿红"颜色序列 | 民间口诀、散户科普文 | 定性描述"放量更可靠"，无具体胜率统计 | **未查到任何严谨验证来源**，属于股评/民间口诀性质 |

## 5. 结论

- 不需要自己发明形态定义或重新设计验证方法：`tech_score/rules/cdl_patterns.py` 已经用 TA-Lib 的61个 `CDL_*` 函数实现了晨星、锤子线、红三兵、刺透形态等经典识别；`tech_score/rules/divergence.py`（RSI+MACD背离）和 `tech_score/rules/demark.py`（神奇九转）本身就是上面表格里"已验证"的两类信号，只是原本按日线写的，传入周线重采样后的 DataFrame 即可复现"周线级别"的版本。
- A股场景下，Deng et al. (2022) 的"形态 + 趋势/超买超卖过滤 + 分持有期收益"框架是现成的、可直接参考的验证范式。
- 用户提出的具体颜色序列"红绿红绿绿绿绿红"没有查到任何验证来源，性质上跟"低档五连阳"一类民间口诀相同——有直觉道理，但缺乏统计支撑，需要自己用真实数据跑一遍才知道有没有效。见仓库根目录 `analyze_weekly_bottom_signals.py`。

## 参考来源

- thepatternsite.com（Bulkowski）
- quantifiedstrategies.com
- Zhu, Zhang & Xiang (2016), *Physica A* 457
- Deng et al. (2022), *SAGE Open* 12(3)
- 东方财富、广发证券公开研报
- Kirkpatrick & Dahlquist (2016), *Technical Analysis: The Complete Resource for Financial Market Technicians*
