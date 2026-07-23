# Trading Analysis Agent

支持美股（含期权）| A股（含资金流）| 恒生科技指数

---

## 安装依赖

```bash
pip install ta-lib yfinance akshare pandas numpy mplfinance openai-whisper yt-dlp anthropic

# macOS 需要 ffmpeg（Whisper 转录音频用）
brew install ffmpeg
```

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `analyzer.py` | 主分析模块：技术指标 + 蜡烛图 + 资金流 + 期权 |
| `get_youtube_transcript.py` | 获取 YouTube 会员视频文字稿 |
| `transcript_strategy_agent.py` | 聚合所有 transcript，构建“方法论技术分析 Agent” |

---

## 使用方法

### 1. 分析股票

```python
from analyzer import analyze

analyze("NVDA")       # 美股（含期权建议）
analyze("600519")     # A股茅台（含主力资金流）
analyze("AAPL")       # 美股苹果
analyze("HSTECH")     # 恒生科技指数（代理：3033.HK）
```

命令行：
```bash
python analyzer.py NVDA
python analyzer.py 600519
python analyzer.py NVDA AAPL TSLA   # 批量分析
```

### 2. 获取 YouTube 会员视频文字稿

**前提：在 Chrome 中登录 YouTube 会员账号**

> 合规提示：仅处理你有合法访问权限且符合平台条款的内容。

```bash
# 单个视频
python get_youtube_transcript.py https://youtu.be/xxxx

# 转录 + 自动提炼方法论（需要 ANTHROPIC_API_KEY）
export ANTHROPIC_API_KEY=your_key
python get_youtube_transcript.py https://youtu.be/xxxx --extract-method

# 指定模型（large 质量最好，medium 速度更快）
python get_youtube_transcript.py https://youtu.be/xxxx --model medium

# 批量处理（urls.txt 每行一个链接）
python get_youtube_transcript.py --batch urls.txt
```

文字稿保存在 `transcripts/` 目录，包含：
- `xxx.txt` — 纯文字稿
- `xxx.srt` — 带时间戳字幕
- `xxx_timestamped.txt` — 带时间轴的文字稿
- `xxx_methodology.txt` — 投资方法论提炼（需要 `--extract-method`）

### 3. 基于全部 transcript 构建技术分析 Agent

```bash
# 生成策略画像 + system prompt
python transcript_strategy_agent.py --build

# 询问：从历史 transcript 中检索证据并回答
python transcript_strategy_agent.py --ask "这个体系更重视MACD还是RSI？"

# 可选：用 Claude 生成高质量方法论（需要 ANTHROPIC_API_KEY）
export ANTHROPIC_API_KEY=your_key
python transcript_strategy_agent.py --build --claude
```

会输出：
- `transcripts/strategy_profile.json` — 聚合统计画像
- `transcripts/strategy_system_prompt.md` — 可直接注入 Agent 的 system prompt
- `transcripts/strategy_claude_synthesis.md` — Claude 深度综合（可选）

---

## 技术分析覆盖范围

### 指标
- 均线：MA5 / MA10 / MA20 / MA60 / EMA20
- MACD（金叉/死叉检测）
- RSI(6) / RSI(14)
- KDJ（K/D/J）
- 布林带（位置百分比 + 带宽）
- ATR(14) 波动性
- 成交量比（放量/缩量判断）

### 蜡烛图形态（TA-Lib，61种，含中文名）
锤子线、吊人线、吞没、十字星、晨星、黄昏星、乌云盖顶、刺透形态、三白兵、三黑鸦……等全部 61 种，并标注看涨/看跌/中性

### A股专属
- 个股主力净流入（今日 + 3日 + 5日趋势）
- 超大单（机构）/ 大单 / 中单 / 小单分层
- 北向资金（沪股通 + 深股通）

### 美股期权专属
- ATM IV 均值估算
- 最大痛点（Max Pain）
- 最活跃 Call / Put 合约
- 基于技术面自动建议期权策略：
  - Bull Put Spread / Bear Call Spread（IV 高时）
  - Long Call / Long Put（IV 正常时）
  - Iron Condor / Short Strangle（方向不明 + IV 高）

---

## 数据源

| 市场 | 价格数据 | 资金/期权数据 |
|------|---------|-------------|
| 美股 | yfinance | yfinance 期权链 |
| A股 | AkShare（东方财富） | AkShare 资金流向 |
| 恒生科技 | yfinance (3033.HK) | — |

---

## 进阶：将博主方法论注入 Agent

1. 用 `get_youtube_transcript.py --extract-method` 处理几期核心视频
2. 将生成的 `*_methodology.txt` 整理成一份方法论文档
3. 在 `analyzer.py` 的综合分析部分，将方法论作为 system prompt 调用 Claude API
4. 或者直接把方法论文档粘贴到 Claude Cowork 对话开头，再分析具体标的
