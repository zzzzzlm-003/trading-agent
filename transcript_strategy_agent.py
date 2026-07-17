"""
Transcript Strategy Agent
=========================
将多个 YouTube 视频 transcript 聚合成“技术分析方法论 Agent”。

⚠️ 合规说明：
- 仅处理你有合法访问权限、且符合平台条款的内容。
- 不提供绕过付费/权限控制的能力。

核心能力：
1) 从 transcripts 目录加载全部文字稿
2) 构建本地检索索引（无需向量数据库）
3) 生成策略画像（常见指标、偏好、风险控制）
4) 交互式问答（基于 transcript 证据）
5) 可选：调用 Claude 生成更高质量的总结

依赖：
- 必需：Python 标准库
- 可选：anthropic（用于高质量综合总结）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


STOPWORDS = {
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与", "着", "或", "一个", "我们", "你", "我",
    "the", "a", "an", "to", "of", "is", "are", "and", "or", "for", "on", "in", "with", "that", "this",
}

TECH_KEYWORDS = {
    "趋势": ["均线", "ma", "ema", "趋势", "突破", "回踩", "支撑", "阻力"],
    "动量": ["macd", "rsi", "kdj", "动量", "背离", "金叉", "死叉"],
    "波动": ["布林", "boll", "atr", "波动", "震荡", "带宽"],
    "量价": ["成交量", "放量", "缩量", "量价", "换手"],
    "风险": ["止损", "仓位", "回撤", "风控", "纪律", "杠杆", "分批"],
    "交易执行": ["开仓", "加仓", "减仓", "止盈", "出场", "确认", "信号"],
}

EN_PATTERN = re.compile(r"[A-Za-z]{2,}")
ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Chunk:
    file_name: str
    chunk_id: int
    text: str


class TranscriptStrategyAgent:
    def __init__(self, transcript_dir: str = "transcripts", out_dir: str = "transcripts") -> None:
        self.transcript_dir = Path(transcript_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: List[Chunk] = []
        self.doc_freq: Counter = Counter()
        self.chunk_term_freq: List[Counter] = []
        self.chunk_norm: List[float] = []

    def load_transcripts(self) -> int:
        files = sorted(self.transcript_dir.glob("*.txt"))
        files = [f for f in files if not f.name.endswith("_methodology.txt") and not f.name.endswith("_timestamped.txt")]

        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            for i, ch in enumerate(self._split_into_chunks(text, chunk_size=1200, overlap=150)):
                self.chunks.append(Chunk(file_name=f.name, chunk_id=i, text=ch))

        return len(files)

    def build_index(self) -> None:
        if not self.chunks:
            return

        for chunk in self.chunks:
            tf = Counter(self._tokenize(chunk.text))
            self.chunk_term_freq.append(tf)
            for t in tf.keys():
                self.doc_freq[t] += 1

        n = len(self.chunks)
        for tf in self.chunk_term_freq:
            sq = 0.0
            for t, c in tf.items():
                weight = self._tfidf_weight(c, self.doc_freq[t], n)
                sq += weight * weight
            self.chunk_norm.append(math.sqrt(sq) + 1e-12)

    def ask(self, question: str, top_k: int = 5) -> Dict:
        if not self.chunks:
            return {"answer": "未加载到 transcript。", "evidence": []}

        q_tf = Counter(self._tokenize(question))
        n = len(self.chunks)

        q_vec = {}
        q_norm_sq = 0.0
        for t, c in q_tf.items():
            df = self.doc_freq.get(t, 0)
            w = self._tfidf_weight(c, df, n)
            q_vec[t] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) + 1e-12

        scored: List[Tuple[float, int]] = []
        for i, tf in enumerate(self.chunk_term_freq):
            dot = 0.0
            for t, q_w in q_vec.items():
                if t not in tf:
                    continue
                d_w = self._tfidf_weight(tf[t], self.doc_freq[t], n)
                dot += q_w * d_w
            sim = dot / (q_norm * self.chunk_norm[i])
            if sim > 0:
                scored.append((sim, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:top_k]

        evidence = []
        for sim, idx in top:
            c = self.chunks[idx]
            evidence.append({
                "score": round(sim, 4),
                "file": c.file_name,
                "chunk_id": c.chunk_id,
                "text": c.text[:700].strip(),
            })

        answer = self._build_grounded_answer(question, evidence)
        return {"answer": answer, "evidence": evidence}

    def generate_strategy_profile(self) -> Dict:
        full_text = "\n".join(c.text for c in self.chunks)
        tokens = self._tokenize(full_text)
        overall_counter = Counter(tokens)

        category_scores = {}
        for cat, kws in TECH_KEYWORDS.items():
            score = 0
            hit_words = {}
            for kw in kws:
                hit = overall_counter.get(kw.lower(), 0) + overall_counter.get(kw, 0)
                if hit > 0:
                    hit_words[kw] = hit
                    score += hit
            category_scores[cat] = {
                "score": score,
                "hits": dict(sorted(hit_words.items(), key=lambda x: x[1], reverse=True)[:10]),
            }

        top_terms = [t for t, c in overall_counter.most_common(200) if t not in STOPWORDS and len(t) > 1][:30]

        profile = {
            "transcript_chunks": len(self.chunks),
            "category_scores": category_scores,
            "top_terms": top_terms,
            "generated_at": self._now(),
        }

        json_path = self.out_dir / "strategy_profile.json"
        json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

        md_path = self.out_dir / "strategy_system_prompt.md"
        md_path.write_text(self._to_system_prompt(profile), encoding="utf-8")

        return {
            "profile_path": str(json_path),
            "prompt_path": str(md_path),
            "profile": profile,
        }

    def synthesize_with_claude(self, model: str = "claude-sonnet-4-5") -> str:
        try:
            import anthropic  # type: ignore
        except Exception:
            return "未安装 anthropic，跳过 Claude 综合。"

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return "未设置 ANTHROPIC_API_KEY，跳过 Claude 综合。"

        sample = "\n\n".join(c.text[:1200] for c in self.chunks[:30])
        prompt = f"""
你是资深技术分析研究员。基于以下多个 transcript 摘录，输出一份“可执行交易方法论”，要求：
1) 指标体系（趋势/动量/波动/量价）
2) 入场条件（必须可量化）
3) 出场与止损（必须可量化）
4) 仓位与风险控制
5) 多时间框架确认
6) 常见失效场景与规避
7) 最终给出可直接作为 AI Agent system prompt 的版本（中文）

Transcript 摘录：
{sample}
"""

        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.content[0].text
        out_path = self.out_dir / "strategy_claude_synthesis.md"
        out_path.write_text(text, encoding="utf-8")
        return f"Claude 综合完成：{out_path}"

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            piece = text[start:end]
            if end < n:
                cut = max(piece.rfind("。"), piece.rfind("."), piece.rfind("!"), piece.rfind("?"))
                if cut > int(chunk_size * 0.65):
                    end = start + cut + 1
                    piece = text[start:end]
            chunks.append(piece.strip())
            if end >= n:
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()

        tokens: List[str] = []

        # 英文词
        tokens.extend(m.group(0) for m in EN_PATTERN.finditer(text))

        # 中文：单字 + 双字，提升检索召回
        zh_chars = ZH_PATTERN.findall(text)
        tokens.extend(zh_chars)
        tokens.extend("".join(zh_chars[i:i + 2]) for i in range(len(zh_chars) - 1))

        # 保留常见技术分析关键词（避免被停用词误删）
        keep = {
            "macd", "rsi", "kdj", "atr", "boll", "ma", "ema",
            "止损", "仓位", "入场", "出场", "加仓", "减仓", "突破", "回撤", "成交量", "均线",
        }

        filtered = []
        for t in tokens:
            if not t:
                continue
            if t in keep:
                filtered.append(t)
                continue
            if t in STOPWORDS:
                continue
            filtered.append(t)
        return filtered

    @staticmethod
    def _tfidf_weight(tf: int, df: int, n_docs: int) -> float:
        if tf <= 0:
            return 0.0
        idf = math.log((n_docs + 1) / (df + 1)) + 1
        return (1 + math.log(tf)) * idf

    @staticmethod
    def _build_grounded_answer(question: str, evidence: List[Dict]) -> str:
        if not evidence:
            return "没有检索到足够相关内容。请换个问法，或先增加更多 transcript。"

        lines = [f"问题：{question}", "", "基于 transcript 的关键信息："]
        for i, e in enumerate(evidence[:3], 1):
            snippet = e["text"].replace("\n", " ").strip()
            lines.append(f"{i}. ({e['file']}#chunk{e['chunk_id']}) {snippet[:180]}...")

        lines.append("")
        lines.append("建议：优先把以上规则转成可量化条件（如 MA20 上方、MACD 柱翻红、ATR 止损倍数），再接入你的 `signal_generator.py` 做回测验证。")
        return "\n".join(lines)

    @staticmethod
    def _to_system_prompt(profile: Dict) -> str:
        cats = profile.get("category_scores", {})
        top_terms = "、".join(profile.get("top_terms", [])[:20])

        cat_rank = sorted(cats.items(), key=lambda x: x[1].get("score", 0), reverse=True)
        focus = "\n".join(
            f"- {name}：score={meta.get('score', 0)}，关键词={', '.join(list(meta.get('hits', {}).keys())[:6])}"
            for name, meta in cat_rank
        )

        return f"""# Transcript 驱动技术分析 Agent System Prompt

你是一个“以用户历史视频 transcript 方法论”为核心的技术分析助手。
请遵守以下优先级：
1. 先复述证据：先给出与问题最相关的 transcript 依据。
2. 再给结论：给出方向判断（偏多/偏空/中性）与置信度。
3. 必须量化：所有建议都要有可执行阈值。
4. 强制风控：给出止损、仓位、失效条件。

## 从 transcript 聚合得到的重点
{focus}

## 高频术语
{top_terms}

## 输出模板
- 结论：
- 依据（来自 transcript）：
- 入场条件：
- 止损与止盈：
- 仓位建议：
- 失效条件：
"""

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="用 transcript 构建技术分析 Agent")
    parser.add_argument("--transcript-dir", default="transcripts", help="transcript 输入目录")
    parser.add_argument("--out-dir", default="transcripts", help="输出目录")
    parser.add_argument("--build", action="store_true", help="构建策略画像与 system prompt")
    parser.add_argument("--ask", type=str, help="提问（会返回证据片段）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回数量")
    parser.add_argument("--claude", action="store_true", help="调用 Claude 综合总结")
    args = parser.parse_args()

    agent = TranscriptStrategyAgent(args.transcript_dir, args.out_dir)
    file_count = agent.load_transcripts()
    if file_count == 0:
        print("❌ 未找到 transcript 文件。请先运行 get_youtube_transcript.py 生成 *.txt")
        return

    agent.build_index()
    print(f"✅ 已加载 {file_count} 个 transcript，切分为 {len(agent.chunks)} 个文本块")

    if args.build:
        out = agent.generate_strategy_profile()
        print(f"✅ 策略画像: {out['profile_path']}")
        print(f"✅ System Prompt: {out['prompt_path']}")

    if args.claude:
        msg = agent.synthesize_with_claude()
        print(f"🤖 {msg}")

    if args.ask:
        result = agent.ask(args.ask, top_k=args.top_k)
        print("\n=== 回答 ===")
        print(result["answer"])
        print("\n=== 证据 ===")
        for e in result["evidence"]:
            print(f"- score={e['score']} | {e['file']}#chunk{e['chunk_id']}")


if __name__ == "__main__":
    main()
