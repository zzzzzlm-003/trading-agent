"""
notion_uploader.py
读取 transcripts/.progress.json，找出未上传到 Notion 的转录稿
输出 JSON，供 Claude 用 Notion MCP 工具上传

用法：
    python notion_uploader.py pending     # 列出待上传条目
    python notion_uploader.py content <vid_id>   # 输出某条目的 Markdown 内容
    python notion_uploader.py mark <vid_id> <notion_url>  # 标记为已上传
"""

import sys
import re
import json
from pathlib import Path

PROGRESS_FILE  = "transcripts/.progress.json"
UPLOADED_FILE  = "transcripts/.notion_uploaded.json"
CHAR_LIMIT     = 80
GAP_THRESHOLD  = 1.5

# ── 分类关键词 ──────────────────────────────────────────
CATEGORY_RULES = [
    ("量价关系", ["成交量", "量价", "流动性", "OBV", "量能", "缩量", "放量", "威廉变异"]),
    ("仓位管理", ["仓位", "加仓", "轻仓", "重仓", "满仓", "管理仓"]),
    ("风险管理", ["止盈", "止损", "风险", "Beta"]),
    ("投资方法", ["投资方法", "散户", "交易策略", "13F", "VWAP", "Rolling VWAP",
                  "日内", "短线", "波段", "头皮"]),
    ("技术分析", ["支撑", "压力", "趋势", "K线", "均线", "形态", "斐波", "波浪",
                  "布林", "高低点", "头肩", "三角", "旗形", "杯柄", "江恩", "一目",
                  "Demark", "DeMark", "ATR", "BBI", "SAR", "CCI", "KDJ", "MACD",
                  "顾比", "肯特纳", "KC", "收敛", "逃顶", "抄底", "见底", "见顶",
                  "岛形", "钻石顶", "启明星", "乖离率", "开盘价", "收盘价",
                  "有效高低点", "密集交易区", "分时图"]),
]


def classify(title: str) -> list[str]:
    cats = []
    for cat, keywords in CATEGORY_RULES:
        if any(kw in title for kw in keywords):
            cats.append(cat)
    return cats if cats else ["投资方法"]


def get_period(title: str) -> int | None:
    m = re.search(r"[（(]会员第(\d+)期[)）]", title)
    return int(m.group(1)) if m else None


def _srt2sec(t: str) -> float:
    t = t.replace(",", ".")
    h, m, rest = t.split(":")
    s, ms = rest.split(".")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def _ts(secs: float) -> str:
    m, s = int(secs // 60), int(secs % 60)
    return f"{m:02d}:{s:02d}"


def parse_srt(srt_path: str):
    text   = Path(srt_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text.strip())
    segs   = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            t0, t1 = lines[1].split(" --> ")
            segs.append({
                "start": _srt2sec(t0.strip()),
                "end":   _srt2sec(t1.strip()),
                "text":  " ".join(l.strip() for l in lines[2:]),
            })
        except (IndexError, ValueError):
            continue
    return segs


def _clean_intra_repeat(text: str) -> str:
    """截断单段内 Whisper 幻觉重复（如同一短句连续出现 4+ 次）。
    匹配 4-20 字符的子串连续重复 ≥4 次，从重复起点截断。"""
    m = re.search(r'(.{4,20})\1{3,}', text)
    if m:
        return text[:m.start()].strip()
    return text


def build_paragraphs(segs):
    SENTENCE_END = set("。！？…")
    # 先在 segment 级别过滤 Whisper 连续重复幻觉
    # 用非空段的最近2个文本判断是否连续重复，避免已被清空的段破坏窗口
    filtered = []
    for seg in segs:
        txt = seg["text"].strip()
        non_blank = [s["text"].strip() for s in filtered if s["text"].strip()]
        if txt and len(non_blank) >= 2 and non_blank[-1] == txt and non_blank[-2] == txt:
            filtered.append({**seg, "text": ""})  # 静音掉重复段
        else:
            filtered.append(seg)
    segs = filtered

    paras, p_start, p_text = [], None, ""
    for i, seg in enumerate(segs):
        txt = _clean_intra_repeat(seg["text"].strip())
        if not txt:
            continue
        if p_start is None:
            p_start = seg["start"]
        p_text += txt
        ends_sent = p_text[-1] in SENTENCE_END
        long_gap  = (i+1 < len(segs) and segs[i+1]["start"] - seg["end"] > GAP_THRESHOLD)
        soft_cut  = (len(p_text) >= CHAR_LIMIT and
                     i+1 < len(segs) and segs[i+1]["start"] - seg["end"] > 0.3)
        if ends_sent or long_gap or soft_cut:
            paras.append((p_start, p_text.strip()))
            p_start = p_text = None, ""
            p_start = None
            p_text  = ""
    if p_text.strip():
        paras.append((p_start, p_text.strip()))

    # 去除 Whisper 重复幻觉：同一段落出现 3 次以上则截断
    seen: dict[str, int] = {}
    clean = []
    for para in paras:
        key = para[1][:40]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            clean.append((para[0], "*(转录重复，此处内容已截断)*"))
            break
        clean.append(para)
    return clean


def srt_to_markdown(srt_path: str, title: str, url: str) -> str:
    segs  = parse_srt(srt_path)
    paras = build_paragraphs(segs)
    lines = [f"# {title}\n", f"> {url}\n\n---\n\n"]
    prev_minute = -1
    for start, text in paras:
        m = int(start // 60)
        if m > 0 and m != prev_minute and (start - m*60) < 30:
            lines.append(f"\n## {_ts(start)}\n\n")
            prev_minute = m
        lines.append(f"`{_ts(start)}` {text}\n\n")
    total = paras[-1][0] if paras else 0
    lines.append(f"\n---\n*全长约 {int(total//60)} 分钟，共 {len(paras)} 段*\n")
    return "".join(lines)


def load_progress():
    if not Path(PROGRESS_FILE).exists():
        return {}
    return json.loads(Path(PROGRESS_FILE).read_text(encoding="utf-8"))


def load_uploaded():
    if not Path(UPLOADED_FILE).exists():
        return {}
    return json.loads(Path(UPLOADED_FILE).read_text(encoding="utf-8"))


def save_uploaded(data):
    Path(UPLOADED_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pending"

    progress = load_progress()
    uploaded = load_uploaded()

    if cmd == "pending":
        pending = []
        for vid_id, v in progress.items():
            if not v.get("transcribed"):
                continue
            if vid_id in uploaded:
                continue
            txt_path = v.get("txt_path", "")
            srt_path = txt_path.replace(".txt", ".srt")
            if not Path(srt_path).exists():
                # try to find any matching srt
                stem = Path(txt_path).stem
                srts = list(Path("transcripts").glob(f"{stem}.srt"))
                srt_path = str(srts[0]) if srts else ""
            pending.append({
                "vid_id":   vid_id,
                "title":    v["title"],
                "url":      v["url"],
                "period":   get_period(v["title"]),
                "cats":     classify(v["title"]),
                "srt_path": srt_path,
                "has_srt":  Path(srt_path).exists() if srt_path else False,
            })
        print(json.dumps(pending, ensure_ascii=False, indent=2))

    elif cmd == "content":
        vid_id = sys.argv[2]
        v = progress.get(vid_id)
        if not v:
            print(json.dumps({"error": "not found"}))
            return
        txt_path = v.get("txt_path", "")
        srt_path = txt_path.replace(".txt", ".srt")
        if not Path(srt_path).exists():
            print(json.dumps({"error": f"srt not found: {srt_path}"}))
            return
        content = srt_to_markdown(srt_path, v["title"], v["url"])
        print(json.dumps({"content": content, "title": v["title"]}, ensure_ascii=False))

    elif cmd == "content_only":
        # 只输出正文内容（去掉 # 标题 / > URL / --- 头部），供直接写入 Notion content 字段
        vid_id = sys.argv[2]
        v = progress.get(vid_id)
        if not v:
            sys.exit(f"ERROR: vid_id {vid_id} not found")
        txt_path = v.get("txt_path", "")
        srt_path = txt_path.replace(".txt", ".srt")
        if not Path(srt_path).exists():
            sys.exit(f"ERROR: srt not found: {srt_path}")
        full = srt_to_markdown(srt_path, v["title"], v["url"])
        # 去掉前三行（# title / > url / 空行 / --- / 空行）
        sep = "---\n\n"
        idx = full.find(sep)
        body = full[idx + len(sep):] if idx >= 0 else full
        sys.stdout.write(body)

    elif cmd == "meta":
        # 输出单个视频的元数据（不含内容），供批量脚本使用
        vid_id = sys.argv[2]
        v = progress.get(vid_id)
        if not v:
            print(json.dumps({"error": "not found"}))
            return
        txt_path = v.get("txt_path", "")
        srt_path = txt_path.replace(".txt", ".srt")
        print(json.dumps({
            "vid_id":   vid_id,
            "title":    v["title"],
            "url":      v["url"],
            "period":   get_period(v["title"]),
            "cats":     classify(v["title"]),
            "has_srt":  Path(srt_path).exists(),
        }, ensure_ascii=False))

    elif cmd == "mark":
        vid_id     = sys.argv[2]
        notion_url = sys.argv[3] if len(sys.argv) > 3 else ""
        uploaded[vid_id] = {"notion_url": notion_url, "title": progress.get(vid_id, {}).get("title", "")}
        save_uploaded(uploaded)
        print(f"marked: {vid_id}")

    elif cmd == "stats":
        total     = sum(1 for v in progress.values() if v.get("transcribed"))
        done_up   = len(uploaded)
        print(f"已转录: {total}  已上传Notion: {done_up}  待上传: {total - done_up}")


if __name__ == "__main__":
    main()
