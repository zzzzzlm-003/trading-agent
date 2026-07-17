"""
将 .srt 转成可读 Markdown
用法：
    python reformat_transcript.py               # 处理 transcripts/ 下所有 .srt
    python reformat_transcript.py path/to.srt   # 处理单个文件
"""
import sys
import re
from pathlib import Path


GAP_THRESHOLD = 1.5   # 静音超过这么多秒 → 强制换段
CHAR_LIMIT    = 80    # 累积字数超过这个且有停顿 → 换段


def parse_srt(srt_path: str) -> list[dict]:
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


def _srt2sec(t: str) -> float:
    t = t.replace(",", ".")
    h, m, rest = t.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _ts(secs: float) -> str:
    m, s = int(secs // 60), int(secs % 60)
    return f"{m:02d}:{s:02d}"


def build_paragraphs(segs: list) -> list[tuple]:
    """返回 [(start_secs, text), ...] 每个元素是一段"""
    SENTENCE_END = set("。！？…")
    paras   = []
    p_start = None
    p_text  = ""

    for i, seg in enumerate(segs):
        txt = seg["text"].strip()
        if not txt:
            continue
        if p_start is None:
            p_start = seg["start"]
        p_text += txt

        ends_sent = p_text[-1] in SENTENCE_END
        long_gap  = (i + 1 < len(segs) and
                     segs[i + 1]["start"] - seg["end"] > GAP_THRESHOLD)
        soft_cut  = (len(p_text) >= CHAR_LIMIT and
                     i + 1 < len(segs) and
                     segs[i + 1]["start"] - seg["end"] > 0.3)

        if ends_sent or long_gap or soft_cut:
            paras.append((p_start, p_text.strip()))
            p_start = None
            p_text  = ""

    if p_text.strip():
        paras.append((p_start, p_text.strip()))
    return paras


def to_markdown(title: str, url: str, paras: list) -> str:
    lines = []
    lines.append(f"# {title}\n")
    if url:
        lines.append(f"> {url}\n")
    lines.append("\n---\n\n")

    total = paras[-1][0] if paras else 0

    for start, text in paras:
        # 每整分钟加一个小标题作为导航锚点
        m = int(start // 60)
        if m > 0 and start - (m * 60) < 30 and (not lines or f"## {m:02d}:" not in lines[-3]):
            lines.append(f"\n## {_ts(start)}\n\n")

        lines.append(f"`{_ts(start)}` {text}\n\n")

    lines.append(f"\n---\n*全长约 {int(total//60)} 分钟，共 {len(paras)} 段*\n")
    return "".join(lines)


def process(srt_path: Path):
    segs  = parse_srt(str(srt_path))
    if not segs:
        print(f"⚠️  无法解析：{srt_path}")
        return

    # 标题：去掉文件名末尾的时间戳 _YYYYMMDD_HHMM
    stem  = re.sub(r"_\d{8}_\d{4}$", "", srt_path.stem)
    title = stem.lstrip("_").replace("_", " ").strip()
    url   = ""

    paras   = build_paragraphs(segs)
    content = to_markdown(title, url, paras)

    out = srt_path.with_name(stem + ".md")
    out.write_text(content, encoding="utf-8")
    print(f"✅ {out}  ({len(paras)} 段)")


def main():
    if len(sys.argv) >= 2:
        srts = [Path(sys.argv[1])]
    else:
        srts = sorted(Path("transcripts").glob("*.srt"))
        if not srts:
            print("没有找到 .srt 文件，用法：python reformat_transcript.py transcripts/xxx.srt")
            sys.exit(0)

    for srt in srts:
        process(srt)


if __name__ == "__main__":
    main()
