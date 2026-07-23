"""
频道批量下载 + 转录工具
========================
用法：
    # 先列出频道所有视频（不下载，用于确认）
    python batch_channel_transcribe.py --list-only <频道URL或@handle>

    # 批量下载并转录
    python batch_channel_transcribe.py <频道URL或@handle>

    # 只转录已下载的 mp3（跳过 yt-dlp）
    python batch_channel_transcribe.py --transcribe-only

示例：
    python batch_channel_transcribe.py https://www.youtube.com/@huanqiucaijing
    python batch_channel_transcribe.py --list-only https://www.youtube.com/@huanqiucaijing
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

TRANSCRIPTS_DIR = "transcripts"
PROGRESS_FILE   = "transcripts/.progress.json"  # 记录已完成的视频，避免重复下载


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _yt_dlp_cmd():
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]

def _yt_dlp_extra():
    """node 存在时启用 EJS 运行时，解决会员视频格式问题"""
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    if Path(node).exists():
        return ["--js-runtimes", f"node:{node}"]
    return []

def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 步骤1：获取频道视频列表
# ──────────────────────────────────────────────

def _cookie_args(browser: str, profile: str, cookies_file: str) -> list[str]:
    """优先用 cookies.txt 文件，否则回落 --cookies-from-browser"""
    if cookies_file:
        return ["--cookies", cookies_file]
    browser_spec = browser if not profile else f"{browser}:{profile}"
    return ["--cookies-from-browser", browser_spec]


def fetch_channel_videos(channel_url: str, browser: str = "chrome", profile: str = "",
                         cookies_file: str = "") -> list[dict]:
    """
    返回频道所有视频的 [{id, title, url, upload_date}] 列表
    会员视频也会被列出（但下载需要登录）
    """
    cmd = _yt_dlp_cmd() + _yt_dlp_extra() + _cookie_args(browser, profile, cookies_file) + [
        "--flat-playlist",          # 只获取列表，不下载
        "--dump-json",
        "--no-warnings",
        channel_url,
    ]

    print(f"📡 获取频道视频列表：{channel_url}")
    auth_desc = f"cookies.txt={cookies_file}" if cookies_file else f"{browser}{':'+profile if profile else ''}"
    print(f"   使用 {auth_desc} 的登录状态（会员视频需要登录）...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取列表失败：{e.stderr[-500:]}")
        print("   请确认：1) 已在 Chrome 登录会员账号  2) yt-dlp 版本最新")
        sys.exit(1)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            vid_id = info.get("id", "")
            title  = info.get("title", "")
            if vid_id:
                videos.append({
                    "id":          vid_id,
                    "title":       title,
                    "url":         f"https://www.youtube.com/watch?v={vid_id}",
                    "upload_date": info.get("upload_date", ""),
                })
        except json.JSONDecodeError:
            continue

    print(f"   ✅ 共找到 {len(videos)} 个视频")
    return videos


# ──────────────────────────────────────────────
# 步骤2：下载单个视频的音频
# ──────────────────────────────────────────────

def download_audio(video: dict, browser: str = "chrome", profile: str = "",
                   cookies_file: str = "") -> str | None:
    """下载音频到 transcripts/，返回 mp3 路径；失败返回 None

    反反爬：
    - 下载前随机 sleep 15-45s 降低请求频率
    - 多 player_client 兜底 (web/ios/android/mweb)
    - 遇到 'Join this channel' 伪 403 自动等 90-150s 重试 2 次
    """
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video["title"])[:60]
    audio_path = os.path.join(TRANSCRIPTS_DIR, f"{safe_title}.mp3")

    if Path(audio_path).exists():
        print(f"   ⏭️  已存在，跳过下载：{safe_title[:40]}...")
        return audio_path

    base_cmd = _yt_dlp_cmd() + _yt_dlp_extra() + _cookie_args(browser, profile, cookies_file) + [
        "--extractor-args", "youtube:player_client=web,ios,android,mweb,web_safari",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "--sleep-interval", "20",
        "--max-sleep-interval", "60",
        "-o", audio_path,
        video["url"],
    ]

    print(f"   ⬇️  下载：{video['title'][:50]}...")

    MAX_TRIES = 3
    for attempt in range(1, MAX_TRIES + 1):
        # 非首次尝试前的等待已在下方处理；首次也抖一下避免顶速连发
        if attempt == 1:
            pre_wait = random.uniform(5, 15)
        else:
            pre_wait = 0
        if pre_wait > 0:
            time.sleep(pre_wait)

        try:
            subprocess.run(base_cmd, check=True, capture_output=True)
            print(f"   ✅ 下载完成" + (f"（第 {attempt} 次）" if attempt > 1 else ""))
            return audio_path
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            throttled = "Join this channel" in stderr or "members-only" in stderr
            if throttled and attempt < MAX_TRIES:
                wait = random.uniform(90, 150) * attempt
                print(f"   ⏳ 疑似限速（第 {attempt}/{MAX_TRIES} 次），等 {wait:.0f}s 重试...")
                time.sleep(wait)
                continue
            print(f"   ❌ 下载失败（第 {attempt} 次终止）：{stderr[-200:]}")
            return None
    return None


# ──────────────────────────────────────────────
# 步骤3：Whisper 转录
# ──────────────────────────────────────────────

_whisper_model_cache = None

def get_whisper_model(model_size: str, device: str = "cpu"):
    global _whisper_model_cache
    if _whisper_model_cache is None:
        try:
            import whisper
        except ImportError:
            print("❌ 请安装 whisper：pip install openai-whisper")
            sys.exit(1)
        print(f"\n🔧 加载 Whisper {model_size} 模型（device={device}）...")
        _whisper_model_cache = whisper.load_model(model_size, device=device)
        print("   ✅ 模型已加载")
    return _whisper_model_cache


def transcribe_audio(audio_path: str, model_size: str = "large", language: str = "zh",
                     device: str = "cpu") -> dict | None:
    """转录音频，返回 whisper result dict；失败返回 None"""
    model = get_whisper_model(model_size, device)
    print(f"   🎙️  转录中：{Path(audio_path).name[:50]}...")
    # MPS 不支持 float64，word_timestamps (DTW) 会走 float64 → 关掉
    # SRT 和 timestamped.txt 只用 segment 级时间戳，word 级时间不缺
    use_word_ts = (device == "cpu")
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            word_timestamps=use_word_ts,
            fp16=(device != "cpu"),
        )
        print(f"   ✅ 转录完成（{len(result['text'])} 字）")
        return result
    except Exception as e:
        print(f"   ❌ 转录失败：{e}")
        return None


def save_transcript(result: dict, video: dict) -> str:
    """保存 txt + srt + timestamped.txt，返回 txt 路径"""
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video["title"])[:60]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = os.path.join(TRANSCRIPTS_DIR, f"{safe_title}_{ts}")

    # 纯文字
    txt_path = f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# {video['title']}\n")
        f.write(f"# URL: {video['url']}\n")
        f.write(f"# 转录时间: {ts}\n\n")
        f.write(result["text"])

    # SRT
    with open(f"{base}.srt", "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text'].strip()}\n\n")

    # 可读版：合并到句末标点再换行，每段开头带时间戳
    with open(f"{base}_timestamped.txt", "w", encoding="utf-8") as f:
        f.write(f"# {video['title']}\n")
        f.write(f"# {video['url']}\n\n")
        _write_readable(f, result["segments"])


    return txt_path


def _write_readable(f, segments: list):
    """
    将 Whisper segments 合并成自然段落写入文件。
    规则：
    - 遇到句末标点（。！？…）或沉默超过1.5秒时结束当前段落
    - 每段开头写时间戳，结尾空一行
    """
    SENTENCE_END = set("。！？…")
    GAP_THRESHOLD = 1.5  # 秒

    para_start_secs = None
    para_text = ""

    def flush(f, start, text):
        if not text.strip():
            return
        m, s = int(start // 60), int(start % 60)
        f.write(f"[{m:02d}:{s:02d}] {text.strip()}\n\n")

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        if para_start_secs is None:
            para_start_secs = seg["start"]

        para_text += text

        # 检测段落结束：句末标点 或 与下一段之间有较长沉默
        ends_sentence = para_text and para_text[-1] in SENTENCE_END
        long_gap = False
        if i + 1 < len(segments):
            long_gap = (segments[i + 1]["start"] - seg["end"]) > GAP_THRESHOLD

        if ends_sentence or long_gap:
            flush(f, para_start_secs, para_text)
            para_start_secs = None
            para_text = ""

    # 收尾
    if para_text:
        flush(f, para_start_secs, para_text)


def _srt_time(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ──────────────────────────────────────────────
# 主流程：批量处理频道
# ──────────────────────────────────────────────

def run_batch(channel_url: str, browser: str, profile: str,
              model_size: str, language: str, skip_download: bool,
              list_only: bool, limit: int, device: str = "cpu",
              cookies_file: str = ""):

    Path(TRANSCRIPTS_DIR).mkdir(exist_ok=True)
    progress = load_progress()

    # 1. 获取视频列表
    videos = fetch_channel_videos(channel_url, browser, profile, cookies_file)

    if limit > 0:
        videos = videos[:limit]
        print(f"   (--limit {limit}：只处理前 {limit} 个)")

    if list_only:
        print(f"\n{'='*60}")
        print(f"{'#':>4}  {'上传日期':10}  标题")
        print(f"{'='*60}")
        for i, v in enumerate(videos, 1):
            done = "✅" if v["id"] in progress else "  "
            print(f"{i:>4} {done}  {v['upload_date']}  {v['title'][:55]}")
        already = sum(1 for v in videos if v["id"] in progress)
        print(f"\n合计 {len(videos)} 个，已转录 {already} 个，待处理 {len(videos)-already} 个")
        return

    # 2. 逐个处理
    total = len(videos)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, video in enumerate(videos, 1):
        vid_id = video["id"]
        print(f"\n{'─'*60}")
        print(f"[{i}/{total}] {video['title'][:55]}")

        # 已转录过则跳过
        if vid_id in progress and progress[vid_id].get("transcribed"):
            print(f"   ⏭️  已完成，跳过")
            skip_count += 1
            continue

        # 下载
        if not skip_download:
            audio_path = download_audio(video, browser, profile, cookies_file)
            if audio_path is None:
                fail_count += 1
                progress[vid_id] = {"title": video["title"], "transcribed": False, "error": "download_failed"}
                save_progress(progress)
                continue
        else:
            # 只转录模式：找已存在的 mp3
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video["title"])[:60]
            audio_path = os.path.join(TRANSCRIPTS_DIR, f"{safe_title}.mp3")
            if not Path(audio_path).exists():
                print(f"   ⏭️  没有找到 mp3，跳过（用不带 --transcribe-only 的命令先下载）")
                skip_count += 1
                continue

        # 转录
        result = transcribe_audio(audio_path, model_size, language, device)
        if result is None:
            fail_count += 1
            progress[vid_id] = {"title": video["title"], "transcribed": False, "error": "transcribe_failed"}
            save_progress(progress)
            continue

        # 保存
        txt_path = save_transcript(result, video)
        print(f"   💾 已保存：{txt_path}")

        progress[vid_id] = {
            "title":       video["title"],
            "url":         video["url"],
            "upload_date": video["upload_date"],
            "transcribed": True,
            "txt_path":    txt_path,
            "done_at":     datetime.now().isoformat(),
        }
        save_progress(progress)
        ok_count += 1

        # 礼貌性间隔，避免频繁请求
        if not skip_download and i < total:
            time.sleep(2)

    # 汇总
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"  ✅ 成功：{ok_count}  ⏭️ 跳过：{skip_count}  ❌ 失败：{fail_count}")
    print(f"  文字稿保存在：{TRANSCRIPTS_DIR}/")
    if fail_count > 0:
        print(f"  失败记录见：{PROGRESS_FILE}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="批量下载 YouTube 频道视频并用 Whisper 转录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("channel_url", nargs="?",
                        help="频道 URL，如 https://www.youtube.com/@handle 或频道 /videos 页")
    parser.add_argument("--browser",  default="chrome",
                        choices=["chrome","safari","firefox","edge"])
    parser.add_argument("--profile",  default="", help="浏览器 Profile 名称")
    parser.add_argument("--model",    default="large",
                        choices=["tiny","base","small","medium","large"],
                        help="Whisper 模型（默认 large）")
    parser.add_argument("--lang",     default="zh")
    parser.add_argument("--list-only",       action="store_true",
                        help="只列出视频列表，不下载不转录")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="跳过下载，只转录 transcripts/ 下已有的 mp3")
    parser.add_argument("--limit",    type=int, default=0,
                        help="只处理前 N 个视频（0=全部）")
    parser.add_argument("--device",   default="cpu",
                        choices=["cpu","mps","cuda"],
                        help="Whisper 推理设备（默认 cpu；Apple Silicon 可用 mps）")
    parser.add_argument("--cookies-file", default="",
                        help="cookies.txt 路径；若提供则优先于 --browser")

    args = parser.parse_args()

    if not args.channel_url:
        parser.print_help()
        print("\n💡 示例：")
        print("  python batch_channel_transcribe.py https://www.youtube.com/@huanqiucaijing --list-only")
        print("  python batch_channel_transcribe.py https://www.youtube.com/@huanqiucaijing")
        sys.exit(0)

    run_batch(
        channel_url=args.channel_url,
        browser=args.browser,
        profile=args.profile,
        model_size=args.model,
        language=args.lang,
        skip_download=args.transcribe_only,
        list_only=args.list_only,
        limit=args.limit,
        device=args.device,
        cookies_file=args.cookies_file,
    )


if __name__ == "__main__":
    main()
