"""
YouTube 会员视频 Transcript 获取工具
=====================================
适用场景：无自动字幕的会员视频
流程：yt-dlp 下载音频 → OpenAI Whisper 本地转录 → 输出文字稿

依赖安装：
    pip install yt-dlp openai-whisper
    # macOS 还需要 ffmpeg：
    brew install ffmpeg

使用方法：
    python get_youtube_transcript.py <YouTube链接> [--lang zh]
    python get_youtube_transcript.py https://youtu.be/xxxx
    python get_youtube_transcript.py https://www.youtube.com/watch?v=xxxx --model large
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 步骤1：用 yt-dlp 从浏览器提取 cookies 并下载音频
# ─────────────────────────────────────────────
def download_audio(url: str, output_dir: str = "transcripts", browser: str = "chrome", profile: str = "") -> str:
    """
    下载会员视频音频

    注意：需要在 Chrome/Safari 里登录 YouTube 账号且有会员资格
    yt-dlp 会自动读取浏览器 cookies，无需手动导出
    """
    Path(output_dir).mkdir(exist_ok=True)

    # 先获取视频信息（标题）
    yt_dlp_cmd = _get_yt_dlp_command()
    yt_dlp_runtime_args = _get_yt_dlp_runtime_args()
    browser_cookie_spec = browser if not profile else f"{browser}:{profile}"
    info_cmd = yt_dlp_cmd + yt_dlp_runtime_args + [
        "--cookies-from-browser", browser_cookie_spec,
        "--dump-json",
        "--no-playlist",
        url
    ]
    print(f"📡 获取视频信息（使用 {browser_cookie_spec} 的登录状态）...")
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        title = info.get("title", "video")
        video_id = info.get("id", "unknown")
        upload_date = info.get("upload_date", "")
        print(f"   标题：{title}")
        print(f"   ID：{video_id}  上传日期：{upload_date}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取视频信息失败：{e.stderr}")
        print("   请确认：1) yt-dlp 已安装  2) 已在 Chrome 登录 YouTube 会员账号")
        sys.exit(1)
    except json.JSONDecodeError:
        title = "video"
        video_id = "unknown"

    # 安全化文件名
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:60]
    audio_path = os.path.join(output_dir, f"{safe_title}.mp3")

    # 下载最佳音频（转为 mp3）
    download_cmd = yt_dlp_cmd + yt_dlp_runtime_args + [
        "--cookies-from-browser", browser_cookie_spec,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",           # 最高质量
        "--no-playlist",
        "-o", audio_path,
        url
    ]

    print(f"\n⬇️  下载音频到：{audio_path}")
    try:
        subprocess.run(download_cmd, check=True)
        print("   ✅ 音频下载完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败：{e}")
        print("   如果是会员视频，确认已购买会员且在浏览器中保持登录状态")
        sys.exit(1)

    return audio_path, safe_title


def _get_yt_dlp_command() -> list:
    """返回可用的 yt-dlp 命令：优先可执行文件，否则回退到 `python -m yt_dlp`。"""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def _get_yt_dlp_runtime_args() -> list:
    """自动开启 EJS 运行时。"""
    if shutil.which("node"):
        return ["--js-runtimes", "node"]
    return []


# ─────────────────────────────────────────────
# 步骤2：Whisper 本地转录
# ─────────────────────────────────────────────
def transcribe_audio(
    audio_path: str,
    model_size: str = "large",
    language: str = "zh",
    output_dir: str = "transcripts"
) -> str:
    """
    使用 OpenAI Whisper 本地转录

    模型大小对比（中文）：
    - tiny:   最快，质量一般，约1GB显存
    - base:   较快，质量尚可
    - small:  均衡
    - medium: 推荐，中文效果好
    - large:  最好，但较慢，约10GB显存（M1/M2 Mac 可以跑）

    对于财经内容（有专业术语），建议使用 large 或 medium
    """
    print(f"\n🎙️  开始转录（模型：{model_size}，语言：{language}）")
    print("   首次运行会下载模型，large 约 2.9GB，请耐心等待...")

    try:
        import whisper
    except ImportError:
        print("❌ 未找到 whisper，请先安装：pip install openai-whisper")
        sys.exit(1)

    print("   加载模型...")
    model = whisper.load_model(model_size)

    print(f"   转录中：{audio_path}")
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False,
        # 以下参数优化财经内容转录质量
        temperature=0.0,         # 降低随机性，提高准确度
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        word_timestamps=True,    # 输出词级时间戳
    )

    return result


def save_transcript(result: dict, title: str, output_dir: str = "transcripts") -> str:
    """保存转录结果为多种格式"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"{output_dir}/{title}_{timestamp}"

    # 1. 纯文字稿（最常用）
    txt_path = f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\n   ✅ 纯文字稿：{txt_path}")

    # 2. 带时间戳的 SRT 字幕
    srt_path = f"{base_name}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            start = _seconds_to_srt_time(seg["start"])
            end   = _seconds_to_srt_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
    print(f"   ✅ SRT字幕：{srt_path}")

    # 3. 带时间戳的结构化文本（便于参考时间节点）
    timestamped_path = f"{base_name}_timestamped.txt"
    with open(timestamped_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            f.write(f"[{mins:02d}:{secs:02d}] {seg['text'].strip()}\n")
    print(f"   ✅ 带时间戳：{timestamped_path}")

    return txt_path


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ─────────────────────────────────────────────
# 步骤3：用 Claude API 提炼方法论（可选）
# ─────────────────────────────────────────────
def extract_methodology(transcript_path: str, api_key: str = None) -> str:
    """
    用 Claude API 从 transcript 中提炼投资方法论
    供后续注入 trading agent 的 system prompt

    需要设置环境变量：export ANTHROPIC_API_KEY=your_key
    或传入 api_key 参数
    """
    try:
        import anthropic
    except ImportError:
        print("❌ 未安装 anthropic SDK：pip install anthropic")
        return ""

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("⚠️  未设置 ANTHROPIC_API_KEY，跳过方法论提炼步骤")
        return ""

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # 如果内容太长，截取前2万字
    if len(transcript) > 20000:
        transcript = transcript[:20000] + "\n...(截断)"

    client = anthropic.Anthropic(api_key=key)

    prompt = f"""以下是一段财经视频的转录文字稿。请从中提炼出：

1. **选股逻辑与标准**：他如何筛选标的？关注哪些指标？
2. **入场条件**：什么情况下建仓？技术面和基本面分别看什么？
3. **止损原则**：如何设置止损？跌破什么位置离场？
4. **仓位管理**：如何分批建仓？满仓还是轻仓？
5. **市场判断框架**：如何判断大盘/板块方向？
6. **常用工具/指标**：提到了哪些具体的分析工具？
7. **其他核心观点**：值得记录的投资原则

输出格式：结构化要点，每条尽量引用原文关键词，方便后续作为 AI agent 的 system prompt 使用。

---
文字稿内容：
{transcript}
"""

    print("\n🤖 提炼投资方法论中...")
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    methodology = response.content[0].text

    # 保存方法论
    method_path = transcript_path.replace(".txt", "_methodology.txt")
    with open(method_path, "w", encoding="utf-8") as f:
        f.write("# 投资方法论提炼\n\n")
        f.write(methodology)
    print(f"   ✅ 方法论已保存：{method_path}")
    print(f"\n{methodology}")

    return methodology


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YouTube会员视频 → Transcript → 投资方法论提炼",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基本用法（默认 large 模型，中文）
  python get_youtube_transcript.py https://youtu.be/xxxx

  # 用 medium 模型（速度更快）
  python get_youtube_transcript.py https://youtu.be/xxxx --model medium

  # Safari 浏览器登录的账号
  python get_youtube_transcript.py https://youtu.be/xxxx --browser safari

  # 转录后自动提炼方法论（需要 ANTHROPIC_API_KEY）
  python get_youtube_transcript.py https://youtu.be/xxxx --extract-method

  # 批量处理多个视频（从文件读取链接列表）
  python get_youtube_transcript.py --batch urls.txt
        """
    )
    parser.add_argument("url", nargs="?", help="YouTube 视频链接")
    parser.add_argument("--model",   default="large", choices=["tiny","base","small","medium","large"],
                        help="Whisper 模型大小（默认 large，中文质量最好）")
    parser.add_argument("--lang",    default="zh", help="语言代码（默认 zh=中文）")
    parser.add_argument("--browser", default="chrome", choices=["chrome","safari","firefox","edge"],
                        help="从哪个浏览器读取 cookies（默认 chrome）")
    parser.add_argument("--profile", default="",
                        help="浏览器 Profile 名称（如 'Default' 或 'Profile 1'）")
    parser.add_argument("--output",  default="transcripts", help="输出目录（默认 transcripts/）")
    parser.add_argument("--extract-method", action="store_true",
                        help="转录完成后自动用 Claude API 提炼方法论")
    parser.add_argument("--batch",   metavar="FILE",
                        help="批量处理：从文件读取 YouTube 链接（每行一个）")

    args = parser.parse_args()

    # 批量模式
    if args.batch:
        with open(args.batch, "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"📋 批量处理 {len(urls)} 个视频")
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*50}")
            print(f"[{i}/{len(urls)}] {url}")
            print(f"{'='*50}")
            process_single(url, args)
        return

    if not args.url:
        parser.print_help()
        print("\n💡 快速开始：\n  python get_youtube_transcript.py https://youtu.be/xxxx")
        sys.exit(0)

    process_single(args.url, args)


def process_single(url: str, args) -> None:
    print(f"\n{'='*55}")
    print(f"  YouTube Transcript 提取工具")
    print(f"{'='*55}")

    # 下载音频
    audio_path, title = download_audio(url, args.output, args.browser, args.profile)

    # 转录
    result = transcribe_audio(audio_path, args.model, args.lang, args.output)

    # 保存
    txt_path = save_transcript(result, title, args.output)

    # 提炼方法论
    if args.extract_method:
        extract_methodology(txt_path)

    print(f"\n✅ 全部完成！文字稿保存在：{args.output}/")
    print("   下一步：将 *_methodology.txt 的内容作为 trading agent 的 system prompt")


if __name__ == "__main__":
    main()
