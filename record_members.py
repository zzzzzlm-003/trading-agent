"""
浏览器倍速录制 YouTube 会员视频
================================
绕过 yt-dlp 的会员视频 attestation（老期视频拿不到 playback stream 时走这个）

流程：
  Playwright 启动 Chromium（注入 cookies.txt 会员登录）
  → 打开视频，设 playbackRate=2.0，等 video `ended`
  → ffmpeg 从 BlackHole 2ch 录音
  → 录完用 ffmpeg atempo=0.5 还原到 1x
  → 复用 batch_channel_transcribe 的 Whisper MPS 转录 + 保存

用法：
  # 从一个 playlist 拉全部视频，剔除 progress.json 里已 transcribed 的，录剩下的
  python record_members.py --playlist <playlist_url>

  # 录特定 video IDs（调试用）
  python record_members.py --ids abc123,def456

  # 只录前 N 个
  python record_members.py --playlist <url> --limit 3

  # 不录音，仅打开视频诊断（看能否播放）
  python record_members.py --playlist <url> --dry-run

注意：
  - 脚本运行期间会把系统默认输出切到 BlackHole 2ch，你听不到声音，结束时恢复
  - caffeinate -dis 防止休眠；MacBook 盖子必须开着
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_channel_transcribe as bct

COOKIES_FILE     = os.path.expanduser("~/youtube_cookies.txt")
REC_RAW_DIR      = "transcripts/_rec_raw"
PLAYBACK_RATE    = 2.0
BLACKHOLE_NAME   = "BlackHole 2ch"
PAGE_LOAD_TIMEOUT_MS = 60_000
RECORD_POLL_SEC  = 10          # JS 端轮询间隔
RECORD_MAX_GRACE = 30          # 结束后多录几秒尾巴

# 用真实 Chrome（含 Widevine）+ 专用 user-data-dir，独立于你日常的 Chrome
CHROME_BINARY    = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_PROFILE   = os.path.expanduser("~/chrome-ytrec-profile")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────
# cookies
# ──────────────────────────────────────────────
def parse_netscape_cookies(path: str) -> list[dict]:
    cookies = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, include_sub, path_, secure, expiry, name, value = parts[:7]
        c = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path_,
            "secure": secure.upper() == "TRUE",
            "httpOnly": False,
            "sameSite": "Lax",
        }
        if expiry.isdigit() and int(expiry) > 0:
            c["expires"] = int(expiry)
        cookies.append(c)
    return cookies


# ──────────────────────────────────────────────
# audio 路由 & 录音
# ──────────────────────────────────────────────
def get_current_output() -> str:
    try:
        return subprocess.check_output(
            ["SwitchAudioSource", "-c"], text=True).strip()
    except Exception:
        return ""


def set_output(device: str):
    subprocess.run(["SwitchAudioSource", "-s", device],
                   check=True, stdout=subprocess.DEVNULL)


def start_recording(out_mp3: str) -> subprocess.Popen:
    """从 BlackHole 2ch 录 mp3，返回 Popen（stdin 保持打开以便 q 退出）"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-i", f":{BLACKHOLE_NAME}",
        "-ac", "2", "-ar", "44100",
        "-c:a", "libmp3lame", "-q:a", "4",
        out_mp3,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_recording(proc: subprocess.Popen):
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write(b"q")
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def atempo_restore(src_mp3: str, dst_mp3: str, factor: float = 2.0):
    """用 ffmpeg atempo 把 k×x 录音拉回 1x。

    单次 atempo=0.5 对语音有明显畸变（Whisper VAD 几乎失灵 → 整段合为 1 段），
    改用两级 atempo=sqrt(1/factor) 级联，音质大幅改善。
    """
    import math
    inv = 1.0 / factor
    # 两级等效 atempo=inv，单级范围 [0.5, 100]，用 sqrt 保证落在合法区间
    single = math.sqrt(inv)
    cmd = ["ffmpeg", "-y", "-i", src_mp3,
           "-af", f"atempo={single:.9f},atempo={single:.9f}",
           "-c:a", "libmp3lame", "-q:a", "4",
           dst_mp3]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ──────────────────────────────────────────────
# Playwright：打开视频 + 2x 播放 + 等结束
# ──────────────────────────────────────────────
JS_SET_PLAYBACK = r"""
({rate}) => new Promise((resolve, reject) => {
    let tries = 0;
    const tick = () => {
        const v = document.querySelector('video');
        if (!v || isNaN(v.duration) || v.duration === 0) {
            if (++tries > 60) return reject('video not ready');
            return setTimeout(tick, 500);
        }
        v.muted = false;
        v.volume = 1.0;
        v.playbackRate = rate;
        // 关键：YouTube 会从上次观看位置恢复，强制 seek 到 0
        try { v.currentTime = 0; } catch(e) {}
        // 触发 user gesture 替代：直接 click player 元素
        const player = document.querySelector('.html5-video-player') || v;
        try { player.click(); } catch(e) {}
        v.play().catch(() => {});
        resolve({duration: v.duration, rate: v.playbackRate, startAt: v.currentTime});
    };
    tick();
});
"""

# 关键改进：每次轮询都重新 querySelector('video')，避免 YouTube 重建 video
# 时抓着失效的旧引用导致 currentTime 永远停在 0。
JS_WAIT_END = r"""
({rate, pollMs}) => new Promise((resolve) => {
    let last = -1, stuck = 0, totalStuck = 0, ticks = 0;
    const iv = setInterval(() => {
        ticks++;
        const v = document.querySelector('video');  // 每次重查！
        if (!v) {
            totalStuck++;
            if (totalStuck > 30) { clearInterval(iv); return resolve({reason:'no-video'}); }
            return;
        }
        const dur = v.duration;
        const ct = v.currentTime;
        // ended 条件
        if (v.ended || (dur && !isNaN(dur) && ct >= dur - 0.5)) {
            clearInterval(iv);
            return resolve({reason:'ended', currentTime: ct, duration: dur, ticks});
        }
        // 保持 2x + 不 pause
        if (v.playbackRate !== rate) v.playbackRate = rate;
        if (v.paused) v.play().catch(() => {});
        // 卡住检测（基于 currentTime 不前进）
        if (ct === last) {
            stuck++; totalStuck++;
            if (stuck >= 3) {
                // 尝试抢救：重新 play + 再 setPlaybackRate
                v.play().catch(() => {});
                v.playbackRate = rate;
                stuck = 0;
            }
        } else {
            stuck = 0;
            totalStuck = 0;  // 有进度就重置全局卡计数
        }
        last = ct;
        // 60 轮（10 分钟）毫无进度才放弃
        if (totalStuck > 60) {
            clearInterval(iv);
            resolve({reason:'timeout', currentTime: ct, duration: dur, ticks});
        }
    }, pollMs);
});
"""


def check_blocked(page) -> str | None:
    """页面上是否有"加入此频道"等拦截状态。返回描述或 None"""
    try:
        msg = page.evaluate(r"""
            () => {
                const t = document.body.innerText || '';
                const needles = [
                    '加入此频道', '加入频道',
                    'Join this channel',
                    'members-only',
                    'This video is available',
                    'private video',
                    'unavailable',
                ];
                for (const n of needles) if (t.includes(n)) return n;
                return null;
            }
        """)
        return msg
    except Exception:
        return None


def record_and_transcribe(ctx, video: dict, args, progress: dict):
    vid = video["id"]
    title = video["title"]
    raw_path = os.path.join(REC_RAW_DIR, f"{vid}.mp3")

    log(f"▶ 打开 {vid} {title[:45]}")
    page = ctx.new_page()
    rec = None
    try:
        page.goto(video["url"], wait_until="domcontentloaded",
                  timeout=PAGE_LOAD_TIMEOUT_MS)
        page.wait_for_selector("video", timeout=30_000)
        time.sleep(1.5)  # 让页面 JS 初始化

        blocked = check_blocked(page)
        # 只有在同时没有 video 元素的情况下才算真的被拦截
        # 会员视频页面也会包含"加入此频道"按钮文本，所以不能只凭文本判断
        has_video = page.evaluate("!!document.querySelector('video')")
        if blocked and not has_video:
            log(f"  ⛔ 页面似乎被拦截: '{blocked}'，跳过")
            progress[vid] = {"title": title, "url": video["url"],
                             "transcribed": False, "error": f"blocked:{blocked}"}
            bct.save_progress(progress)
            return

        # 主动 user gesture：鼠标移到页面中央再点一下，让 autoplay 被批准
        try:
            page.mouse.move(640, 390)
            page.mouse.click(640, 390)
            time.sleep(0.3)
        except Exception:
            pass

        try:
            meta = page.evaluate(JS_SET_PLAYBACK, {"rate": PLAYBACK_RATE})
        except Exception as e:
            log(f"  ❌ 无法启动播放: {e}")
            try:
                page.screenshot(path=os.path.join(REC_RAW_DIR, f"{vid}_init_fail.png"))
            except Exception:
                pass
            progress[vid] = {"title": title, "url": video["url"],
                             "transcribed": False, "error": "playback_init_failed"}
            bct.save_progress(progress)
            return

        duration = float(meta.get("duration", 0))
        log(f"  ⏱  原长 {int(duration)}s，预计录 {int(duration/PLAYBACK_RATE)}s")

        if args.dry_run:
            log("  (dry-run) 不录不转，跳过")
            time.sleep(3)
            return

        rec = start_recording(raw_path)
        time.sleep(0.5)  # 让 ffmpeg 打开设备

        end_info = page.evaluate(
            JS_WAIT_END,
            {"rate": PLAYBACK_RATE, "pollMs": RECORD_POLL_SEC * 1000},
        )
        reason = end_info.get("reason")
        ct = end_info.get("currentTime", 0) or 0
        dur = end_info.get("duration", 0) or 0
        log(f"  ⏹  结束（{reason}），currentTime={ct:.1f}/{dur:.1f}，ticks={end_info.get('ticks','?')}")
        if reason in ("timeout", "no-video"):
            try:
                page.screenshot(path=os.path.join(REC_RAW_DIR, f"{vid}_final.png"))
            except Exception:
                pass
            # 卡住就别继续转录浪费时间
            stop_recording(rec); rec = None
            progress[vid] = {"title": title, "url": video["url"],
                             "transcribed": False, "error": f"playback_stuck:{reason}"}
            bct.save_progress(progress)
            return

        # 多录几秒，保证尾巴完整
        time.sleep(RECORD_MAX_GRACE)
        stop_recording(rec)
        rec = None

        if not Path(raw_path).exists() or Path(raw_path).stat().st_size < 10_000:
            log(f"  ❌ 录音文件太小 / 不存在，跳过")
            progress[vid] = {"title": title, "url": video["url"],
                             "transcribed": False, "error": "record_empty"}
            bct.save_progress(progress)
            return

        # atempo 还原
        safe = "".join(c if c.isalnum() or c in " -_" else "_"
                       for c in title)[:60]
        final_mp3 = os.path.join(bct.TRANSCRIPTS_DIR, f"{safe}.mp3")
        if Path(final_mp3).exists():
            Path(final_mp3).unlink()
        log(f"  🎚 atempo=0.5 还原 1x → {final_mp3}")
        atempo_restore(raw_path, final_mp3, factor=PLAYBACK_RATE)

        # Whisper 转录
        result = bct.transcribe_audio(final_mp3, args.model, "zh", args.device)
        if result is None:
            progress[vid] = {"title": title, "url": video["url"],
                             "transcribed": False,
                             "error": "transcribe_failed_after_record"}
        else:
            txt_path = bct.save_transcript(result, video)
            progress[vid] = {
                "title": title, "url": video["url"],
                "upload_date": video.get("upload_date", ""),
                "transcribed": True, "txt_path": txt_path,
                "done_at": datetime.now().isoformat(),
                "source": "browser_record_2x",
            }
            log(f"  ✅ 保存 {txt_path}")
        bct.save_progress(progress)

        # raw 2x 文件保留，便于万一要重转；想节省空间可删
        # Path(raw_path).unlink(missing_ok=True)

    except Exception as e:
        log(f"  ❌ 异常: {e}")
        progress[vid] = {"title": title, "url": video["url"],
                         "transcribed": False, "error": f"exception:{e}"}
        bct.save_progress(progress)
    finally:
        if rec is not None:
            stop_recording(rec)
        try:
            page.close()
        except Exception:
            pass


# ──────────────────────────────────────────────
# 目标视频列表
# ──────────────────────────────────────────────
def targets_from_playlist(url: str, progress: dict) -> list[dict]:
    videos = bct.fetch_channel_videos(
        url, browser="chrome", profile="", cookies_file=COOKIES_FILE)
    done = {k for k, v in progress.items() if v.get("transcribed")}
    return [v for v in videos if v["id"] not in done]


def targets_from_ids(ids: list[str]) -> list[dict]:
    return [{"id": i, "title": i,
             "url": f"https://www.youtube.com/watch?v={i}",
             "upload_date": ""} for i in ids]


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", default="",
                    help="playlist URL；列出其全部视频并剔除已转录的")
    ap.add_argument("--ids", default="",
                    help="逗号分隔 video IDs（覆盖 playlist）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只录前 N 个（0=全部）")
    ap.add_argument("--model", default="large",
                    choices=["tiny", "base", "small", "medium", "large"])
    ap.add_argument("--device", default="mps",
                    choices=["cpu", "mps", "cuda"])
    ap.add_argument("--dry-run", action="store_true",
                    help="只打开视频不录不转录")
    ap.add_argument("--keep-audio", action="store_true",
                    help="保留系统原音频输出（默认切到 BlackHole 让你静音）")
    args = ap.parse_args()

    Path(REC_RAW_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(COOKIES_FILE).exists():
        log(f"❌ cookies 文件不存在: {COOKIES_FILE}")
        sys.exit(1)

    progress = bct.load_progress()

    if args.ids:
        videos = targets_from_ids([x.strip() for x in args.ids.split(",") if x.strip()])
    elif args.playlist:
        videos = targets_from_playlist(args.playlist, progress)
    else:
        ap.print_help()
        sys.exit(0)

    if args.limit > 0:
        videos = videos[:args.limit]

    if not videos:
        log("没有待录的视频")
        return

    log(f"目标 {len(videos)} 个视频，device={args.device}, model={args.model}")

    # 切音频 + 防休眠
    prev_output = "" if args.keep_audio else get_current_output()
    if not args.keep_audio:
        try:
            set_output(BLACKHOLE_NAME)
            log(f"🔇 系统输出 {prev_output} → {BLACKHOLE_NAME}")
        except Exception as e:
            log(f"⚠ 切音频失败（{e}），将继续但你会听到播放声")
            prev_output = ""
    caffeinate = subprocess.Popen(["caffeinate", "-dis"])

    # Playwright
    from playwright.sync_api import sync_playwright

    stop_requested = {"v": False}

    def on_sigint(sig, frame):
        log("收到 Ctrl+C，结束当前视频后退出...")
        stop_requested["v"] = True
    signal.signal(signal.SIGINT, on_sigint)

    try:
        with sync_playwright() as p:
            # 用真实 Chrome + 专用 profile（独立于日常 Chrome）
            # 第一次运行时浏览器弹窗里手动登录 YouTube，以后登录态会持久保存
            Path(CHROME_PROFILE).mkdir(exist_ok=True)
            log(f"🦊 启动 Chrome（profile: {CHROME_PROFILE}）")
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=CHROME_PROFILE,
                executable_path=CHROME_BINARY,
                headless=False,
                viewport={"width": 1280, "height": 780},
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            # 如果你首次未登录，脚本会在第一个视频打开时显示登录提示并等 120 秒
            first_page = ctx.pages[0] if ctx.pages else ctx.new_page()
            first_page.goto("https://www.youtube.com/", wait_until="domcontentloaded")
            time.sleep(2)
            signed_in = first_page.evaluate(
                "!!document.querySelector('ytd-topbar-menu-button-renderer button img, #avatar-btn img')"
            )
            if not signed_in:
                log("⚠ 检测到未登录 YouTube。请在打开的 Chrome 窗口里手动登录你的会员账号，然后按回车继续...")
                log("  （登录状态会保存在 ~/chrome-ytrec-profile，下次自动用）")
                try:
                    input("登录完成后按回车继续: ")
                except EOFError:
                    log("⚠ 非交互式会话，等 180 秒让你登录...")
                    time.sleep(180)
            else:
                log("✅ 已登录")
            first_page.close()

            for i, v in enumerate(videos, 1):
                if stop_requested["v"]:
                    break
                log(f"\n===== [{i}/{len(videos)}] {v['id']} =====")
                record_and_transcribe(ctx, v, args, progress)

            ctx.close()
    finally:
        try:
            caffeinate.terminate()
        except Exception:
            pass
        if prev_output and not args.keep_audio:
            try:
                set_output(prev_output)
                log(f"🔊 系统输出恢复 → {prev_output}")
            except Exception:
                pass
        log("全部结束")


if __name__ == "__main__":
    main()
