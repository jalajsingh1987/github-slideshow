#!/usr/bin/env python3
"""Render a short talking-character film from one persistent master still."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from gtts import gTTS
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets" / "characters-master.png"
OUT_DIR = ROOT / "out"
FPS = 24
WIDTH, HEIGHT = 1920, 1080
# iOS/Android HTML5 players reject 24 kHz mono AAC and High-profile H.264.
AUDIO_RATE = 44100

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Lower-face search boxes on the original painting (x0,y0,x1,y1).
PIP_FACE = (320, 360, 560, 520)
JUN_FACE = (1000, 370, 1260, 530)

# Mouth centers — replaced at render time by detect_smile_center().
PIP = {
    "mouth": (476, 438),
    "eyes": ((400, 372), (455, 370)),
    "color": (232, 176, 128),
    "lip": (176, 86, 78),
    "bubble": (255, 244, 214),
    "name": "Pip",
    "pitch": 1.12,
    "tld": "co.uk",
}
JUN = {
    "mouth": (1048, 436),
    "eyes": ((1095, 372), (1155, 370)),
    "color": (176, 112, 72),
    "lip": (132, 62, 58),
    "bubble": (232, 242, 255),
    "name": "Jun",
    "pitch": 0.94,
    "tld": "com",
}

LINES = [
    {"who": "pip", "text": "Hey. We stay ourselves the whole time."},
    {"who": "jun", "text": "Same hair. Same outfits. That's persistence."},
    {"who": "pip", "text": "And when we talk, only the mouth should move."},
    {"who": "jun", "text": "See? I'm still me."},
    {"who": "pip", "text": "Persistent characters. That's us."},
]

INTRO = 1.15
GAP = 0.42
OUTRO = 1.6

VISEME = {
    "rest": None,
    "M": "closed",
    "A": "open",
    "E": "wide",
    "O": "round",
    "U": "small",
    "F": "teeth",
}


def who_cfg(who: str) -> dict:
    return PIP if who == "pip" else JUN


def viseme_for_char(ch: str) -> str:
    c = ch.lower()
    if c in "mbp":
        return "M"
    if c in "fv":
        return "F"
    if c in "w":
        return "U"
    if c in "o":
        return "O"
    if c in "u":
        return "U"
    if c in "aei":
        return "A" if c == "a" else "E"
    if c in "y":
        return "E"
    if c in " .,'!?-":
        return "rest"
    return "M"


def viseme_track(text: str, duration: float, fps: int) -> list[str]:
    frames = max(1, int(round(duration * fps)))
    letters = [c for c in text if c.strip()] or ["."]
    track = []
    for i in range(frames):
        t = i / frames
        # Ease into and out of the line.
        if t < 0.06 or t > 0.94:
            track.append("rest")
            continue
        idx = min(len(letters) - 1, int((t - 0.06) / 0.88 * len(letters)))
        track.append(viseme_for_char(letters[idx]))
        # Brief closures between syllables.
        if letters[idx] == " " or (i % 5 == 0 and viseme_for_char(letters[idx]) != "M"):
            if i % 9 == 0:
                track[-1] = "M"
    return track


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def ffprobe_sample_rate(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(float(out))


def pitch_shift(src: Path, dst: Path, factor: float) -> None:
    """Change pitch without the chipmunk effect from a mismatched sample rate.

    gTTS writes ~24 kHz MP3. Pitch must use that source rate, then resample
    to 44.1 kHz stereo so phones will play the file.
    """
    src_rate = ffprobe_sample_rate(src)
    # rubberband keeps duration; asetrate+atempo is the fallback.
    rubber = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-filter:a",
        f"rubberband=pitch={factor}:channels=1,aformat=sample_rates={AUDIO_RATE}:channel_layouts=stereo",
        str(dst),
    ]
    try:
        subprocess.check_call(rubber, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except subprocess.CalledProcessError:
        pass
    tempo = min(2.0, max(0.5, 1.0 / factor))
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter:a",
            f"asetrate={src_rate}*{factor},aresample={AUDIO_RATE},atempo={tempo},"
            f"aformat=channel_layouts=stereo",
            str(dst),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_line_audio(text: str, cfg: dict, dest: Path) -> float:
    raw = dest.with_suffix(".raw.mp3")
    gTTS(text=text, lang="en", tld=cfg["tld"]).save(str(raw))
    pitch_shift(raw, dest, cfg["pitch"])
    raw.unlink(missing_ok=True)
    return ffprobe_duration(dest)


def crop_scene(master: Image.Image) -> Image.Image:
    w, h = master.size
    target_h = int(w * 9 / 16)
    top = max(0, (h - target_h) // 2 - 36)
    box = (0, top, w, min(h, top + target_h))
    return master.crop(box).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def map_pt(pt: tuple[int, int], master: Image.Image) -> tuple[int, int]:
    """Map a point on the original painting into the cropped 1920x1080 frame."""
    w, h = master.size
    target_h = int(w * 9 / 16)
    top = max(0, (h - target_h) // 2 - 36)
    x, y = pt
    nx = x / w * WIDTH
    ny = (y - top) / target_h * HEIGHT
    return int(nx), int(ny)


def detect_smile_center(rgb: np.ndarray, box: tuple[int, int, int, int]) -> dict | None:
    """Find the painted closed-mouth stroke in a lower-face box.

    These characters use a 1–3px dark smile on skin, not photographic lips, so
    generic face-mesh models miss them. For each row in the middle of the box
    we look for a dark run sandwiched by brighter skin above and below, merge
    nearby corners of the same smile, and take the centroid of the best stroke.
    """
    x0, y0, x1, y1 = box
    rgb = rgb.astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    merge_gap, min_w, max_w = 42, 18, 110
    best = None
    for y in range(y0 + int((y1 - y0) * 0.28), y0 + int((y1 - y0) * 0.72)):
        row = lum[y, x0:x1]
        med = np.median(lum[max(0, y - 5) : y + 6, x0:x1], axis=0)
        dark = np.clip(med - row, 0, None)
        above = lum[max(0, y - 8), x0:x1]
        below = lum[min(lum.shape[0] - 1, y + 8), x0:x1]
        sandwich = (above > row + 6) & (below > row + 6) & (np.abs(above - below) < 35)
        score = dark * sandwich
        on = score > 8
        runs = []
        i, n = 0, len(on)
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                runs.append([i, j, float(score[i:j].sum())])
                i = j
            else:
                i += 1
        if not runs:
            continue
        merged = [runs[0]]
        for a, b, s in runs[1:]:
            if a - merged[-1][1] <= merge_gap:
                merged[-1][1] = b
                merged[-1][2] += s
            else:
                merged.append([a, b, s])
        for a, b, s in merged:
            w = b - a
            if w < min_w or w > max_w:
                continue
            xs = np.arange(a, b)
            wts = score[a:b] + 1e-6
            cx = x0 + int(np.average(xs, weights=wts))
            quality = s * np.sqrt(w)
            cand = (quality, cx, y, x0 + a, x0 + b)
            if best is None or quality > best[0]:
                best = cand
    if best is None:
        return None
    return {"xy": (best[1], best[2]), "span": (best[3], best[4])}


def locate_mouths(master: Image.Image) -> None:
    rgb = np.array(master.convert("RGB"))
    pip = detect_smile_center(rgb, PIP_FACE)
    jun = detect_smile_center(rgb, JUN_FACE)
    if pip:
        a, b = pip["span"]
        # 3/4 facing right: sit on the smile, slightly toward the visible right.
        PIP["mouth"] = (a + int((b - a) * 0.52), pip["xy"][1])
        PIP["smile_w"] = b - a
        print(f"detected Pip smile at {PIP['mouth']} span {pip['span']}")
    if jun:
        a, b = jun["span"]
        # 3/4 facing left: sit on the smile, slightly toward the visible left.
        JUN["mouth"] = (a + int((b - a) * 0.30), jun["xy"][1])
        JUN["smile_w"] = b - a
        print(f"detected Jun smile at {JUN['mouth']} span {jun['span']}")
    if not pip or not jun:
        print("warning: smile detector missed a face; using fallback coords")


def ellipse_mask(size: tuple[int, int], blur: float) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.ellipse((1, 1, size[0] - 2, size[1] - 2), fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur))


def draw_mouth(layer: Image.Image, cx: int, cy: int, kind: str, cfg: dict, open_amt: float) -> None:
    if kind in (None, "rest") and open_amt < 0.12:
        return
    overlay = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # Center visemes on the detected smile line so they replace it, not sit on the chin.
    smile_w = max(14, int(cfg.get("smile_w", 32) * WIDTH / 1536 * 0.42))
    cover = (*cfg["color"], 240)
    d.ellipse([cx - smile_w - 4, cy - 5, cx + smile_w + 4, cy + 6], fill=cover)
    lip = (*cfg["lip"], 235)
    cavity = (42, 18, 16, 235)
    teeth = (245, 236, 220, 220)

    if kind == "closed" or kind == "M":
        d.ellipse([cx - smile_w, cy - 3, cx + smile_w, cy + 3], fill=lip)
    elif kind == "wide" or kind == "E":
        h = int(5 + 6 * open_amt)
        d.ellipse([cx - smile_w - 4, cy - h, cx + smile_w + 4, cy + h], fill=lip)
        d.ellipse([cx - smile_w, cy - h + 3, cx + smile_w, cy + h - 2], fill=cavity)
        d.rectangle([cx - smile_w + 2, cy - h + 3, cx + smile_w - 2, cy - 1], fill=teeth)
    elif kind == "round" or kind == "O":
        r = int(7 + 6 * open_amt)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=lip)
        d.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 3], fill=cavity)
    elif kind == "small" or kind == "U":
        r = int(5 + 4 * open_amt)
        d.ellipse([cx - r - 2, cy - r + 1, cx + r + 2, cy + r], fill=lip)
        d.ellipse([cx - r + 2, cy - r + 3, cx + r - 2, cy + r - 2], fill=cavity)
    elif kind == "teeth" or kind == "F":
        d.ellipse([cx - smile_w, cy - 3, cx + smile_w, cy + 6], fill=lip)
        d.rectangle([cx - smile_w + 2, cy - 1, cx + smile_w - 2, cy + 3], fill=teeth)
    else:  # open / A
        h = int(6 + 8 * open_amt)
        d.ellipse([cx - smile_w, cy - h, cx + smile_w, cy + h], fill=lip)
        d.ellipse([cx - smile_w + 4, cy - h + 3, cx + smile_w - 4, cy + h - 3], fill=cavity)
        d.rectangle([cx - smile_w + 5, cy - h + 3, cx + smile_w - 5, cy - 1], fill=teeth)

    overlay = overlay.filter(ImageFilter.GaussianBlur(0.5))
    layer.alpha_composite(overlay)


def blink_amount(t: float, offset: float) -> float:
    # Quick blinks ~ every 3.3s, 0.12s long.
    cycle = (t + offset) % 3.35
    if 0.0 <= cycle <= 0.12:
        x = cycle / 0.12
        return math.sin(x * math.pi)
    return 0.0


def draw_blinks(layer: Image.Image, cfg: dict, mapped_eyes: tuple, amt: float, master_map) -> None:
    if amt <= 0.05:
        return
    overlay = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    lid = (*cfg["color"], int(210 * amt))
    for ex, ey in mapped_eyes:
        d.ellipse([ex - 22, ey - int(6 * amt) - 2, ex + 22, ey + int(7 * amt) + 1], fill=lid)
    layer.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(1.2)))


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for w in words:
        trial = (cur + " " + w).strip()
        if dummy.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def draw_bubble(layer: Image.Image, who: str, text: str, mouth_xy: tuple[int, int]) -> None:
    cfg = who_cfg(who)
    font = ImageFont.truetype(FONT_REG, 36)
    name_font = ImageFont.truetype(FONT_BOLD, 22)
    lines = wrap_text(text, font, 620)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = int(max(dummy.textlength(line, font=font) for line in lines))
    th = 48 * len(lines)
    pad_x, pad_y = 28, 20
    bw, bh = tw + pad_x * 2, th + pad_y * 2 + 18
    mx, my = mouth_xy
    if who == "pip":
        x1 = max(48, mx - 40)
        y1 = max(48, my - 280)
    else:
        x1 = min(WIDTH - bw - 48, mx - bw + 80)
        y1 = max(48, my - 300)
    overlay = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    shadow = (x1 + 6, y1 + 8, x1 + bw + 6, y1 + bh + 8)
    rounded_rect(d, shadow, 22, (30, 18, 10, 50))
    box = (x1, y1, x1 + bw, y1 + bh)
    rounded_rect(d, box, 22, (*cfg["bubble"], 236), outline=(255, 255, 255, 200), width=3)
    # Tail toward the mouth.
    tx = mx
    ty = my - 36
    mid = x1 + (80 if who == "pip" else bw - 80)
    d.polygon([(mid - 16, y1 + bh - 2), (mid + 16, y1 + bh - 2), (tx, ty)], fill=(*cfg["bubble"], 236))
    d.text((x1 + pad_x, y1 + 10), cfg["name"].upper(), font=name_font, fill=(90, 60, 40, 220))
    y = y1 + 34
    for line in lines:
        d.text((x1 + pad_x, y), line, font=font, fill=(40, 28, 22, 255))
        y += 48
    layer.alpha_composite(overlay)


def draw_captions(layer: Image.Image, text: str) -> None:
    font = ImageFont.truetype(FONT_BOLD, 34)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = dummy.textlength(text, font=font)
    x = (WIDTH - tw) / 2
    y = HEIGHT - 92
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([x - 22, y - 12, x + tw + 22, y + 50], 16, fill=(20, 12, 8, 150))
    d.text((x, y), text, font=font, fill=(255, 248, 236, 255))


def ken_burns(base: Image.Image, t: float, total: float) -> Image.Image:
    z = 1.0 + 0.045 * (t / max(total, 0.01))
    nw, nh = int(WIDTH * z), int(HEIGHT * z)
    grown = base.resize((nw, nh), Image.Resampling.LANCZOS)
    # Drift slightly right so both faces stay in frame.
    left = int((nw - WIDTH) * 0.35)
    top = int((nh - HEIGHT) * 0.25)
    return grown.crop((left, top, left + WIDTH, top + HEIGHT))


def compose_frame(scene: Image.Image, t: float, total: float, active: dict | None, maps: dict) -> Image.Image:
    # Animate mouths/blinks on the locked still, then zoom the whole plate so
    # lip overlays never drift off the faces.
    plate = scene.convert("RGBA")
    pulse = 1.0 + 0.012 * math.sin(t * 1.3)
    plate = ImageEnhance.Brightness(plate).enhance(pulse)

    for who, cfg in (("pip", PIP), ("jun", JUN)):
        eyes = tuple(maps[who]["eyes"])
        blink = blink_amount(t, 0.2 if who == "pip" else 1.4)
        if active and active["who"] == who:
            blink *= 0.25
        draw_blinks(plate, cfg, eyes, blink, maps)

    hud_mouth = None
    if active:
        who = active["who"]
        cfg = who_cfg(who)
        mx, my = maps[who]["mouth"]
        kind = VISEME.get(active["viseme"], "open")
        open_amt = 0.35 + 0.65 * abs(math.sin(t * 17.0))
        if active["viseme"] in ("M", "rest"):
            open_amt = 0.08
        draw_mouth(plate, mx, my, kind or "rest", cfg, open_amt)
        hud_mouth = (mx, my)

    frame = ken_burns(plate, t, total).convert("RGBA")
    if active and hud_mouth:
        who = active["who"]
        cfg = who_cfg(who)
        z = 1.0 + 0.045 * (t / max(total, 0.01))
        nw, nh = int(WIDTH * z), int(HEIGHT * z)
        left = int((nw - WIDTH) * 0.35)
        top = int((nh - HEIGHT) * 0.25)
        mx = int(hud_mouth[0] * z - left)
        my = int(hud_mouth[1] * z - top)
        draw_bubble(frame, who, active["text"], (mx, my))
        draw_captions(frame, f'{cfg["name"]}: {active["text"]}')
    else:
        font = ImageFont.truetype(FONT_BOLD, 42)
        d = ImageDraw.Draw(frame)
        label = "Pip & Jun  ·  persistent characters"
        tw = d.textlength(label, font=font)
        d.rounded_rectangle(
            [(WIDTH - tw) / 2 - 28, HEIGHT - 110, (WIDTH + tw) / 2 + 28, HEIGHT - 48],
            18,
            fill=(20, 12, 8, 140),
        )
        d.text(((WIDTH - tw) / 2, HEIGHT - 100), label, font=font, fill=(255, 246, 230, 255))
    return frame.convert("RGB")


def concat_audio(clips: list[tuple[Path, float, float]], total: float, dest: Path) -> None:
    """clips: (path, start_time, duration)"""
    parts = []
    filter_bits = []
    cmd = ["ffmpeg", "-y"]
    cmd += [
        "-f",
        "lavfi",
        "-t",
        f"{total:.3f}",
        "-i",
        f"anullsrc=r={AUDIO_RATE}:cl=stereo",
    ]
    for i, (path, _start, _dur) in enumerate(clips, start=1):
        cmd += ["-i", str(path)]
    mix = ["[0:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0[base]"]
    for i, (path, start, _dur) in enumerate(clips, start=1):
        delay = int(start * 1000)
        mix.append(
            f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
            f"adelay={delay}|{delay},volume=1[a{i}]"
        )
    inputs = "[base]" + "".join(f"[a{i}]" for i in range(1, len(clips) + 1))
    mix.append(f"{inputs}amix=inputs={len(clips)+1}:dropout_transition=0:normalize=0[out]")
    cmd += [
        "-filter_complex",
        ";".join(mix),
        "-map",
        "[out]",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "2",
        "-b:a",
        "128k",
        str(dest),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(preview: bool = False) -> Path:
    if not ASSET.exists():
        raise SystemExit(f"Missing {ASSET}")
    master = Image.open(ASSET).convert("RGB")
    locate_mouths(master)
    scene = crop_scene(master)
    maps = {}
    for who, cfg in (("pip", PIP), ("jun", JUN)):
        maps[who] = {
            "mouth": map_pt(cfg["mouth"], master),
            "eyes": tuple(map_pt(e, master) for e in cfg["eyes"]),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="talkvid-"))
    try:
        timeline = []
        t = INTRO
        audio_clips = []
        for i, line in enumerate(LINES):
            cfg = who_cfg(line["who"])
            mp3 = work / f"line{i}.mp3"
            dur = make_line_audio(line["text"], cfg, mp3)
            vis = viseme_track(line["text"], dur, FPS)
            timeline.append({"who": line["who"], "text": line["text"], "start": t, "end": t + dur, "visemes": vis})
            audio_clips.append((mp3, t, dur))
            t = t + dur + GAP
        total = t + OUTRO

        if preview:
            # One talking frame per line for alignment checks.
            preview_dir = OUT_DIR / "preview"
            preview_dir.mkdir(exist_ok=True)
            dbg = master.copy()
            dr = ImageDraw.Draw(dbg)
            for cfg, col in ((PIP, (255, 60, 40)), (JUN, (40, 200, 90))):
                x, y = cfg["mouth"]
                dr.ellipse([x - 12, y - 8, x + 12, y + 8], outline=col, width=3)
            dbg.save(preview_dir / "detected-smiles.png")
            for i, item in enumerate(timeline):
                mid = (item["start"] + item["end"]) / 2
                active = {"who": item["who"], "text": item["text"], "viseme": "A"}
                frame = compose_frame(scene, mid, total, active, maps)
                frame.save(preview_dir / f"line_{i}_{item['who']}.png")
                # tight face crops for alignment
                mx, my = maps[item["who"]]["mouth"]
                z = 1.0 + 0.045 * (mid / max(total, 0.01))
                nw, nh = int(WIDTH * z), int(HEIGHT * z)
                left = int((nw - WIDTH) * 0.35)
                top = int((nh - HEIGHT) * 0.25)
                sx, sy = int(mx * z - left), int(my * z - top)
                crop = frame.crop((sx - 140, sy - 160, sx + 140, sy + 140))
                crop.save(preview_dir / f"face_{i}_{item['who']}.png")
            idle = compose_frame(scene, 0.4, total, None, maps)
            idle.save(preview_dir / "idle.png")
            print(f"Wrote previews to {preview_dir}")
            return preview_dir

        frames_dir = work / "frames"
        frames_dir.mkdir()
        nframes = int(total * FPS)
        for fi in range(nframes):
            now = fi / FPS
            active = None
            for item in timeline:
                if item["start"] <= now < item["end"]:
                    idx = min(len(item["visemes"]) - 1, int((now - item["start"]) * FPS))
                    active = {
                        "who": item["who"],
                        "text": item["text"],
                        "viseme": item["visemes"][idx],
                    }
                    break
            frame = compose_frame(scene, now, total, active, maps)
            frame.save(frames_dir / f"f{fi:05d}.png")
            if fi % 24 == 0:
                print(f"  frame {fi}/{nframes}")

        audio_path = work / "mix.m4a"
        concat_audio(audio_clips, total, audio_path)
        mp4 = OUT_DIR / "pip-and-jun-talking.mp4"
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(frames_dir / "f%05d.png"),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-level",
                "4.0",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-profile:a",
                "aac_low",
                "-ar",
                str(AUDIO_RATE),
                "-ac",
                "2",
                "-shortest",
                "-movflags",
                "+faststart",
                "-brand",
                "mp42",
                str(mp4),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Wrote {mp4}")
        return mp4
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import sys

    preview = "--preview" in sys.argv
    render(preview=preview)
