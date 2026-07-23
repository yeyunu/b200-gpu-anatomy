from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "gpu-sm-warp-memory-flow-1080p.mp4"
COVER = ROOT / "gpu-sm-warp-memory-video-cover.png"
WIDTH, HEIGHT = 1920, 1080
FPS = 24
INTRO_SECONDS = 3.5
SCENE_SECONDS = 7.0
OUTRO_SECONDS = 3.5

NAVY = (7, 18, 32)
NAVY_2 = (11, 31, 52)
WHITE = (246, 250, 255)
MUTED = (166, 188, 210)
BLUE = (55, 153, 255)
CYAN = (63, 221, 211)
ORANGE = (255, 139, 61)
GREEN = (79, 202, 124)
PINK = (247, 112, 170)


@dataclass(frozen=True)
class Scene:
    number: int
    title: str
    caption: str
    bullets: tuple[str, ...]
    image: str
    path: tuple[tuple[float, float], ...]
    focus: tuple[int, int, int, int]
    location: str


SCENES = (
    Scene(
        1,
        "全局分工",
        "CPU 发出 Kernel；HBM 保存 1…64；一个 Block 被调度到 SM 5。",
        ("位置：CPU、HBM、GPU die", "64 个线程 = 2 个 Warp", "先看清谁负责什么"),
        "gpu-parallel-01-mapping.png",
        ((95, 232), (282, 275), (545, 225), (512, 366), (840, 340), (986, 340)),
        (180, 115, 1428, 530),
        "全局 → SM 5",
    ),
    Scene(
        2,
        "Load：HBM → 寄存器",
        "两个 Warp 发起读取。数据经过 L2/L1，进入 T0…T63 各自的私有寄存器。",
        ("HBM：原始输入 1…64", "L2/L1：可能命中的缓存路径", "寄存器：每个线程的临时值"),
        "gpu-parallel-02-registers.png",
        ((282, 275), (545, 225), (512, 366), (840, 340), (930, 390), (1070, 390)),
        (225, 170, 1140, 445),
        "HBM → L2/L1 → Register",
    ),
    Scene(
        3,
        "写入 Shared Memory",
        "线程把寄存器中的值写进同一 Block 共享的 s[0…63]，为并行归约做准备。",
        ("寄存器：线程私有", "Shared：Block 内共享", "数据仍然位于 SM 5"),
        "gpu-parallel-03-shared-storage.png",
        ((1070, 390), (1160, 390), (1225, 390)),
        (990, 320, 1308, 460),
        "Register → Shared",
    ),
    Scene(
        4,
        "32 路同时加法",
        "Warp 0 的 32 条 Lane 同时执行同一条加法：前半区和后半区逐项相加。",
        ("T0：1 + 33 = 34", "T1：2 + 34 = 36", "……直到 T31：32 + 64 = 96"),
        "gpu-parallel-04-round32.png",
        ((1220, 390), (1370, 390), (1280, 390), (1220, 390)),
        (45, 550, 1435, 1015),
        "SM 5：32 条 Lane 并行",
    ),
    Scene(
        5,
        "继续归约",
        "32 个结果继续缩成 16、8、4、2、1；中间值始终留在 SM 5 的 Shared 中。",
        ("stride：16 → 8 → 4 → 2 → 1", "每一轮减少一半活跃 Lane", "最后 s[0] = 2080"),
        "gpu-parallel-05-rounds.png",
        ((100, 620), (1370, 620), (100, 705), (1370, 705), (100, 970), (1370, 970)),
        (45, 555, 1425, 1005),
        "Shared ↔ ALU，循环 5 轮",
    ),
    Scene(
        6,
        "数据究竟放在哪里",
        "HBM 长期保存；寄存器线程私有；Shared 属于当前 Block；L1/L2 只是缓存路径。",
        ("HBM：全局、容量大、延迟高", "Register：线程私有、最快", "Shared：Block 共享、片上高速"),
        "gpu-parallel-06-storage-map.png",
        ((350, 670), (690, 670), (1030, 670), (1320, 670), (1320, 950)),
        (45, 555, 1425, 1045),
        "位置表：阶段 × 存储层级",
    ),
    Scene(
        7,
        "Warp 如何隐藏等待",
        "Warp 0 等待 HBM 时，调度器可运行 Warp 1 或其他已就绪 Warp；这叫延迟隐藏。",
        ("等待发生在全局内存路径", "计算发生在放大的 SM 内", "切换 Warp 不等于搬走数据"),
        "gpu-parallel-07-warp-timeline.png",
        ((310, 650), (520, 650), (740, 735), (965, 650), (1180, 735), (1290, 825)),
        (280, 600, 1310, 1015),
        "等待 Warp A → 运行 Warp B",
    ),
    Scene(
        8,
        "写回 HBM",
        "最终 2080 从 Shared 进入寄存器，经 L2 写入同一个 HBM 内存池的 out[0]。",
        ("输入 x[0…63] 仍然不变", "输出 out[0] = 2080", "输入区和输出区属于同一 HBM"),
        "gpu-parallel-08-overview.png",
        ((1220, 390), (1070, 390), (840, 360), (545, 225), (282, 395)),
        (180, 115, 1428, 530),
        "Shared → Register → L2 → HBM",
    ),
)


def choose_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


FONT_HERO = choose_font(72, True)
FONT_H1 = choose_font(54, True)
FONT_H2 = choose_font(32, True)
FONT_BODY = choose_font(30)
FONT_SMALL = choose_font(23)
FONT_PILL = choose_font(22, True)
FONT_TINY = choose_font(19)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def fit_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 12,
) -> int:
    x, y = xy
    bbox = draw.textbbox((0, 0), "国Ag", font=font)
    line_height = bbox[3] - bbox[1] + line_gap
    for line in fit_text(text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def rounded_image(image: Image.Image, radius: int = 24) -> Image.Image:
    rgba = image.convert("RGBA")
    mask = Image.new("L", rgba.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, rgba.width - 1, rgba.height - 1), radius, fill=255)
    rgba.putalpha(mask)
    return rgba


def image_layout(image: Image.Image) -> tuple[int, int, float, Image.Image]:
    target_h = 1038
    scale = target_h / image.height
    target_w = int(image.width * scale)
    x = 1920 - target_w - 34
    y = 21
    resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return x, y, scale, rounded_image(resized, 22)


def map_point(point: tuple[float, float], x: int, y: int, scale: float) -> tuple[float, float]:
    return x + point[0] * scale, y + point[1] * scale


def point_along_path(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]
    lengths: list[float] = []
    total = 0.0
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        lengths.append(length)
        total += length
    target = (t % 1.0) * total
    traversed = 0.0
    for index, length in enumerate(lengths):
        if target <= traversed + length:
            local = 0.0 if length == 0 else (target - traversed) / length
            a, b = points[index], points[index + 1]
            return a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local
        traversed += length
    return points[-1]


def overlay_flow(
    frame: Image.Image,
    scene: Scene,
    progress: float,
    img_x: int,
    img_y: int,
    scale: float,
) -> None:
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [map_point(point, img_x, img_y, scale) for point in scene.path]
    focus = scene.focus
    fx1, fy1 = map_point((focus[0], focus[1]), img_x, img_y, scale)
    fx2, fy2 = map_point((focus[2], focus[3]), img_x, img_y, scale)
    pulse = 0.5 + 0.5 * math.sin(progress * math.tau * 2.0)
    draw.rounded_rectangle(
        (fx1, fy1, fx2, fy2),
        radius=18,
        fill=(55, 153, 255, int(12 + 10 * pulse)),
        outline=(55, 153, 255, int(150 + 80 * pulse)),
        width=5,
    )
    draw.line(points, fill=(255, 139, 61, 115), width=7, joint="curve")
    for offset, radius in ((0.0, 13), (0.22, 10), (0.44, 8)):
        px, py = point_along_path(points, progress * 1.35 - offset)
        draw.ellipse((px - radius * 2.0, py - radius * 2.0, px + radius * 2.0, py + radius * 2.0), fill=(255, 139, 61, 40))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 245, 215, 255), outline=(255, 139, 61, 255), width=4)
    frame.alpha_composite(overlay)


def draw_progress(draw: ImageDraw.ImageDraw, active: int, scene_progress: float) -> None:
    x0, x1, y = 54, 506, 1006
    segment_gap = 8
    segment_w = (x1 - x0 - segment_gap * 7) / 8
    for index in range(8):
        left = x0 + index * (segment_w + segment_gap)
        right = left + segment_w
        draw.rounded_rectangle((left, y, right, y + 8), radius=4, fill=(51, 73, 94))
        if index < active:
            fill_right = right
        elif index == active:
            fill_right = left + segment_w * scene_progress
        else:
            continue
        draw.rounded_rectangle((left, y, fill_right, y + 8), radius=4, fill=BLUE)


def render_scene(scene: Scene, progress: float, source: Image.Image) -> Image.Image:
    frame = Image.new("RGBA", (WIDTH, HEIGHT), NAVY + (255,))
    draw = ImageDraw.Draw(frame)

    draw.rectangle((0, 0, 560, HEIGHT), fill=NAVY_2)
    draw.rectangle((558, 0, 560, HEIGHT), fill=BLUE)

    img_x, img_y, scale, visual = image_layout(source)
    shadow = Image.new("RGBA", visual.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((8, 10, visual.width - 2, visual.height - 2), 24, fill=(0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    frame.alpha_composite(shadow, (img_x - 8, img_y - 2))
    frame.alpha_composite(visual, (img_x, img_y))

    pill = (54, 62, 180, 104)
    draw.rounded_rectangle(pill, radius=21, fill=BLUE)
    draw.text((78, 68), f"步骤 {scene.number}/8", font=FONT_PILL, fill=WHITE)

    draw.text((54, 142), scene.title, font=FONT_H1, fill=WHITE)
    caption_y = draw_wrapped(draw, (54, 230), scene.caption, FONT_BODY, WHITE, 450, 14)
    caption_y += 34
    for bullet in scene.bullets:
        draw.ellipse((58, caption_y + 10, 70, caption_y + 22), fill=CYAN)
        caption_y = draw_wrapped(draw, (88, caption_y), bullet, FONT_SMALL, MUTED, 410, 10) + 18

    location_y = 830
    draw.text((54, location_y), "当前数据路径", font=FONT_TINY, fill=MUTED)
    draw.rounded_rectangle((54, location_y + 38, 506, location_y + 100), radius=18, fill=(19, 55, 78), outline=CYAN, width=2)
    draw.text((76, location_y + 53), scene.location, font=FONT_PILL, fill=WHITE)
    draw_progress(draw, scene.number - 1, progress)
    draw.text((54, 1030), "64 个数并行归约为 2080", font=FONT_TINY, fill=(118, 150, 180))

    overlay_flow(frame, scene, progress, img_x, img_y, scale)

    fade = min(1.0, progress / 0.08, (1.0 - progress) / 0.08)
    if fade < 1.0:
        veil = Image.new("RGBA", frame.size, NAVY + (int(255 * (1.0 - fade)),))
        frame.alpha_composite(veil)
    return frame.convert("RGB")


def render_intro(progress: float, cover: Image.Image) -> Image.Image:
    background = cover.resize((WIDTH, int(cover.height * WIDTH / cover.width)), Image.Resampling.LANCZOS)
    crop_y = max(0, (background.height - HEIGHT) // 2)
    frame = background.crop((0, crop_y, WIDTH, crop_y + HEIGHT)).convert("RGBA")
    frame = frame.filter(ImageFilter.GaussianBlur(4))
    frame.alpha_composite(Image.new("RGBA", frame.size, (4, 15, 28, 220)))
    draw = ImageDraw.Draw(frame)

    appear = ease(min(1.0, progress / 0.45))
    y_shift = int(28 * (1.0 - appear))
    draw.rounded_rectangle((720, 170 + y_shift, 1200, 224 + y_shift), radius=27, fill=(55, 153, 255, int(220 * appear)))
    label = "GPU 并行计算 · 形象化演示"
    label_box = draw.textbbox((0, 0), label, font=FONT_H2)
    draw.text(((WIDTH - (label_box[2] - label_box[0])) / 2, 179 + y_shift), label, font=FONT_H2, fill=WHITE)
    title = "一笔数据，如何在 GPU 里完成并行归约？"
    title_box = draw.textbbox((0, 0), title, font=FONT_HERO)
    draw.text(((WIDTH - (title_box[2] - title_box[0])) / 2, 315 + y_shift), title, font=FONT_HERO, fill=WHITE)
    subtitle = "64 个数  →  2 个 Warp  →  1 个结果（2080）"
    subtitle_box = draw.textbbox((0, 0), subtitle, font=FONT_H1)
    draw.text(((WIDTH - (subtitle_box[2] - subtitle_box[0])) / 2, 455 + y_shift), subtitle, font=FONT_H1, fill=(157, 222, 255))

    cards = (("HBM", "全局数据仓库", BLUE), ("SM 5", "真正执行计算", ORANGE), ("Shared", "Block 的工作台", CYAN))
    for index, (title_card, body, color) in enumerate(cards):
        x = 370 + index * 410
        draw.rounded_rectangle((x, 650, x + 350, 800), radius=26, fill=(12, 38, 62, 235), outline=color, width=3)
        draw.text((x + 28, 680), title_card, font=FONT_H2, fill=color)
        draw.text((x + 28, 738), body, font=FONT_SMALL, fill=WHITE)

    fade = min(1.0, progress / 0.15, (1.0 - progress) / 0.14)
    if fade < 1.0:
        frame.alpha_composite(Image.new("RGBA", frame.size, NAVY + (int(255 * (1.0 - fade)),)))
    return frame.convert("RGB")


def render_outro(progress: float, cover: Image.Image) -> Image.Image:
    frame = Image.new("RGBA", (WIDTH, HEIGHT), NAVY + (255,))
    draw = ImageDraw.Draw(frame)
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    radius = 360 + int(40 * math.sin(progress * math.pi))
    glow_draw.ellipse((WIDTH // 2 - radius, 110, WIDTH // 2 + radius, 110 + radius * 2), fill=(55, 153, 255, 45))
    frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(90)))
    draw = ImageDraw.Draw(frame)
    draw.text((110, 100), "最终记住三句话", font=FONT_HERO, fill=WHITE)

    summary = (
        ("01", "HBM 是远端仓库", "容量和带宽很大，但访问延迟高。", BLUE),
        ("02", "SM 是并行计算现场", "Warp、寄存器、Shared 和算术单元都在这里协作。", ORANGE),
        ("03", "并行 + 数据复用 = 高吞吐", "让数据留在片上，并用其他 Warp 隐藏等待。", CYAN),
    )
    for index, (number, title, body, color) in enumerate(summary):
        y = 270 + index * 205
        draw.rounded_rectangle((110, y, 1810, y + 160), radius=30, fill=(13, 37, 59), outline=color, width=3)
        draw.rounded_rectangle((145, y + 38, 235, y + 122), radius=22, fill=color)
        draw.text((165, y + 60), number, font=FONT_PILL, fill=NAVY)
        draw.text((285, y + 27), title, font=FONT_H2, fill=WHITE)
        draw.text((285, y + 87), body, font=FONT_BODY, fill=MUTED)
    footer = "完整交互图：yeyunu.github.io/b200-gpu-anatomy/"
    footer_box = draw.textbbox((0, 0), footer, font=FONT_SMALL)
    draw.text(((WIDTH - footer_box[2]) / 2, 955), footer, font=FONT_SMALL, fill=(119, 161, 199))
    fade = min(1.0, progress / 0.15, (1.0 - progress) / 0.16)
    if fade < 1.0:
        frame.alpha_composite(Image.new("RGBA", frame.size, NAVY + (int(255 * (1.0 - fade)),)))
    return frame.convert("RGB")


def main() -> None:
    images = {scene.image: Image.open(ROOT / scene.image).convert("RGB") for scene in SCENES}
    render_scene(SCENES[3], 0.45, images[SCENES[3].image]).save(COVER)
    total_seconds = INTRO_SECONDS + len(SCENES) * SCENE_SECONDS + OUTRO_SECONDS
    total_frames = int(total_seconds * FPS)
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    assert process.stdin is not None

    intro_frames = int(INTRO_SECONDS * FPS)
    scene_frames = int(SCENE_SECONDS * FPS)
    outro_frames = int(OUTRO_SECONDS * FPS)
    written = 0

    try:
        for frame_index in range(intro_frames):
            frame = render_intro(frame_index / max(1, intro_frames - 1), images[SCENES[0].image])
            process.stdin.write(frame.tobytes())
            written += 1
        for scene in SCENES:
            source = images[scene.image]
            for frame_index in range(scene_frames):
                frame = render_scene(scene, frame_index / max(1, scene_frames - 1), source)
                process.stdin.write(frame.tobytes())
                written += 1
            print(f"Rendered scene {scene.number}/8 ({written}/{total_frames} frames)", flush=True)
        for frame_index in range(outro_frames):
            frame = render_outro(frame_index / max(1, outro_frames - 1), images[SCENES[-1].image])
            process.stdin.write(frame.tobytes())
            written += 1
    finally:
        process.stdin.close()

    result = process.wait()
    if result != 0:
        raise SystemExit(result)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
