"""生成应用图标 assets/app.ico（仅用于重新生成，构建使用已提交的 .ico）。

用法：cd 仓库根目录 && python packaging/make_icon.py
输出：assets/app.ico

图标设计：深蓝圆角方块 + 白色 CC 字样 + 底部 TXT 徽标，寓意「为视频补全同名 TXT」。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "app.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)

BG_TOP = (59, 130, 246, 255)     # 顶部渐变亮蓝
BG_BOTTOM = (30, 64, 175, 255)   # 底部深蓝
TEXT = (255, 255, 255, 255)      # 白色 CC
BADGE = (249, 250, 251, 255)     # 徽标底色
BADGE_TEXT = (30, 64, 175, 255)  # 徽标文字色


def _font(size: int) -> ImageFont.FreeTypeFont:
    """取 Windows 常见粗体字体；均不存在时回退默认字体。"""
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_icon() -> Image.Image:
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 垂直渐变背景
    for y in range(size):
        t = y / (size - 1)
        color = tuple(
            int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        )
        draw.line([(0, y), (size, y)], fill=color + (255,))

    # 圆角遮罩
    radius = int(size * 0.22)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    image.putalpha(mask)
    draw = ImageDraw.Draw(image)

    # CC 字样（水平居中，略靠上）
    font = _font(int(size * 0.52))
    text = "CC"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2 - bbox[0]
    y = int(size * 0.15) - bbox[1]
    draw.text((x, y), text, font=font, fill=TEXT)

    # 底部 TXT 徽标
    badge_font = _font(int(size * 0.17))
    badge_text = "TXT"
    bb = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_width = bb[2] - bb[0]
    badge_height = bb[3] - bb[1]
    pad = int(size * 0.035)
    bx0 = (size - badge_width) // 2 - pad - bb[0]
    by0 = int(size * 0.64)
    draw.rounded_rectangle(
        [bx0, by0, bx0 + badge_width + pad * 2, by0 + badge_height + pad],
        radius=int(size * 0.045),
        fill=BADGE,
    )
    draw.text(
        (bx0 + pad - bb[0], by0 + pad // 2 - bb[1]),
        badge_text,
        font=badge_font,
        fill=BADGE_TEXT,
    )
    return image


def main() -> int:
    if not (ROOT / "src" / "cc_cover").is_dir():
        print("请在仓库根目录运行：python packaging/make_icon.py")
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _draw_icon().save(OUTPUT, format="ICO", sizes=[(s, s) for s in SIZES])
    print("图标已生成：", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
