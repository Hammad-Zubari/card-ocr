"""Generate synthetic business-card PDFs + ground truth for testing.

    python scripts/make_sample_cards.py            # writes samples/*.pdf and samples/ground_truth.json

Creates:
  cards_one_per_page.pdf    20 cards, one per card-sized page, varied layouts/fonts, slight skew
  cards_multi_per_page.pdf  20 cards on A4 "scanner" pages (8 + 8 with visible edges, 4 white-on-white)
  cards_low_quality.pdf     6 degraded cards: blur, heavy JPEG, low DPI, noise, 4° skew, upside down
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples"
FONTS = Path("C:/Windows/Fonts")
W, H = 1050, 600  # 3.5 x 2 in at 300 dpi

CARDS = [
    dict(name="Muhammad Ali Khan", title="Senior Sales Manager", company="Crescent Textile Mills Ltd.",
         address=["Plot 23, Sector 15, Korangi Industrial Area", "Karachi 74900, Pakistan"],
         phones=[("Tel", "+92 21 3507 1234"), ("Mob", "+92 300 1234567")],
         email="ali.khan@crescenttextile.com.pk", web="www.crescenttextile.com.pk"),
    dict(name="Sarah Johnson", title="Marketing Director", company="BrightPath Solutions Inc.",
         address=["1250 Market Street, Suite 400", "San Francisco, CA 94103, USA"],
         phones=[("Phone", "+1 (415) 555-0132")], email="sarah.johnson@brightpath.io", web="brightpath.io"),
    dict(name="James O'Connor", title="Managing Partner", company="O'Connor & Hale LLP",
         address=["22 King Street", "London EC2V 8EH, United Kingdom"],
         phones=[("T", "+44 20 7946 0958")], fax="+44 20 7946 0959",
         email="j.oconnor@oconnorhale.co.uk", web="www.oconnorhale.co.uk"),
    dict(name="王伟", alt_name="Wang Wei", title="外贸经理 Export Manager", company="Ningbo Sunrise Trading Co., Ltd.",
         address=["Room 1203, No. 88 Zhongshan Road", "Ningbo, Zhejiang 315000, China"],
         phones=[("Tel", "+86 574 8765 4321"), ("Mob", "+86 138 0574 1234")],
         email="wangwei@sunrise-trading.cn", web=""),
    dict(name="Dr. Ayesha Siddiqui", title="Consultant Cardiologist", company="City Heart Clinic",
         address=["45-B Main Boulevard, Gulberg III", "Lahore 54660, Pakistan"],
         phones=[("Tel", "+92 42 3578 9900")], email="", web="www.cityheartclinic.pk"),
    dict(name="Lukas Weber", title="Head of Procurement", company="Kraftwerk Maschinenbau GmbH",
         address=["Industriestraße 14", "70565 Stuttgart, Germany"],
         phones=[("Tel", "+49 711 1234560")], fax="+49 711 1234569",
         email="l.weber@kraftwerk-mb.de", web="www.kraftwerk-mb.de"),
    dict(name="Priya Sharma", title="Software Engineer", company="Infinitum Technologies Pvt Ltd",
         address=["3rd Floor, Tower B, Cyber City", "Gurugram, Haryana 122002, India"],
         phones=[("Mobile", "+91 98765 43210")], email="priya.sharma@infinitum.tech", web=""),
    dict(name="Ahmed Al Mansoori", title="Business Development Manager", company="Gulf Horizon Trading LLC",
         address=["Office 1504, Burj Gate Tower, Sheikh Zayed Road", "Dubai, UAE"],
         phones=[("Tel", "+971 4 321 5678"), ("Mob", "+971 50 123 4567")],
         email="ahmed@gulfhorizon.ae", web="www.gulfhorizon.ae"),
    dict(name="Emily Chen", title="Chief Financial Officer", company="Northstar Capital Partners",
         address=["200 Bay Street, Suite 1800", "Toronto, ON M5J 2J2, Canada"],
         phones=[("P", "+1 416 555 0199")], email="emily.chen@northstarcap.com", web=""),
    dict(name="Carlos Mendes", title="Operations Manager", company="Atlântico Logística S.A.",
         address=["Rua Augusta 120", "1100-053 Lisboa, Portugal"],
         phones=[("Tel", "+351 21 123 4567")], email="carlos.mendes@atlantico-log.pt", web=""),
    dict(name="Hannah Müller", title="Freelance Photographer", company="",
         address=["Torstraße 55", "10119 Berlin, Germany"],
         phones=[], email="hannah@hannahmueller.photo", web="hannahmueller.photo"),
    dict(name="Kenji Tanaka", title="Product Designer", company="Sakura Design Studio",
         address=["2-11-3 Shibuya, Shibuya-ku", "Tokyo 150-0002, Japan"],
         phones=[("Tel", "+81 3 1234 5678")], email="k.tanaka@sakura-design.jp", web=""),
    dict(name="Fatima Zahra Benali", title="HR Coordinator", company="Atlas Pharma SARL",
         address=["18 Boulevard Zerktouni", "Casablanca 20250, Morocco"],
         phones=[("Tel", "+212 522 123 456")], email="f.benali@atlaspharma.ma", web=""),
    dict(name="Robert Williams", title="Owner", company="Williams Plumbing & Heating",
         address=["87 Oak Avenue", "Springfield, IL 62704"],
         phones=[("Tel", "(217) 555-0147"), ("Cell", "(217) 555-0190")],
         email="bob@williamsplumbing.com", web="williamsplumbing.com"),
    dict(name="Zainab Hussain", title="Merchandiser", company="Al-Karam Garments",
         address=["Shop 12, Liberty Market", "Faisalabad 38000, Pakistan"],
         phones=[("Mob", "0300-1234567")], email="zainab.hussain@alkaramgarments.pk", web=""),
    dict(name="Oliver Brown", title="Account Executive", company="Pinnacle Insurance Group",
         address=["Level 9, 1 Martin Place", "Sydney NSW 2000, Australia"],
         phones=[("Ph", "+61 2 9876 5432")], email="oliver.brown@pinnacleins.com.au", web=""),
    dict(name="李娜", alt_name="Li Na", title="销售经理 Sales Manager", company="深圳华美电子有限公司",
         address=["深圳市南山区科技园南路18号"],
         phones=[("电话", "+86 755 2345 6789")], email="lina@huamei-elec.com", web=""),
    dict(name="Michael Thompson", title="Vice President, Sales", company="Summit Analytics Corp.",
         address=[], phones=[("Direct", "+1 617 555 0123")],
         email="mthompson@summitanalytics.com", web="www.summitanalytics.com"),
    dict(name="Sofia Rossi", title="Architect", company="Studio Rossi Architettura",
         address=["Via Roma 45", "20121 Milano, Italy"],
         phones=[("Tel", "+39 02 1234 5678")], email="sofia@studiorossi.it", web=""),
    dict(name="Daniel Kim", title="Logistics Coordinator", company="Hanbit Shipping Co., Ltd.",
         address=["15F, 123 Teheran-ro, Gangnam-gu", "Seoul 06133, South Korea"],
         phones=[("Tel", "+82 2 555 1234")], email="daniel.kim@hanbitshipping.kr", web=""),
]

FONT_SETS = [
    ("arialbd.ttf", "arial.ttf"), ("georgiab.ttf", "georgia.ttf"), ("calibrib.ttf", "calibri.ttf"),
    ("verdanab.ttf", "verdana.ttf"), ("timesbd.ttf", "times.ttf"), ("segoeuib.ttf", "segoeui.ttf"),
    ("tahomabd.ttf", "tahoma.ttf"), ("trebucbd.ttf", "trebuc.ttf"), ("courbd.ttf", "cour.ttf"),
    ("bahnschrift.ttf", "bahnschrift.ttf"),
]
ACCENTS = [(20, 60, 120), (150, 30, 40), (10, 110, 90), (90, 40, 130), (30, 30, 30), (200, 110, 0)]


def font(name: str, size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    if cjk:
        name = "msyhbd.ttc" if "bd" in name or "b." in name else "msyh.ttc"
    return ImageFont.truetype(str(FONTS / name), size)


def has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def contact_lines(c: dict) -> list[str]:
    lines = [f"{lab}: {num}" if len(lab) > 1 or not lab.isalpha() else f"{lab} {num}" for lab, num in c["phones"]]
    if c.get("fax"):
        lines.append(f"Fax: {c['fax']}")
    if c["email"]:
        lines.append(("E: " if len(lines) % 2 else "Email: ") + c["email"])
    if c.get("web"):
        lines.append(c["web"])
    return lines


def draw_text(d: ImageDraw.ImageDraw, xy, text, f, fill, anchor="la"):
    d.text(xy, text, font=f if not has_cjk(text) else font("msyh.ttc", f.size, cjk=True), fill=fill, anchor=anchor)


def render_card(c: dict, layout: int, fonts: tuple[str, str], accent) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    bold, reg = fonts
    name_text = c["name"] + (f"  {c['alt_name']}" if c.get("alt_name") else "")
    f_name, f_title = font(bold, 52), font(reg, 30)
    f_comp, f_body = font(bold, 34), font(reg, 25)
    contacts = contact_lines(c)

    if layout == 0:  # classic left: company top, name middle, contacts bottom
        draw_text(d, (60, 50), c["company"], f_comp, accent)
        draw_text(d, (60, 150), name_text, f_name, (20, 20, 20))
        draw_text(d, (60, 215), c["title"], f_title, (90, 90, 90))
        y = 300
        for line in c["address"] + contacts:
            draw_text(d, (60, y), line, f_body, (40, 40, 40))
            y += 36
    elif layout == 1:  # centered
        draw_text(d, (W // 2, 70), name_text, f_name, (20, 20, 20), "ma")
        draw_text(d, (W // 2, 140), c["title"], f_title, accent, "ma")
        d.line([(W // 2 - 200, 190), (W // 2 + 200, 190)], fill=accent, width=3)
        draw_text(d, (W // 2, 210), c["company"], f_comp, (50, 50, 50), "ma")
        y = 275
        for line in c["address"] + contacts:
            draw_text(d, (W // 2, y), line, f_body, (40, 40, 40), "ma")
            y += 34
    elif layout == 2:  # two columns: identity left, contacts right, address bottom
        draw_text(d, (50, 60), name_text, f_name if len(name_text) < 22 else font(bold, 44), (20, 20, 20))
        draw_text(d, (50, 130), c["title"], f_title, accent)
        draw_text(d, (50, 180), c["company"], f_body, (60, 60, 60))
        y = 270
        for line in contacts:
            draw_text(d, (560, y), line, font(reg, 23), (40, 40, 40))
            y += 34
        y = max(y, 430) + 10
        draw_text(d, (50, y), ", ".join(c["address"]), font(reg, 23), (40, 40, 40))
    elif layout == 3:  # logo block right, text left, company name bigger than person name
        d.rectangle([W - 300, 0, W, H], fill=accent)
        initials = "".join(w[0] for w in c["company"].split()[:2] if w[0].isalpha()) or "C"
        draw_text(d, (W - 150, H // 2 - 60), initials, font(bold, 110), "white", "ma")
        draw_text(d, (50, 50), c["company"], font(bold, 46) if len(c["company"]) < 26 else font(bold, 34), accent)
        draw_text(d, (50, 130), name_text, font(bold, 38), (20, 20, 20))
        draw_text(d, (50, 180), c["title"], font(reg, 26), (90, 90, 90))
        y = 250
        for line in c["address"] + contacts:
            draw_text(d, (50, y), line, font(reg, 23), (40, 40, 40))
            y += 32
    else:  # coloured header band
        d.rectangle([0, 0, W, 120], fill=accent)
        draw_text(d, (50, 38), c["company"], f_comp, "white")
        draw_text(d, (50, 160), name_text, f_name, (20, 20, 20))
        draw_text(d, (50, 225), c["title"], f_title, (90, 90, 90))
        y = 300
        for line in contacts + c["address"]:
            draw_text(d, (50, y), line, f_body, (40, 40, 40))
            y += 36
    return img


def jpeg(img: Image.Image, quality=88) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def truth(c: dict) -> dict:
    return {
        "name": c["name"], "alt_name": c.get("alt_name", ""),
        "address": ", ".join(c["address"]),
        "telephone": [n for _, n in c["phones"]],
        "email": c["email"],
    }


def scanned(img: Image.Image, rng: random.Random, max_angle=1.5) -> Image.Image:
    a = rng.uniform(-max_angle, max_angle)
    img = img.rotate(a, resample=Image.BICUBIC, expand=True, fillcolor="white")
    arr = np.asarray(img).astype(np.int16)
    arr = arr + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 4, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build() -> None:
    OUT.mkdir(exist_ok=True)
    rng = random.Random(42)
    gt: dict[str, list] = {}
    cards = [render_card(c, i % 5, FONT_SETS[i % len(FONT_SETS)], ACCENTS[i % len(ACCENTS)]) for i, c in enumerate(CARDS)]

    # 1) one card per page
    doc = pymupdf.open()
    gt["cards_one_per_page.pdf"] = []
    for i, (c, img) in enumerate(zip(CARDS, cards)):
        im = scanned(img, rng)
        page = doc.new_page(width=252 * im.width / W, height=144 * im.height / H)
        page.insert_image(page.rect, stream=jpeg(im))
        gt["cards_one_per_page.pdf"].append({"page": i + 1, "card": i + 1, **truth(c)})
    doc.save(OUT / "cards_one_per_page.pdf", deflate=True)

    # 2) several cards per A4 page (scanner-bed background + shadows), last page white-on-white
    a4w, a4h, dpi = 595, 842, 200
    pxw, pxh = int(a4w / 72 * dpi), int(a4h / 72 * dpi)
    cw, ch = int(3.5 * dpi), int(2 * dpi)
    doc = pymupdf.open()
    gt["cards_multi_per_page.pdf"] = []
    groups = [list(range(0, 8)), list(range(8, 16)), list(range(16, 20))]
    for p, idxs in enumerate(groups):
        edges = p < 2
        bg = Image.new("RGB", (pxw, pxh), (205, 205, 200) if edges else "white")
        for k, ci in enumerate(idxs):
            col, row = k % 2, k // 2
            card = cards[ci].resize((cw, ch), Image.LANCZOS)
            angle = rng.uniform(-3, 3) if edges else 0
            rot = card.convert("RGBA").rotate(angle, resample=Image.BICUBIC, expand=True)
            x = int(0.45 * dpi + col * (cw + 0.45 * dpi))
            y = int(0.5 * dpi + row * (ch + 0.45 * dpi))
            if edges:
                shadow = Image.new("RGBA", rot.size, (0, 0, 0, 0))
                shadow.paste((90, 90, 90, 160), mask=rot.split()[3])
                bg.paste(shadow.filter(ImageFilter.GaussianBlur(6)), (x + 6, y + 6), shadow.filter(ImageFilter.GaussianBlur(6)))
            bg.paste(rot, (x, y), rot)
            gt["cards_multi_per_page.pdf"].append({"page": p + 1, "card": len(gt["cards_multi_per_page.pdf"]) + 1,
                                                   **truth(CARDS[ci])})
        page = doc.new_page(width=a4w, height=a4h)
        page.insert_image(page.rect, stream=jpeg(scanned(bg, rng, 0.0)))
    doc.save(OUT / "cards_multi_per_page.pdf", deflate=True)

    # 3) low-quality variants
    doc = pymupdf.open()
    gt["cards_low_quality.pdf"] = []
    variants = [
        (0, lambda im: im.filter(ImageFilter.GaussianBlur(2.2))),
        (2, lambda im: Image.open(io.BytesIO(jpeg(im, 12)))),
        (7, lambda im: im.resize((W // 3, H // 3), Image.BILINEAR)),
        (12, lambda im: Image.fromarray(np.clip(np.asarray(im).astype(np.int16) + np.random.default_rng(3).normal(
            0, 28, (H, W, 3)).astype(np.int16), 0, 255).astype(np.uint8))),
        (15, lambda im: im.rotate(4.0, resample=Image.BICUBIC, expand=True, fillcolor="white")),
        (18, lambda im: im.rotate(180)),
    ]
    for i, (ci, fn) in enumerate(variants):
        im = fn(cards[ci])
        page = doc.new_page(width=252 * im.width / (W if im.width > W // 2 else W // 3),
                            height=144 * im.height / (H if im.height > H // 2 else H // 3))
        page.insert_image(page.rect, stream=jpeg(im))
        gt["cards_low_quality.pdf"].append({"page": i + 1, "card": i + 1, **truth(CARDS[ci])})
    doc.save(OUT / "cards_low_quality.pdf", deflate=True)

    (OUT / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    sys.exit(build())
