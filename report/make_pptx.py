"""Build the presentation as a PowerPoint file (P05_presentation.pptx) from the slide figures, equation images
and speaker notes. Run from P05_code/:  python report/make_pptx.py
"""
import json
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
NOTES = json.load(open(ROOT / "report" / "speaker_notes.json"))
FRAME = ROOT.parent / "Ra1e15_f100.png"

DARK, LIGHT, PANEL = RGBColor(0x14, 0x21, 0x3D), RGBColor(0xF7, 0xF6, 0xF2), RGBColor(0xEC, 0xEA, 0xE3)
BODY, ACCENT, ORANGE = RGBColor(0x3F, 0x4A, 0x5C), RGBColor(0x8A, 0x1C, 0x5C), RGBColor(0xF2, 0xA6, 0x5A)
C9, C12, C15 = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xC4, 0x50, 0x1F), RGBColor(0x13, 0x8A, 0x60)
SUB = RGBColor(0xC9, 0xD3, 0xE3)
FONT, HFONT = "Calibri", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=DARK, bold=False, font=FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.1):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, dict) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.01)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        for t, st in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = t
            r.font.size = Pt(st.get("size", size)); r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color); r.font.name = st.get("font", font)
    return tb


def rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background(); s.shadow.inherit = False
    if radius is not None:
        s.adjustments[0] = radius
    return s


def picture(slide, path, x, y, w=None, h=None):
    W, H = Image.open(path).size
    if w and not h:
        h = w * H / W
    if h and not w:
        w = h * W / H
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h)), w, h


def notes(slide, key):
    slide.notes_slide.notes_text_frame.text = NOTES[key]


def header(slide, eyebrow, title):
    text(slide, 0.6, 0.38, 12, 0.35, eyebrow.upper(), size=13, color=ACCENT, bold=True)
    text(slide, 0.6, 0.68, 12.2, 0.8, title, size=32, color=DARK, bold=True, font=HFONT)


def method(slide, steps, top=1.62, bottom=7.05, x=0.6, w=4.55):
    """steps: list of (number, text, equation image or None, equation display height in inches)."""
    rect(slide, x, top, w, bottom - top, PANEL, MSO_SHAPE.ROUNDED_RECTANGLE, 0.04)
    text(slide, x + 0.28, top + 0.18, w - 0.5, 0.3, "METHOD", size=12, color=ACCENT, bold=True)
    tw = w - 1.0
    est = []
    for n, t, eq, eh in steps:
        lines = max(1, -(-len(t) // 40))
        est.append(lines * 0.29 + (eh + 0.08 if eq else 0))
    free = (bottom - top - 0.75) - sum(est)
    gap = max(0.12, free / max(1, len(steps)))
    y = top + 0.62
    for (n, t, eq, eh), h in zip(steps, est):
        c = rect(slide, x + 0.28, y, 0.36, 0.36, DARK, MSO_SHAPE.OVAL)
        tf = c.text_frame; tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(n); r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = LIGHT; r.font.name = FONT
        lines = max(1, -(-len(t) // 40))
        text(slide, x + 0.78, y - 0.02, tw, lines * 0.29, t, size=15, color=DARK)
        if eq:
            picture(slide, FIG / eq, x + 0.8, y + lines * 0.29 + 0.06, h=eh)
        y += h + gap


def result(slide, fig, chips, top=1.62, x=5.45, w=7.3):
    pic, pw, ph = picture(slide, FIG / fig, x, top, h=4.25)
    if pw > w:
        pic.width = Inches(w); pic.height = Inches(w * ph / pw); pw, ph = w, w * ph / pw
    pic.left = Inches(x + (w - pw) / 2)
    cw, gap, cy = (w - 0.5) / 3, 0.25, top + 4.42
    for i, (v, t, colr) in enumerate(chips):
        cx = x + i * (cw + gap)
        rect(slide, cx, cy, cw, 0.05, colr)
        text(slide, cx, cy + 0.1, cw, 0.45, v, size=22, color=colr if colr != DARK else DARK, bold=True, font=HFONT)
        text(slide, cx, cy + 0.55, cw, 0.5, t, size=11.5, color=BODY)


# 1 cover -----------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, DARK)
text(s, 0.65, 0.75, 5.6, 0.35, "IDC612 · PROJECT P05", size=14, color=ORANGE, bold=True)
text(s, 0.65, 1.15, 5.7, 2.0, "Spectra of compressible convection, read from movie pixels", size=36, color=LIGHT, bold=True, font=HFONT, spacing=1.0)
text(s, 0.65, 3.25, 5.5, 0.9, "Four simulation movies: 2D at Ra 10⁹, 10¹², 10¹⁵ and 3D at Ra 10⁸. Ideal gas with γ = 1.3, D = 0.5, ε = 0.1.", size=15, color=SUB)
rect(s, 0.65, 4.3, 5.2, 0.02, RGBColor(0x34, 0x45, 0x6B))
text(s, 0.65, 4.42, 5, 0.3, "TEAM", size=12, color=ORANGE, bold=True)
text(s, 0.65, 4.75, 5.4, 1.5, ["Suvid Goel", "Abhay Tripathi", "Vibhor Srivastava"], size=20, color=LIGHT, bold=True, spacing=1.15)
text(s, 0.65, 6.5, 5, 0.35, "14 November 2026", size=13, color=SUB)
picture(s, FRAME, 6.55, 1.25, w=6.2)
text(s, 6.55, 4.92, 6.2, 0.7, "One frame of the Ra 10¹⁵ movie: temperature (top) and density (bottom). Every number in this talk comes from these pixels.", size=13, color=SUB)
notes(s, "cover")

# 2 spectrum -------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, LIGHT)
header(s, "Result 1 of 4 · 2D FFT with the mirror trick", "The full 2D temperature spectrum")
method(s, [(0, "Turn each pixel’s colour into temperature using the colour bar in the frame.", None, 0),
           (1, "Shift the temperature so it is zero on both plates:", "eq_zero.png", 0.62),
           (2, "Mirror it about the top plate with a sign flip: θ′(2 − z) = −θ′(z). Now z is periodic (period 2), so the FFT in z becomes a sine series.", None, 0),
           (3, "FFT in x × sine series in z gives every 2D mode.", None, 0),
           (4, "Add the energy in rings of equal K and fit the slope at K/π = 5–15.", None, 0)])
result(s, "deck_spec2d.png", [("−1.27", "Ra 10⁹  [−1.38, −1.16]", C9), ("−1.62", "Ra 10¹²  [−2.04, −1.20]. Matches −5/3.", C12),
                              ("−2.27", "Ra 10¹⁵  [−2.34, −2.20]. Steeper.", C15)])
notes(s, "spec2d")

# 3 entropy --------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, LIGHT)
header(s, "Result 2 of 4 · Entropy and the density panel", "Entropy falls and vortex cores take over")
method(s, [(1, "Treat the spectrum as probabilities over k and take the Shannon entropy:", "eq_ent.png", 0.5),
           (2, "Read the density panel too. Since p = ρT, the temperature splits into a plume part and a vortex-core (pressure) part.", None, 0),
           (3, "Pressure share = how much of the temperature variance the vortex cores carry.", None, 0)])
result(s, "deck_entropy.png", [("4.44 → 2.82", "bits: fewer, bigger structures as Ra rises", DARK),
                               ("28% → 97%", "vortex cores; steep spectrum at Ra 10¹⁵. 3D: 7%", C15),
                               ("0.29, 0.48", "γ − 1 and D from pixels (printed: 0.30, 0.50)", DARK)])
notes(s, "entropy")

# 4 mean profile ---------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, LIGHT)
header(s, "Result 3 of 4 · Mean temperature profile", "Boundary layers give a k⁻² branch in z")
method(s, [(1, "At each height z, average the temperature θ′ across the whole box (along x) and over all frames. This gives one curve: the mean temperature at each height, Θ(z).", None, 0),
           (2, "Write Θ(z) as a sum of sine waves sin(nπz), n = 1, 2, 3, … Eₙ is the energy of the n-th sine wave.", None, 0),
           (3, "Read the two jumps J₀, J₁ at the plates from the profile. They predict every mode, with nothing fitted:", "eq_jump.png", 1.05)])
result(s, "deck_upper.png", [("0.98, 1.00", "measured ÷ model at Ra 10¹², 10¹⁵. Nothing fitted.", DARK),
                             ("≈ −2.1", "slope in n (−2.15, −2.09). Theory: −2.", DARK),
                             ("≈ 0.25, 0.75", "jumps J₀, J₁ in ε, all three movies. Boussinesq: 0.5 each.", DARK)])
notes(s, "branch")

# 5 sound ----------------------------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); bg(s, LIGHT)
header(s, "Result 4 of 4 · Frequency–wavenumber spectrum", "Sound waves move at the ideal-gas speed")
method(s, [(1, "Take each 2D wave from slide 2 and follow its strength through all 1001 frames. FFT that in time: how fast does the wave wobble?", None, 0),
           (2, "Plot the power against wavenumber K and frequency.", None, 0),
           (3, "Sound shows as a straight ridge; its slope is the speed c:", "eq_ridge.png", 0.3),
           (4, "Compare c with the ideal-gas sound speed from the printed γ, D, ε.", None, 0)])
result(s, "deck_sound.png", [("1.93–2.08", "measured c, T and ρ panels, Ra 10¹², 10¹⁵", DARK),
                             ("2.09", "ideal-gas theory, nothing fitted", DARK),
                             ("20%", "of T variance is sound at Ra 10¹⁵ (10¹²: 4%)", C15)])
notes(s, "sound")

out = ROOT / "report" / "P05_presentation.pptx"
prs.save(out)
print("saved", out)
