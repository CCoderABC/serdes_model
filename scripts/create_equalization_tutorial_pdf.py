"""Create a beginner tutorial PDF for FFE, DFE, and MLSE in an ADC SerDes."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "adc_serdes_ffe_dfe_mlse_tutorial.pdf"

NAVY = HexColor("#17365D")
BLUE = HexColor("#2F75B5")
TEAL = HexColor("#007C91")
ORANGE = HexColor("#C55A11")
GREEN = HexColor("#3B7A57")
RED = HexColor("#B7403A")
LIGHT_BLUE = HexColor("#EAF2F8")
LIGHT_TEAL = HexColor("#E8F5F6")
LIGHT_ORANGE = HexColor("#FBEFE3")
LIGHT_GRAY = HexColor("#F3F5F7")
MID_GRAY = HexColor("#667085")
DARK = HexColor("#17212B")


class FigureFlowable(Flowable):
    """Small vector diagrams used throughout the tutorial."""

    def __init__(self, width: float, height: float, kind: str):
        super().__init__()
        self.width = width
        self.height = height
        self.kind = kind

    def _arrow(self, c, x1, y1, x2, y2, color=NAVY, width=1.6):
        c.setStrokeColor(color)
        c.setLineWidth(width)
        c.line(x1, y1, x2, y2)
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        head = 6
        c.setFillColor(color)
        c.line(x2, y2, x2 - head * ux + 0.5 * head * px, y2 - head * uy + 0.5 * head * py)
        c.line(x2, y2, x2 - head * ux - 0.5 * head * px, y2 - head * uy - 0.5 * head * py)

    def _box(self, c, x, y, w, h, label, fill, border=NAVY, size=8.2):
        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.setLineWidth(1.1)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", size)
        lines = label.split("\n")
        line_h = size + 2
        start = y + h / 2 + (len(lines) - 1) * line_h / 2 - size * 0.35
        for index, line in enumerate(lines):
            c.drawCentredString(x + w / 2, start - index * line_h, line)

    def _caption_text(self, c, text):
        c.setFillColor(MID_GRAY)
        c.setFont("Helvetica-Oblique", 7.4)
        c.drawString(0, 3, text)

    def draw(self):
        c = self.canv
        if self.kind == "system":
            y, h, w = 45, 34, 78
            blocks = [
                (4, "ADC\nsamples", LIGHT_BLUE),
                (105, "FFE\nreshape", LIGHT_TEAL),
                (206, "DFE\nsubtract", LIGHT_ORANGE),
                (307, "PAM4\nslicer", LIGHT_BLUE),
                (408, "decoded\nsymbols", LIGHT_TEAL),
            ]
            for x, label, fill in blocks:
                self._box(c, x, y, w, h, label, fill)
            for index in range(len(blocks) - 1):
                self._arrow(c, blocks[index][0] + w, y + h / 2, blocks[index + 1][0] - 6, y + h / 2)
            c.setStrokeColor(ORANGE)
            c.setDash(3, 2)
            self._arrow(c, 447, y, 447, 22, ORANGE, 1.3)
            self._arrow(c, 447, 22, 245, 22, ORANGE, 1.3)
            self._arrow(c, 245, 22, 245, y - 2, ORANGE, 1.3)
            c.setDash()
            c.setFillColor(ORANGE)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(295, 13, "DFE uses only past detected symbols")
            self._box(c, 205, 97, 102, 31, "Alternative: MLSE / Viterbi\nchooses a sequence", LIGHT_TEAL, TEAL, 8)
            self._arrow(c, 144, y + h, 144, 112, TEAL, 1.3)
            self._arrow(c, 144, 112, 199, 112, TEAL, 1.3)
            self._caption_text(c, "Fig. 1 - Two common digital receiver choices after an ADC.")

        elif self.kind == "pulse":
            left, bottom, top, right = 57, 31, 124, self.width - 18
            axis_y = 73
            c.setStrokeColor(MID_GRAY)
            c.setLineWidth(0.7)
            c.line(left, axis_y, right, axis_y)
            c.line(left, bottom, left, top)
            positions = [105, 176, 247, 318, 389]
            values = [0.20, 0.20, 1.00, 0.45, -0.20]
            labels = ["-2 UI", "-1 UI", "0 UI", "+1 UI", "+2 UI"]
            types = ["pre", "pre", "main", "post", "post"]
            for x, val, label, kind in zip(positions, values, labels, types):
                color = BLUE if kind == "main" else (TEAL if kind == "pre" else ORANGE)
                y2 = axis_y + val * 42
                c.setStrokeColor(color)
                c.setLineWidth(2.4)
                c.line(x, axis_y, x, y2)
                c.setFillColor(color)
                c.circle(x, y2, 3.5, fill=1, stroke=0)
                c.setFont("Helvetica-Bold", 8)
                c.drawCentredString(x, y2 + (8 if val >= 0 else -14), f"{val:+.2f}")
                c.setFillColor(MID_GRAY)
                c.setFont("Helvetica", 7.6)
                c.drawCentredString(x, 19, label)
                c.drawCentredString(x, 8, kind)
            c.setFillColor(DARK)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(8, 118, "cursor")
            self._caption_text(c, "Fig. 2 - Symbol-spaced channel pulse. Energy before 0 UI is precursor ISI; later energy is postcursor ISI.")

        elif self.kind == "ffe":
            y, box_w, box_h = 43, 62, 28
            positions = [8, 83, 158, 233]
            labels = ["r[n-2]", "r[n-1]", "r[n]", "r[n+1]"]
            for x, label in zip(positions, labels):
                self._box(c, x, y, box_w, box_h, label, LIGHT_BLUE)
            for x, tap in zip([39, 114, 189, 264], ["c2", "c1", "c0", "c-1"]):
                self._arrow(c, x, y, x, 21, TEAL, 1.1)
                c.setFillColor(TEAL)
                c.setFont("Helvetica-Bold", 7.5)
                c.drawCentredString(x, 29, tap)
            self._box(c, 315, 35, 60, 33, "sum", LIGHT_TEAL, TEAL)
            for x in (39, 114, 189, 264):
                self._arrow(c, x, 21, 315, 52, TEAL, 0.95)
            self._arrow(c, 375, 52, 393, 52, NAVY)
            self._box(c, 399, 35, 44, 33, "y[n]", LIGHT_BLUE, NAVY, 7.7)
            c.setFillColor(ORANGE)
            c.setFont("Helvetica-Bold", 7.2)
            c.drawCentredString(225, 96, "Decision delay makes future-relative samples available causally.")
            self._caption_text(c, "Fig. 3 - FFE is a weighted sum of ADC samples. It can reshape precursor and postcursor ISI, but may enhance noise.")

        elif self.kind == "dfe":
            self._box(c, 14, 83, 85, 34, "FFE output\ny[n]", LIGHT_BLUE)
            self._box(c, 146, 83, 68, 34, "subtract", LIGHT_ORANGE, ORANGE)
            self._box(c, 262, 83, 82, 34, "slicer\nz[n]", LIGHT_BLUE)
            self._box(c, 392, 83, 88, 34, "decision\na_hat[n]", LIGHT_TEAL, TEAL)
            self._arrow(c, 99, 100, 140, 100)
            self._arrow(c, 214, 100, 256, 100)
            self._arrow(c, 344, 100, 386, 100)
            self._box(c, 250, 23, 112, 34, "DFE taps\nb1, b2, ...", LIGHT_ORANGE, ORANGE)
            self._arrow(c, 436, 83, 436, 60, ORANGE, 1.4)
            self._arrow(c, 436, 60, 362, 40, ORANGE, 1.4)
            self._arrow(c, 250, 40, 180, 83, ORANGE, 1.4)
            c.setFillColor(RED)
            c.setFont("Helvetica-Bold", 6.9)
            c.drawString(318, 15, "wrong decision can cause a burst")
            self._caption_text(c, "Fig. 4 - DFE removes predicted postcursor ISI using past hard decisions. This avoids FFE noise enhancement but introduces feedback risk.")

        elif self.kind == "trellis":
            xs = [32, 125, 218, 311, 404]
            ys = [38, 62, 86, 110]
            c.setStrokeColor(HexColor("#B9C7D4"))
            c.setLineWidth(0.8)
            for column in range(len(xs) - 1):
                for y1 in ys:
                    for y2 in ys:
                        c.line(xs[column], y1, xs[column + 1], y2)
            survivor = [(32, 62), (125, 86), (218, 62), (311, 38), (404, 62)]
            c.setStrokeColor(GREEN)
            c.setLineWidth(3.0)
            for (x1, y1), (x2, y2) in zip(survivor, survivor[1:]):
                c.line(x1, y1, x2, y2)
            for x in xs:
                for y in ys:
                    c.setFillColor(LIGHT_TEAL)
                    c.setStrokeColor(TEAL)
                    c.circle(x, y, 5, fill=1, stroke=1)
            for index, x in enumerate(xs):
                c.setFillColor(MID_GRAY)
                c.setFont("Helvetica", 7.4)
                c.drawCentredString(x, 16, f"time {index}")
            c.setFillColor(GREEN)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawCentredString(225, 128, "survivor path: lowest accumulated prediction error")
            c.setFillColor(MID_GRAY)
            c.setFont("Helvetica", 7.3)
            c.drawCentredString(225, 141, "One-memory PAM4 MLSE has four states: the previous PAM4 symbol.")
            self._caption_text(c, "Fig. 5 - Simplified Viterbi trellis. MLSE retains the best candidate path entering each state instead of committing immediately.")

    def wrap(self, avail_width, avail_height):
        return self.width, self.height


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
            leading=28, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11,
            leading=15, textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
            leading=20, textColor=NAVY, spaceBefore=8, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
            leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13.1, textColor=DARK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.1,
            leading=10.7, textColor=DARK, spaceAfter=3,
        ),
        "eq": ParagraphStyle(
            "Equation", parent=base["Code"], fontName="Courier", fontSize=8.5,
            leading=12, textColor=NAVY, alignment=TA_CENTER, backColor=HexColor("#F7F9FB"),
            borderColor=HexColor("#D8E1E8"), borderWidth=0.5, borderPadding=5,
            spaceBefore=3, spaceAfter=7,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.2,
            leading=13.2, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=HexColor("#B8D3EA"),
            borderWidth=0.7, borderPadding=7, spaceBefore=4, spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.7,
            leading=9.5, textColor=MID_GRAY, alignment=TA_CENTER, spaceBefore=1, spaceAfter=7,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"], fontName="Helvetica", fontSize=7.5,
            textColor=MID_GRAY, alignment=TA_CENTER,
        ),
    }


def p(text: str, style):
    return Paragraph(text, style)


def bullet(text: str, style):
    return Paragraph(f'<font color="#007C91">&#8226;</font>&nbsp;&nbsp;{text}', style)


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(HexColor("#D6DEE5"))
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, 0.52 * inch, letter[0] - doc.rightMargin, 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(doc.leftMargin, 0.34 * inch, "ADC SerDes tutorial - FFE, DFE, and MLSE")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def section_table(rows, widths, styles, header_fill=NAVY):
    table_rows = []
    for row_index, row in enumerate(rows):
        row_style = styles["small"]
        table_rows.append([p(cell, row_style) for cell in row])
    table = Table(table_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#CFD8E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row in range(1, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=0.62 * inch, rightMargin=0.62 * inch,
        topMargin=0.58 * inch, bottomMargin=0.7 * inch,
        title="ADC SerDes Tutorial: FFE, DFE, and MLSE",
        author="Codex",
    )
    story = []

    mlse_state_table = section_table(
        [
            ["Residual memory L", "PAM4 states", "Branches / detected symbol", "Typical use"],
            ["1", "4", "16", "Strong single postcursor"],
            ["2", "16", "64", "Two deliberate residual cursors"],
            ["3", "64", "256", "Only if gain justifies cost"],
            ["4", "256", "1,024", "Usually expensive for a per-lane receiver"],
        ], [1.18 * inch, 1.05 * inch, 1.6 * inch, 2.35 * inch], styles, TEAL,
    )
    story.extend([
        Spacer(1, 0.36 * inch),
        p("ADC SerDes Tutorial", styles["title"]),
        p("Feed-forward equalization, decision-feedback equalization, and maximum-likelihood sequence estimation", styles["subtitle"]),
        FigureFlowable(6.25 * inch, 1.82 * inch, "system"),
        Spacer(1, 0.06 * inch),
        p("Purpose", styles["h1"]),
        p(
            "This tutorial builds intuition for the three main digital tools used after an ADC in a high-speed wireline receiver. "
            "It uses PAM4 examples, but the ideas apply equally to NRZ and other multi-level formats.", styles["body"]
        ),
        p(
            "Key idea: an FFE reshapes the channel, a DFE subtracts echoes from already detected symbols, and MLSE makes a delayed decision by comparing complete candidate sequences.",
            styles["callout"],
        ),
        p("Learning goals", styles["h2"]),
        bullet("Read a symbol-spaced pulse response and identify main cursor, precursor ISI, and postcursor ISI.", styles["body"]),
        bullet("Understand why FFE can cancel precursor ISI but may enhance noise.", styles["body"]),
        bullet("Understand why DFE is efficient for postcursor ISI but can create error bursts.", styles["body"]),
        bullet("Understand when a short-memory MLSE detector can outperform aggressive equalization.", styles["body"]),
        Spacer(1, 0.09 * inch),
        p("Notation", styles["h2"]),
        p(
            "a[n] is the transmitted PAM4 symbol in {-3, -1, +1, +3}; r[n] is the ADC sample; h[k] is the channel cursor; "
            "w[n] is additive noise; and a_hat[n] is a receiver decision.", styles["body"]
        ),
        PageBreak(),
    ])

    story.extend([
        p("1. The common channel model", styles["h1"]),
        p(
            "At the ADC sampling times, a linear channel is represented by a weighted sum of symbols. The current symbol is not observed in isolation because energy from nearby symbols overlaps it.",
            styles["body"],
        ),
        p("r[n] = sum(k=-Kpre to Kpost) h[k] a[n-k] + w[n]     (Eq. 1)", styles["eq"]),
        FigureFlowable(6.25 * inch, 1.82 * inch, "pulse"),
        p("How to read Fig. 2", styles["h2"]),
        bullet("The main cursor h[0] carries the desired symbol a[n].", styles["body"]),
        bullet("Precursor cursors arrive before the main pulse. They depend on future-relative symbols when the decision is referenced to a[n].", styles["body"]),
        bullet("Postcursor cursors are trailing echoes from earlier symbols. They depend on a[n-1], a[n-2], and so on.", styles["body"]),
        bullet("Noise cannot be removed deterministically. A good equalizer trades residual ISI against noise enhancement.", styles["body"]),
        p("Worked PAM4 sample", styles["h2"]),
        p(
            "Use the channel h[-1]=0.20, h[0]=1.00, h[1]=0.45, h[2]=-0.20. For a[n+1]=-3, a[n]=+1, a[n-1]=+3, and a[n-2]=-1:",
            styles["body"],
        ),
        p("r[n] = 0.20(-3) + 1.00(+1) + 0.45(+3) - 0.20(-1) = 1.95     (Eq. 2)", styles["eq"]),
        p(
            "The wanted contribution is only +1.00. The remaining +0.95 is structured ISI. The receiver methods below differ in how they deal with that structured interference.",
            styles["callout"],
        ),
        PageBreak(),
    ])

    story.extend([
        p("2. Feed-forward equalization (FFE)", styles["h1"]),
        p(
            "An FFE is an FIR filter applied directly to ADC samples. It does not depend on prior hard decisions, so it cannot propagate a wrong slicer decision through feedback.",
            styles["body"],
        ),
        p("y[n] = sum(m=0 to NF-1) c[m] r[n-m]     (Eq. 3)", styles["eq"]),
        FigureFlowable(6.25 * inch, 1.54 * inch, "ffe"),
        p("Why FFE can cancel precursor ISI", styles["h2"]),
        p(
            "The receiver delays the decision for a[n]. By the time it makes that delayed decision, samples that were future-relative to a[n] have arrived. "
            "This is why a symmetric analysis FFE can be implemented causally with a matching output latency.", styles["body"]
        ),
        p("Zero forcing versus MMSE", styles["h2"]),
        p(
            "A zero-forcing FFE tries to make the combined response c*h look like one isolated cursor. An MMSE FFE accepts some residual ISI when removing it would excessively amplify noise.",
            styles["body"],
        ),
        p("c_MMSE = arg min_c E{|a[n-d] - c^T r[n]|^2}     (Eq. 4)", styles["eq"]),
        p("For white input noise: sigma_out^2 = sigma_w^2 sum(m) |c[m]|^2     (Eq. 5)", styles["eq"]),
        p(
            "Eq. 5 is the FFE penalty: large coefficients can improve the pulse response while degrading actual BER. For long, high-loss channels, MMSE is normally the useful baseline.",
            styles["callout"],
        ),
        p("Practical FFE notes", styles["h2"]),
        bullet("Use known training data, least squares, LMS, or NLMS to obtain coefficients.", styles["body"]),
        bullet("T-spaced taps are simple; T/2-spaced taps are more tolerant of timing phase and fractional-delay effects.", styles["body"]),
        bullet("Include coefficient limits, rounding, saturation and output latency before using FFE results for hardware optimization.", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        p("3. Decision-feedback equalization (DFE)", styles["h1"]),
        p(
            "A DFE uses already detected symbols to predict and subtract the trailing channel echoes. It is especially efficient when postcursor ISI is large.", styles["body"]
        ),
        p("z[n] = y[n] - sum(k=1 to NB) b[k] a_hat[n-k]     (Eq. 6)", styles["eq"]),
        FigureFlowable(6.25 * inch, 1.72 * inch, "dfe"),
        p("What a DFE can and cannot cancel", styles["h2"]),
        bullet("If the effective postcursor is g[1]=0.45, a first DFE tap near +0.45 removes 0.45 a_hat[n-1].", styles["body"]),
        bullet("The DFE avoids the linear noise enhancement in Eq. 5 because it subtracts reconstructed symbol interference.", styles["body"]),
        bullet("A conventional DFE cannot cancel precursor ISI, because future symbols have not yet been detected.", styles["body"]),
        p("Error propagation example", styles["h2"]),
        p(
            "Suppose b[1]=0.45 and the correct past symbol is +3, but the slicer decides +1. The correct feedback is 0.45(3)=1.35; the applied feedback is 0.45(1)=0.45. "
            "The next decision has a residual error of +0.90.", styles["body"]
        ),
        p("feedback error = b[1] (a[n-1] - a_hat[n-1]) = 0.45(3 - 1) = 0.90     (Eq. 7)", styles["eq"]),
        p(
            "One wrong decision can therefore increase the chance of later wrong decisions. This creates burst errors, which matter directly to RS-FEC codeword failure probability.",
            styles["callout"],
        ),
        p("Training versus payload operation", styles["h2"]),
        p(
            "During training, fit DFE taps with known transmitted symbols. During payload operation, replace those symbols with a_hat. Always report both genie-aided and decision-directed results; only the decision-directed result is a credible receiver result.",
            styles["body"],
        ),
        PageBreak(),
    ])

    story.extend([
        p("4. Maximum-likelihood sequence estimation (MLSE)", styles["h1"]),
        p(
            "MLSE does not try to eliminate every residual cursor. Instead, it asks which sequence of PAM4 symbols best explains the received samples given a short channel model.",
            styles["body"],
        ),
        p("y[n] = sum(k=0 to L) g[k] a[n-k] + v[n]     (Eq. 8)", styles["eq"]),
        p("a_hat = arg min_a sum(n) |y[n] - sum(k=0 to L) g[k] a[n-k]|^2 / sigma_v^2     (Eq. 9)", styles["eq"]),
        FigureFlowable(6.25 * inch, 2.1 * inch, "trellis"),
        p("The Viterbi idea", styles["h2"]),
        bullet("A state stores the last L candidate PAM4 symbols.", styles["body"]),
        bullet("For each possible new PAM4 symbol, predict the next received sample and calculate its squared-error branch metric.", styles["body"]),
        bullet("Accumulate metrics and keep the lowest-metric survivor entering each state.", styles["body"]),
        bullet("Trace back through surviving paths after a delay to output reliable decisions.", styles["body"]),
        p("State count", styles["h2"]),
        mlse_state_table,
        Spacer(1, 0.08 * inch),
        p(
            "MLSE complexity grows exponentially with memory. This is why the FFE before MLSE should shorten the long physical channel into a deliberately short target response.",
            styles["callout"],
        ),
        PageBreak(),
    ])

    comparison_table = section_table(
        [
            ["Property", "FFE", "DFE", "MLSE"],
            ["Primary action", "Reshape samples", "Subtract past-symbol echoes", "Choose most likely path"],
            ["Precursor ISI", "Can cancel with latency", "Cannot cancel", "Handled with decision delay"],
            ["Postcursor ISI", "Can reduce", "Efficiently cancels", "Uses known memory"],
            ["Noise enhancement", "Possible", "No linear enhancement", "Only through any front-end filter"],
            ["Error propagation", "No", "Yes", "No direct feedback propagation"],
            ["Complexity", "Linear in taps", "Linear in feedback taps", "Exponential in memory"],
            ["Best fit", "Precursor removal and shaping", "Strong early postcursors", "Short strong residual memory"],
        ], [1.35 * inch, 1.63 * inch, 1.63 * inch, 1.63 * inch], styles,
    )
    story.extend([
        p("5. Channel shortening and receiver choices", styles["h1"]),
        p(
            "A slicer-oriented FFE tries to produce one isolated cursor. An MLSE-oriented FFE instead aims for a short controlled response, such as [1, 0.35]. The detector then uses that known remaining memory.",
            styles["body"],
        ),
        p("c*h approximately equals [0, ..., 0, g[0], g[1], ..., g[L], 0, ...]     (Eq. 10)", styles["eq"]),
        p("Example", styles["h2"]),
        p(
            "Rather than using extreme FFE gain to force g[1] to zero, retain g=[1, 0.35] and use a 4-state, one-memory PAM4 MLSE. "
            "This can reduce noise enhancement while preserving detection reliability.", styles["body"]
        ),
        comparison_table,
        Spacer(1, 0.11 * inch),
        p("Important ADC SerDes insight", styles["h2"]),
        p(
            "The ADC samples are already quantized. Neither FFE, DFE nor MLSE can recover amplitude information lost to clipping, too little ENOB, or a poor recovered clock phase. "
            "Equalizer, ADC range, threshold adaptation and CDR must be treated as a coupled system.", styles["callout"]
        ),
        p("Metric selection", styles["h2"]),
        bullet("Use dpSNR to understand residual linear error and noise, especially for an FFE.", styles["body"]),
        bullet("Use empirical DER and BER for DFE because error propagation can make residuals non-Gaussian.", styles["body"]),
        bullet("Use BER, error-event length, RS-symbol errors per codeword, and latency/operations for MLSE tradeoffs. An eye alone can underestimate MLSE capability.", styles["body"]),
        PageBreak(),
    ])

    plan_table = section_table(
        [
            ["Case", "Receiver path", "What it teaches"],
            ["A", "ADC -> FFE -> slicer", "Baseline residual ISI, noise enhancement, precursor cancellation"],
            ["B", "ADC -> FFE -> DFE -> slicer", "Postcursor cancellation and decision-error propagation"],
            ["C", "ADC -> channel-shortening FFE -> 1-memory MLSE", "Value of keeping one controlled residual cursor"],
            ["D", "ADC -> channel-shortening FFE -> 2-memory MLSE", "BER gain versus 16-state detector cost"],
        ], [0.55 * inch, 2.55 * inch, 3.14 * inch], styles, TEAL,
    )
    story.extend([
        p("6. A practical experiment plan", styles["h1"]),
        p(
            "Use the exact same channel, CTLE, ADC codes, noise realization and held-out data set to compare receiver choices. This prevents a favorable random sequence from looking like an algorithm improvement.",
            styles["body"],
        ),
        plan_table,
        Spacer(1, 0.11 * inch),
        p("Recommended reporting", styles["h2"]),
        bullet("FFE coefficients, FFE output noise variance, cursor response and effective group delay.", styles["body"]),
        bullet("DFE genie-aided and decision-directed BER separately, plus error-burst length distribution.", styles["body"]),
        bullet("MLSE target response, state count, branches per UI, traceback depth and BER.", styles["body"]),
        bullet("For RS(544,514), record erroneous 10-bit symbols per codeword, not only raw bit BER.", styles["body"]),
        p("Recommended first architecture for this model", styles["h2"]),
        p(
            "Keep the current 17-tap code-domain FFE and 12-tap DFE as the conventional baseline. Add a separate channel-shortening FFE mode and compare it with a 4-state one-memory PAM4 MLSE first. "
            "Only add a 16-state two-memory MLSE if the first experiment leaves a strong second residual cursor.", styles["body"]
        ),
        p("Beginner checklist", styles["h2"]),
        bullet("FFE first: remove precursor ISI and shape the channel without excessive noise gain.", styles["body"]),
        bullet("DFE next: remove postcursor ISI only when the feedback path is implementable.", styles["body"]),
        bullet("MLSE last: use it for intentionally retained, short channel memory - not for an arbitrarily long tail.", styles["body"]),
        bullet("Do not compare algorithms using only a noiseless eye diagram.", styles["body"]),
        bullet("For ADC-based links, include sampling phase and converter clipping before claiming DSP gain.", styles["body"]),
        Spacer(1, 0.12 * inch),
        p("End of tutorial", styles["caption"]),
    ])

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
