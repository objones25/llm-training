#!/usr/bin/env python3
"""Generate SVG architecture diagrams for README/assets."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "io":   ("#e5e7eb", "#6b7280", "#374151"),
    "embed":("#fed7aa", "#c2410c", "#7c2d12"),
    "norm": ("#ede9fe", "#7c3aed", "#4c1d95"),
    "attn": ("#fbcfe8", "#be185d", "#831843"),
    "ff":   ("#fef08a", "#a16207", "#713f12"),
    "head": ("#bbf7d0", "#15803d", "#14532d"),
    "rope": ("#ede9fe", "#7c3aed", "#4c1d95"),
    "proj": ("#fbcfe8", "#be185d", "#831843"),
    "sdpa": ("#bfdbfe", "#2563eb", "#1e3a8a"),
}

ARROW_KW = dict(
    arrowstyle="-|>",
    color="#6b7280",
    lw=1.5,
    mutation_scale=12,
    shrinkA=0,
    shrinkB=1,
)

NODE_W = 8.0   # data-unit width (canvas is 0–10)
X_C    = 5.0   # center x
GAP    = 0.55  # gap between consecutive nodes
H1, H2, H3 = 1.1, 1.65, 2.2   # heights for 1 / 2 / 3 text lines
PAD    = 0.18  # FancyBboxPatch pad (adds to each side)


# ── helpers ───────────────────────────────────────────────────────────────────
def _node(ax, y_top, height, style, lines, bold_first=False):
    fill, stroke, text_col = COLORS[style]
    # FancyBboxPatch expands by `pad` on every side, so shrink the rect to compensate
    p = PAD
    patch = FancyBboxPatch(
        (X_C - NODE_W / 2 + p, y_top + p),
        NODE_W - 2 * p, height - 2 * p,
        boxstyle=f"round,pad={p}",
        facecolor=fill, edgecolor=stroke, linewidth=1.5, zorder=3,
    )
    ax.add_patch(patch)

    y_c = y_top + height / 2
    n = len(lines)
    spacing = height / (n + 0.4)

    for i, line in enumerate(lines):
        y_txt = y_top + spacing * (i + 0.7)
        fw = "bold" if (bold_first and i == 0) else "normal"
        fs = 10.5 if i == 0 else 9.0
        ax.text(X_C, y_txt, line,
                ha="center", va="center",
                fontsize=fs, color=text_col, fontweight=fw,
                fontfamily="DejaVu Sans", zorder=4)

    return y_top + height   # bottom edge


def _arrow(ax, y_from, y_to, x=X_C):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=ARROW_KW, zorder=2)


def _subgraph(ax, y_top, y_bot, line1, line2=None):
    p = 0.25
    x_left = X_C - NODE_W / 2 - 0.5
    w = NODE_W + 1.0
    h = y_bot - y_top
    patch = FancyBboxPatch(
        (x_left + p, y_top + p), w - 2 * p, h - 2 * p,
        boxstyle=f"round,pad={p}",
        facecolor="#dbeafe", edgecolor="#93c5fd",
        linewidth=1.5, zorder=1,
    )
    ax.add_patch(patch)
    if line2:
        ax.text(X_C, y_top + 0.27, line1,
                ha="center", va="center", fontsize=8.5,
                color="#1e40af", fontweight="bold",
                fontfamily="DejaVu Sans", zorder=2)
        ax.text(X_C, y_top + 0.55, line2,
                ha="center", va="center", fontsize=7.5,
                color="#1e40af", fontweight="normal",
                fontfamily="DejaVu Sans", zorder=2)
    else:
        ax.text(X_C, y_top + 0.35, line1,
                ha="center", va="center", fontsize=8.5,
                color="#1e40af", fontweight="bold",
                fontfamily="DejaVu Sans", zorder=2)


def _make_fig(total_h, width_in=4.3):
    height_in = total_h * (width_in / 10.0)
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(total_h, 0)   # invert: y=0 at top
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


# ── main architecture diagram ─────────────────────────────────────────────────
def make_arch(out_path: Path) -> None:
    # ----- layout pass (calculate y positions) -----
    y = 0.4

    def place(h):
        nonlocal y
        top = y; y += h + GAP
        return (top, h)

    p_in  = place(H2)
    p_em  = place(H2)

    sg_top = y - GAP + 0.3   # subgraph starts just before LN1
    y += 0.65                 # extra headroom for the two-line subgraph label

    p_ln1 = place(H1)
    p_att = place(H2)
    p_ln2 = place(H1)
    p_ff  = place(H2)

    sg_bot = y - GAP + 0.3   # subgraph ends just after FF

    y = sg_bot + 0.5          # resume below subgraph

    p_lnf = place(H1)
    p_lmh = place(H3)
    p_out = place(H2)

    total_h = y - GAP + 0.3

    # ----- render -----
    fig, ax = _make_fig(total_h)

    _subgraph(ax, sg_top, sg_bot,
              "TransformerBlock \u00d7 12",
              "pre-norm \u00b7 residual connections")

    def d(pos, style, lines, bold=False):
        return _node(ax, pos[0], pos[1], style, lines, bold)

    b_in  = d(p_in,  "io",    ["Input Token IDs", "[B, T]"])
    _arrow(ax, b_in, p_em[0])
    b_em  = d(p_em,  "embed", ["Token Embedding",
                                "vocab_size = 32 768   \u2192   d_model = 2 048"], True)
    _arrow(ax, b_em, p_ln1[0])
    b_ln1 = d(p_ln1, "norm",  ["RMSNorm  ln_1"])
    _arrow(ax, b_ln1, p_att[0])
    b_att = d(p_att, "attn",  ["CausalSelfAttention",
                                "32 heads   \u00b7   head_dim = 64"], True)
    _arrow(ax, b_att, p_ln2[0])
    b_ln2 = d(p_ln2, "norm",  ["RMSNorm  ln_2"])
    _arrow(ax, b_ln2, p_ff[0])
    b_ff  = d(p_ff,  "ff",    ["FeedForward",
                                "2 048 \u2192 8 192 \u2192 2 048   \u00b7   GELU"])
    _arrow(ax, b_ff, p_lnf[0])
    b_lnf = d(p_lnf, "norm",  ["RMSNorm   ln_f"])
    _arrow(ax, b_lnf, p_lmh[0])
    b_lmh = d(p_lmh, "head",  ["LM Head",
                                "2 048 \u2192 32 768",
                                "weight-tied to token embedding"], True)
    _arrow(ax, b_lmh, p_out[0])
    d(p_out, "io",   ["Output Logits", "[B, T, 32 768]"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── CausalSelfAttention detail diagram ────────────────────────────────────────
def make_attention(out_path: Path) -> None:
    y = 0.4

    def place(h):
        nonlocal y
        top = y; y += h + GAP
        return (top, h)

    p_xin    = place(H1)
    p_qkv    = place(H2)
    p_split  = place(H2)
    p_reshape= place(H2)
    p_rope   = place(H1)
    p_sdpa   = place(H3)
    p_reshp2 = place(H2)
    p_outprj = place(H2)
    p_xout   = place(H1)

    total_h = y - GAP + 0.3

    fig, ax = _make_fig(total_h)

    def d(pos, style, lines, bold=False):
        return _node(ax, pos[0], pos[1], style, lines, bold)

    b = d(p_xin,    "io",   ["x   [B, T, 2048]"])
    _arrow(ax, b, p_qkv[0])
    b = d(p_qkv,    "proj", ["QKV Projection",
                              "Linear 2048 \u2192 6144   \u00b7   no bias"], True)
    _arrow(ax, b, p_split[0])
    b = d(p_split,  "proj", ["Split Q, K, V",
                              "each   [B, T, 2048]"])
    _arrow(ax, b, p_reshape[0])
    b = d(p_reshape,"proj", ["Reshape + Transpose",
                              "[B, 32, T, 64]"])
    _arrow(ax, b, p_rope[0])
    b = d(p_rope,   "rope", ["Apply RoPE to Q and K"])
    _arrow(ax, b, p_sdpa[0])
    b = d(p_sdpa,   "sdpa", ["Scaled Dot-Product Attention",
                              "softmax(QK\u1d40 / \u221a64) \u00b7 V",
                              "FlashAttention-2 on CUDA"], True)
    _arrow(ax, b, p_reshp2[0])
    b = d(p_reshp2, "proj", ["Transpose + Reshape",
                              "[B, T, 2048]"])
    _arrow(ax, b, p_outprj[0])
    b = d(p_outprj, "head", ["out_proj",
                              "Linear 2048 \u2192 2048   \u00b7   no bias"], True)
    _arrow(ax, b, p_xout[0])
    d(p_xout,       "io",   ["output   [B, T, 2048]"])

    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    assets = Path(__file__).resolve().parent.parent / "assets"
    make_arch(assets / "arch.svg")
    make_attention(assets / "attention.svg")
    print("Done.")
