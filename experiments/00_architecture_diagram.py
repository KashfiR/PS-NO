"""Draw the spectral operator diagram used in the paper.

    python experiments/00_architecture_diagram.py
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

NAVY, OCHRE, SAGE, BRICK, STONE = "#1B3A5C", "#A87225", "#46654F", "#7E3B36", "#4C4A45"
INK = "#1A1A17"


def block(ax, x, y, w, h, lines, colour, big=14, small=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                fc=colour, ec="none", zorder=3))
    for i, (text, bold) in enumerate(lines):
        ax.text(x + w / 2, y + h * (1 - (i + 0.5) / len(lines)), text, ha="center",
                va="center", zorder=4, color="w",
                fontsize=big if bold else small, weight="bold" if bold else "normal")


def hollow(ax, x, y, w, h, lines, colour, big=14, small=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                fc="white", ec=colour, lw=1.9, zorder=3))
    for i, (text, bold) in enumerate(lines):
        ax.text(x + w / 2, y + h * (1 - (i + 0.5) / len(lines)), text, ha="center",
                va="center", zorder=4, color=colour,
                fontsize=big if bold else small, weight="bold" if bold else "normal")


def arrow(ax, p1, p2, colour=INK, lw=1.9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                 color=colour, lw=lw, shrinkA=0, shrinkB=0, zorder=5))


def elbow(ax, points, colour=INK, lw=1.9):
    for a, b in zip(points[:-1], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=lw, zorder=5,
                solid_capstyle="round")
    arrow(ax, points[-2], points[-1], colour, lw)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(15.0, 7.2))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.text(6, 68.4, "Regulatory network", fontsize=14.5, weight="bold", color=SAGE)
    ax.text(6, 65.2, "built once from prior knowledge, then held fixed",
            fontsize=11.5, color=SAGE, style="italic")

    block(ax, 6, 53, 35, 9, [("CollecTRI subnetwork", True), ("64 genes, 241 edges", False)],
          SAGE, 13.5, 11)
    arrow(ax, (41.3, 57.5), (48.7, 57.5))
    block(ax, 49, 53, 34, 9, [("Graph Laplacian", True),
                              ("L = I \u2212 D$^{-1/2}$AD$^{-1/2}$", False)], SAGE, 13.5, 11)
    arrow(ax, (83.3, 57.5), (90.7, 57.5))
    block(ax, 91, 53, 34, 9, [("Eigenbasis", True),
                              ("\u03a6$_k$ (64\u00d732),   \u03bb \u2208 [0, 2]", False)],
          SAGE, 13.5, 11)

    elbow(ax, [(99, 52.8), (99, 49.5), (32, 49.5), (32, 33.4)], SAGE)
    ax.text(34.2, 46.4, "\u03a6", fontsize=13, color=SAGE, weight="bold")
    elbow(ax, [(117, 52.8), (117, 45.5), (60, 45.5), (60, 41.6)], SAGE)
    ax.text(62.2, 46.6, "\u03bb", fontsize=13, color=SAGE, weight="bold")

    ax.add_patch(FancyBboxPatch((19, 4.5), 100, 38, boxstyle="round,pad=0,rounding_size=1",
                                fc="#F2F1EC", ec=NAVY, lw=1.5, zorder=1))
    ax.text(21.5, 9.6, "Spectral block, repeated three times", fontsize=13.5,
            weight="bold", color=NAVY)
    ax.text(21.5, 6.6, "all weights inside are learned", fontsize=11.5, color=NAVY,
            style="italic")

    hollow(ax, 45, 33, 30, 8.5, [("Filter  R(\u03bb)", True),
                                 ("a small network reads \u03bb", False)], OCHRE)
    elbow(ax, [(60, 32.8), (60, 29.7)], OCHRE)

    block(ax, 23, 19, 22, 10.5, [("Project onto", False), ("eigenvectors", True),
                                 ("\u03a6$^{T}$v", False)], NAVY, 13.5, 10.5)
    arrow(ax, (45.3, 24.2), (48.7, 24.2))
    block(ax, 49, 19, 22, 10.5, [("Scale each", False), ("mode", True),
                                 ("R(\u03bb) v\u0302", False)], OCHRE, 13.5, 10.5)
    arrow(ax, (71.3, 24.2), (74.7, 24.2))
    block(ax, 75, 19, 22, 10.5, [("Project back", False), ("to genes", True),
                                 ("\u03a6 v\u0302", False)], NAVY, 13.5, 10.5)
    arrow(ax, (97.3, 24.2), (100.7, 24.2))
    block(ax, 101, 19, 15, 10.5, [("Combine", True), ("GELU + LN", False)], NAVY, 13.5, 10.5)

    block(ax, 49, 11, 22, 7.5, [("Per-gene mixing", True), ("W v", False)], BRICK, 12.5, 10.5)
    ax.add_patch(Circle((21.3, 24.2), 0.75, fc=INK, ec="none", zorder=6))
    elbow(ax, [(21.3, 24.2), (21.3, 14.7), (48.7, 14.7)])
    elbow(ax, [(71, 14.7), (108.5, 14.7), (108.5, 18.7)])

    block(ax, 2, 19, 15, 10.5, [("Input data", True), ("64 genes \u00d7 5", False),
                                ("x\u2080, k, \u03b3, dose", False)], STONE, 13.5, 10.5)
    arrow(ax, (17.3, 24.2), (20.4, 24.2))
    arrow(ax, (119.3, 24.2), (122.7, 24.2))
    block(ax, 123, 19, 12, 10.5, [("Head", True), ("32\u2192128\u21921", False)],
          STONE, 13.5, 10.5)
    arrow(ax, (135.3, 24.2), (138.4, 24.2))
    block(ax, 138.7, 19, 10, 10.5, [("Output", True), ("\u0177", False)], BRICK, 13.5, 15)
    ax.text(143.7, 16.2, "steady state", fontsize=11, color=BRICK, ha="center", style="italic")

    plt.savefig(C.ensure(C.path("figures/architecture.png")), dpi=210,
                bbox_inches="tight", facecolor="white")
    print("wrote figures/architecture.png")


if __name__ == "__main__":
    main()
