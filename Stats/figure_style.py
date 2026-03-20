from __future__ import annotations

from typing import Iterable, Optional

from matplotlib.colors import LinearSegmentedColormap


DEFAULT_GROUP_COLORS = [
    "#1B6CA8",
    "#D1495B",
    "#2F7D32",
    "#C17C00",
    "#6C5B7B",
    "#008B8B",
    "#B05D1E",
    "#7A3E9D",
    "#4D6A6D",
    "#A23B72",
]

VOLCANO_UP = "#C0392B"
VOLCANO_DOWN = "#2166AC"
VOLCANO_NS = "#9AA0A6"
QC_COLOR = "#1A1A1A"

SHARED_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "shared_diverging",
    ["#2166AC", "#F7F7F7", "#B2182B"],
    N=256,
)


def get_figure_style(publication_theme: bool = False, dpi: int = 100) -> dict:
    dpi = int(dpi)
    if publication_theme:
        return {
            "dpi": dpi,
            "title_size": 22,
            "label_size": 18,
            "tick_size": 18,
            "legend_size": 18,
            "line_width": 1.1,
            "grid_width": 0.7,
            "marker_size": 42,
            "box_alpha": 0.30,
            "diverging_cmap": SHARED_DIVERGING_CMAP,
            "font_family": "Arial",
            "base_font_size": 18,
        }
    return {
        "dpi": dpi,
        "title_size": 18,
        "label_size": 16,
        "tick_size": 16,
        "legend_size": 16,
        "line_width": 1.0,
        "grid_width": 0.5,
        "marker_size": 34,
        "box_alpha": 0.35,
        "diverging_cmap": SHARED_DIVERGING_CMAP,
        "font_family": "Arial",
        "base_font_size": 16,
    }


def build_group_palette(
    groups_like: Iterable[str],
    group_colors: Optional[dict] = None,
    group_order: Optional[list[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    natural = [str(g) for g in (groups_like.tolist() if hasattr(groups_like, "tolist") else list(groups_like))]
    unique_natural = list(dict.fromkeys(natural))
    if group_order:
        order = [g for g in group_order if g in unique_natural] + [g for g in unique_natural if g not in group_order]
    else:
        order = unique_natural

    palette = {}
    fallback_order = sorted(unique_natural, key=lambda g: str(g).strip().casefold())
    fallback_index = {group: i for i, group in enumerate(fallback_order)}

    for group in order:
        group_norm = str(group).strip().lower()
        if "qc" in group_norm:
            palette[group] = QC_COLOR
        elif group_colors and group_colors.get(group):
            palette[group] = group_colors[group]
        else:
            palette[group] = DEFAULT_GROUP_COLORS[fallback_index[group] % len(DEFAULT_GROUP_COLORS)]
    return order, palette
