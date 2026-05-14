import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX fonts
        "font.family": "serif",  # Matches Latex default
        "font.serif": ["Times"],  # Times New Roman usually matches body
        "font.size": 9,  # ICML caption size is usually 9pt
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
    }
)

ICML_TEXT_WIDTH = 6.75133
ICML_COLUMN_WIDTH = 3.25063
ICML_COLUMN_HEIGHT = 4.2


# Standard Model Order
MODEL_ORDER = [
    "fbp",
    "mle",
    "unet",
    "unet_ensemble",
    "diffusion",
    "bootstrapping_fbp",
    "bootstrapping_unet",
    "bootstrapping_unet_ensemble",
    "equivariant_bootstrapping_fbp",
    "boundary",
    "skrock",
    "distance_maximization",
]

# Display Names
MODEL_NAMES = {
    "fbp": "FBP",
    "mle": "MLE",
    "unet": "U-Net",
    "unet_ensemble": "U-Net Ens.",
    "diffusion": "Diffusion",
    "bootstrapping_fbp": "FBP (Boot.)",
    "bootstrapping_unet": "U-Net (Boot.)",
    "bootstrapping_unet_ensemble": "U-Net Ens. (Boot.)",
    "equivariant_bootstrapping_fbp": "Equiv. Bootstr.",
    "boundary": "Boundary",
    "skrock": "SK-ROCK",
    "distance_maximization": "Worst-Case",
}


# Colors -- explicit per-model dict so adding new methods doesn't shift the
# colors of existing ones (which would change the accepted paper figures).
#
# Most colors are the matplotlib tab10 values that the paper already uses,
# preserved verbatim. The one change from the accepted version is `unet_ensemble`:
# previously tab:red (#d62728), now Okabe-Ito vermillion (#D55E00). Both read as
# "warm red" so the visual change is minimal, but vermillion is distinguishable
# from blue under common color-vision deficiencies (a reviewer flagged the
# original red/blue pair as indistinguishable).
#
# Order in the dict matches MODEL_ORDER for cleanliness; the dict itself is
# what get_model_colors() returns.
_MODEL_COLORS_EXPLICIT: dict[str, str] = {
    # Paper analysis-level method names -> paper colors (faithfully restored).
    # In uqct.vis.plot_uq.METHODS, 'fbp' and 'unet' represent the BOOTSTRAP
    # variants (displayed as 'FBP Bootstr.' / 'U-Net Bootstr.'), so they get
    # the paper's bootstrap colors.
    "fbp":                           "#1f77b4",  # tab:blue   -> 'FBP Bootstr.'
    "unet":                          "#2ca02c",  # tab:green  -> 'U-Net Bootstr.'
    "unet_ensemble":                 "#D55E00",  # Okabe-Ito vermillion (paper:
                                                 # tab:red; this is the one
                                                 # minimal change for colorblind
                                                 # red/blue distinguishability).
    "boundary":                      "#9467bd",  # tab:purple
    "distance_maximization":         "#17becf",  # tab:cyan   -> 'Worst-Case'
    # Other paper methods (not currently in METHODS but used elsewhere).
    "mle":                           "#ff7f0e",  # tab:orange
    "diffusion":                     "#9467bd",  # tab:purple (same hue as boundary;
                                                 # never co-plotted with boundary)
    "bootstrapping_fbp":             "#1f77b4",  # same as 'fbp' alias
    "bootstrapping_unet":            "#2ca02c",  # same as 'unet' alias
    "bootstrapping_unet_ensemble":   "#7f7f7f",  # tab:gray
    # NEW methods, chosen to be distinct from paper hues + colorblind-friendly.
    "equivariant_bootstrapping_fbp": "#8c564b",  # tab:brown
    "skrock":                        "#000000",  # black
}


def get_model_colors() -> dict[str, str]:
    """Explicit per-model color map (unaffected by additions to MODEL_ORDER)."""
    return dict(_MODEL_COLORS_EXPLICIT)


MODEL_COLORS = get_model_colors()


def get_style(model: str) -> dict[str, str]:
    """Returns dict of style kwargs for a given model (color, label)."""
    return {
        "color": MODEL_COLORS.get(model, "gray"),
        "label": MODEL_NAMES.get(model, model.title()),
    }
