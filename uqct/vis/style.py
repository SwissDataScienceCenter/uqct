import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX fonts
        "font.family": "serif",  # Matches Latex default
        "font.serif": ["Times"],  # Times New Roman usually matches body
        "font.size": 9,  # ICML caption size is usually 9pt
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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
    "boundary",
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
    "boundary": "Boundary",
    "distance_maximization": "Worst-Case",
}


# Colors
# We derive them from the default property cycle (usually tab10)
# to match what plotting functions do when iterating sorted models.
def get_model_colors() -> dict[str, str]:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Map purely by index in sorted list
    model_colors = {}
    for i, m in enumerate(MODEL_ORDER):
        model_colors[m] = colors[i % len(colors)]
    return model_colors


MODEL_COLORS = get_model_colors()


def get_style(model: str) -> dict[str, str]:
    """Returns dict of style kwargs for a given model (color, label)."""
    return {
        "color": MODEL_COLORS.get(model, "gray"),
        "label": MODEL_NAMES.get(model, model.title()),
    }
