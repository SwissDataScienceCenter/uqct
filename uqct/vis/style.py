import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Standard Model Order
MODEL_ORDER = ["diffusion", "fbp", "map", "mle", "unet"]

# Display Names
MODEL_NAMES = {
    "diffusion": "Diffusion",
    "fbp": "FBP",
    "map": "MAP",
    "mle": "MLE",
    "unet": "U-Net",
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
