import tomllib
from pathlib import Path


def verify():
    settings_path = Path("uqct/settings.toml")
    section = "eval-sparse"

    print(f"Reading {settings_path}...")
    with open(settings_path, "rb") as f:
        full_config = tomllib.load(f)

    print(f"Full config keys: {list(full_config.keys())}")

    # Simulate the logic in cli.py
    settings = full_config[section].copy()

    print(f"Settings (before merge) keys: {list(settings.keys())}")

    if "eval" in full_config:
        print("Merging 'eval' section...")
        settings.update(full_config["eval"])

    print(f"Settings (after merge) keys: {list(settings.keys())}")

    # Check for specific model keys
    models = ["mle", "map", "unet", "diffusion"]
    for m in models:
        if m in settings:
            print(f"SUCCESS: '{m}' found in settings: {settings[m]}")
        else:
            print(f"FAILURE: '{m}' NOT found in settings")


if __name__ == "__main__":
    verify()
