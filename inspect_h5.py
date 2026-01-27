import h5py
import sys


def inspect_h5(path):
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print("Keys:", list(f.keys()))
        for k in f.keys():
            print(f"  {k}: {f[k].shape}")
            # If attrs exist
            if len(f[k].attrs) > 0:
                print(f"    Attrs: {list(f[k].attrs.keys())}")


if __name__ == "__main__":
    inspect_h5(sys.argv[1])
