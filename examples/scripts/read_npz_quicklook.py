#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Quick viewer for NPZ outputs")
    ap.add_argument("--file", default="sim.npz")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--figdir", default="_figs")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    data = np.load(args.file, allow_pickle=True)
    print("Keys:", list(data.keys()))
    for k in data.files:
        v = data[k]
        if hasattr(v, "shape"):
            print(f"{k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{k}: {type(v)}")

    if args.save_plots:
        os.makedirs(args.figdir, exist_ok=True)

    if "power" in data:
        plt.figure(); plt.imshow(data["power"], aspect="auto", origin="lower"); plt.title("power")
        if args.save_plots: plt.savefig(os.path.join(args.figdir, "npz_power.png"), dpi=150, bbox_inches="tight")
        if args.show: plt.show()

    if "sk" in data:
        plt.figure(); plt.plot(data["sk"]); plt.title("sk (proxy)")
        if args.save_plots: plt.savefig(os.path.join(args.figdir, "npz_sk.png"), dpi=150, bbox_inches="tight")
        if args.show: plt.show()

if __name__ == "__main__":
    main()
