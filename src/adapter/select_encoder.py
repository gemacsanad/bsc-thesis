"""Select the GNN variant with the highest validation cosine.

Usage:
    python select_encoder.py ../../checkpoints/run11-gnn
"""

import glob
import os
import shutil
import sys

import torch


def main():
    directory = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        if os.path.basename(path) == "winner.pt":
            continue
        c = torch.load(path, map_location="cpu")
        rows.append((c["valid"]["mean_cosine"], c["valid"]["top1"], path, c))
    assert rows, f"no pretrain_encoder.py checkpoints in {directory}"

    ref = rows[0][3]
    print(f"{'variant':<24} {'epoch':>5} {'val cos':>8} {'val top1':>8} "
          f"{'eval cos':>8} {'eval top1':>9} {'out cos':>7}")
    print(f"{'plain reference mean':<24} {'':>5} {ref['valid_refs_mean']['mean_cosine']:>8.4f} "
          f"{ref['valid_refs_mean']['top1']:>8.2%} {ref['eval_refs_mean']['mean_cosine']:>8.4f} "
          f"{ref['eval_refs_mean']['top1']:>9.2%} {ref['eval_refs_mean']['output_cosine']:>7.3f}")
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    for _, _, path, c in rows:
        print(f"{os.path.basename(path)[:-3]:<24} {c['epoch']:>5} {c['valid']['mean_cosine']:>8.4f} "
              f"{c['valid']['top1']:>8.2%} {c['eval']['mean_cosine']:>8.4f} "
              f"{c['eval']['top1']:>9.2%} {c['eval']['output_cosine']:>7.3f}")

    winner = rows[0][2]
    shutil.copyfile(winner, os.path.join(directory, "winner.pt"))
    print(f"\nwinner: {winner} -> {os.path.join(directory, 'winner.pt')}")


if __name__ == "__main__":
    main()
