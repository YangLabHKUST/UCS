import matplotlib.pyplot as plt
import argparse
import tifffile
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation', type=str, required=True, help="segmentation file path")
    parser.add_argument('--gene_map', type=str, required=True, help="gene map file path")
    # Region: [x_min, x_max, y_min, y_max]
    parser.add_argument('--region', type=int, nargs=4, required=True, help="region to visualize")
    parser.add_argument('--out_dir', type=str, required=True, help="output file path")
    return parser.parse_args()

def visualize(segmentation, gene_map, region, out_dir):
    seg = tifffile.imread(segmentation)
    gene = tifffile.imread(gene_map)
    # Sum the last channel
    gene = np.log1p(gene.sum(axis=-1))
    x_min, x_max, y_min, y_max = region
    seg = seg[y_min:y_max, x_min:x_max]
    gene = gene[y_min:y_max, x_min:x_max]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(seg > 0, cmap='gray')
    ax[0].set_title("Segmentation")
    ax[1].imshow(gene, cmap='gray')
    ax[1].set_title("Gene Map")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check_paired_gene_seg.png"))
    plt.close()

if __name__ == "__main__":
    args = get_args()
    visualize(args.segmentation, args.gene_map, args.region, args.out_dir)