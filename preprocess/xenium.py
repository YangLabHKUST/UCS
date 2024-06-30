# Preprocess the Xenium dataset to gene_map and nuclei_mask.
import cv2
import natsort
import pandas as pd
import argparse
import os
import tifffile
import multiprocessing as mp
import numpy as np
import tempfile

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', type=str, required=True, help="transcripts file path")
    parser.add_argument('--cell_boundary_10X', type=str, required=True, help="cell boundary 10X file path (cell_boundaries.parquet)")
    parser.add_argument('--nucleus_boundary_10X', type=str, required=True, help="nucleus boundary 10X file path (nucleus_boundaries.parquet)")
    parser.add_argument('--out_dir', type=str, required=True, help="output directory")
    # Default values
    parser.add_argument('--min_qv', type=int, default=20, help="minimum qv value")

    return parser.parse_args()

def process_gene_chunk(gene_chunk, df, map_height, map_width, temp_dir):
    for i_fe, fe in enumerate(gene_chunk):
        df_fe = df.loc[df['feature_name'] == fe]
        map_fe = np.zeros((map_height, map_width))

        for idx in df_fe.index:
            idx_x = np.round(df.iloc[idx]['x_location']).astype(int)
            idx_y = np.round(df.iloc[idx]['y_location']).astype(int)
            # idx_qv = df.iloc[idx]['qv']

            map_fe[idx_y, idx_x] += 1
        tifffile.imwrite(os.path.join(temp_dir, str(fe) + '.tif'), map_fe.astype(np.uint8),
                         photometric='minisblack')

def get_gene_map(args):
    # Load transcripts
    print("======> Loading transcripts file")
    # Load transcripts
    df = pd.read_csv(args.transcripts)
    print("Filtering transcripts")
    df = df[(df["qv"] >= args.min_qv) &
            (~df["feature_name"].str.startswith("NegControlProbe_")) &
            (~df["feature_name"].str.startswith("antisense_")) &
            (~df["feature_name"].str.startswith("NegControlCodeword_")) &
            (~df["feature_name"].str.startswith("BLANK_"))]
    df.reset_index(inplace=True, drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, 'transcripts_filtered.csv'))
    gene_names = df['feature_name'].unique()
    print('%d unique genes' % len(gene_names))
    gene_names = natsort.natsorted(gene_names)
    # Write gene names to file
    with open(os.path.join(args.out_dir, 'gene_names.txt'), 'w') as f:
        for gene in gene_names:
            f.write(gene + '\n')

    # Generate expression maps
    map_width = int(np.ceil(df['x_location'].max())) + 1
    map_height = int(np.ceil(df['y_location'].max())) + 1
    print(f"Map shape: ({map_height}, {map_width})")

    print("Converting to expression maps")
    n_processes = mp.cpu_count()
    gene_names_chunks = np.array_split(gene_names, n_processes)
    processes = []
    # Create temp dir
    temp_dir = tempfile.mkdtemp(dir="/import/home2/yhchenmath/Dataset/CellSeg/GSE264334_RAW/1")
    for gene_chunk in gene_names_chunks:
        p = mp.Process(target=process_gene_chunk, args=(gene_chunk, df,
                                                             map_height, map_width, temp_dir))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    # Combine channel-wise
    map_all_genes = np.zeros((map_height, map_width, len(gene_names)), dtype=np.uint8)
    for i_fe, fe in enumerate(tqdm(gene_names)):
        map_all_genes[:, :, i_fe] = tifffile.imread(os.path.join(temp_dir, str(fe) + '.tif'))
    # Save the combined map
    tifffile.imwrite(os.path.join(args.out_dir, "gene_map.tif"), map_all_genes)
    print("Saved gene map")

def get_10X_segmentation(args):
    if not os.path.exists(args.cell_boundary_10X):
        print("10X Cell boundary file not found")
        return
    else:
        cell_boundaries = pd.read_parquet(args.cell_boundary_10X)
        nucleus_boundaries = pd.read_parquet(args.nucleus_boundary_10X)

    # Size: equals to gene_map
    gene_map = tifffile.imread(os.path.join(args.out_dir, "gene_map.tif"))
    height, width, _ = gene_map.shape

    # All cell_id
    cell_ids = cell_boundaries['cell_id'].unique()

    nuclei_mask_whole = np.zeros((height, width))
    cell_mask_whole = np.zeros((height, width))
    bar = tqdm(enumerate(cell_ids), total=len(cell_ids))

    for m, id in bar:
        mask_id = m + 1
        single_nuclei_boundary = nucleus_boundaries[nucleus_boundaries['cell_id'] == id]
        single_nuclei_boundary = single_nuclei_boundary.reset_index(drop=True)
        single_cell_boundary = cell_boundaries[cell_boundaries['cell_id'] == id]
        single_cell_boundary = single_cell_boundary.reset_index(drop=True)
        if len(single_nuclei_boundary) > 0:
            # Detect how many nuclei in one cell
            df_nb = single_nuclei_boundary.copy()
            # Find none unique coordinates and its index
            df_nb = df_nb[df_nb.duplicated(subset=['vertex_x', 'vertex_y'], keep=False)]
            poly_num = int(len(df_nb) / 2)

            nuclei_poly_list = []
            for i in range(poly_num):
                poly_index = (df_nb.index[i * 2], df_nb.index[i * 2 + 1])
                df_poly = single_nuclei_boundary.loc[poly_index[0]:poly_index[1]]
                x = df_poly.vertex_x
                y = df_poly.vertex_y
                polygons = [list(zip(x, y))]
                polygons = np.array(polygons, 'int32')
                nuclei_poly_list.append(polygons)

            cv2.fillPoly(nuclei_mask_whole, nuclei_poly_list, mask_id)

        x = single_cell_boundary.vertex_x
        y = single_cell_boundary.vertex_y
        polygons = [list(zip(x, y))]
        polygons = np.array(polygons, 'int32')
        cv2.fillPoly(cell_mask_whole, polygons, mask_id)

    # Save
    tifffile.imsave(os.path.join(args.out_dir, "nuclei_10X_mask.tif"), nuclei_mask_whole.astype(np.uint32))
    tifffile.imsave(os.path.join(args.out_dir, "cell_10X_mask.tif"), cell_mask_whole.astype(np.uint32))

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    get_gene_map(args)
    get_10X_segmentation(args)

