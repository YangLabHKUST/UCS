# Preprocess the Vizgen dataset to gene_map and nuclei_mask.
import warnings
warnings.filterwarnings("ignore")
import cv2
import natsort
import pandas as pd
import argparse
import os
import tifffile
import multiprocessing as mp
import numpy as np
import tempfile
import h5py
from shapely.geometry import Polygon
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', type=str, required=True, help="detected_transcripts.csv file path")
    parser.add_argument('--cell_boundaries', type=str, required=True, help="cell boundaries dir path, e.g MERSCOPE_ovarian_cancer/cell_boundaries/")
    parser.add_argument('--cell_meta', type=str, required=True, help="cell meta file path, e.g MERSCOPE_ovarian_cancer/cell_metadata.csv")
    parser.add_argument('--transform_matrix', type=str, required=True, help="micron_to_mosaic_pixel_transform.csv file path")
    parser.add_argument('--images', type=str, required=True, help="image file path")
    parser.add_argument('--out_dir', type=str, required=True, help="output directory")
    return parser.parse_args()

def process_gene_chunk(gene_chunk, df, map_height, map_width, temp_dir):
    for i_fe, fe in enumerate(gene_chunk):
        df_fe = df.loc[df['gene'] == fe]
        map_fe = np.zeros((map_height, map_width))

        for idx in df_fe.index:
            idx_x = np.round(df.iloc[idx]['global_x']).astype(int)
            idx_y = np.round(df.iloc[idx]['global_y']).astype(int)

            map_fe[idx_y, idx_x] += 1
        tifffile.imwrite(os.path.join(temp_dir, str(fe) + '.tif'), map_fe.astype(np.uint8),
                         photometric='minisblack')

def get_gene_map(args):
    # Load transcripts
    df_transcripts = pd.read_csv(args.transcripts, index_col=0)
    df_transcripts = df_transcripts[~df_transcripts["gene"].str.startswith("Blank")]
    df_transcripts.reset_index(inplace=True, drop=True)
    print("X range: ", df_transcripts["global_x"].max(), df_transcripts["global_x"].min())
    print("Y range: ", df_transcripts["global_y"].max(), df_transcripts["global_y"].min())
    print("Unique gene number: ", len(df_transcripts["gene"].unique()))
    transformation_matrix = pd.read_csv(args.transform_matrix, header=None, sep=' ').values
    df_transcripts.loc[:, "global_x"] = df_transcripts["global_x"] + transformation_matrix[0, 2] / \
                                        transformation_matrix[0, 0]
    df_transcripts.loc[:, "global_y"] = df_transcripts["global_y"] + transformation_matrix[1, 2] / \
                                        transformation_matrix[1, 1]
    df_transcripts.to_csv(os.path.join(args.out_dir, "transcripts_filtered.csv"))

    df = df_transcripts.copy()
    gene_names = df["gene"].unique()
    gene_names = natsort.natsorted(gene_names)
    # Write gene names to file
    with open(os.path.join(args.out_dir, 'gene_names.txt'), 'w') as f:
        for gene in gene_names:
            f.write(gene + '\n')

    # Generate expression maps
    map_width = int(np.ceil(df['global_x'].max())) + 1
    map_height = int(np.ceil(df['global_y'].max())) + 1
    print(f"Map shape: ({map_height}, {map_width})")

    print("Converting to expression maps")
    n_processes = mp.cpu_count()
    gene_names_chunks = np.array_split(gene_names, n_processes)
    processes = []
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
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

def get_vigzen_polygon(fov, cell_boundary_h5_path, cell_meta_path, transform_mat_path, z_index=2):
    cellBoundaries = h5py.File(cell_boundary_h5_path)
    meta_cell = pd.read_csv(cell_meta_path,
                            index_col=0)
    meta_cell = meta_cell[meta_cell["fov"] == fov]
    z_index = 'zIndex_' + str(z_index)
    transformation_matrix = pd.read_csv(transform_mat_path,
                                        header=None, sep=' ').values
    currentCellsNormalized = []
    for inst_cell in meta_cell.index.tolist():
        # If p_0 exist
        if 'p_0' not in cellBoundaries['featuredata'][str(inst_cell)][z_index]:
            continue
        temp = cellBoundaries['featuredata'][str(inst_cell)][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1]))
        boundaryPolygon[:, :] = temp
        boundaryPolygon[:, 0] += (transformation_matrix[0, 2] / transformation_matrix[0, 0])
        boundaryPolygon[:, 1] += (transformation_matrix[1, 2] / transformation_matrix[1, 1])
        currentCellsNormalized.append(boundaryPolygon)
    x_max = 0
    y_max = 0
    x_min = np.inf
    y_min = np.inf
    for boundaryPolygon in currentCellsNormalized:
        x_max = max(x_max, np.max(boundaryPolygon[:, 0]))
        y_max = max(y_max, np.max(boundaryPolygon[:, 1]))
        x_min = min(x_min, np.min(boundaryPolygon[:, 0]))
        y_min = min(y_min, np.min(boundaryPolygon[:, 1]))

    x_min = int(np.floor(x_min))
    y_min = int(np.floor(y_min))
    x_max = int(np.ceil(x_max))+1
    y_max = int(np.ceil(y_max))+1

    polygon_list = []
    for inst_index in range(len(currentCellsNormalized)):
        inst_cell = currentCellsNormalized[inst_index]
        polygon = Polygon(inst_cell)
        polygon_list.append(polygon)

    return polygon_list, x_min, x_max, y_min, y_max

def get_vizgen_segmentation(args):
    gene_map = tifffile.imread(os.path.join(args.out_dir, "gene_map.tif"))
    mask_all = np.zeros((gene_map.shape[0], gene_map.shape[1]))
    cell_meta_path = args.cell_meta
    transform_mat_path = args.transform_matrix
    all_boundary_files = os.listdir(args.cell_boundaries)

    cell_count = 1
    for f in tqdm(all_boundary_files):
        fov = int(f.split('_')[-1].split('.')[0])
        cell_boundary_h5_path = os.path.join(args.cell_boundaries, f)
        try:
            polygon_list, x_min, x_max, y_min, y_max = get_vigzen_polygon(fov, cell_boundary_h5_path, cell_meta_path,
                                                                          transform_mat_path)
        except Exception as e:
            print(f + " failed: " + str(e))
            continue
        for polygon in polygon_list:
            points = [[x, y] for x, y in zip(*polygon.boundary.coords.xy)]
            mask_all = cv2.fillPoly(mask_all, np.array([points]).astype(np.int32), color=cell_count)
            cell_count += 1

    # Save
    mask_all = mask_all.astype(np.uint32)
    tifffile.imsave(os.path.join(args.out_dir, "cell_vizgen_mask.tif"), mask_all)

def prepare_dapi_for_cellpose(args):
    z_list = ["z0", "z1", "z2", "z3", "z4", "z5", "z6"]
    dapi_0 = tifffile.imread(os.path.join(args.images, "mosaic_DAPI_z0.tif"))
    for z in z_list[1:]:
        dapi = tifffile.imread(os.path.join(args.images, f"mosaic_DAPI_{z}.tif"))
        dapi_0 = np.maximum(dapi_0, dapi)
    # Scale
    dapi = dapi_0
    transformation_matrix = pd.read_csv(args.transform_matrix, header=None, sep=' ').values
    shape_0 = int(np.ceil(dapi.shape[0] / transformation_matrix[1, 1]))
    shape_1 = int(np.ceil(dapi.shape[1] / transformation_matrix[0, 0]))
    resized = cv2.resize(dapi_0, (shape_1, shape_0), interpolation = cv2.INTER_AREA)
    gene_map = tifffile.imread(os.path.join(args.out_dir, "gene_map.tif"))
    shape_0, shape_1 = gene_map.shape[0], gene_map.shape[1]
    resized = resized[:shape_0, :shape_1]
    # Save
    tifffile.imwrite(os.path.join(args.out_dir, "mosaic_DAPI_mip_resized.tif"), resized)


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    get_gene_map(args)
    get_vizgen_segmentation(args)
    prepare_dapi_for_cellpose(args)
