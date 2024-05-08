# Here is all the code for the analysis of the data
from utils import read_json_to_namespace, create_cell_info_csv_cuda_multi, celltype_annotation_scVI
import tifffile
import os

# Xenium Breast Cancer
config = read_json_to_namespace("./downstream/config/xenium_breast_cancer.json")
os.makedirs(config.output_dir, exist_ok=True)
gene_map = tifffile.imread(config.gene_map)
with open(config.gene_names, 'r') as f:
    gene_names = f.readlines()
gene_names = [x.strip() for x in gene_names]

for method in config.segmentation_results.keys():
    print(f"Processing method mask: {method}")
    cell_info_path = os.path.join(config.output_dir, f"{method}_cell_feature.csv")
    if not os.path.exists(config.segmentation_results[method]):
        print(f"Skip missing mask {method}")
        continue
    if os.path.exists(cell_info_path) and method not in config.reload_results:
        print(f"Skip existing csv {method}")
    else:
        if method in config.reload_results:
            print(f"Reload csv {method}")
        create_cell_info_csv_cuda_multi(gene_map, gene_names, tifffile.imread(config.segmentation_results[method]),
                                        cell_info_path)
    print(f"Finish turning mask to csv {method}")
    print(f"Method: {method}, Dataset: human_breast, Celltype annotation with scVI")
    celltype_annotation_scVI(config, method, dataset_name="human_breast", cal_ecc=True)
