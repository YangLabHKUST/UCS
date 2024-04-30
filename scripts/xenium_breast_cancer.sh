# Download 10X Xenium Breast Cancer Dataset from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast

# Run the preprocess to get the gene map and official nuclei mask
python preprocess/xenium.py \
--transcripts /import/home2/yhchenmath/Dataset/CellSeg/TestSeg/Xenium-BreastCancer1/outs/transcripts.csv \
--cell_boundary_10X /import/home2/yhchenmath/Dataset/CellSeg/TestSeg/Xenium-BreastCancer1/outs/cell_boundaries.parquet \
--nucleus_boundary_10X /import/home2/yhchenmath/Dataset/CellSeg/TestSeg/Xenium-BreastCancer1/outs/nucleus_boundaries.parquet \
--out_dir ./data/xenium_breast_cancer

# [Optional]Check the gene map and nuclei mask by visualizing the region [x_min, x_max, y_min, y_max]
python preprocess/check_paired.py \
--gene_map ./data/xenium_breast_cancer/gene_map.tif \
--segmentation ./data/xenium_breast_cancer/nuclei_10X_mask.tif \
--region 2000 3000 2000 3000 \
--out_dir ./data/xenium_breast_cancer

# Run UCS on Xenium Breast Cancer dataset
python run.py --gene_map ./data/xenium_breast_cancer/gene_map.tif \
--nuclei_mask ./data/xenium_breast_cancer/nuclei_10X_mask.tif \
--log_dir ./log/xenium_breast_cancer


