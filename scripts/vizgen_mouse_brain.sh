# Download Vizgen Mouse Brain S2R1 from https://console.cloud.google.com/storage/browser/public-datasets-vizgen-merfish/

# Run the preprocess to get the gene map and vizgen segmentation
python preprocess/vizgen.py \
--transcripts /import/home2/yhchenmath/Dataset/CellSeg/MERFISH_mouse_brain/detected_transcripts_S2R1.csv \
--cell_boundaries /import/home2/yhchenmath/Dataset/CellSeg/MERFISH_mouse_brain/cell_boundaries/ \
--cell_meta /import/home2/yhchenmath/Dataset/CellSeg/MERFISH_mouse_brain/cell_metadata_S2R1.csv \
--transform_matrix /import/home2/yhchenmath/Dataset/CellSeg/MERFISH_mouse_brain/images/micron_to_mosaic_pixel_transform.csv \
--images /import/home2/yhchenmath/Dataset/CellSeg/MERFISH_mouse_brain/images/ \
--out_dir ./data/vizgen_mouse_brain

# You can also use the Cellpose Nuclei for UCS with the Maximum Intensity Projection (MIP) of DAPI channel
python preprocess/cellpose_dapi.py --dapi ./data/vizgen_mouse_brain/mosaic_DAPI_mip_resized.tif --out_dir ./data/vizgen_mouse_brain


# ========================================================================================================
# [Optional]Check the gene map and nuclei mask by visualizing the region [x_min, x_max, y_min, y_max]
python preprocess/check_paired.py \
--gene_map ./data/vizgen_mouse_brain/gene_map.tif \
--segmentation ./data/vizgen_mouse_brain/cell_vizgen_mask.tif \
--region 2000 3000 2000 3000 \
--out_dir ./data/vizgen_mouse_brain

# Run UCS on Vizgen Mouse Brain dataset
python run.py --gene_map ./data/vizgen_mouse_brain/gene_map.tif \
--nuclei_mask ./data/vizgen_mouse_brain/cell_vizgen_mask.tif \
--log_dir ./log/vizgen_mouse_brain \
--fg_net_batch_size 200


# ========================================================================================================
# If you want to use the Cellpose Nuclei for UCS, you can run the following commands:
# [Optional]Check
python preprocess/check_paired.py \
--gene_map ./data/vizgen_mouse_brain/gene_map.tif \
--segmentation ./data/vizgen_mouse_brain/cellpose_segmentation.tif \
--region 2000 3000 2000 3000 \
--out_dir ./data/vizgen_mouse_brain
# Run UCS
python run.py \
--gene_map ./data/vizgen_mouse_brain/gene_map.tif \
--nuclei_mask ./data/vizgen_mouse_brain/cellpose_segmentation.tif \
--log_dir ./log/vizgen_mouse_brain_cellpose

