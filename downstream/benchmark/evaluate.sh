 python benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/10X_cell_mask.tif --seg_path /import/home2/yhchenmath/Log/CellSeg/UCS_human_lung_calibration_10X/predict/0/pred/epoch_0_step_0_connected.tif
 python benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/10X_cell_mask.tif --seg_path /import/home2/yhchenmath/Log/CellSeg/UCS_human_lung_default_10X/predict/0/pred/epoch_0_step_0_connected.tif
 python benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/10X_cell_mask.tif --seg_path  /import/home2/yhchenmath/Log/CellSeg/UCS_human_lung_default/predict/0/pred/epoch_0_step_0_connected.tif
 python benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/10X_cell_mask.tif --seg_path  /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/baysor_segmentation_mask_resized.tif
 python benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/10X_cell_mask.tif --seg_path  /import/home2/yhchenmath/Log/CellSeg/result_collection/xenium_human_lung/voroni_mask.tif


python downstream/benchmark/f1_score.py --gt_path /import/home2/yhchenmath/Log/CellSeg/UCS_merfish_mouse_brain_default_vigzen/predict/0/pred/epoch_0_step_0_connected.tif --seg_path  /import/home2/yhchenmath/Code/ucs/log/vizgen_mouse_brain/pred/segmentation_mask.tif
