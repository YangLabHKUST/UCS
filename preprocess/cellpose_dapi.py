# If using Cellpose Nuclei for segmentation,
# you must make sure it is paired with the transcriptome data.
import argparse
import os
import tifffile
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dapi', type=str, required=True, help="DAPI file path")
    parser.add_argument('--out_dir', type=str, required=True, help="output file path")
    return parser.parse_args()

from cellpose import models

def segment_dapi(img, diameter=None, use_cpu=False):
    """Segment nuclei in DAPI image using Cellpose"""
    use_gpu = True if not use_cpu else False
    model = models.Cellpose(gpu=use_gpu, model_type="cyto")
    channels = [0, 0]
    mask, _, _, _ = model.eval(img, diameter=diameter, channels=channels)
    print(mask.max())
    return mask


if __name__ == "__main__":
    args = get_args()
    dapi = tifffile.imread(args.dapi)
    print(dapi.shape)
    dapi_mask = segment_dapi(dapi)
    tifffile.imsave(os.path.join(args.out_dir, "cellpose_segmentation.tif"), dapi_mask.astype(np.uint32))

