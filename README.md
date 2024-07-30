# UCS: a unified approach to cell segmentation for subcellular spatial transcriptomics

This repository contains the code for the UCS method, a unified approach to cell segmentation for spatially resolved transcriptomics. UCS is a deep learning-based method that can be used to segment cells in spatially resolved transcriptomics data. UCS can be used to segment cells in spatially resolved transcriptomics data from different platforms.

![Workflow](workflow.png)

## Installation
```bash
git clone https://github.com/YangLabHKUST/UCS.git
cd /path/to/UCS
conda create -n ucs python=3.9
conda activate ucs
pip install -r requirements.txt
```
Then you can run the UCS method on your data.

## Run
![Workflow](github_workflow.png)

See `scrips` dir for examples on how to run UCS on Xenium data and Vizgen data.

Actually, the only things you should provide are the gene map and a nuclei segmentation mask of the same height and width. You can obtain
the nuclei segmentation either by using Cellpose on DAPI image or directly obtain it from the platform like Vizgen or Xenium.


**Note that it is important to make sure the gene map and nuclei mask are aligned correctly. The `check_paired.py` script can be used 
to check the alignment of the gene map and nuclei mask by visualization.**
```bash
python run.py --gene_map YOUR_PATH/gene_map.tif \  # The gene map is a 3D image with shape (height, width, n_genes)
--nuclei_mask YOUR_PATH/nuclei_mask.tif \      # The nuclei mask is a 2D image with shape (height, width)
--log_dir ./log/LOG_NAME
```
Remember to replace `YOUR_PATH` with the path to your data.

## Example data
***To be updated***

Here is some processed data of several datasets as examples. You can download the data and run the UCS method on them.

| Dataset                                                                                                                   | Link                                                                                                | Data                                                                                                                |
|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Xenium Breast Cancer                                                                                                      | [Download](https://drive.google.com/drive/folders/1cQ0qVT6tyOu9_O2viHYCvaLXrZILFE2-?usp=drive_link) | Gene Map, Nuclei segmentation mask from 10X, UCS segmentaion, H&E, DAPI, Visualization                              |                                                                                                                    |
| Xenium human IgAN kidney from Paper "Multiscale topology classifies cells in subcellular spatial transcriptomics", Nature | [Download](https://drive.google.com/drive/folders/1JYp8tIWDKE0N58XZ0FKXhvqv61hRPa-y?usp=drive_link) | Gene Map, Nuclei segmentation mask from Cellpose, UCS segmentation, DAPI                                            |
| Vizgen Mouse Brain                                                                                                        | [Download](https://drive.google.com/drive/folders/19Bc-3AYILrVW2f9y7VCrWKKBW3NMDA2V?usp=drive_link) | Gene Map, Nuclei segmentation mask from Cellpose, Nuclei segmentation mask from Vizgen, UCS segmentation, DAPI                            | 
| One 1200x1200 patch of Stereo-seq                                                                                         | [Download](https://drive.google.com/drive/folders/1E2-ya-n9eCMjpZCEzj5qOs0NC_1UD9kh?usp=drive_link) | Gene Map, Nuclei segmentation mask from Cellpose, UCS segmentation, DAPI |
| One FOV of NanoString CosMx Human Pancreas                                                                                | [Download](https://drive.google.com/drive/folders/1C-FcheVxaHMaH13PSvzdeC65wPIDmZy6?usp=drive_link) | Gene Map, Nuclei segmentation mask from Cellpose, UCS segmentation |

A simple visualization script:
```python
import tifffile
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

gene_map = tifffile.imread("path/gene_map.tif")
dapi = tifffile.imread("path/dapi.tif")
seg_mask = tifffile.imread("path/segmentation_mask.tif")


gene_sum = gene_map.sum(axis=2)
boundary = find_boundaries(seg_mask, mode='inner')

x_min, x_max, y_min, y_max = 0, 1200, 0, 1200
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.axis('off')
# Background: gene or dapi
# ax.imshow(gene_sum[x_min:x_max, y_min:y_max] > 0, aspect='auto', cmap='Greys')
ax.imshow(dapi[x_min:x_max, y_min:y_max], aspect='auto', cmap='Greys')
ax.imshow(boundary[x_min:x_max, y_min:y_max], cmap='Blues', alpha=0.5)
plt.show()
```

## Downstream analysis
See `downstream` dir README for examples on how to run downstream analysis to reproduce most of the results in the paper.


## Acknowledgements
Thanks the contributors of [BIDCell](https://github.com/SydneyBioX/BIDCell) for their valuable resources and inspiration.