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

Note that it is important to make sure the gene map and nuclei mask are aligned correctly. The `check_paired.py` script can be used to check the alignment of the gene map and nuclei mask.
```bash
python run.py --gene_map YOUR_PATH/gene_map.tif \  # The gene map is a 3D image with shape (height, width, n_genes)
--nuclei_mask YOUR_PATH/nuclei_mask.tif \      # The nuclei mask is a 2D image with shape (height, width)
--log_dir ./log/LOG_NAME
```
Remember to replace `YOUR_PATH` with the path to your data.

## Example data
Here is some processed data of several datasets as examples. You can download the data and run the UCS method on them.

| Dataset                                                                                                                   | Data         | UCS segmentation  result                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------|
| Xenium Breast Cancer                                                                                                      | [Gene Map](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv), [Nuclei Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv) | [UCS Segmentation Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv) |                                                                                                                    |
| Xenium human IgAN kidney from Paper "Multiscale topology classifies cells in subcellular spatial transcriptomics", Nature | [Gene Map](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv), [Nuclei Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv)              | [UCS Segmentation Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv)                                                                                                                    |
| Vizgen Mouse Brain                                                                                                        | [Gene Map](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv), [Nuclei Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv) | [UCS Segmentation Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv) | 
| One 1200x1200 patch of Stereo-seq                                                                                         | [Gene Map](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv), [Nuclei Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv) |[UCS Segmentation Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv)                                                                                                                     |
| One FOV of NanoString CosMx Human Pancreas                                                                                | [Gene Map](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv), [Nuclei Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv)              | [UCS Segmentation Mask](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yyangaj_connect_ust_hk/EQ1Q1Z6Q1QdKv)                                                                                                                    |


## Downstream analysis
See `downstream` dir README for examples on how to run downstream analysis to reproduce most of the results in the paper.


## Acknowledgements
Thanks the contributors of [BIDCell](https://github.com/SydneyBioX/BIDCell) for their valuable resources and inspiration.