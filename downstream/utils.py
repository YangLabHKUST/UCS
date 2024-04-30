import json
import argparse
import os
import cv2
import scanpy as sc
import numpy as np
import pandas as pd
import tifffile
import natsort
import torch
import scvi
import anndata as ad
from scvi.model.utils import mde
from tqdm import tqdm

def read_json_to_namespace(json_path):
    with open(json_path, "r") as f:
        js = f.read()
    config = json.loads(js)
    return argparse.Namespace(**config)

def create_cell_info_csv_cuda_multi(gene_map, gene_names, mask, save_path, gpu_ids = [1, 3]):
    # Check gene_map shape
    assert gene_map.shape[0] == mask.shape[0]
    assert gene_map.shape[1] == mask.shape[1]

    gpu_num = len(gpu_ids)

    per_gpu_genes = np.linspace(0, gene_map.shape[2], gpu_num + 1).astype(int)
    gene_map_gpu = [torch.from_numpy(gene_map[:,:,per_gpu_genes[i]:per_gpu_genes[i+1]]).to(f'cuda:{gpu_ids[i]}') for i in range(gpu_num)]
    print(f"All gene chennel: {gene_map.shape[2]}")
    print(f"Gene chennel on each gpu: {[gene_map_gpu[i].shape[2] for i in range(gpu_num)]}")

    cell_mask = mask.copy()
    # Convert to int32
    cell_mask = cell_mask.astype(np.int32)
    df = pd.DataFrame(columns=[*gene_names, 'center_x', 'center_y', 'area'], index=range(1, int(cell_mask.max()) + 1))

    cell_mask_gpu = [torch.from_numpy(cell_mask).to(f'cuda:{gpu_ids[i]}') for i in range(gpu_num)]

    for cell_id in tqdm(range(1, int(cell_mask.max()) + 1)):
        mask_gpu = [cell_mask_gpu[i] == cell_id for i in range(gpu_num)]

        if mask_gpu[0].sum() == 0:
            continue
        gene_gpu = [gene_map_gpu[i][mask_gpu[i]].sum(dim=0) for i in range(gpu_num)]
        gene_sum = sum([gene_gpu[i].sum().cpu() for i in range(gpu_num)])
        if gene_sum == 0:
            continue
        gene = torch.cat([gene_gpu[i].cpu() for i in range(gpu_num)], dim=0)
        x, y = torch.where(cell_mask_gpu[0] == cell_id)
        center = (x.float().mean(), y.float().mean())
        df.loc[cell_id] = [*gene.cpu().numpy(), int(center[0].cpu().numpy()), int(center[1].cpu().numpy()),
                           int(mask_gpu[0].sum().cpu().numpy())]
    df.to_csv(save_path)

def celltype_annotation_scVI(config, method, dataset_name="human_breast", cal_ecc=True):
    output_dir = os.path.join(config.output_dir, "scVI_output", method)
    os.makedirs(output_dir, exist_ok=True)
    with open(config.gene_names, 'r') as f:
        gene_names = f.readlines()
    gene_names = [x.strip() for x in gene_names]
    # Different dataset with different preprocessing
    if dataset_name == "human_breast":
        if os.path.exists(os.path.join(output_dir, "filtered_sc.h5ad")):
            anndata_sc = sc.read_h5ad(os.path.join(output_dir, "filtered_sc.h5ad"))
            # This order is important, it will change in different run
            gene_list = list(set(gene_names) & set(anndata_sc.var_names))
        else:
            anndata_sc = sc.read_10x_h5(config.scRNA_ref)
            # Read celltype_csv: xlsx format
            celltype_df = pd.read_excel(config.celltype_csv)
            # Filtered out Annotation with hybrid celltype
            celltype_df = celltype_df[~celltype_df['Annotation'].str.contains("Hybrid")]
            print(f"Filtered out hybrid celltype")
            print(f"Total cell: {len(celltype_df)}")
            print(f"Total celltype: {celltype_df['Annotation'].unique()}")
            # Select the anndata where obs_names in df barcode
            anndata_sc = anndata_sc[anndata_sc.obs_names.isin(celltype_df['Barcode'])]
            # Celltype: celltype_df['Annotation']
            anndata_sc.obs['celltype'] = celltype_df.set_index('Barcode')['Annotation']
            # Make unique var_names
            anndata_sc.var_names_make_unique()
            # Pickup the gene both in st and scRNA
            gene_list = list(set(gene_names) & set(anndata_sc.var_names))
            # Select anndata var_names
            anndata_sc = anndata_sc[:, gene_list]
            # Save the anndata
            anndata_sc.write(os.path.join(output_dir, "filtered_sc.h5ad"))
        print("scRNA dataset loaded")
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")
    # Sort gene_list
    gene_list = natsort.natsorted(gene_list)
    # ST dataset
    # You must create CSV cell-feature file before running this function
    df = pd.read_csv(os.path.join(config.output_dir, f"{method}_cell_feature.csv"), index_col=0)
    anndata_st = ad.AnnData(X=df[gene_list].values, obs=df[['center_x', 'center_y']])
    # Var_names: gene_list
    anndata_st.var_names = gene_list
    sc.pp.filter_cells(anndata_st, min_genes=1)
    anndata_sc.obs["tech"] = "sc"
    anndata_st.obs["tech"] = "st"
    scvi.settings.seed = 0
    print("Last run with scvi-tools version:", scvi.__version__)
    adata = ad.concat([anndata_sc, anndata_st])
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full dimension safe
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="tech")
    scvi_model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    print("Training scVI")
    scvi_model.train()
    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = scvi_model.get_latent_representation()
    SCVI_MDE_KEY = "X_scVI_mde"
    adata.obsm[SCVI_MDE_KEY] = mde(adata.obsm[SCVI_LATENT_KEY])
    fig = sc.pl.embedding(
        adata,
        basis=SCVI_MDE_KEY,
        color=["tech"],
        frameon=False,
        ncols=1,
        return_fig=True
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scVI_embedding.png"))
    print("Saving embedding to scVI_embedding.png")
    # scANVI
    SCANVI_CELLTYPE_KEY = "celltype_scanvi"
    adata.obs[SCANVI_CELLTYPE_KEY] = "Unknown"
    sc_mask = adata.obs["tech"] == "sc"
    adata.obs[SCANVI_CELLTYPE_KEY][sc_mask] = anndata_sc.obs['celltype'][
        sc_mask
    ].values
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        unlabeled_category="Unknown",
        labels_key=SCANVI_CELLTYPE_KEY,
    )
    print("Training scANVI")
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)
    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTION_KEY = "C_scANVI"

    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
    adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)
    SCANVI_MDE_KEY = "X_mde_scanvi"
    adata.obsm[SCANVI_MDE_KEY] = mde(adata.obsm[SCANVI_LATENT_KEY])
    fig = sc.pl.embedding(
        adata,
        basis=SCANVI_MDE_KEY,
        color=[SCANVI_PREDICTION_KEY],
        ncols=1,
        frameon=False,
        return_fig=True
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scANVI_embedding.png"))
    print("Saving embedding to scANVI_embedding.png")
    adata.write(os.path.join(output_dir, "adata.h5ad"))
    # Spatial
    annotated_adata_st = adata[adata.obs["tech"] == "st"]
    annotated_adata_st.obsm['spatial'] = anndata_st.obs[['center_x', 'center_y']].values

    # Get shape info
    if cal_ecc:
        cell_dir = os.path.join(config.output_dir, "all_cell_mask", method)
        os.makedirs(cell_dir, exist_ok=True)
        # For all celltype, create dir
        for celltype in adata.obs[SCANVI_PREDICTION_KEY].unique():
            os.makedirs(os.path.join(cell_dir, celltype), exist_ok=True)
        ad_ = annotated_adata_st.copy()
        print("Getting eccentricity info")
        ad_.obs["elongate_x"] = np.zeros((ad_.obs.shape[0]))
        ad_.obs["elongate_y"] = np.zeros((ad_.obs.shape[0]))
        mask = tifffile.imread(config.segmentation_results[method])
        mask_gpu = torch.tensor(mask.astype("float32")).cuda()
        bar = tqdm(np.unique(ad_.obs.index))
        fail_count = 0
        for cell_id in bar:
            # Find cell type
            cell_type = ad_.obs.loc[cell_id, SCANVI_PREDICTION_KEY]
            cell_mask = mask_gpu == int(cell_id)
            # Find the sub mask
            x, y = torch.where(cell_mask)
            x_min, x_max = x.min().item(), x.max().item()
            y_min, y_max = y.min().item(), y.max().item()
            cell_mask = cell_mask[x_min - 5:x_max + 5, y_min - 5:y_max + 5].cpu().numpy()
            # Save the cell mask with cell type
            bar.set_postfix_str(f"Cell {cell_id}, Size {cell_mask.sum()}, Fail {fail_count}")
            # Find the rectangle
            try:
                contours, _ = cv2.findContours((cell_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                ellipse = cv2.fitEllipse(contours[0])
                ad_.obs.loc[cell_id, "elongate_x"] = ellipse[1][0]
                ad_.obs.loc[cell_id, "elongate_y"] = ellipse[1][1]
                tifffile.imwrite(os.path.join(cell_dir, cell_type, f"{cell_id}_{x_min-5}_{x_max + 5}_{y_min - 5}_{y_max + 5}.tif"), cell_mask.astype(np.uint8),
                                 photometric='minisblack')
            except:
                fail_count += 1

        ad_.write(os.path.join(output_dir, "annotated_adata_st_with_ecc.h5ad"))
        print("Saving annotated_adata_st with ecc info to annotated_adata_st_with_ecc.h5ad")

    # Save
    annotated_adata_st.write(os.path.join(output_dir, "annotated_adata_st.h5ad"))
    print("Saving annotated_adata_st to annotated_adata_st.h5ad")


