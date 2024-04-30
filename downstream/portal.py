import os
import time
import numpy as np
import scanpy as sc
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torch
from sklearn.decomposition import PCA, IncrementalPCA

import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import umap
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


# from downstream.portal.networks import *

class Model(object):
    def __init__(self, batch_size=500, training_steps=2000, seed=1234, npcs=30, n_latent=20, lambdacos=20.0,
                 model_path="models", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.npcs = npcs
        self.n_latent = n_latent
        self.lambdacos = lambdacos
        self.lambdaAE = 10.0
        self.lambdaLA = 10.0
        self.lambdaGAN = 1.0
        self.margin = 5.0
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path


    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   hvg_num=4000, # number of highly variable genes for each anndata
                   save_embedding=False # save low-dimensional embeddings or not
                   ):
        '''
        Performing preprocess for a pair of datasets.
        To integrate multiple datasets, use function preprocess_multiple_anndata in utils.py
        '''
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        # print("Finding highly variable genes...")
        # sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
        # sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
        # hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        # hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        # hvg_total = hvg_A & hvg_B
        # if len(hvg_total) < 100:
        #     raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))
        #
        print("Normalizing and scaling...")
        sc.pp.normalize_total(adata_A, target_sum=1e4)
        sc.pp.log1p(adata_A)
        # adata_A = adata_A[:, hvg_total]
        sc.pp.scale(adata_A, max_value=10)

        sc.pp.normalize_total(adata_B, target_sum=1e4)
        sc.pp.log1p(adata_B)
        # adata_B = adata_B[:, hvg_total]
        sc.pp.scale(adata_B, max_value=10)

        adata_total = adata_A.concatenate(adata_B, index_unique=None)

        print("Dimensionality reduction via PCA...")
        pca = PCA(n_components=self.npcs, svd_solver="arpack", random_state=0)
        adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

        self.emb_A = adata_total.obsm["X_pca"][:adata_A.shape[0], :self.npcs].copy()
        self.emb_B = adata_total.obsm["X_pca"][adata_A.shape[0]:, :self.npcs].copy()

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if save_embedding:
            np.save(os.path.join(self.data_path, "lowdim_A.npy"), self.emb_A)
            np.save(os.path.join(self.data_path, "lowdim_B.npy"), self.emb_B)


    def preprocess_memory_efficient(self, 
                                    adata_A_path, 
                                    adata_B_path, 
                                    hvg_num=4000, 
                                    chunk_size=20000,
                                    save_embedding=True
                                    ):
        '''
        Performing preprocess for a pair of datasets with efficient memory usage.
        To improve time efficiency, use a larger chunk_size.
        '''
        adata_A_input = sc.read_h5ad(adata_A_path, backed="r+", chunk_size=chunk_size)
        adata_B_input = sc.read_h5ad(adata_B_path, backed="r+", chunk_size=chunk_size)

        print("Finding highly variable genes...")
        subsample_idx_A = np.random.choice(adata_A_input.shape[0], size=np.minimum(adata_A_input.shape[0], chunk_size), replace=False)
        subsample_idx_B = np.random.choice(adata_B_input.shape[0], size=np.minimum(adata_B_input.shape[0], chunk_size), replace=False)

        adata_A_subsample = adata_A_input[subsample_idx_A].to_memory().copy()
        adata_B_subsample = adata_B_input[subsample_idx_B].to_memory().copy()

        sc.pp.highly_variable_genes(adata_A_subsample, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(adata_B_subsample, flavor='seurat_v3', n_top_genes=hvg_num)

        hvg_A = adata_A_subsample.var[adata_A_subsample.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_B = adata_B_subsample.var[adata_B_subsample.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg = hvg_A & hvg_B

        del adata_A_subsample, adata_B_subsample, subsample_idx_A, subsample_idx_B

        print("Normalizing and scaling...")
        adata_A = adata_A_input.copy(adata_A_path)
        adata_B = adata_B_input.copy(adata_B_path)

        adata_A_hvg_idx = adata_A.var.index.get_indexer(hvg)
        adata_B_hvg_idx = adata_B.var.index.get_indexer(hvg)

        mean_A = np.zeros((1, len(hvg)))
        sq_A = np.zeros((1, len(hvg)))
        mean_B = np.zeros((1, len(hvg)))
        sq_B = np.zeros((1, len(hvg)))

        for i in range(adata_A.shape[0] // chunk_size):
            X_norm = sc.pp.normalize_total(adata_A[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_A_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_A = mean_A + X_norm.sum(axis=0) / adata_A.shape[0]
            sq_A = sq_A + X_norm.power(2).sum(axis=0) / adata_A.shape[0]

        if (adata_A.shape[0] % chunk_size) > 0:
            X_norm = sc.pp.normalize_total(adata_A[(adata_A.shape[0] // chunk_size) * chunk_size: adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_A_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_A = mean_A + X_norm.sum(axis=0) / adata_A.shape[0]
            sq_A = sq_A + X_norm.power(2).sum(axis=0) / adata_A.shape[0]

        std_A = np.sqrt(sq_A - np.square(mean_A))

        for i in range(adata_B.shape[0] // chunk_size):
            X_norm = sc.pp.normalize_total(adata_B[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_B_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_B = mean_B + X_norm.sum(axis=0) / adata_B.shape[0]
            sq_B = sq_B + X_norm.power(2).sum(axis=0) / adata_B.shape[0]

        if (adata_B.shape[0] % chunk_size) > 0:
            X_norm = sc.pp.normalize_total(adata_B[(adata_B.shape[0] // chunk_size) * chunk_size: adata_B.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            X_norm = X_norm[:, adata_B_hvg_idx]
            X_norm = sc.pp.log1p(X_norm)
            mean_B = mean_B + X_norm.sum(axis=0) / adata_B.shape[0]
            sq_B = sq_B + X_norm.power(2).sum(axis=0) / adata_B.shape[0]

        std_B = np.sqrt(sq_B - np.square(mean_B))

        del X_norm, sq_A, sq_B 

        print("Dimensionality reduction via Incremental PCA...")
        ipca = IncrementalPCA(n_components=self.npcs, batch_size=chunk_size)
        total_ncells = adata_A.shape[0] + adata_B.shape[0]
        order = np.arange(total_ncells)
        np.random.RandomState(1234).shuffle(order)

        for i in range(total_ncells // chunk_size):
            idx = order[i * chunk_size : (i + 1) * chunk_size]
            idx_is_A = (idx < adata_A.shape[0])
            data_A = sc.pp.normalize_total(adata_A[idx[idx_is_A]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_A = data_A[:, adata_A_hvg_idx]
            data_A = sc.pp.log1p(data_A)
            data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
            idx_is_B = (idx >= adata_A.shape[0])
            data_B = sc.pp.normalize_total(adata_B[idx[idx_is_B] - adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_B = data_B[:, adata_B_hvg_idx]
            data_B = sc.pp.log1p(data_B)
            data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
            data = np.concatenate((data_A, data_B), axis=0)
            ipca.partial_fit(data)

        if (total_ncells % chunk_size) > 0:
            idx = order[(total_ncells // chunk_size) * chunk_size: total_ncells]
            idx_is_A = (idx < adata_A.shape[0])
            data_A = sc.pp.normalize_total(adata_A[idx[idx_is_A]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_A = data_A[:, adata_A_hvg_idx]
            data_A = sc.pp.log1p(data_A)
            data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
            idx_is_B = (idx >= adata_A.shape[0])
            data_B = sc.pp.normalize_total(adata_B[idx[idx_is_B] - adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
            data_B = data_B[:, adata_B_hvg_idx]
            data_B = sc.pp.log1p(data_B)
            data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
            data = np.concatenate((data_A, data_B), axis=0)
            ipca.partial_fit(data)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # if save_embedding:
        #     h5filename_A = os.path.join(self.data_path, "lowdim_A.h5")
        #     f = tables.open_file(h5filename_A, mode='w')
        #     atom = tables.Float64Atom()
        #     f.create_earray(f.root, 'data', atom, (0, self.npcs))
        #     f.close()
        #     # transform
        #     f = tables.open_file(h5filename_A, mode='a')
        #     for i in range(adata_A.shape[0] // chunk_size):
        #         data_A = sc.pp.normalize_total(adata_A[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
        #         data_A = data_A[:, adata_A_hvg_idx]
        #         data_A = sc.pp.log1p(data_A)
        #         data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
        #         data_A = ipca.transform(data_A)
        #         f.root.data.append(data_A)
        #     if (adata_A.shape[0] % chunk_size) > 0:
        #         data_A = sc.pp.normalize_total(adata_A[(adata_A.shape[0] // chunk_size) * chunk_size: adata_A.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
        #         data_A = data_A[:, adata_A_hvg_idx]
        #         data_A = sc.pp.log1p(data_A)
        #         data_A = np.clip((data_A - mean_A) / std_A, -10, 10)
        #         data_A = ipca.transform(data_A)
        #         f.root.data.append(data_A)
        #         f.close()
        #     del data_A
        #
        #     h5filename_B = os.path.join(self.data_path, "lowdim_B.h5")
        #     f = tables.open_file(h5filename_B, mode='w')
        #     atom = tables.Float64Atom()
        #     f.create_earray(f.root, 'data', atom, (0, self.npcs))
        #     f.close()
        #     # transform
        #     f = tables.open_file(h5filename_B, mode='a')
        #     for i in range(adata_B.shape[0] // chunk_size):
        #         data_B = sc.pp.normalize_total(adata_B[i * chunk_size: (i + 1) * chunk_size].to_memory(), target_sum=1e4, inplace=False)["X"]
        #         data_B = data_B[:, adata_B_hvg_idx]
        #         data_B = sc.pp.log1p(data_B)
        #         data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
        #         data_B = ipca.transform(data_B)
        #         f.root.data.append(data_B)
        #     if (adata_B.shape[0] % chunk_size) > 0:
        #         data_B = sc.pp.normalize_total(adata_B[(adata_B.shape[0] // chunk_size) * chunk_size: adata_B.shape[0]].to_memory(), target_sum=1e4, inplace=False)["X"]
        #         data_B = data_B[:, adata_B_hvg_idx]
        #         data_B = sc.pp.log1p(data_B)
        #         data_B = np.clip((data_B - mean_B) / std_B, -10, 10)
        #         data_B = ipca.transform(data_B)
        #         f.root.data.append(data_B)
        #         f.close()
        #     del data_B


    def train(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            # discriminator loss:
            optimizer_D.zero_grad()
            if step <= 5:
                # Warm-up
                loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
            else:
                loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # cosine correspondence:
            loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
            loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
            loss_cos = loss_cos_A + loss_cos_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            optimizer_G.zero_grad()
            if step <= 5:
                # Warm-up
                loss_G_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA))) + torch.log(1 + torch.exp(-self.D_B(x_AtoB)))).mean()
            else:
                loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin))) + torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA
            loss_G.backward()
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_cos=%f, loss_LA=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdacos*loss_cos, self.lambdaLA*loss_LA))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(), 
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def train_memory_efficient(self):
        import tables
        f_A = tables.open_file(os.path.join(self.data_path, "lowdim_A.h5"))
        f_B = tables.open_file(os.path.join(self.data_path, "lowdim_B.h5"))

        self.emb_A = np.array(f_A.root.data)
        self.emb_B = np.array(f_B.root.data)

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]
        
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.D_A = discriminator(self.npcs).to(self.device)
        self.D_B = discriminator(self.npcs).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.)
        params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        for step in range(self.training_steps):
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size, replace=False)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size, replace=False)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            # discriminator loss:
            optimizer_D.zero_grad()
            if step <= 5:
                # Warm-up
                loss_D_A = (torch.log(1 + torch.exp(-self.D_A(x_A))) + torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-self.D_B(x_B))) + torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
            else:
                loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_A), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_B), -self.margin, self.margin))) + torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # cosine correspondence:
            loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(x_A, p=2), 1)).mean()
            loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(x_B, p=2), 1)).mean()
            loss_cos = loss_cos_A + loss_cos_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            optimizer_G.zero_grad()
            if step <= 5:
                # Warm-up
                loss_G_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA))) + torch.log(1 + torch.exp(-self.D_B(x_AtoB)))).mean()
            else:
                loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin))) + torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA
            loss_G.backward()
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_cos=%f, loss_LA=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdacos*loss_cos, self.lambdaLA*loss_LA))

        f_A.close()
        f_B.close()

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(), 
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    def eval(self, D_score=False, save_results=False):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
        self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
        self.G_A = generator(self.npcs, self.n_latent).to(self.device)
        self.G_B = generator(self.npcs, self.n_latent).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
        self.G_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_A'])
        self.G_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_B'])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

        if D_score:
            self.D_A = discriminator(self.npcs).to(self.device)
            self.D_B = discriminator(self.npcs).to(self.device)
            self.D_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_A'])
            self.D_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['D_B'])

            score_D_A_A = self.D_A(x_A)
            score_D_B_A = self.D_B(x_AtoB)
            score_D_B_B = self.D_B(x_B)
            score_D_A_B = self.D_A(x_BtoA)

            self.score_Aspace = np.concatenate((score_D_A_A.detach().cpu().numpy(), score_D_A_B.detach().cpu().numpy()), axis=0)
            self.score_Bspace = np.concatenate((score_D_B_A.detach().cpu().numpy(), score_D_B_B.detach().cpu().numpy()), axis=0)

        if save_results:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)

            np.save(os.path.join(self.result_path, "latent_A.npy"), z_A.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "latent_B.npy"), z_B.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_AtoB.npy"), x_AtoB.detach().cpu().numpy())
            np.save(os.path.join(self.result_path, "x_BtoA.npy"), x_BtoA.detach().cpu().numpy())
            if D_score:
                np.save(os.path.join(self.result_path, "score_Aspace_A.npy"), score_D_A_A.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Bspace_A.npy"), score_D_B_A.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Bspace_B.npy"), score_D_B_B.detach().cpu().numpy())
                np.save(os.path.join(self.result_path, "score_Aspace_B.npy"), score_D_A_B.detach().cpu().numpy())


    # def eval_memory_efficient(self):
    #     begin_time = time.time()
    #     print("Begining time: ", time.asctime(time.localtime(begin_time)))
    #
    #     self.E_A = encoder(self.npcs, self.n_latent).to(self.device)
    #     self.E_B = encoder(self.npcs, self.n_latent).to(self.device)
    #     self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
    #     self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
    #
    #     if not os.path.exists(self.result_path):
    #         os.makedirs(self.result_path)
    #
    #     f_A = tables.open_file(os.path.join(self.data_path, "lowdim_A.h5"))
    #     f_B = tables.open_file(os.path.join(self.data_path, "lowdim_B.h5"))
    #
    #     N_A = f_A.root.data.shape[0]
    #     N_B = f_B.root.data.shape[0]
    #
    #     h5_latent_A = os.path.join(self.result_path, "latent_A.h5")
    #     f_latent_A = tables.open_file(h5_latent_A, mode='w')
    #     atom = tables.Float64Atom()
    #     f_latent_A.create_earray(f_latent_A.root, 'data', atom, (0, self.n_latent))
    #     f_latent_A.close()
    #
    #     f_latent_A = tables.open_file(h5_latent_A, mode='a')
    #     # f_x_AtoB = tables.open_file(h5_x_AtoB, mode='a')
    #     for i in range(N_A // self.batch_size):
    #         x_A = torch.from_numpy(f_A.root.data[i * self.batch_size: (i + 1) * self.batch_size]).float().to(self.device)
    #         z_A = self.E_A(x_A)
    #         f_latent_A.root.data.append(z_A.detach().cpu().numpy())
    #     if (N_A % self.batch_size) > 0:
    #         x_A = torch.from_numpy(f_A.root.data[(N_A // self.batch_size) * self.batch_size: N_A]).float().to(self.device)
    #         z_A = self.E_A(x_A)
    #         f_latent_A.root.data.append(z_A.detach().cpu().numpy())
    #         f_latent_A.close()
    #
    #     h5_latent_B = os.path.join(self.result_path, "latent_B.h5")
    #     f_latent_B = tables.open_file(h5_latent_B, mode='w')
    #     atom = tables.Float64Atom()
    #     f_latent_B.create_earray(f_latent_B.root, 'data', atom, (0, self.n_latent))
    #     f_latent_B.close()
    #
    #     f_latent_B = tables.open_file(h5_latent_B, mode='a')
    #     for i in range(N_B // self.batch_size):
    #         x_B = torch.from_numpy(f_B.root.data[i * self.batch_size: (i + 1) * self.batch_size]).float().to(self.device)
    #         z_B = self.E_B(x_B)
    #         f_latent_B.root.data.append(z_B.detach().cpu().numpy())
    #     if (N_B % self.batch_size) > 0:
    #         x_B = torch.from_numpy(f_B.root.data[(N_B // self.batch_size) * self.batch_size: N_B]).float().to(self.device)
    #         z_B = self.E_B(x_B)
    #         f_latent_B.root.data.append(z_B.detach().cpu().numpy())
    #         f_latent_B.close()
    #
    #     end_time = time.time()
    #
    #     f_A.close()
    #     f_B.close()
    #
    #     print("Ending time: ", time.asctime(time.localtime(end_time)))
    #     self.eval_time = end_time - begin_time
    #     print("Evaluating takes %.2f seconds" % self.eval_time)

class encoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        z = F.linear(h, self.W_2, self.b_2)
        return z

class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x

class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)



# Utils
def preprocess_datasets(adata_list,  # list of anndata to be integrated
                        hvg_num=4000,  # number of highly variable genes for each anndata
                        save_embedding=False,  # save low-dimensional embeddings or not
                        data_path="data"
                        ):
    if len(adata_list) < 2:
        raise ValueError("There should be at least two datasets for integration!")

    sample_size_list = []

    print("Finding highly variable genes...")
    for i, adata in enumerate(adata_list):
        sample_size_list.append(adata.shape[0])
        # adata = adata_input.copy()
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg = adata.var[adata.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvg_total = hvg
        else:
            hvg_total = hvg_total & hvg
        if len(hvg_total) < 100:
            raise ValueError(
                "The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(
                    hvg_total))

    print("Normalizing and scaling...")
    for i, adata in enumerate(adata_list):
        # adata = adata_input.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, hvg_total]
        sc.pp.scale(adata, max_value=10)
        if i == 0:
            adata_total = adata
        else:
            adata_total = adata_total.concatenate(adata, index_unique=None)

    print("Dimensionality reduction via PCA...")
    npcs = 30
    pca = PCA(n_components=npcs, svd_solver="arpack", random_state=0)
    adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

    indices = np.cumsum(sample_size_list)

    data_path = os.path.join(data_path, "preprocess")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if save_embedding:
        for i in range(len(indices)):
            if i == 0:
                np.save(os.path.join(data_path, "lowdim_1.npy"),
                        adata_total.obsm["X_pca"][:indices[0], :npcs])
            else:
                np.save(os.path.join(data_path, "lowdim_%d.npy" % (i + 1)),
                        adata_total.obsm["X_pca"][indices[i - 1]:indices[i], :npcs])

    lowdim = adata_total.obsm["X_pca"].copy()
    lowdim_list = [lowdim[:indices[0], :npcs] if i == 0 else lowdim[indices[i - 1]:indices[i], :npcs] for i in
                   range(len(indices))]

    return lowdim_list


def preprocess_recover_expression(adata_list,  # list of anndata to be integrated
                                  hvg_num=4000,  # number of highly variable genes for each anndata
                                  save_embedding=False,  # save low-dimensional embeddings or not
                                  data_path="data"
                                  ):
    if len(adata_list) < 2:
        raise ValueError("There should be at least two datasets for integration!")

    sample_size_list = []

    print("Finding highly variable genes...")
    for i, adata in enumerate(adata_list):
        sample_size_list.append(adata.shape[0])
        # adata = adata_input.copy()
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hvg_num)
        hvg = adata.var[adata.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvg_total = hvg
        else:
            hvg_total = hvg_total & hvg
        if len(hvg_total) < 100:
            raise ValueError(
                "The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(
                    hvg_total))

    print("Normalizing and scaling...")
    for i, adata in enumerate(adata_list):
        # adata = adata_input.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, hvg_total]
        sc.pp.scale(adata, max_value=10)
        if i == 0:
            adata_total = adata
            mean = adata.var["mean"]
            std = adata.var["std"]
        else:
            adata_total = adata_total.concatenate(adata, index_unique=None)

    print("Dimensionality reduction via PCA...")
    npcs = 30
    pca = PCA(n_components=npcs, svd_solver="arpack", random_state=0)
    adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

    indices = np.cumsum(sample_size_list)

    data_path = os.path.join(data_path, "preprocess")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if save_embedding:
        for i in range(len(indices)):
            if i == 0:
                np.save(os.path.join(data_path, "lowdim_1.npy"),
                        adata_total.obsm["X_pca"][:indices[0], :npcs])
            else:
                np.save(os.path.join(data_path, "lowdim_%d.npy" % (i + 1)),
                        adata_total.obsm["X_pca"][indices[i - 1]:indices[i], :npcs])

    lowdim = adata_total.obsm["X_pca"].copy()
    lowdim_list = [lowdim[:indices[0], :npcs] if i == 0 else lowdim[indices[i - 1]:indices[i], :npcs] for i in
                   range(len(indices))]

    return lowdim_list, hvg_total, mean.values.reshape(1, -1), std.values.reshape(1, -1), pca


def integrate_datasets(lowdim_list,  # list of low-dimensional representations
                       search_cos=False,  # searching for an optimal lambdacos
                       lambda_cos=20.0,
                       training_steps=2000,
                       space=None,  # None or "reference" or "latent"
                       data_path="data",
                       mixingmetric_subsample=True
                       ):
    if space == None:
        if len(lowdim_list) == 2:
            space = "latent"
        else:
            space = "reference"

    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        if isinstance(lambda_cos, float) or isinstance(lambda_cos, int):
            lambda_cos_tmp = lambda_cos

        for i in range(len(lowdim_list) - 1):

            if isinstance(lambda_cos, list):
                lambda_cos_tmp = lambda_cos[i]

            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos_tmp,
                          training_steps=training_steps,
                          data_path=os.path.join(data_path, "preprocess"),
                          model_path="models/%d_datasets" % (i + 2),
                          result_path="results/%d_datasets" % (i + 2))
            if i == 0:
                model.emb_A = lowdim_list[0]
            else:
                model.emb_A = emb_total
            model.emb_B = lowdim_list[i + 1]
            model.train()
            model.eval()
            emb_total = model.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")
    else:
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                model = Model(lambdacos=lambda_cos,
                              training_steps=training_steps,
                              data_path=os.path.join(data_path, "preprocess"),
                              model_path="models/%d_datasets" % (i + 2),
                              result_path="results/%d_datasets" % (i + 2))
                if i == 0:
                    model.emb_A = lowdim_list[0]
                else:
                    model.emb_A = emb_total
                model.emb_B = lowdim_list[i + 1]
                model.train()
                model.eval()
                meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)),
                                                 subsample=mixingmetric_subsample)
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 15.0:
                    model_opt = model
                    mixing_metric_opt = mixing
                elif mixing < mixing_metric_opt:
                    model_opt = model
                    mixing_metric_opt = mixing
            emb_total = model_opt.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model_opt.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")


def integrate_recover_expression(lowdim_list,  # list of low-dimensional representations
                                 mean, std, pca,  # information for recovering expression
                                 search_cos=False,  # searching for an optimal lambdacos
                                 lambda_cos=20.0,
                                 training_steps=2000,
                                 data_path="data",
                                 mixingmetric_subsample=True
                                 ):
    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        if isinstance(lambda_cos, float) or isinstance(lambda_cos, int):
            lambda_cos_tmp = lambda_cos

        for i in range(len(lowdim_list) - 1):

            if isinstance(lambda_cos, list):
                lambda_cos_tmp = lambda_cos[i]

            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos_tmp,
                          training_steps=training_steps,
                          data_path=os.path.join(data_path, "preprocess"),
                          model_path="models/%d_datasets" % (i + 2),
                          result_path="results/%d_datasets" % (i + 2))
            if i == 0:
                model.emb_A = lowdim_list[0]
            else:
                model.emb_A = emb_total
            model.emb_B = lowdim_list[i + 1]
            model.train()
            model.eval()
            emb_total = model.data_Aspace
    else:
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                model = Model(lambdacos=lambda_cos,
                              training_steps=training_steps,
                              data_path=os.path.join(data_path, "preprocess"),
                              model_path="models/%d_datasets" % (i + 2),
                              result_path="results/%d_datasets" % (i + 2))
                if i == 0:
                    model.emb_A = lowdim_list[0]
                else:
                    model.emb_A = emb_total
                model.emb_B = lowdim_list[i + 1]
                model.train()
                model.eval()
                meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)),
                                                 subsample=mixingmetric_subsample)
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 15.0:
                    model_opt = model
                    mixing_metric_opt = mixing
                elif mixing < mixing_metric_opt:
                    model_opt = model
                    mixing_metric_opt = mixing
            emb_total = model_opt.data_Aspace

    expression_scaled = pca.inverse_transform(emb_total)
    expression_log_normalized = expression_scaled * std + mean

    return expression_scaled, expression_log_normalized


def calculate_mixing_metric(data, meta, methods, k=5, max_k=300, subsample=True):
    if subsample:
        if data.shape[0] >= 1e4:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 10000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            meta.index = np.arange(len(subsample_idx))
    lowdim = data

    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='kd_tree').fit(lowdim)
    _, indices = nbrs.kneighbors(lowdim)
    indices = indices[:, 1:]
    mixing = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        if len(np.where(meta.method[indices[i, :]] == methods[0])[0]) > k - 1:
            mixing[i, 0] = np.where(meta.method[indices[i, :]] == methods[0])[0][k - 1]
        else:
            mixing[i, 0] = max_k - 1
        if len(np.where(meta.method[indices[i, :]] == methods[1])[0]) > k - 1:
            mixing[i, 1] = np.where(meta.method[indices[i, :]] == methods[1])[0][k - 1]
        else:
            mixing[i, 1] = max_k - 1
    return np.mean(np.median(mixing, axis=1) + 1)


def calculate_ARI(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
    # np.random.seed(1234)
    if data.shape[0] > 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno_A].astype(str)
    if (anno_B != anno_A):
        cluster_B = meta[anno_B].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)
    if (anno_B != anno_A):
        cluster_B = ro.StrVector(cluster_B)
        ro.r.assign("cluster_B", cluster_B)

    ro.r("set.seed(1234)")
    ro.r['library']("Seurat")
    ro.r['library']("mclust")

    ro.r("comb_normalized <- t(data)")
    ro.r('''rownames(comb_normalized) <- paste("gene", 1:nrow(comb_normalized), sep = "")''')
    ro.r("colnames(comb_normalized) <- as.vector(cellid)")

    ro.r("comb_raw <- matrix(0, nrow = nrow(comb_normalized), ncol = ncol(comb_normalized))")
    ro.r("rownames(comb_raw) <- rownames(comb_normalized)")
    ro.r("colnames(comb_raw) <- colnames(comb_normalized)")

    ro.r("comb <- CreateSeuratObject(comb_raw)")
    ro.r('''scunitdata <- Seurat::CreateDimReducObject(
                embeddings = t(comb_normalized),
                stdev = as.numeric(apply(comb_normalized, 2, stats::sd)),
                assay = "RNA",
                key = "scunit")''')
    ro.r('''comb[["scunit"]] <- scunitdata''')

    ro.r("comb@meta.data$method <- method")

    ro.r("comb@meta.data$cluster_A <- cluster_A")
    if (anno_B != anno_A):
        ro.r("comb@meta.data$cluster_B <- cluster_B")

    ro.r(
        '''comb <- FindNeighbors(comb, reduction = "scunit", dims = 1:ncol(data), force.recalc = TRUE, verbose = FALSE)''')
    ro.r('''comb <- FindClusters(comb, verbose = FALSE)''')

    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(comb$method == method_A)''')
        ro.r('''indx_B <- which(comb$method == method_B)''')

        ro.r("ARI_A <- adjustedRandIndex(comb$cluster_A[indx_A], comb$seurat_clusters[indx_A])")
        ro.r("ARI_B <- adjustedRandIndex(comb$cluster_B[indx_B], comb$seurat_clusters[indx_B])")
        ARI_A = np.array(ro.r("ARI_A"))[0]
        ARI_B = np.array(ro.r("ARI_B"))[0]

        return ARI_A, ARI_B
    else:
        ro.r("ARI_A <- adjustedRandIndex(comb$cluster_A, comb$seurat_clusters)")
        ARI_A = np.array(ro.r("ARI_A"))[0]

        return ARI_A


def calculate_kBET(data, meta):
    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = data.shape
    data = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", data)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)

    ro.r("set.seed(1234)")
    ro.r['library']("kBET")

    accept_rate = []
    for _ in range(100):
        ro.r("subset_id <- sample.int(n = length(method), size = 1000, replace=FALSE)")

        ro.r("batch.estimate <- kBET(data[subset_id,], method[subset_id], do.pca = FALSE, plot=FALSE)")
        accept_rate.append(np.array(ro.r("mean(batch.estimate$results$kBET.pvalue.test > 0.05)")))

    return np.median(accept_rate)


def calculate_ASW(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
    if data.shape[0] >= 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno_A].astype(str)
    if (anno_B != anno_A):
        cluster_B = meta[anno_B].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)
    if (anno_B != anno_A):
        cluster_B = ro.StrVector(cluster_B)
        ro.r.assign("cluster_B", cluster_B)

    ro.r("set.seed(1234)")
    ro.r['library']("cluster")

    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(method == method_A)''')
        ro.r('''indx_B <- which(method == method_B)''')
        ro.r(
            '''ASW_A <- summary(silhouette(as.numeric(as.factor(cluster_A[indx_A])), dist(data[indx_A, 1:20])))[["avg.width"]]''')
        ro.r(
            '''ASW_B <- summary(silhouette(as.numeric(as.factor(cluster_B[indx_B])), dist(data[indx_B, 1:20])))[["avg.width"]]''')
        ASW_A = np.array(ro.r("ASW_A"))[0]
        ASW_B = np.array(ro.r("ASW_B"))[0]

        return ASW_A, ASW_B
    else:
        ro.r('''ASW_A <- summary(silhouette(as.numeric(as.factor(cluster_A)), dist(data[, 1:20])))[["avg.width"]]''')
        ASW_A = np.array(ro.r("ASW_A"))[0]

        return ASW_A


def calculate_cellcycleconservation(data, meta, adata_raw, organism="mouse", resources_path="./cell_cycle_resources"):
    # adata
    cellid = list(meta.index.astype(str))
    geneid = ["gene_" + str(i) for i in range(data.shape[1])]
    adata = anndata.AnnData(X=data, obs=cellid, var=geneid)

    # score cell cycle
    cc_files = {'mouse': [os.path.join(resources_path, 's_genes_tirosh.txt'),
                          os.path.join(resources_path, 'g2m_genes_tirosh.txt')]}
    with open(cc_files[organism][0], "r") as f:
        s_genes = [x.strip() for x in f.readlines() if x.strip() in adata_raw.var.index]
    with open(cc_files[organism][1], "r") as f:
        g2m_genes = [x.strip() for x in f.readlines() if x.strip() in adata_raw.var.index]
    sc.tl.score_genes_cell_cycle(adata_raw, s_genes, g2m_genes)

    adata_raw.obs["method"] = meta["method"].values.astype(str)
    adata.obs["method"] = meta["method"].values.astype(str)
    batches = adata_raw.obs["method"].unique()

    scores_final = []
    scores_before = []
    scores_after = []
    for batch in batches:
        raw_sub = adata_raw[adata_raw.obs["method"] == batch]
        int_sub = adata[adata.obs["method"] == batch].copy()
        int_sub = int_sub.X

        # regression variable
        covariate_values = raw_sub.obs[['S_score', 'G2M_score']]
        if pd.api.types.is_numeric_dtype(covariate_values):
            covariate_values = np.array(covariate_values).reshape(-1, 1)
        else:
            covariate_values = pd.get_dummies(covariate_values)

        # PCR on data before integration
        n_comps = 50
        svd_solver = 'arpack'
        pca = sc.tl.pca(raw_sub.X, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver,
                        copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        before = sum(r2 * Var) / 100

        # PCR on data after integration
        n_comps = min(data.shape)
        svd_solver = 'full'
        pca = sc.tl.pca(int_sub, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver,
                        copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        after = sum(r2 * Var) / 100

        # scale result
        score = 1 - abs(before - after) / before
        if score < 0:
            score = 0
        scores_before.append(before)
        scores_after.append(after)
        scores_final.append(score)

    score_out = np.mean(scores_final)
    return score_out


def calculate_isolatedASW(data, meta, anno):
    tmp = meta[[anno, "method"]].drop_duplicates()
    batch_per_lab = tmp.groupby(anno).agg({"method": "count"})
    iso_threshold = batch_per_lab.min().tolist()[0]
    labels = batch_per_lab[batch_per_lab["method"] <= iso_threshold].index.tolist()

    scores = {}
    for label_tar in labels:
        iso_label = np.array(meta[anno] == label_tar).astype(int)
        asw = silhouette_score(
            X=data,
            labels=iso_label,
            metric='euclidean'
        )
        asw = (asw + 1) / 2
        scores[label_tar] = asw

    scores = pd.Series(scores)
    score = scores.mean()
    return score


def calculate_isolatedF1(data, meta, anno):
    if data.shape[0] > 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    tmp = meta[[anno, "method"]].drop_duplicates()
    batch_per_lab = tmp.groupby(anno).agg({"method": "count"})
    iso_threshold = batch_per_lab.min().tolist()[0]
    labels = batch_per_lab[batch_per_lab["method"] <= iso_threshold].index.tolist()

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)

    ro.r("set.seed(1234)")
    ro.r['library']("Seurat")

    ro.r("comb_normalized <- t(data)")
    ro.r('''rownames(comb_normalized) <- paste("gene", 1:nrow(comb_normalized), sep = "")''')
    ro.r("colnames(comb_normalized) <- as.vector(cellid)")

    ro.r("comb_raw <- matrix(0, nrow = nrow(comb_normalized), ncol = ncol(comb_normalized))")
    ro.r("rownames(comb_raw) <- rownames(comb_normalized)")
    ro.r("colnames(comb_raw) <- colnames(comb_normalized)")

    ro.r("comb <- CreateSeuratObject(comb_raw)")
    ro.r('''scunitdata <- Seurat::CreateDimReducObject(
                embeddings = t(comb_normalized),
                stdev = as.numeric(apply(comb_normalized, 2, stats::sd)),
                assay = "RNA",
                key = "scunit")''')
    ro.r('''comb[["scunit"]] <- scunitdata''')

    ro.r(
        '''comb <- FindNeighbors(comb, reduction = "scunit", dims = 1:ncol(data), force.recalc = TRUE, verbose = FALSE)''')
    ro.r('''comb <- FindClusters(comb, verbose = FALSE)''')

    louvain_clusters = np.array(ro.r("comb$seurat_clusters")).astype("str")
    louvain_list = list(set(louvain_clusters))

    scores = {}
    for label_tar in labels:
        max_f1 = 0
        for cluster in louvain_list:
            y_pred = louvain_clusters == cluster
            y_true = meta[anno].values.astype(str) == label_tar
            f1 = f1_score(y_pred, y_true)
            if f1 > max_f1:
                max_f1 = f1
        scores[label_tar] = max_f1

    scores = pd.Series(scores)
    score = scores.mean()
    return score


def calculate_graphconnectivity(data, meta, anno):
    cellid = list(meta.index.astype(str))
    geneid = ["gene_" + str(i) for i in range(data.shape[1])]
    adata = anndata.AnnData(X=data, obs=cellid, var=geneid)

    adata.obsm["X_emb"] = data
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_emb")

    adata.obs["anno"] = meta[anno].values.astype(str)
    anno_list = list(set(adata.obs["anno"]))

    clust_res = []

    for label in anno_list:
        adata_sub = adata[adata.obs["anno"].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp['connectivities'],
            connection='strong'
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    score = np.mean(clust_res)
    return score


def calculate_PCRbatch(data, meta, data_before=None):
    covariate_values = meta["method"]

    n_comps = min(data.shape)
    svd_solver = 'full'
    pca = sc.tl.pca(data, n_comps=n_comps, use_highly_variable=False, return_info=True, svd_solver=svd_solver,
                    copy=True)
    X_pca = pca[0].copy()
    pca_var = pca[3].copy()
    del pca

    if pd.api.types.is_numeric_dtype(covariate_values):
        covariate_values = np.array(covariate_values).reshape(-1, 1)
    else:
        covariate_values = pd.get_dummies(covariate_values)

    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = LinearRegression()
        lm.fit(covariate_values, pc)
        r2_score = np.maximum(0, lm.score(covariate_values, pc))
        r2.append(r2_score)

    Var = pca_var / sum(pca_var) * 100
    R2Var = sum(r2 * Var) / 100

    if data_before is not None:
        n_comps = 50
        svd_solver = 'arpack'
        pca = sc.tl.pca(data_before, n_comps=n_comps, use_highly_variable=False, return_info=True,
                        svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca

        r2 = []
        for i in range(n_comps):
            pc = X_pca[:, [i]]
            lm = LinearRegression()
            lm.fit(covariate_values, pc)
            r2_score = np.maximum(0, lm.score(covariate_values, pc))
            r2.append(r2_score)

        Var = pca_var / sum(pca_var) * 100
        R2Var_before = sum(r2 * Var) / 100

        score = (R2Var_before - R2Var) / R2Var_before
        return score, R2Var, R2Var_before
    else:
        return R2Var


def calculate_NMI(data, meta, anno_A="drop_subcluster", anno_B="subcluster"):
    # np.random.seed(1234)
    if data.shape[0] > 1e5:
        np.random.seed(1234)
        subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
        data = data[subsample_idx]
        meta = meta.iloc[subsample_idx]
    lowdim = data

    cellid = meta.index.astype(str)
    method = meta["method"].astype(str)
    cluster_A = meta[anno_A].astype(str)
    if (anno_B != anno_A):
        cluster_B = meta[anno_B].astype(str)

    rpy2.robjects.numpy2ri.activate()
    nr, nc = lowdim.shape
    lowdim = ro.r.matrix(lowdim, nrow=nr, ncol=nc)
    ro.r.assign("data", lowdim)
    rpy2.robjects.numpy2ri.deactivate()

    cellid = ro.StrVector(cellid)
    ro.r.assign("cellid", cellid)
    method = ro.StrVector(method)
    ro.r.assign("method", method)
    cluster_A = ro.StrVector(cluster_A)
    ro.r.assign("cluster_A", cluster_A)
    if (anno_B != anno_A):
        cluster_B = ro.StrVector(cluster_B)
        ro.r.assign("cluster_B", cluster_B)

    ro.r("set.seed(1234)")
    ro.r['library']("Seurat")

    ro.r("comb_normalized <- t(data)")
    ro.r('''rownames(comb_normalized) <- paste("gene", 1:nrow(comb_normalized), sep = "")''')
    ro.r("colnames(comb_normalized) <- as.vector(cellid)")

    ro.r("comb_raw <- matrix(0, nrow = nrow(comb_normalized), ncol = ncol(comb_normalized))")
    ro.r("rownames(comb_raw) <- rownames(comb_normalized)")
    ro.r("colnames(comb_raw) <- colnames(comb_normalized)")

    ro.r("comb <- CreateSeuratObject(comb_raw)")
    ro.r('''scunitdata <- Seurat::CreateDimReducObject(
                embeddings = t(comb_normalized),
                stdev = as.numeric(apply(comb_normalized, 2, stats::sd)),
                assay = "RNA",
                key = "scunit")''')
    ro.r('''comb[["scunit"]] <- scunitdata''')

    ro.r("comb@meta.data$method <- method")
    ro.r("comb@meta.data$cluster_A <- cluster_A")
    if (anno_B != anno_A):
        ro.r("comb@meta.data$cluster_B <- cluster_B")

    ro.r(
        '''comb <- FindNeighbors(comb, reduction = "scunit", dims = 1:ncol(data), force.recalc = TRUE, verbose = FALSE)''')
    ro.r('''comb <- FindClusters(comb, verbose = FALSE)''')

    np.random.seed(1234)
    if (anno_B != anno_A):
        method_set = pd.unique(meta["method"])
        method_A = method_set[0]
        ro.r.assign("method_A", method_A)
        method_B = method_set[1]
        ro.r.assign("method_B", method_B)
        ro.r('''indx_A <- which(comb$method == method_A)''')
        ro.r('''indx_B <- which(comb$method == method_B)''')

        # A
        louvain_A = np.array(ro.r("comb$seurat_clusters[indx_A]")).astype("str")
        cluster_A = np.array(ro.r("comb$cluster_A[indx_A]")).astype("str")
        df_A = pd.DataFrame({'louvain_A': louvain_A, 'cluster_A': cluster_A})
        df_A.louvain_A = pd.Categorical(df_A.louvain_A)
        df_A.cluster_A = pd.Categorical(df_A.cluster_A)
        df_A['louvain_code'] = df_A.louvain_A.cat.codes
        df_A['A_code'] = df_A.cluster_A.cat.codes
        NMI_A = NMI(df_A['A_code'].values, df_A['louvain_code'].values)

        # B
        louvain_B = np.array(ro.r("comb$seurat_clusters[indx_B]")).astype("str")
        cluster_B = np.array(ro.r("comb$cluster_B[indx_B]")).astype("str")
        df_B = pd.DataFrame({'louvain_B': louvain_B, 'cluster_B': cluster_B})
        df_B.louvain_B = pd.Categorical(df_B.louvain_B)
        df_B.cluster_B = pd.Categorical(df_B.cluster_B)
        df_B['louvain_code'] = df_B.louvain_B.cat.codes
        df_B['B_code'] = df_B.cluster_B.cat.codes
        NMI_B = NMI(df_B['B_code'].values, df_B['louvain_code'].values)

        return NMI_A, NMI_B
    else:
        louvain_clusters = np.array(ro.r("comb$seurat_clusters")).astype("str")
        cluster_A = np.array(ro.r("comb$cluster_A")).astype("str")

        df_fornmi = pd.DataFrame({'louvain_clusters': louvain_clusters,
                                  'cluster_A': cluster_A})
        df_fornmi.louvain_clusters = pd.Categorical(df_fornmi.louvain_clusters)
        df_fornmi.cluster_A = pd.Categorical(df_fornmi.cluster_A)
        df_fornmi['louvain_code'] = df_fornmi.louvain_clusters.cat.codes
        df_fornmi['A_code'] = df_fornmi.cluster_A.cat.codes

        NMI_A = NMI(df_fornmi['A_code'].values, df_fornmi['louvain_code'].values)
        return NMI_A


def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels


def plot_UMAP(data, meta, space="latent", score=None, colors=["method"], subsample=False,
              save=False, result_path=None, filename_suffix=None):
    if filename_suffix is not None:
        filenames = [os.path.join(result_path, "%s-%s-%s.pdf" % (space, c, filename_suffix)) for c in colors]
    else:
        filenames = [os.path.join(result_path, "%s-%s.pdf" % (space, c)) for c in colors]

    if subsample:
        if data.shape[0] >= 1e5:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            if score is not None:
                score = score[subsample_idx]

    adata = anndata.AnnData(X=data)
    adata.obs.index = meta.index
    adata.obs = pd.concat([adata.obs, meta], axis=1)
    adata.var.index = "dim-" + adata.var.index
    adata.obsm["latent"] = data

    # run UMAP
    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    embedding = reducer.fit_transform(adata.obsm["latent"])
    adata.obsm["X_umap"] = embedding

    n_cells = embedding.shape[0]
    if n_cells >= 10000:
        size = 120000 / n_cells
    else:
        size = 12

    for i, c in enumerate(colors):
        groups = sorted(set(adata.obs[c].astype(str)))
        if "nan" in groups:
            groups.remove("nan")
        palette = "rainbow"
        if save:
            fig = sc.pl.umap(adata, color=c, palette=palette, groups=groups, return_fig=True, size=size)
            fig.savefig(filenames[i], bbox_inches='tight', dpi=300)
        else:
            sc.pl.umap(adata, color=c, palette=palette, groups=groups, size=size)

    if space == "Aspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"] == method_set[1]], color="score", palette=palette, groups=groups,
                         return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"] == method_set[1]], color="margin", palette=palette, groups=groups,
                         return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)
    if space == "Bspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"] == method_set[0]], color="score", palette=palette, groups=groups,
                         return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"] == method_set[0]], color="margin", palette=palette, groups=groups,
                         return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)

