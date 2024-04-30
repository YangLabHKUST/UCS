import os
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data as data
import tifffile

class UCSDataset(data.Dataset):
    def __init__(self, manager, shift_patches=0):
        self.manager = manager
        # opt, logger, log_dir
        self.opt = manager.get_opt()
        self.logger = manager.get_logger()
        self.log_dir = manager.get_log_dir()
        # Gene Map data, Nuclei Mask data
        self.logger.info(f"Loading gene map from {self.opt.gene_map}")
        self.gene_map = tifffile.imread(self.opt.gene_map)
        self.logger.info(f"Gene map shape: {self.gene_map.shape}")
        self.logger.info(f"Loading nuclei mask from {self.opt.nuclei_mask}")
        self.nuclei_mask = tifffile.imread(self.opt.nuclei_mask)
        self.nuclei_mask = self.nuclei_mask.astype(np.int32)
        self.logger.info(f"Nuclei mask unique values: {len(np.unique(self.nuclei_mask))}, Shape: {self.nuclei_mask.shape}")
        # Assert the shape of gene_map and nuclei_mask_fp
        assert self.gene_map.shape[:2] == self.nuclei_mask.shape, "Gene map and nuclei mask shape mismatch"
        self.patch_size = self.opt.patch_size

        if shift_patches == 0:
            h_starts = list(np.arange(0, self.gene_map.shape[0] - self.patch_size, self.patch_size))
            w_starts = list(np.arange(0, self.gene_map.shape[1] - self.patch_size, self.patch_size))
            h_starts.append(self.gene_map.shape[0] - self.patch_size)
            w_starts.append(self.gene_map.shape[1] - self.patch_size)
        else:
            h_starts = list(np.arange(shift_patches, self.gene_map.shape[0] - self.patch_size, self.patch_size))
            w_starts = list(np.arange(shift_patches, self.gene_map.shape[1] - self.patch_size, self.patch_size))

        self.coords_starts = [(x, y) for x in h_starts for y in w_starts]
        self.logger.info(f"Total {len(self.coords_starts)} patches with shift {shift_patches}")

    def __len__(self):
        return len(self.coords_starts)

    def set_train(self, is_train):
        self.is_train = is_train

    def augment_data(self, batch_raw):
        batch_raw = np.expand_dims(batch_raw, 0)
        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]

        # Flips
        if random_flip == 0:
            batch_flip = batch_raw * 1
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)

        # Rotations
        if random_rotate == 0:
            batch_rotate = batch_flip * 1
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)

        images_aug_array = np.array(batch_rotate)

        return images_aug_array, random_flip, random_rotate

    def get_gene_num(self):
        return self.gene_map.shape[-1]

    def get_nuclei_shape(self):
        return self.nuclei_mask.shape

    def __getitem__(self, index):
        coords = self.coords_starts[index]
        coords_h1 = coords[0]
        coords_w1 = coords[1]
        coords_h2 = coords_h1 + self.patch_size
        coords_w2 = coords_w1 + self.patch_size

        expr = self.gene_map[coords_h1:coords_h2, coords_w1:coords_w2]
        nucl = self.nuclei_mask[coords_h1:coords_h2, coords_w1:coords_w2]

        # Append the nuclei to the last channel
        img = np.concatenate((expr, np.expand_dims(nucl, -1)), -1)

        if self.is_train:
            img, _, _ = self.augment_data(img)
            img = img[0, :, :, :]

        expr_aug = img[:, :, :-1]
        nucl_aug = img[:, :, -1]

        mask = np.where(nucl_aug > 0, 1, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.opt.dilation_kernel_size, self.opt.dilation_kernel_size))
        bg_mask = (1 - cv2.dilate(mask.astype('uint8'), kernel, iterations=self.opt.dilation_iter_num))

        expr_torch = torch.from_numpy(expr_aug).float()
        nucl_torch = torch.from_numpy(nucl_aug).long()
        mask_torch = torch.from_numpy(mask).long()
        bg_mask_torch = torch.from_numpy(bg_mask).long()

        return expr_torch, nucl_torch, mask_torch, bg_mask_torch, coords_h1, coords_w1




