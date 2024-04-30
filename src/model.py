import glob
import os
import random
from collections import Counter

import cv2
from scipy import ndimage as ndi
import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader
import multiprocessing as mp
from src.utils import ForegroundPredictNet, CellPredictNet, get_softmask, get_seg_mask

class UCSModel(nn.Module):
    def __init__(self, manager, gene_map_shape):
        super(UCSModel, self).__init__()
        self.manager = manager
        self.opt = manager.get_opt()
        self.logger = manager.get_logger()

        self.mask_size = (gene_map_shape[0], gene_map_shape[1])
        self.gene_num = gene_map_shape[2]

        # You must use cuda
        self.device = torch.device("cuda")

        self._build_network()
        self.optimizer_1, self.optimizer_2 = self.get_optimizer()

    def _build_network(self):
        self.fg_net = ForegroundPredictNet(n_channels=self.gene_num).to(self.device)
        self.cell_net = CellPredictNet().to(self.device)
        # Initialize weights
        self.logger.info('Initialize weights')
        self.fg_net.apply(self.init_weights)
        self.cell_net.apply(self.init_weights)
    def get_optimizer(self):
        optimizer_1 = torch.optim.Adam(self.fg_net.parameters(), lr=self.opt.fg_net_lr,
                                       betas=(0.9,
                                              0.999),
                                       weight_decay=0.0001)
        optimizer_2 = torch.optim.Adam(self.cell_net.parameters(), lr=self.opt.cell_net_lr,
                                       betas=(0.9,
                                              0.999),
                                       weight_decay=0.0001)
        return optimizer_1, optimizer_2

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        return 0

    def train_model(self, train_dataset):
        print("Length of train dataset: ", len(train_dataset))
        # Train foreground net
        self.logger.info('Start training foreground net')
        if self.opt.using_filtered_background:
            self.logger.info('Using filtered background')

        train_loader = DataLoader(train_dataset, batch_size=self.opt.fg_net_batch_size, shuffle=True, num_workers=0)
        self.fg_net.train()
        num_epochs = self.opt.fg_net_epoch
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_loader):
                expr = batch[0].permute(0, 3, 1, 2).cuda()
                nuclei = batch[1].cuda()
                if torch.unique(nuclei).shape[0] == 1:
                    continue
                mask = batch[2].cuda()
                bg_mask = batch[3].cuda()
                if self.opt.using_filtered_background:
                    expr_bg = bg_mask * expr.sum(dim=1)
                    bg_size = bg_mask.sum()
                    bg_mask = ((expr_bg <= (expr_bg.sum() / bg_size)) * bg_mask)
                loss_mask = mask + bg_mask
                pred = self.fg_net(expr)
                criterion_ce = torch.nn.CrossEntropyLoss(reduction='none',
                                                         weight=torch.tensor(
                                                             [1., self.opt.fg_net_nuclei_weight]).cuda())
                loss = criterion_ce(pred, mask)
                loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
                self.optimizer_1.zero_grad()
                loss.backward()
                self.optimizer_1.step()
                self.logger.info(f'Epoch {epoch}, iter {i}, loss {loss.item()}')

        # Save model
        self.save(self.manager.get_checkpoint_dir() + '/model.pth')
        self.logger.info(f"Model saved to {self.manager.get_checkpoint_dir() + '/model.pth'}")

        # Train cell net
        self.logger.info('Start training cell net')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.cell_net.train()
        self.fg_net.eval()
        num_epochs = self.opt.cell_net_epoch
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_loader):
                expr = batch[0].permute(0, 3, 1, 2).cuda()
                nuclei = batch[1].cuda()
                if torch.unique(nuclei).shape[0] == 1:
                    continue
                # Tile cells individually along channel axis
                nucl_aug = nuclei.squeeze(0).cpu().numpy()
                cell_ids, _ = np.unique(nucl_aug, return_index=True)
                cell_ids = cell_ids[cell_ids != 0]  # remove background: cell_ids == 0
                n_cells = len(cell_ids)
                nucl_split = np.zeros((n_cells, nucl_aug.shape[0], nucl_aug.shape[1]), dtype=np.float32)
                nucl_soft_mask = np.zeros((n_cells, nucl_aug.shape[0], nucl_aug.shape[1]), dtype=np.float32)
                for i_cell, c_id in enumerate(cell_ids):
                    nucl_split[i_cell, :, :] = np.where(nucl_aug == c_id, 1, 0)
                    nucl_soft_mask[i_cell, :, :] = get_softmask(nucl_split[i_cell, :, :], self.opt.tau)
                nucl_split = torch.from_numpy(nucl_split).long().cuda()
                nucl_soft_mask = torch.from_numpy(nucl_soft_mask).float().cuda()
                expr = expr.repeat(n_cells, 1, 1, 1)
                pred = self.fg_net.get_feature(expr).detach()
                pred_bg_mask = self.fg_net(expr).detach().cpu().numpy()
                pred_bg_mask = np.argmax(pred_bg_mask, axis=1)
                pred_bg_mask = torch.from_numpy(pred_bg_mask).long().cuda()
                cell_pred = self.cell_net(nucl_soft_mask.unsqueeze(1), pred)
                criterion_ce = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(
                    [1., self.opt.cell_net_nuclei_weight]).cuda())
                loss = criterion_ce(cell_pred, nucl_split)
                loss = torch.sum(
                    loss * (nucl_split + (1 - pred_bg_mask))
                    / torch.sum(nucl_split + (1 - pred_bg_mask)))
                self.optimizer_2.zero_grad()
                loss.backward()
                self.optimizer_2.step()
                if i % 10 == 0:
                    self.logger.info(f'Epoch {epoch}, iter {i}, loss {loss.item()}')
        # Save
        self.save(self.manager.get_checkpoint_dir() + '/model.pth')
        self.logger.info(f"Model saved to {self.manager.get_checkpoint_dir() + '/model.pth'}")

    # ============================== Predict and postprocess ===============================
    def predict_whole(self, pred_dataset_shift_0, pred_dataset_shift_half_patch):
        self.pred_dir = os.path.join(self.manager.get_log_dir(), "pred")
        os.makedirs(self.pred_dir, exist_ok=True)
        for s, pred_dataset in enumerate([pred_dataset_shift_0, pred_dataset_shift_half_patch]):
            pred_dataset = torch.utils.data.DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=0)
            self.eval()
            for i, batch in enumerate(pred_dataset):
                if i == 0:
                    whole_seg = np.zeros(self.mask_size, dtype=np.uint32)
                expr = batch[0].permute(0, 3, 1, 2).cuda()
                nuclei = batch[1].cuda()
                if torch.unique(nuclei).shape[0] == 1:
                    # No nuclei in the patch
                    pass
                else:
                    # Tile cells individually along channel axis
                    nucl_aug = nuclei.squeeze(0).cpu().numpy()
                    cell_ids, _ = np.unique(nucl_aug, return_index=True)
                    cell_ids = cell_ids[cell_ids != 0]  # remove background: cell_ids == 0
                    n_cells = len(cell_ids)
                    nucl_split = np.zeros((n_cells, nucl_aug.shape[0], nucl_aug.shape[1]), dtype=np.float32)
                    nucl_soft_mask = np.zeros((n_cells, nucl_aug.shape[0], nucl_aug.shape[1]), dtype=np.float32)
                    for i_cell, c_id in enumerate(cell_ids):
                        nucl_split[i_cell, :, :] = np.where(nucl_aug == c_id, 1, 0)
                        nucl_soft_mask[i_cell, :, :] = get_softmask(nucl_split[i_cell, :, :],self.opt.tau)
                    nucl_soft_mask = torch.from_numpy(nucl_soft_mask).float().cuda()
                    expr = expr.repeat(n_cells, 1, 1, 1)
                    pred = self.fg_net.get_feature(expr).detach()
                    cell_pred = self.cell_net(nucl_soft_mask.unsqueeze(1), pred)

                    coords_h1 = batch[4].detach().cpu().squeeze().numpy()
                    coords_w1 = batch[5].detach().cpu().squeeze().numpy()
                    sample_seg = cell_pred.detach().cpu().numpy()
                    sample_n = batch[1].detach().cpu().numpy()
                    seg_patch, _ = get_seg_mask(sample_seg, sample_n)
                    whole_seg[coords_h1:coords_h1 + self.opt.patch_size,
                        coords_w1:coords_w1 + self.opt.patch_size] = seg_patch.copy()
            seg_fp = os.path.join(self.pred_dir,
                                  f"mask_shift_{0 if s == 0 else int(self.opt.patch_size // 2)}.tif")
            tifffile.imwrite(seg_fp, whole_seg.astype(np.uint32), photometric='minisblack')
        self.logger.info("Prediction finished")
        self.logger.info("Filling grid")
        self.fill_grid()

    def fill_grid(self):
        """
        Combine predictions from unshifted and shifted patches to remove
        border effects
        """
        patch_size = self.opt.patch_size
        self.patch_size = patch_size
        shift = patch_size // 2
        pred_fp = os.path.join(self.pred_dir, f"mask_shift_{0}.tif")
        pred_fp_sf = os.path.join(self.pred_dir, f"mask_shift_{int(shift)}.tif")
        output_fp = os.path.join(self.pred_dir, "segmentation_mask.tif")
        pred = tifffile.imread(pred_fp)
        pred_sf = tifffile.imread(pred_fp_sf)

        # Get coordinates of non-overlapping patches
        h_starts = list(np.arange(0, pred.shape[0] - patch_size, patch_size))
        w_starts = list(np.arange(0, pred.shape[1] - patch_size, patch_size))

        h_starts.append(pred.shape[0] - patch_size)
        w_starts.append(pred.shape[1] - patch_size)

        # Fill along grid
        h_starts_wide = []
        w_starts_wide = []

        for i in range(-8, 9):
            h_starts_wide.extend([x + i for x in h_starts])
            w_starts_wide.extend([x + i for x in w_starts])

        fill = np.zeros(pred.shape)
        fill[h_starts_wide, :] = 1
        fill[:, w_starts_wide] = 1

        # border
        fill[:patch_size, :] = 0
        fill[-patch_size:, :] = 0
        fill[:, :patch_size] = 0
        fill[:, -patch_size:] = 0

        result = np.zeros(pred.shape, dtype=np.uint32)
        result = np.where(fill > 0, pred_sf, pred)

        tifffile.imwrite(output_fp, result.astype(np.uint32), photometric='minisblack')

    def postprocess(self):
        nuclei_img = tifffile.imread(self.opt.nuclei_mask)
        img_whole = tifffile.imread(os.path.join(self.pred_dir, "segmentation_mask.tif"))

        patch_size = self.opt.patch_size
        h_starts = list(np.arange(0, img_whole.shape[0] - patch_size, patch_size))
        w_starts = list(np.arange(0, img_whole.shape[1] - patch_size, patch_size))
        h_starts.append(img_whole.shape[0] - patch_size)
        w_starts.append(img_whole.shape[1] - patch_size)
        coords_starts = [(x, y) for x in h_starts for y in w_starts]
        self.logger.info('%d patches available' %len(coords_starts))

        num_processes = mp.cpu_count()
        self.logger.info('Num multiprocessing splits: %d' % num_processes)
        coords_splits = np.array_split(coords_starts, num_processes)
        processes = []
        # Make a temporary directory to store the results of each process
        output_dir = "./temp_dir/"
        os.makedirs(output_dir, exist_ok=True)
        for chunk in coords_splits:
            p = mp.Process(target=self.process_chunk, args=(chunk, patch_size, img_whole, nuclei_img, output_dir))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        self.logger.info('Combining results')
        seg_final, to_check_ids = self.combine(patch_size, output_dir)
        ids_splits = np.array_split(to_check_ids, num_processes)
        processes = []
        for chunk_ids in ids_splits:
            p = mp.Process(target=self.process_check_splits, args=(nuclei_img, seg_final,
                                                                   chunk_ids))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        check_mask = np.isin(seg_final, to_check_ids)
        seg_final = np.where(check_mask == 1, 0, seg_final)
        fp_checked_splits = glob.glob(self.pred_dir + '/*_checked_splits.tif', recursive=True)
        for fp in fp_checked_splits:
            checked_split = tifffile.imread(fp)
            seg_final = np.where(checked_split > 0, checked_split, seg_final)
            os.remove(fp)
        fp_dir = os.path.join(self.pred_dir, 'segmentation_mask.tif')
        self.logger.info('Saved final segmentation to %s' % fp_dir)
        tifffile.imwrite(fp_dir, seg_final.astype(np.uint32), photometric='minisblack')
        # Remove temporary directory with all the files
        for fp in glob.glob(output_dir + '/*.tif', recursive=True):
            os.remove(fp)
        os.rmdir(output_dir)

    def process_chunk(self, chunk, patch_size, img_whole, nuclei_img, output_dir):

        for index in range(len(chunk)):

            coords = chunk[index]
            coords_x1 = coords[0]
            coords_y1 = coords[1]
            coords_x2 = coords_x1 + patch_size
            coords_y2 = coords_y1 + patch_size

            img = img_whole[coords_x1:coords_x2, coords_y1:coords_y2]

            nuclei = nuclei_img[coords_x1:coords_x2, coords_y1:coords_y2]
            output_fp = output_dir + '%d_%d.tif' % (coords_x1, coords_y1)

            # print('Filling holes')
            filled = self.postprocess_connect(img, nuclei)

            # print('Removing islands')
            final = self.remove_islands(filled, nuclei)

            tifffile.imwrite(output_fp, final.astype(np.uint32), photometric='minisblack')

    def combine(self, patch_size, output_dir):
        """
        Combine the patches previously output by the connect function
        """

        fp_dir = output_dir
        fp_unconnected = self.pred_dir + '/segmentation_mask.tif'

        dl_pred = tifffile.imread(fp_unconnected)
        height = dl_pred.shape[0]
        width = dl_pred.shape[1]

        seg_final = np.zeros((height, width), dtype=np.uint32)

        fp_seg = glob.glob(fp_dir + '/*.tif', recursive=True)

        sample = tifffile.imread(fp_seg[0])
        patch_h = sample.shape[0]
        patch_w = sample.shape[1]

        cell_ids = []

        for fp in fp_seg:
            patch = tifffile.imread(fp)

            patch_ids = np.unique(patch)
            patch_ids = patch_ids[patch_ids != 0]
            cell_ids.extend(patch_ids)

            fp_coords = os.path.basename(fp).split('.')[0]
            fp_x = int(fp_coords.split('_')[0])
            fp_y = int(fp_coords.split('_')[1])

            # Place into appropriate location
            seg_final[fp_x:fp_x + patch_h, fp_y:fp_y + patch_w] = patch[:]

        # If cell is split by windowing, keep component with nucleus
        count_ids = Counter(cell_ids)
        windowed_ids = [k for k, v in count_ids.items() if v > 1]

        # Check along borders
        h_starts = list(np.arange(0, height - patch_size, patch_size))
        w_starts = list(np.arange(0, width - patch_size, patch_size))
        h_starts.append(height - patch_size)
        w_starts.append(width - patch_size)

        # Mask along grid
        h_starts_wide = []
        w_starts_wide = []
        for i in range(-10, 11):
            h_starts_wide.extend([x + i for x in h_starts])
            w_starts_wide.extend([x + i for x in w_starts])

        mask = np.zeros(seg_final.shape)
        mask[h_starts_wide, :] = 1
        mask[:, w_starts_wide] = 1

        masked = mask * seg_final
        masked_ids = np.unique(masked)[1:]

        # IDs to check for split bodies
        to_check_ids = list(set(masked_ids) & set(windowed_ids))

        return seg_final, to_check_ids

    def process_check_splits(self, nuclei_img, seg_final, chunk_ids):
        """
        Check and fix cells split by windowing
        """

        chunk_seg = np.zeros(seg_final.shape, dtype=np.uint32)

        # Touch diagonally = same object
        s = ndi.generate_binary_structure(2, 2)

        for i in chunk_ids:
            i_mask = np.where(seg_final == i, 1, 0).astype(np.uint8)

            # Number of blobs belonging to cell
            unique_ids, num_blobs = ndi.label(i_mask, structure=s)

            # Bounding box
            bb = np.argwhere(unique_ids)
            (ystart, xstart), (ystop, xstop) = bb.min(0), bb.max(0) + 1
            unique_ids_crop = unique_ids[ystart:ystop, xstart:xstop]

            nucleus_mask = np.where(nuclei_img == i, 1, 0).astype(np.uint8)
            nucleus_mask = nucleus_mask[ystart:ystop, xstart:xstop]

            if num_blobs > 1:
                # Keep the blob with max overlap to nucleus
                amount_overlap = np.zeros(num_blobs)

                for i_blob in range(1, num_blobs + 1):
                    blob = np.where(unique_ids_crop == i_blob, 1, 0)
                    amount_overlap[i_blob - 1] = np.sum(blob * nucleus_mask)
                blob_keep = np.argmax(amount_overlap) + 1

                # Put into final segmentation
                final_mask = np.where(unique_ids_crop == blob_keep, 1, 0)
                chunk_seg[ystart:ystop, xstart:xstop] = np.where(final_mask == 1, i,
                                                                 chunk_seg[ystart:ystop, xstart:xstop])

            else:
                chunk_seg = np.where(i_mask == 1, i, chunk_seg)

        tifffile.imwrite(self.pred_dir + '/' + str(chunk_ids[0]) + '_checked_splits.tif', chunk_seg,
                         photometric='minisblack')

    def remove_islands(self, img, nuclei):
        cell_ids = np.unique(img)
        cell_ids = cell_ids[1:]

        random.shuffle(cell_ids)

        # Touch diagonally = same object
        s = ndi.generate_binary_structure(2, 2)

        final = np.zeros(img.shape, dtype=np.uint32)

        for i in cell_ids:
            i_mask = np.where(img == i, 1, 0).astype(np.uint8)

            nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

            # Number of blobs belonging to cell
            unique_ids, num_blobs = ndi.label(i_mask, structure=s)
            if num_blobs > 1:
                # Keep the blob with max overlap to nucleus
                amount_overlap = np.zeros(num_blobs)

                for i_blob in range(1, num_blobs + 1):
                    blob = np.where(unique_ids == i_blob, 1, 0)
                    amount_overlap[i_blob - 1] = np.sum(blob * nucleus_mask)
                blob_keep = np.argmax(amount_overlap) + 1

                final_mask = np.where(unique_ids == blob_keep, 1, 0)

            else:
                blob_size = np.count_nonzero(i_mask)
                if blob_size > 2:
                    final_mask = i_mask.copy()
                else:
                    final_mask = i_mask * 0

            final_mask = ndi.binary_fill_holes(final_mask).astype(int)

            final = np.where(final_mask > 0, i, final)

        return final

    def postprocess_connect(self, img, nuclei):
        cell_ids = np.unique(img)
        cell_ids = cell_ids[1:]

        random.shuffle(cell_ids)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Touch diagonally = same object
        s = ndi.generate_binary_structure(2, 2)

        final = np.zeros(img.shape, dtype=np.uint32)

        for i in cell_ids:
            i_mask = np.where(img == i, 1, 0).astype(np.uint8)

            connected_mask = cv2.dilate(i_mask, kernel, iterations=2)
            connected_mask = cv2.erode(connected_mask, kernel, iterations=2)

            # Add nucleus as predicted by cellpose
            nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

            connected_mask = connected_mask + nucleus_mask
            connected_mask[connected_mask > 0] = 1

            unique_ids, num_ids = ndi.label(connected_mask, structure=s)
            if num_ids > 1:
                # The first element is always 0 (background)
                unique, counts = np.unique(unique_ids, return_counts=True)

                # Ensure counts in descending order
                counts, unique = (list(t) for t in zip(*sorted(zip(counts, unique))))
                counts.reverse()
                unique.reverse()
                counts = np.array(counts)
                unique = np.array(unique)

                no_overlap = False

                # Rarely, the nucleus is not the largest segment
                for i_part in range(1, len(counts)):
                    if i_part > 1:
                        no_overlap = True
                    largest = unique[np.argmax(counts[i_part:]) + i_part]
                    connected_mask = np.where(unique_ids == largest, 1, 0)
                    # Break if current largest region overlaps nucleus
                    if np.sum(connected_mask * nucleus_mask) > 0.5:
                        break

                # Close holes on largest section
                filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

            else:
                filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

            final = np.where(filled_mask > 0, i, final)

        final = np.where(nuclei > 0, nuclei, final)
        return final

    def save(self, path):
        """Save the model state
        """
        save_dict = {'model_state': self.state_dict()}

        # for k, v in self.schedulers.items():
        #     save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, path)

    def load(self, path, strict = True):
        """Load a model state from a checkpoint file
        """
        checkpoint_file = path
        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'], strict=strict)