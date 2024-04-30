import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

# Initialisation
import torch
import torch.nn as nn
from torch.nn import init


# Network
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

class ForegroundPredictNet(nn.Module):
    def __init__(self, n_channels=313):
        super(ForegroundPredictNet, self).__init__()

        self.n_chanels = n_channels

        # Extract feature by 1x1 conv
        self.conv1x1_1 = nn.Conv2d(n_channels, 256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        # Encoder
        self.conv1 = unetConv2(64, 128, False)
        self.conv2 = unetConv2(128, 256, False)
        self.conv3 = unetConv2(256, 128, False)
        self.conv4 = unetConv2(128, 64, False)

        # Out Conv
        self.out_conv1 = nn.Conv2d((64+128+256+128+64), 256, kernel_size=1, stride=1, padding=0)
        self.out_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.out_conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.out_conv4 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = inputs
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        # Concatenate
        x = torch.cat((h0, h1, h2, h3, h4), 1)
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        x = self.out_conv4(x)

        return x

    def get_feature(self, inputs):
        x = inputs
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        # Concatenate
        x = torch.cat((h0, h1, h2, h3, h4), 1)
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        return x

    def get_spot_feature(self, inputs):
        x = inputs
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)
        return h0

    def get_conv_spot_feature(self, inputs):
        x = inputs
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        h0 = self.conv1x1_3(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        return h4

class CellPredictNet(nn.Module):
    def __init__(self, cond_channels = 64):
        super(CellPredictNet, self).__init__()
        self.cond_channel = cond_channels

        # FLiM
        self.FLiMmul = nn.Sequential(
            nn.Conv2d(self.cond_channel, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )

        self.predict = nn.Conv2d(1, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, nuclei_soft_mask, cond):
        f_mul = self.FLiMmul(cond)
        x = nuclei_soft_mask * f_mul
        x = self.predict(x)
        return x


# Other
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def get_softmask(cell_nuclei_mask, tau=5):
    contours, _ = cv2.findContours((cell_nuclei_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    I = np.zeros((cell_nuclei_mask.shape[0], cell_nuclei_mask.shape[1]))
    for i in range(cell_nuclei_mask.shape[0]):
        for j in range(cell_nuclei_mask.shape[1]):
            # Why j, i?
            I[i, j] = sigmoid(cv2.pointPolygonTest(contours[0], [j,i], True) / tau)
    return I

def get_seg_mask(sample_seg, sample_n):
    """
    Generate the segmentation mask with unique cell IDs
    """
    sample_n = np.squeeze(sample_n)

    # Background prob is average probability of all cells EXCEPT FOR NUCLEI
    sample_probs = softmax(sample_seg, axis=1)
    bgd_probs = np.expand_dims(np.mean(sample_probs[:, 0, :, :], axis=0), 0)
    fgd_probs = sample_probs[:, 1, :, :]
    probs = np.concatenate((bgd_probs, fgd_probs), axis=0)
    final_seg = np.argmax(probs, axis=0)

    # Map predictions to original cell IDs
    ids_orig = np.unique(sample_n)
    if ids_orig[0] != 0:
        ids_orig = np.insert(ids_orig, 0, 0)
    ids_pred = np.unique(final_seg)
    if ids_pred[0] != 0:
        ids_pred = np.insert(ids_pred, 0, 0)
    ids_orig = ids_orig[ids_pred]

    dictionary = dict(zip(ids_pred, ids_orig))
    dictionary[0] = 0

    final_seg_raw = np.vectorize(dictionary.get)(final_seg)

    # Add nuclei back in
    final_seg_orig = np.where(sample_n > 0, sample_n, final_seg_raw)

    return final_seg_orig, final_seg_raw