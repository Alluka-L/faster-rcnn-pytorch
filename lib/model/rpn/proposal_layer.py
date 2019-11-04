from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_bboxes_batch
import pdb
# TODO: from model.utils.config import cfg
# TODO: from model.nms.nms_wrapper import nms

DEBUG = False


class _ProposaLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feature_stride, scales, ratios):
        super(_ProposaLayer, self).__init__()

        self.feature_stride = feature_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        """
        For each (H, W) location i, generate A anchor boxes centered on cell i.
        Apply predicted bbox deltas at cell i to each of the A anchors.
        Clip predicted boxes to image.
        Remove predicted boxes with either height or width < threshold.
        Sort all (proposal, score) pairs by scores from highest to lowest
        Take top pre_nms_topN proposals before NMS.
        apply NMS with threshold 0.7 to remaining proposals.
        Take after_nms_topN proposals after NMS
        Return the top proposals (-> RoIs top, scores top)

        The first set of _num_anchors channels are bg probs.
        The second set are the fg probs
        """
