import torch

from ...utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    [
        'iou3d_boxes_overlap_bev_forward',
        'iou3d_boxes_iou_bev_forward',
        'iou3d_nms_forward',
        'iou3d_nms_normal_forward',
    ],
)


def boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou):
    ext_module.iou3d_boxes_iou_bev_forward(
        boxes_a.contiguous(),
        boxes_b.contiguous(),
        ans_iou,
    )


def boxes_overlap_bev_gpu(boxes_a, boxes_b, ans_overlap):
    ext_module.iou3d_boxes_overlap_bev_forward(
        boxes_a.contiguous(),
        boxes_b.contiguous(),
        ans_overlap,
    )


def nms_gpu(boxes, keep, thresh, device_id=None):
    del device_id
    num_out = torch.zeros(size=(), dtype=torch.long, device=keep.device)
    ext_module.iou3d_nms_forward(
        boxes.contiguous(),
        keep,
        num_out,
        nms_overlap_thresh=thresh,
    )
    return int(num_out.item())


def nms_normal_gpu(boxes, keep, thresh, device_id=None):
    del device_id
    num_out = torch.zeros(size=(), dtype=torch.long, device=keep.device)
    ext_module.iou3d_nms_normal_forward(
        boxes.contiguous(),
        keep,
        num_out,
        nms_overlap_thresh=thresh,
    )
    return int(num_out.item())
