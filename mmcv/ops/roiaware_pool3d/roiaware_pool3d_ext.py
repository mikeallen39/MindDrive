from ...utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    [
        'points_in_boxes_cpu_forward',
        'points_in_boxes_part_forward',
        'points_in_boxes_all_forward',
        'roiaware_pool3d_forward',
        'roiaware_pool3d_backward',
    ],
)


def points_in_boxes_cpu(boxes, points, point_indices):
    ext_module.points_in_boxes_cpu_forward(
        boxes.contiguous(),
        points.contiguous(),
        point_indices,
    )


def points_in_boxes_gpu(boxes, points, box_idxs_of_pts):
    ext_module.points_in_boxes_part_forward(
        boxes.contiguous(),
        points.contiguous(),
        box_idxs_of_pts,
    )


def points_in_boxes_batch(boxes, points, box_idxs_of_pts):
    ext_module.points_in_boxes_all_forward(
        boxes.contiguous(),
        points.contiguous(),
        box_idxs_of_pts,
    )


def forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels,
            pooled_features, mode):
    ext_module.roiaware_pool3d_forward(
        rois.contiguous(),
        pts.contiguous(),
        pts_feature.contiguous(),
        argmax,
        pts_idx_of_voxels,
        pooled_features,
        mode,
    )


def backward(pts_idx_of_voxels, argmax, grad_out, grad_in, mode):
    ext_module.roiaware_pool3d_backward(
        pts_idx_of_voxels,
        argmax,
        grad_out.contiguous(),
        grad_in,
        mode,
    )
