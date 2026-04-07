# ------------------------------------------------------------------------
# Derived from Bench2DriveZoo official dataset preparation utilities.
# This local copy keeps only the functions/constants needed by prepare_B2D.py.
# ------------------------------------------------------------------------
import numpy as np


WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

DIS_CAR_SAVE = 50
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def point_in_canvas_hw(pos):
    return (0 <= pos[0] < WINDOW_HEIGHT) and (0 <= pos[1] < WINDOW_WIDTH)


def point_is_occluded(point, vertex_depth, depth_map):
    y, x = map(int, point)
    is_occluded = []
    for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        if point_in_canvas_hw((dy + y, dx + x)):
            is_occluded.append(depth_map[y + dy, x + dx] < vertex_depth)
    return bool(is_occluded) and all(is_occluded)


def calculate_occlusion_stats(bbox_points, depth, depth_map, max_render_depth):
    num_visible_vertices = 0
    num_invisible_vertices = 0
    num_vertices_outside_camera = 0
    points = []

    for i in range(len(bbox_points)):
        x_2d = bbox_points[i][0]
        y_2d = bbox_points[i][1]
        point_depth = depth[i]

        if max_render_depth > point_depth > 0 and point_in_canvas_hw((y_2d, x_2d)):
            is_occluded = point_is_occluded((y_2d, x_2d), point_depth, depth_map)
            if is_occluded:
                vertex_color = (0, 0, 255)
                num_invisible_vertices += 1
            else:
                vertex_color = (0, 255, 0)
                num_visible_vertices += 1
            points.append((x_2d, y_2d, vertex_color))
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_invisible_vertices, num_vertices_outside_camera, points


def calculate_cube_vertices(center, extent):
    cx, cy, cz = center
    x, y, z = extent
    return [
        (cx + x, cy + y, cz + z),
        (cx + x, cy + y, cz - z),
        (cx + x, cy - y, cz + z),
        (cx + x, cy - y, cz - z),
        (cx - x, cy + y, cz + z),
        (cx - x, cy + y, cz - z),
        (cx - x, cy - y, cz + z),
        (cx - x, cy - y, cz - z),
    ]
