# ------------------------------------------------------------------------
# Derived from Bench2DriveZoo official prepare_B2D.py.
# Adapted locally to support disk-aware subset preparation in this repo.
# ------------------------------------------------------------------------
import argparse
import gzip
import json
import multiprocessing
import os
import pickle
from os.path import join

import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from vis_utils import DIS_CAR_SAVE, calculate_cube_vertices, calculate_occlusion_stats, edges


DATAROOT = "../../data/bench2drive"
MAP_ROOT = "../../data/bench2drive/maps"
OUT_DIR = "../../data/infos"
SPLIT_FILE = "../../data/splits/bench2drive_base_train_val_split.json"

MAX_DISTANCE = 75
FILTER_Z_SHRESHOLD = 10
FILTER_INVISINLE = True
NUM_VISIBLE_SHRESHOLD = 1
NUM_OUTPOINT_SHRESHOLD = 7
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAMERA_TO_FOLDER_MAP = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}

stand_to_ue4_rotate = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)
lidar_to_righthand_ego = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
lefthand_ego_to_lidar = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
left2right = np.eye(4)
left2right[1, 1] = -1


def apply_trans(vec, world2ego):
    vec = np.concatenate((vec, np.array([1])))
    transformed = world2ego @ vec
    return transformed[0:3]


def get_npc2world(npc):
    for key in ["world2vehicle", "world2ego", "world2sign", "world2ped"]:
        if key in npc:
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1, 0], npc2world[0, 0])
            yaw = npc["rotation"][-1] / 180 * np.pi
            if abs(yaw - yaw_from_matrix) > 0.01:
                npc2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
            return left2right @ npc2world @ left2right
    npc2world = np.eye(4)
    npc2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=npc["rotation"][-1] / 180 * np.pi).rotation_matrix
    npc2world[0:3, 3] = np.array(npc["location"])
    return left2right @ npc2world @ left2right


def get_image_point(loc, intrinsic, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)[0:3]
    depth = point_camera[2]
    point_img = np.dot(intrinsic, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], depth


def get_action(index):
    discrete_actions = {
        0: (0, 0, 1, False),
        1: (0.7, -0.5, 0, False),
        2: (0.7, -0.3, 0, False),
        3: (0.7, -0.2, 0, False),
        4: (0.7, -0.1, 0, False),
        5: (0.7, 0, 0, False),
        6: (0.7, 0.1, 0, False),
        7: (0.7, 0.2, 0, False),
        8: (0.7, 0.3, 0, False),
        9: (0.7, 0.5, 0, False),
        10: (0.3, -0.7, 0, False),
        11: (0.3, -0.5, 0, False),
        12: (0.3, -0.3, 0, False),
        13: (0.3, -0.2, 0, False),
        14: (0.3, -0.1, 0, False),
        15: (0.3, 0, 0, False),
        16: (0.3, 0.1, 0, False),
        17: (0.3, 0.2, 0, False),
        18: (0.3, 0.3, 0, False),
        19: (0.3, 0.5, 0, False),
        20: (0.3, 0.7, 0, False),
        21: (0, -1, 0, False),
        22: (0, -0.6, 0, False),
        23: (0, -0.3, 0, False),
        24: (0, -0.1, 0, False),
        25: (1, 0, 0, False),
        26: (0, 0.1, 0, False),
        27: (0, 0.3, 0, False),
        28: (0, 0.6, 0, False),
        29: (0, 1.0, 0, False),
        30: (0.5, -0.5, 0, True),
        31: (0.5, -0.3, 0, True),
        32: (0.5, -0.2, 0, True),
        33: (0.5, -0.1, 0, True),
        34: (0.5, 0, 0, True),
        35: (0.5, 0.1, 0, True),
        36: (0.5, 0.2, 0, True),
        37: (0.5, 0.3, 0, True),
        38: (0.5, 0.5, 0, True),
    }
    throttle, steer, brake, _ = discrete_actions[index]
    return throttle, steer, brake


def generate_map(map_root, out_dir):
    map_infos = {}
    for file_name in sorted(os.listdir(map_root)):
        if not file_name.endswith(".npz"):
            continue
        map_info = dict(np.load(join(map_root, file_name), allow_pickle=True)["arr"])
        town_name = file_name.split("_")[0]
        map_infos[town_name] = {}
        lane_points = []
        lane_types = []
        lane_sample_points = []
        trigger_volumes_points = []
        trigger_volumes_types = []
        trigger_volumes_sample_points = []
        for road in map_info.values():
            for lane_id, lane in road.items():
                if lane_id == "Trigger_Volumes":
                    for single_trigger_volume in lane:
                        points = np.array(single_trigger_volume["Points"])
                        points[:, 1] *= -1
                        trigger_volumes_points.append(points)
                        trigger_volumes_sample_points.append(points.mean(axis=0))
                        trigger_volumes_types.append(single_trigger_volume["Type"])
                else:
                    for single_lane in lane:
                        points = np.array([raw_point[0] for raw_point in single_lane["Points"]])
                        points[:, 1] *= -1
                        lane_points.append(points)
                        lane_types.append(single_lane["Type"])
                        lane_length = points.shape[0]
                        divide_points = [50 * i for i in range(lane_length // 50 + (1 if lane_length % 50 != 0 else 0))]
                        divide_points.append(lane_length - 1)
                        lane_sample_points.append(points[divide_points])
        map_infos[town_name]["lane_points"] = lane_points
        map_infos[town_name]["lane_sample_points"] = lane_sample_points
        map_infos[town_name]["lane_types"] = lane_types
        map_infos[town_name]["trigger_volumes_points"] = trigger_volumes_points
        map_infos[town_name]["trigger_volumes_sample_points"] = trigger_volumes_sample_points
        map_infos[town_name]["trigger_volumes_types"] = trigger_volumes_types
    with open(join(out_dir, "b2d_map_infos.pkl"), "wb") as outfile:
        pickle.dump(map_infos, outfile)


def preprocess(folder_list, worker_idx, tmp_dir, train_or_val):
    final_data = []
    folders = tqdm(folder_list) if worker_idx == 0 else folder_list

    for folder_name in folders:
        folder_path = join(DATAROOT, folder_name)
        if not os.path.isdir(join(folder_path, "anno")):
            continue
        last_position_dict = {}
        ann_files = sorted(os.listdir(join(folder_path, "anno")), key=lambda x: int(x.split(".")[0]))
        for ann_name in ann_files:
            position_dict = {}
            frame_data = {}
            cam_gray_depth = {}
            with gzip.open(join(folder_path, "anno", ann_name), "rt", encoding="utf-8") as gz_file:
                anno = json.load(gz_file)

            frame_data["folder"] = folder_name
            frame_data["town_name"] = folder_name.split("/")[1].split("_")[1]
            frame_data["command_far_xy"] = np.array([anno["x_command_far"], -anno["y_command_far"]])
            frame_data["command_far"] = anno["command_far"]
            frame_data["command_near_xy"] = np.array([anno["x_command_near"], -anno["y_command_near"]])
            frame_data["command_near"] = anno["command_near"]
            frame_data["frame_idx"] = int(ann_name.split(".")[0])
            frame_data["ego_yaw"] = -np.nan_to_num(anno["theta"], nan=np.pi) + np.pi / 2
            frame_data["ego_translation"] = np.array([anno["x"], -anno["y"], 0])
            frame_data["ego_vel"] = np.array([anno["speed"], 0, 0])
            frame_data["ego_accel"] = np.array([anno["acceleration"][0], -anno["acceleration"][1], anno["acceleration"][2]])
            frame_data["ego_rotation_rate"] = -np.array(anno["angular_velocity"])
            frame_data["ego_size"] = np.array(
                [
                    anno["bounding_boxes"][0]["extent"][1],
                    anno["bounding_boxes"][0]["extent"][0],
                    anno["bounding_boxes"][0]["extent"][2],
                ]
            ) * 2
            world2ego = left2right @ anno["bounding_boxes"][0]["world2ego"] @ left2right
            frame_data["world2ego"] = world2ego

            if frame_data["frame_idx"] == 0:
                expert_file_path = join(folder_path, "expert_assessment", "-0001.npz")
            else:
                expert_file_path = join(folder_path, "expert_assessment", str(frame_data["frame_idx"] - 1).zfill(5) + ".npz")
            expert_data = np.load(expert_file_path, allow_pickle=True)["arr_0"]
            action_id = expert_data[-1]
            throttle, steer, brake = get_action(action_id)
            frame_data["brake"] = brake
            frame_data["throttle"] = throttle
            frame_data["steer"] = steer

            sensor_infos = {}
            for cam in CAMERAS:
                sensor_infos[cam] = {}
                sensor_infos[cam]["cam2ego"] = left2right @ np.array(anno["sensors"][cam]["cam2ego"]) @ stand_to_ue4_rotate
                sensor_infos[cam]["intrinsic"] = np.array(anno["sensors"][cam]["intrinsic"])
                sensor_infos[cam]["world2cam"] = np.linalg.inv(stand_to_ue4_rotate) @ np.array(anno["sensors"][cam]["world2cam"]) @ left2right
                sensor_infos[cam]["data_path"] = join(folder_name, "camera", CAMERA_TO_FOLDER_MAP[cam], ann_name.split(".")[0] + ".jpg")
                depth_path = join(DATAROOT, sensor_infos[cam]["data_path"]).replace("rgb_", "depth_").replace(".jpg", ".png")
                cam_gray_depth[cam] = cv2.imread(depth_path)[:, :, 0]

            sensor_infos["LIDAR_TOP"] = {}
            sensor_infos["LIDAR_TOP"]["lidar2ego"] = left2right @ np.array(anno["sensors"]["LIDAR_TOP"]["lidar2ego"]) @ left2right @ lidar_to_righthand_ego
            world2lidar = lefthand_ego_to_lidar @ np.array(anno["sensors"]["LIDAR_TOP"]["world2lidar"]) @ left2right
            sensor_infos["LIDAR_TOP"]["world2lidar"] = world2lidar
            frame_data["sensors"] = sensor_infos

            gt_boxes = []
            gt_names = []
            gt_ids = []
            num_points_list = []
            npc2world_list = []

            for npc in anno["bounding_boxes"]:
                if npc["class"] == "ego_vehicle":
                    continue
                if npc["distance"] > MAX_DISTANCE:
                    continue
                if abs(npc["location"][2] - anno["bounding_boxes"][0]["location"][2]) > FILTER_Z_SHRESHOLD:
                    continue

                center = np.array([npc["center"][0], -npc["center"][1], npc["center"][2]])
                extent = np.array([npc["extent"][1], npc["extent"][0], npc["extent"][2]])
                position_dict[npc["id"]] = center
                local_center = apply_trans(center, world2lidar)
                size = extent * 2

                if "world2vehicle" in npc:
                    world2vehicle = left2right @ np.array(npc["world2vehicle"]) @ left2right
                    vehicle2lidar = world2lidar @ np.linalg.inv(world2vehicle)
                    yaw_local = np.arctan2(vehicle2lidar[1, 0], vehicle2lidar[0, 0])
                else:
                    yaw_local = -npc["rotation"][-1] / 180 * np.pi - frame_data["ego_yaw"] + np.pi / 2
                yaw_local_in_lidar_box = -yaw_local - np.pi / 2
                while yaw_local < -np.pi:
                    yaw_local += 2 * np.pi
                while yaw_local > np.pi:
                    yaw_local -= 2 * np.pi

                if "speed" in npc:
                    if "vehicle" in npc["class"]:
                        speed = npc["speed"]
                    elif npc["id"] in last_position_dict:
                        speed = np.linalg.norm((center - last_position_dict[npc["id"]])[0:2]) * 10
                    else:
                        speed = 0
                else:
                    speed = 0

                num_points = npc.get("num_points", -1)
                npc2world = get_npc2world(npc)
                speed_x = speed * np.cos(yaw_local)
                speed_y = speed * np.sin(yaw_local)

                if FILTER_INVISINLE:
                    valid = False
                    box2lidar = np.eye(4)
                    box2lidar[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=yaw_local).rotation_matrix
                    box2lidar[0:3, 3] = local_center
                    lidar2box = np.linalg.inv(box2lidar)
                    raw_verts = calculate_cube_vertices(local_center, extent)
                    verts = []
                    for raw_vert in raw_verts:
                        tmp = np.dot(lidar2box, [raw_vert[0], raw_vert[1], raw_vert[2], 1])
                        tmp[0:3] += local_center
                        verts.append(tmp.tolist()[:-1])
                    for cam in CAMERAS:
                        lidar2cam = np.linalg.inv(frame_data["sensors"][cam]["cam2ego"]) @ sensor_infos["LIDAR_TOP"]["lidar2ego"]
                        test_points = []
                        test_depth = []
                        for vert in verts:
                            point, depth = get_image_point(vert, frame_data["sensors"][cam]["intrinsic"], lidar2cam)
                            if depth > 0:
                                test_points.append(point)
                                test_depth.append(depth)
                        num_visible_vertices, _, num_vertices_outside_camera, _ = calculate_occlusion_stats(
                            np.array(test_points),
                            np.array(test_depth),
                            cam_gray_depth[cam],
                            max_render_depth=MAX_DISTANCE,
                        )
                        if num_visible_vertices > NUM_VISIBLE_SHRESHOLD and num_vertices_outside_camera < NUM_OUTPOINT_SHRESHOLD:
                            valid = True
                            break
                else:
                    valid = True

                if valid:
                    npc2world_list.append(npc2world)
                    num_points_list.append(num_points)
                    gt_boxes.append(np.concatenate([local_center, size, np.array([yaw_local_in_lidar_box, speed_x, speed_y])]))
                    gt_names.append(npc["type_id"])
                    gt_ids.append(int(npc["id"]))

            if len(gt_boxes) == 0:
                continue

            last_position_dict = position_dict.copy()
            frame_data["gt_ids"] = np.array(gt_ids)
            frame_data["gt_boxes"] = np.stack(gt_boxes)
            frame_data["gt_names"] = np.array(gt_names)
            frame_data["num_points"] = np.array(num_points_list)
            frame_data["npc2world"] = np.stack(npc2world_list)
            final_data.append(frame_data)

    os.makedirs(join(OUT_DIR, tmp_dir), exist_ok=True)
    with open(join(OUT_DIR, tmp_dir, f"b2d_infos_{train_or_val}_{worker_idx}.pkl"), "wb") as outfile:
        pickle.dump(final_data, outfile)


def generate_infos(folder_list, workers, train_or_val, tmp_dir):
    folder_num = len(folder_list)
    divide_list = [(folder_num // workers) * i for i in range(workers)]
    divide_list.append(folder_num)

    process_list = []
    for i in range(workers):
        sub_folder_list = folder_list[divide_list[i] : divide_list[i + 1]]
        process = multiprocessing.Process(target=preprocess, args=(sub_folder_list, i, tmp_dir, train_or_val))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()

    union_data = []
    for i in range(workers):
        worker_path = join(OUT_DIR, tmp_dir, f"b2d_infos_{train_or_val}_{i}.pkl")
        if not os.path.exists(worker_path):
            continue
        with open(worker_path, "rb") as infile:
            union_data.extend(pickle.load(infile))
    with open(join(OUT_DIR, f"b2d_infos_{train_or_val}.pkl"), "wb") as outfile:
        pickle.dump(union_data, outfile)


def collect_routes_from_v1():
    v1_root = join(DATAROOT, "v1")
    if not os.path.isdir(v1_root):
        return []
    return sorted(
        join("v1", folder_name)
        for folder_name in os.listdir(v1_root)
        if "Town" in folder_name and "Route" in folder_name and "Weather" in folder_name
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="number of workers used to prepare the dataset")
    parser.add_argument("--tmp-dir", default="tmp_data")
    parser.add_argument("--only-existing-routes", action="store_true", default=False)
    args = parser.parse_args()

    with open(SPLIT_FILE, "r") as infile:
        train_val_split = json.load(infile)

    all_routes = collect_routes_from_v1()
    if args.only_existing_routes:
        val_routes = [route for route in train_val_split["val"] if os.path.isdir(join(DATAROOT, route))]
        train_routes = [route for route in all_routes if route not in set(train_val_split["val"])]
    else:
        val_routes = train_val_split["val"]
        train_routes = [route for route in all_routes if route not in set(train_val_split["val"])]

    print(f"processing train data from {len(train_routes)} routes...")
    generate_infos(train_routes, args.workers, "train", args.tmp_dir)
    print(f"processing val data from {len(val_routes)} routes...")
    generate_infos(val_routes, args.workers, "val", args.tmp_dir)
    print("processing map data...")
    generate_map(MAP_ROOT, OUT_DIR)
    print("finish!")


if __name__ == "__main__":
    main()
