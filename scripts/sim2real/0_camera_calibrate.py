import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from perception.multi_camera_wrapper import MultiCameraWrapper
from perception.pcd_utils import *


if __name__ == "__main__":
    # Number of calibration rounds
    NUM_CALIBRATION_ROUNDS = 10

    # gather cameras
    multi_camera_wrapper = MultiCameraWrapper(rgb=True, depth=True, ir=False, high_res_rgb=False, align="rgb", type="realsense")
    num_cameras = multi_camera_wrapper.num_cameras
    print(f"Number of cameras: {num_cameras}")

    # calibrate using aruco tag
    pcds = []
    all_calib_dicts = []
    
    for round_idx in range(NUM_CALIBRATION_ROUNDS):
        print(f"\nPerforming calibration round {round_idx + 1}/{NUM_CALIBRATION_ROUNDS}")
        round_calib_dict = []
        
        for camera in multi_camera_wrapper._all_cameras:
            intrinsics = camera.calibration["intrinsics"]["rgb"]["cameraMatrix"]
            print(f"Intrinsics:\n{intrinsics}")

            marker_size = 0.15
            tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
                camera, marker_size=marker_size, verbose=True
            )
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotmat
            extrinsics[:3, 3:] = tvec
            print(f"Extrinsics:\n{extrinsics}")
            # camera frame -> aruco frame
            extrinsics_inv = np.linalg.inv(extrinsics)

            frames = camera.read_camera()
            rgb = frames["rgb"]
            depth = frames["depth"]

            # ArUco marker-center position in the SIM (REP-103) base frame -- the authors'
            # convention (their default [0.24, 0, 0] is sim-frame too), so the calibration
            # output is directly the camera pose in the sim frame (what align_cameras and
            # the sim cfgs use; no rotation conversion anywhere).
            #
            # CONTRACT (both parts required):
            #  * marker ORIENTATION: marker +X must point FROM the robot base TOWARD the
            #    marker/workspace (physically: pendant -Y on our rig = sim +X), marker +Y
            #    90deg CCW from that (viewed from above). NOT pendant-aligned.
            #  * offset = marker center in sim frame: 0.463 m from base toward the
            #    workspace => [0.463, 0, 0]. (Pendant reads the same point as [0,-0.463,0];
            #    our rig's workspace sits 90deg from the authors' -- see the rig-orientation
            #    note in diffusion_policy/real_world/ur10e_kinematics.py.)
            aruco_offset = np.array(
                [
                    0.463,
                    0.0,
                    0.0,
                ]
            )

            extrinsics_inv[:3, 3] += aruco_offset

            round_calib_dict.append(
                {
                    "camera_serial_number": camera._serial_number,
                    "intrinsics_raw": intrinsics.tolist(),
                    "extrinsics_raw": extrinsics_inv.tolist(),
                    "intrinsics": {
                        "fx": intrinsics[0, 0],
                        "fy": intrinsics[1, 1],
                        "ppx": intrinsics[0, 2],
                        "ppy": intrinsics[1, 2],
                        "height": rgb.shape[0],
                        "width": rgb.shape[1],
                        "fovy": camera._fovy,
                        "coeffs": camera.calibration["intrinsics"]["rgb"][
                            "distCoeffs"
                        ].tolist(),
                    },
                    "camera_base_ori": extrinsics_inv[:3, :3].tolist(),
                    "camera_base_pos": extrinsics_inv[:3, 3:].tolist(),
                }
            )

            if round_idx == 0:  # Only collect point cloud data in first round
                # 1/depth_scale = units-per-meter for THIS device (D405 ~0.1mm, not the
                # 1mm the old hardcoded 1000.0 assumed -> that made the cloud ~10x too big).
                points = depth_to_points(
                    depth, intrinsics, extrinsics_inv, depth_scale=1.0 / camera._depth_scale
                )
                colors = rgb.reshape(-1, 3) / 255.0
                points, colors = crop_points(points, colors=colors, crop_min=-2*np.ones(3), crop_max=2*np.ones(3))
                pcds.append(points_to_pcd(points, colors=colors))
        
        all_calib_dicts.append(round_calib_dict)

    # Average the calibration results
    final_calib_dict = []
    for camera_idx in range(num_cameras):
        # Collect all measurements for this camera
        camera_measurements = [round_dict[camera_idx] for round_dict in all_calib_dicts]
        
        # Average the extrinsics
        avg_extrinsics_raw = np.mean([np.array(m["extrinsics_raw"]) for m in camera_measurements], axis=0)
        avg_camera_base_ori = np.mean([np.array(m["camera_base_ori"]) for m in camera_measurements], axis=0)
        avg_camera_base_pos = np.mean([np.array(m["camera_base_pos"]) for m in camera_measurements], axis=0)
        
        # Use the first measurement for intrinsics (these shouldn't change)
        first_measurement = camera_measurements[0]
        
        final_calib_dict.append({
            "camera_serial_number": first_measurement["camera_serial_number"],
            "intrinsics_raw": first_measurement["intrinsics_raw"],
            "extrinsics_raw": avg_extrinsics_raw.tolist(),
            "intrinsics": first_measurement["intrinsics"],
            "camera_base_ori": avg_camera_base_ori.tolist(),
            "camera_base_pos": avg_camera_base_pos.tolist(),
        })

    current_time_date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception/calibrations/")
    os.makedirs(calib_path, exist_ok=True)
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"{current_time_date}.json"), "w"),
    )
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"most_recent_calib.json"), "w"),
    )
    print(f"Saved calibration at {os.path.join(calib_path,f'perception/logs/aruco/most_recent_calib.json')}")

    # --- labeled reference triads (disambiguate WHERE each frame origin is) ---
    # A compact solid triad marks each ORIGIN; the arms are just direction hints. This
    # replaces the old 1 m dotted axes, whose long +Z line read as "the frame floating up
    # near the camera" when the origin is actually the point where R/G/B converge.
    import open3d as o3d

    geoms = list(pcds)
    # ROBOT BASE frame (biggest triad) at the origin of the whole point cloud.
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20, origin=[0.0, 0.0, 0.0]))
    # ArUco MARKER center (mid triad) -- should sit ~|aruco_offset| from the base triad.
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10, origin=list(map(float, aruco_offset)))
    )
    # CAMERA position(s) (small triad) -- where each D405 sees the scene from.
    for c in final_calib_dict:
        cam_pos = np.array(c["camera_base_pos"]).ravel()
        cam_ori = np.array(c["camera_base_ori"])  # camera axes expressed in the base frame
        # Build the triad at the origin, ROTATE it into the camera's real orientation, then
        # move it to the camera. (create_coordinate_frame with only `origin` stays base-aligned
        # -- it would show no tilt. The blue axis is the OpenCV optical axis = view direction.)
        cam_tri = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        cam_tri.rotate(cam_ori, center=(0.0, 0.0, 0.0))
        cam_tri.translate(list(map(float, cam_pos)))
        geoms.append(cam_tri)
        opt = cam_ori[:, 2]  # optical axis (view direction) in base frame
        down_deg = float(np.degrees(np.arctan2(-opt[2], np.linalg.norm(opt[:2]))))
        print(
            f"[frames] camera {c['camera_serial_number']}  base pos = {np.round(cam_pos, 3)}"
            f"  view-dir = {np.round(opt, 3)}  (~{down_deg:.0f} deg below horizontal)"
        )
    print(
        f"[frames] BIG triad = ROBOT BASE @ [0,0,0] | MID triad = MARKER @ {list(map(float, aruco_offset))}"
        f" | SMALL triad(s) = CAMERA. Base->marker should measure ~{np.linalg.norm(aruco_offset):.3f} m."
    )
    o3d.visualization.draw_geometries(geoms)
