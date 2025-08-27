import os
import numpy as np
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
# import open3d as o3d
import pandas as pd
from gluefactory.utils.image import read_image
from gluefactory.utils.experiments import load_experiment
from sklearn.metrics import mean_squared_error


##  1, Load_img, depth_maps, camera_poses 
def load_img(id1, id2, images_dir, poses_dir, depths_dir, images_path_list):
    im1 = cv2.imread(images_dir / f"{images_path_list[id1]}")
    pose1 = np.load(poses_dir / f"{images_path_list[id1].replace('.png', '.npy')}")
    depth1 = np.load(depths_dir / f"{images_path_list[id1].replace('.png', '.npy')}")
    

    im2 = cv2.imread(images_dir / f"{images_path_list[id2]}")
    pose2 = np.load(poses_dir / f"{images_path_list[id2].replace('.png', '.npy')}")
    depth2 = np.load(depths_dir / f"{images_path_list[id2].replace('.png', '.npy')}")

    cam1_to_cam2 = pose2 @ np.linalg.inv(pose1)
    cam2_to_cam1 = pose1 @ np.linalg.inv(pose2)

    return im1, im2, cam1_to_cam2, cam2_to_cam1, depth1, depth2
    

##  2, Segment the imahes 
def seg(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)/255.0
    h, s, v = cv2.split(hsv)
    keep_mask = (h < 0.200) | (h > 0.428)
    mrgb = np.ones_like(rgb)*255
    mrgb[keep_mask] = rgb[keep_mask]
    
    return keep_mask.astype(bool), mrgb

def numpy_to_torch(array, dtype=torch.float32):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).type(dtype)
    raise TypeError("Input should be a NumPy array.")


## 3, Detect and match key_point
def glue_factory_matching(img0_whitebg, img1_whitebg, model_path, output_path = None):
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_experiment(model_path).eval().to(device)

    # Normalize for model input
    img0 = img0_whitebg.astype(np.float32) / 255.0
    img1 = img1_whitebg.astype(np.float32) / 255.0

    data = {
        'view0': {'image': numpy_to_torch(img0).unsqueeze(0).permute(0, 3, 1, 2).to(device)},
        'view1': {'image': numpy_to_torch(img1).unsqueeze(0).permute(0, 3, 1, 2).to(device)},
    }

    with torch.no_grad():
        pred = model(data)

    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()

    kpts0_int = np.round(kpts0).astype(int)
    kpts1_int = np.round(kpts1).astype(int)

    valid_matches = matches > 0
    match_pairs = np.column_stack((np.where(valid_matches)[0], matches[valid_matches]))

    print(f"Finish matching...")

    return kpts0, kpts1, matches, match_pairs, img0, img1

## 4, project keypoints from one camera frame to another using depth and pose


def project_to_other(uv, gt, K, depth, P_1to_2):
    """
    Reproject 2D pixels (image 1) with depth to image 2 using intrinsics K and pose P_1to_2.
    Row-vector convention is used: X2_h = [X1 1] @ P_1to_2^T.

    Returns:
        uv2_valid   : (M,2) projected pixels in image 2
        idx_valid   : (M,) indices into the original uv array
        uv_valid    : (M,2) original pixels in image 1 that survived
        gt_valid    : (M,...) corresponding gt entries
    """
    H, W = depth.shape

    if uv.size == 0:
        # 4-tuple for consistency
        return (np.empty((0, 2)),
                np.empty((0,), dtype=int),
                np.empty((0, 2)),
                np.empty((0,), dtype=gt.dtype if hasattr(gt, 'dtype') else object))

    # round to nearest integer pixel for depth sampling
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)

    # mask points inside the image BEFORE sampling depth
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return (np.empty((0, 2)),
                np.empty((0,), dtype=int),
                np.empty((0, 2)),
                np.empty((0,), dtype=gt.dtype if hasattr(gt, 'dtype') else object))

    u = u[in_bounds]
    v = v[in_bounds]
    uv_ib = uv[in_bounds]
    gt_ib = gt[in_bounds]
    idx_ib = np.nonzero(in_bounds)[0]

    depths = depth[v, u]
    has_depth = depths > 0
    if not np.any(has_depth):
        return (np.empty((0, 2)),
                np.empty((0,), dtype=int),
                np.empty((0, 2)),
                np.empty((0,), dtype=gt.dtype if hasattr(gt, 'dtype') else object))

    uv_d = uv_ib[has_depth]
    gt_d = gt_ib[has_depth]
    idx_d = idx_ib[has_depth]
    Z = depths[has_depth]

    # Back-project to cam1
    ones = np.ones((uv_d.shape[0], 1))
    xys_h = np.hstack([uv_d, ones]) * Z[:, None]          # (N,3)
    Xcam1 = xys_h @ np.linalg.inv(K).T                    # row-vector

    # Transform to cam2
    X1_h = np.hstack([Xcam1, ones])                       # (N,4)
    X2_h = X1_h @ P_1to_2.T
    w = X2_h[:, 3:4]
    valid_w = np.abs(w) > 1e-8
    if not np.any(valid_w):
        return (np.empty((0, 2)),
                np.empty((0,), dtype=int),
                np.empty((0, 2)),
                np.empty((0,), dtype=gt.dtype if hasattr(gt, 'dtype') else object))

    X2 = X2_h[:, :3] / np.clip(w, 1e-8, None)

    # Keep only points in front of cam2
    front = X2[:, 2] > 0

    # Project to pixels
    uv2_h = X2 @ K.T
    z = uv2_h[:, 2:3]
    valid_z = np.abs(z) > 1e-8
    uv2 = uv2_h[:, :2] / np.clip(z, 1e-8, None)

    # Image bounds in cam2
    u2_ok = (uv2[:, 0] >= 0) & (uv2[:, 0] < W)
    v2_ok = (uv2[:, 1] >= 0) & (uv2[:, 1] < H)

    valid = valid_w[:,0] & valid_z[:,0] & front & u2_ok & v2_ok
    if not np.any(valid):
        return (np.empty((0, 2)),
                np.empty((0,), dtype=int),
                np.empty((0, 2)),
                np.empty((0,), dtype=gt.dtype if hasattr(gt, 'dtype') else object))

    return uv2[valid], idx_d[valid], uv_d[valid], gt_d[valid]


## 6, Visualize matches and reprojection error


## 5, Compare matches to ground truth projections


# ------------------------------ uv1 → uv2 → uv1 (reprojection) ---------------------------------
def uv1_to_uv2_loop_check(id1, id2, images_dir, poses_dir, depths_dir, images_path_list, model_path, K):
    # Load data
    im1, im2, cam1_to_cam2, cam2_to_cam1, depth1, depth2 = load_img(id1, id2, images_dir, poses_dir, depths_dir, images_path_list)

    img0_uint8 = (im1).astype(np.uint8)
    img1_uint8 = (im2).astype(np.uint8)


    # Perform segmentation
    mask0, img0_whitebg = seg(img0_uint8)
    mask1, img1_whitebg = seg(img1_uint8)

  #counting mask of the tree in both frame to compute overlap_ratio
    #fram1
    ys, xs = np.where(mask0 > 0)
    uv = np.stack([xs, ys], axis=1)
    #fram2
    ys1, xs1 = np.where(mask1 > 0)
    uv1 = np.stack([xs1, ys1], axis=1)
    overlap_ratio = min(len(uv), len(uv1)) / max(len(uv), len(uv1))
    print(f"overlap_ratio: {overlap_ratio}")

    # Run Glue Factory feature matching
    kpts0, kpts1, matches, match_pairs, img0, img1 = glue_factory_matching(img0_whitebg, img1_whitebg, model_path)
    aligned_uv1 = kpts0[match_pairs[:, 0]]
    aligned_uv2 = kpts1[match_pairs[:, 1]]

    
    #  Main goals:
    # - Project keypoints from frame 1 to frame 2 and compare them with the predicted keypoints from the model.  
    # - Reproject the keypoints back to frame 1 to check consistency. If they don’t return close to the original location, the depth is unreliable and that point should not be used for evaluation.



    # uv1_vgt0 is the total gt keypoints in frame1 with the samesize as uv_proj(some keypoint will be loss),it will be use as the input in Backward projection
    _, _, _, uv1_vgt0 = project_to_other(aligned_uv1, aligned_uv1, K, depth1, cam1_to_cam2)

    # Forward projection: cam1 → cam2
    uv_proj, valid_indices1, uv1_valid, uv_nframe = project_to_other(aligned_uv1, aligned_uv2, K, depth1, cam1_to_cam2)
   
    # Backward projection: cam2 → cam1
    uv_Rproj, valid_indices2, _, uv1_vgt = project_to_other(uv_proj, uv1_vgt0, K, depth2, cam2_to_cam1)

    # print(f"uv1_vgt0 shape: {uv1_vgt0.shape}")
    print(f"aligned_uv1 shape: {aligned_uv1.shape}")
    print(f"uv_proj shape: {uv_proj.shape}") 
    print(f"uv1_Rproj shape: {uv_Rproj.shape}") 
    print(f"uv_nframe shape: {uv1_vgt.shape}")

    # print(f"mask0/mask1 shape: {len(mask0)/len(mask1)}")



    # Compute errors
    errors21 = np.linalg.norm(uv_Rproj - uv1_vgt, axis=1)
    errors12 = np.linalg.norm(uv_proj - uv_nframe, axis=1)
    
    # Stats
    n_matches21 = len(errors21)
    n_matches12 = len(errors12)
    # mean_error = np.mean(errors) if n_matches > 0 else 30000
    # median_error = np.median(errors) if n_matches > 0 else 30000
    # mean_error1 = np.mean(errors1) if n_matches1 > 0 else 30000
    # median_error1 = np.median(errors1) if n_matches1 > 0 else 30000

    return overlap_ratio, n_matches12, n_matches21, errors12, errors21

if __name__ == "__main__":

    K = np.array([[397.8731, 0, 320], [0, 396.0224, 240], [0, 0, 1]])

    path = Path("/home/chhaileng/Documents/test_model/glue-factory/testing_tree/tree42_1")
    # path = Path("/home/rscherrer/Downloads/scan_1")
    images_dir = path / "images"
    depths_dir = path / "depth"
    poses_dir = path / "pose"
    images_path_list = sorted(os.listdir(images_dir), key=lambda x: int(x.split(".")[0]))
    model_path = "/home/chhaileng/Documents/test_model/glue-factory/pre_trained/white_background/checkpoint_best.tar"
    result_dir = "/home/chhaileng/Documents/test_model/glue-factory"
    
    images_path_list = sorted(os.listdir(images_dir), key=lambda x: int(x.split(".")[0]))

    # Loop over id2 > id1

    i = 10  # fixed
    results = []
        
    for id1 in range(1, len(images_path_list) - i):  # -4 so id2 is in range
        id2 = id1 + i  # fixed offset
        try:
            print(f"Matching {id1} -> {id2}")

            overlap_ratio, n_matches12, n_matches21, errors12, errors21 = uv1_to_uv2_loop_check(
                id1, id2, images_dir, poses_dir, depths_dir, images_path_list, model_path, K
            )

            results.append({
                "id1": id1,
                "id2": id2,
                "overlap (%)": round(overlap_ratio * 100, 2),
                # "mean_error uv1→uv2 (px)": round(mean_err_12, 2),
                # "median_error uv1→uv2 (px)": round(median_err_12, 2),
                "matches uv1→uv2": n_matches12,
                "errors uv1→uv2 (px)": errors12,
                "matches uv2→uv1": n_matches21,
                "errors uv2→uv1 (px)": errors21,
                
            })

        except Exception as e:
            print(f"Error on {id1}-{id2}: {e}")
            continue


        # Save to Excel
        if results:
            df = pd.DataFrame(results)
            output_file = os.path.join(result_dir,f"matching_results_Pt{i}.csv")
            # df.to_excel(output_file,  sheet_name="uv1_to_uv2", index=False)
            df.to_csv(output_file,index=False)
            print(f"Saved matching results to Excel: {output_file}")

        else:
            print(" No results to save. All matches may have failed.")