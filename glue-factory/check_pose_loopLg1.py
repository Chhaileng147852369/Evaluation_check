import os
import numpy as np
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
# import open3d as o3d
import pandas as pd
from gluefactory.utils.image import read_image,load_image, numpy_image_to_torch
from gluefactory.utils.experiments import load_experiment
from sklearn.metrics import mean_squared_error

# LightGlue and utilities
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)



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


## 3, Detect and match key_point

def Conv_tensor(img):
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return tensor.to(torch.float32).to(device)


def matchLightGlue(mrgb0 , mrgb1 ):

    # Convert segmented images to tensors
    tensor0 = Conv_tensor(mrgb0)
    tensor1 = Conv_tensor(mrgb1)

    # Extract features
    f0 = extractor.extract(tensor0)
    f1 = extractor.extract(tensor1)

    # Match features
    matches01 = matcher({'image0': f0, 'image1': f1})
    feats0, feats1, matches01 = [rbd(x) for x in [f0, f1, matches01]]

    matches = matches01['matches'].cpu()
    scores = matches01['scores'].detach().cpu().numpy()

    # Filter valid matches
    valid = (matches[:, 0] != -1) & (matches[:, 1] != -1)
    matches = matches[valid]
    scores = scores[valid]

    if matches.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), np.empty((0, 2), dtype=int)

    # Get matching points
    pts0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()
    pts1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()

    return mrgb0,mrgb1, pts0, pts1, scores, matches.numpy()

## 4, project keypoints from one camera frame to another using depth and pose

def project_to_other(uv, gt, K, depth, P_1to_2):
    H, W = depth.shape

    # Échantillonnage des profondeurs
    u = uv[:, 0].astype(int)
    v = uv[:, 1].astype(int)
    depths = depth[v, u]  # attention : (y, x) = (v, u)

    # Supprimer les points avec profondeur invalide (0)
    valid_depth = depths > 0
    uv = uv[valid_depth]
    gt = gt[valid_depth]
    depths = depths[valid_depth]


    if len(uv) == 0:
        return np.empty((0, 2)), np.array([], dtype=bool), np.array([], dtype=int)

    # Backprojection
    xys_h = np.hstack([uv, np.ones((uv.shape[0], 1))]) * depths[:, None]  # (N,3)
    Xcam1 = xys_h @ np.linalg.inv(K).T

    # Vers l'autre caméra
    Xcam1_h = np.hstack([Xcam1, np.ones((Xcam1.shape[0], 1))])  # (N,4)
    Xcam2_h = Xcam1_h @ P_1to_2.T
    Xcam2 = Xcam2_h[:, :3] / Xcam2_h[:, 3:4].clip(1e-6, 2000) 


    # Projection
    uv2_h = Xcam2 @ K.T
    uv2 = uv2_h[:, :2] / uv2_h[:, 2:3]
    

    # Filtrage spatial
    u2_valid = (uv2[:, 0] >= 0) & (uv2[:, 0] < W)
    v2_valid = (uv2[:, 1] >= 0) & (uv2[:, 1] < H)
    valid = u2_valid & v2_valid 

    return uv2[valid], valid_depth.nonzero()[0][valid], uv[valid], gt[valid]



## 5, Compare matches to ground truth projections
# ------------------------------ uv1 → uv2 → uv1 (reprojection) ---------------------------------
def uv1_to_uv2_loop_check(id1, id2, images_dir, poses_dir, depths_dir, images_path_list,  K):
   
    im1, im2, cam1_to_cam2, cam2_to_cam1, depth1, depth2 = load_img(id1, id2, images_dir, poses_dir, depths_dir, images_path_list)
    mask0, mrgb0 = seg(im1)
    mask1, mrgb1 = seg(im2)

    #counting mask of the tree in both frame to compute overlap_ratio
    #fram1
    ys, xs = np.where(mask0 > 0)
    uv = np.stack([xs, ys], axis=1) #counting mask of the tree
    #fram2
    ys1, xs1 = np.where(mask1 > 0)
    uv1 = np.stack([xs1, ys1], axis=1) #counting mask of the tree
    overlap_ratio = min(len(uv), len(uv1)) / max(len(uv), len(uv1))
    print(f"overlap_ratio: {overlap_ratio}")

    # Run Glue Factory feature matching
    wimg0, wimg1, aligned_uv1, aligned_uv2, scores, match_pairs = matchLightGlue(mrgb0, mrgb1)

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
    result_dir = "/home/chhaileng/Documents/test_model/glue-factory/Output_error"
    
    
    # Loop over id2 > id1

    i = 20  # fixed
    results = []
    

    
    for id1 in range(1, len(images_path_list) - i):  # -4 so id2 is in range
        id2 = id1 + i  # fixed offset
        try:
            print(f"Matching {id1} -> {id2}")

            overlap_ratio, n_matches12, n_matches21, errors12, errors21 = uv1_to_uv2_loop_check(
                id1, id2, images_dir, poses_dir, depths_dir, images_path_list, K
            )

            results.append({
                "id1": id1,
                "id2": id2,
                "overlap (%)": round(overlap_ratio * 100, 2),
                # "mean_error uv1→uv2 (px)": round(mean_err_12, 2),
                # "median_error uv1→uv2 (px)": round(median_err_12, 2),
                "matches uv2→uv1": n_matches21,
                "errors uv2→uv1 (px)": errors21,
                "matches uv1→uv2": n_matches12,
                "errors uv1→uv2 (px)": errors12,
            })

        except Exception as e:
            print(f"Error on {id1}-{id2}: {e}")
            continue


    # Save to Excel
    if results:
        df = pd.DataFrame(results)
        output_file = os.path.join(result_dir,f"matching_results_Ori{i}.csv")
        # df.to_excel(output_file,  sheet_name="uv1_to_uv2", index=False)
        df.to_csv(output_file,index=False)
        print(f"Saved matching results to Excel: {output_file}")

    else:
        print(" No results to save. All matches may have failed.")