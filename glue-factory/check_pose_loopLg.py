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

# .tar the pth of super point and light glue 

# extractor.load

# matcher.load

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




def matchLightGlue(img0, img1):
    """
    Match keypoints between two RGB images using SuperPoint + LightGlue
    with segmentation preprocessing.
    
    Args:
        img0: RGB image (numpy array)
        img1: RGB image (numpy array)
    
    Returns:
        pts0: Nx2 array of keypoints in image0
        pts1: Nx2 array of corresponding keypoints in image1
        scores: N array of match scores
        match_pairs: Nx2 array of matched keypoint indices [index_in_img0, index_in_img1]
    """

    # Apply segmentation mask
    mask0, mrgb0 = seg(img0)
    mask1, mrgb1 = seg(img1)

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



def LightGlue_matching(img0_rgb, img1_rgb, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the feature extractor and matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Ensure uint8
    img0_uint8 = img0_rgb.astype(np.uint8)
    img1_uint8 = img1_rgb.astype(np.uint8)

    # Perform segmentation
    mask0, img0_whitebg = seg(img0_uint8)
    mask1, img1_whitebg = seg(img1_uint8)

    # Normalize to float32 [0, 1]
    img0 = img0_whitebg.astype(np.float32) / 255.0
    img1 = img1_whitebg.astype(np.float32) / 255.0

    # Convert numpy arrays → torch tensors for LightGlue
    image0 = numpy_image_to_torch(img0).to(device)
    image1 = numpy_image_to_torch(img1).to(device)

    # Feature extraction
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # Matching
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # Remove batch dimension and detach
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Extract keypoints and matches
    kpts0 = feats0["keypoints"].cpu().numpy()
    kpts1 = feats1["keypoints"].cpu().numpy()
    match_pairs = matches01["matches"].cpu().numpy()

    match_scores = matches01["scores"].cpu().numpy()

    print(f"Finished matching: {len(match_pairs)} matches found.")
    #print("Match scores:", match_scores)

  

    return kpts0, kpts1, matches01, match_pairs, img0_whitebg, img1_whitebg

## 4, project keypoints from one camera frame to another using depth and pose
def project_to_other_torch(uv, K, depth, P_1to_2):
    """
    uv : (N, 2) tensor [u, v] en torch.float32
    K : (3,3) camera intrinsic (torch.float32)
    depth : (H,W) tensor
    P_1to_2 : (4,4) tensor (homogeneous transform cam1 → cam2)
    """

    H, W = depth.shape[-2:]

    u = uv[:, 0].long()
    v = uv[:, 1].long()
    depths = depth[v, u]  # (N,)

    valid_depth = (depths > 0.1) & (depths < 1) 
    uv_valid = uv[valid_depth]
    depths_valid = depths[valid_depth]


    if uv_valid.shape[0] == 0:
        # Aucun point valide, retour vide
        return (
            torch.empty((0, 2), device=uv.device),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 2), device=uv.device),
        )

    ones = torch.ones((uv_valid.shape[0], 1), device=uv.device)
    xys_h = torch.cat([uv_valid, ones], dim=1) * depths_valid.unsqueeze(1)  # (N,3)

    K_inv = torch.inverse(K)
    Xcam1 = (K_inv @ xys_h.T).T  # (N,3)

    ones_h = torch.ones((Xcam1.shape[0], 1), device=uv.device)
    Xcam1_h = torch.cat([Xcam1, ones_h], dim=1)  # (N,4)

    Xcam2_h = (P_1to_2 @ Xcam1_h.T).T  # (N,4)
    Xcam2 = Xcam2_h[:, :3] / Xcam2_h[:, 3:].clamp(min=1e-6, max=2000)  # (N,3)

    uv2_h = (K @ Xcam2.T).T  # (N,3)
    uv2 = uv2_h[:, :2] / uv2_h[:, 2:3]  # Normalisation perspective

    # Validité dans image 2
    u2_valid = (uv2[:, 0] >= 0) & (uv2[:, 0] < W)
    v2_valid = (uv2[:, 1] >= 0) & (uv2[:, 1] < H)
    valid = u2_valid & v2_valid

    return uv2[valid], valid_depth.nonzero(as_tuple=True)[0][valid], uv_valid[valid], valid_depth


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


## 6, Visualize matches and reprojection error

def draw_matches1(im1, im2, keypoints1, keypoints2, match_pairs, max_points=None, save_path="results/matches.png"):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    if match_pairs is not None and len(match_pairs) > 0:
        canvas = np.concatenate([im1_rgb, im2_rgb], axis=1)
        offset = im1_rgb.shape[1]

        for idx0, idx1 in match_pairs:
            pt1 = tuple(np.round(keypoints1[idx0]).astype(int))  # left
            pt2 = tuple(np.round(keypoints2[idx1]).astype(int) + np.array([offset, 0]))  # right
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(canvas, pt1, 2, (255, 0, 0), -1)
            cv2.circle(canvas, pt2, 2, (0, 0, 255), -1)

        # Save using OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Saved match visualization to {save_path}")

    else:
        if im1_rgb.shape != im2_rgb.shape:
            im2_rgb = cv2.resize(im2_rgb, (im1_rgb.shape[1], im1_rgb.shape[0]))

        concat_img = np.hstack((im1_rgb, im2_rgb))
        keypoints2_shifted = keypoints2.copy()
        keypoints2_shifted[:, 0] += im1.shape[1]

        if max_points is not None and len(keypoints1) > max_points:
            indices = np.random.choice(len(keypoints1), max_points, replace=False)
            keypoints1 = keypoints1[indices]
            keypoints2_shifted = keypoints2_shifted[indices]

        plt.figure(figsize=(15, 8))
        plt.imshow(concat_img)

        for pt1, pt2 in zip(keypoints1, keypoints2_shifted):
            color = np.random.rand(3,)
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1)
            plt.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, s=5)

        plt.axis("off")
        plt.title(f"Matching points between im1 and im2 (showing {len(keypoints1)} points)")

        # Save using Matplotlib
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved match visualization to {save_path}")


def draw_matches(im1, im2, keypoints1, keypoints2, match_pairs, max_points=None):
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    if match_pairs is not None and len(match_pairs) > 0:
        canvas = np.concatenate([im1_rgb, im2_rgb], axis=1)
        offset = im1_rgb.shape[1]

        for idx0, idx1 in match_pairs:
            pt1 = tuple(np.round(keypoints1[idx0]).astype(int))  # left
            pt2 = tuple(np.round(keypoints2[idx1]).astype(int) + np.array([offset, 0]))  # right
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(canvas, pt1, 2, (255, 0, 0), -1)
            cv2.circle(canvas, pt2, 2, (0, 0, 255), -1)

        # Show using OpenCV
        cv2.imshow("Matches", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:

        if im1_rgb.shape != im2_rgb.shape:
            im2_rgb = cv2.resize(im2_rgb, (im1_rgb.shape[1], im1_rgb.shape[0]))

        concat_img = np.hstack((im1_rgb, im2_rgb))
        keypoints2_shifted = keypoints2.copy()
        keypoints2_shifted[:, 0] += im1.shape[1]

        # Limiter le nombre de points affichés
        if max_points is not None and len(keypoints1) > max_points:
            indices = np.random.choice(len(keypoints1), max_points, replace=False)
            keypoints1 = keypoints1[indices]
            keypoints2_shifted = keypoints2_shifted[indices]
        plt.figure(figsize=(15, 8))
        plt.imshow(concat_img)

        for pt1, pt2 in zip(keypoints1, keypoints2_shifted):
            color = np.random.rand(
                3,
            )
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1)
            plt.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, s=5)
        plt.axis("off")
        plt.title(f"Matching points between im1 and im2 (showing {len(keypoints1)} points)")
        plt.show()    

def draw_combined_window(cur_img, ref_img, kpts0, kpts1, matches, tracking_pts=None):
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    matches = matches.cpu().numpy()
    valid = (matches[:, 0] < len(kpts0)) & (matches[:, 1] < len(kpts1))
    matches = matches[valid]
    canvas = np.concatenate([cur_img, ref_img], axis=1)

    offset = cur_img.shape[1]
    for m in matches:
        pt1 = tuple(np.round(kpts1[m[1]]).astype(int))  # current
        pt2 = tuple(np.round(kpts0[m[0]]).astype(int) + np.array([offset, 0]))  # ref
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(canvas, pt1, 2, (255, 0, 0), -1)
        cv2.circle(canvas, pt2, 2, (0, 0, 255), -1)

    if tracking_pts is not None:
        for (x1, y1), (x2, y2) in tracking_pts:
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(canvas, pt1, pt2, (255, 255, 0), 1)
            cv2.circle(canvas, pt2, 2, (0, 255, 255), -1)

    return canvas

## 5, Compare matches to ground truth projections
# --------------------------------------------- UV1 to UV2 --------------------------------------
def uv1_to_uv2(id1, id2, images_dir, poses_dir, depths_dir, images_path_list, K):

    im1, im2, cam1_to_cam2, _, depth1, _ = load_img(id1, id2, images_dir, poses_dir, depths_dir, images_path_list)

   

    wimg0, wimg1, aligned_uv1, aligned_uv2, scores, match_pairs = matchLightGlue(im1, im2)
    # print("here", match_pairs, type(match_pairs))
    #draw_matches(img0, img1, kpts0, kpts1, match_pairs, max_points=50)

   
   
        
    # cam1 → cam2 
    uv2_gt, _, uv1_valid, uv2_pred = project_to_other(aligned_uv1, aligned_uv2, K, depth1, cam1_to_cam2)
    #ids = np.arange(uv2_gt.shape[0]).reshape(-1,1)
    #new_match_pairs = np.hstack([ids,ids])

    # draw_matches(img0, img1, uv1_valid, uv2_gt, new_match_pairs, max_points=50)
    # draw_matches(img1, img1, uv2_gt, uv2_pred, match_pairs, max_points=20)

    mask1 = seg(im1)[0]
    ys, xs = np.where(mask1 > 0)
    uv = np.stack([xs, ys], axis=1)

    all_uv2, _, _, _ = project_to_other(uv, uv, K, depth1, cam1_to_cam2)

    errors = np.linalg.norm(uv2_gt - uv2_pred, axis=1)
    mse = mean_squared_error(y_true=uv2_gt,y_pred=uv2_pred)
    overlap =len(all_uv2)/len(uv) 
    n_matches = len(errors)
    # mean_error = np.mean(errors)
    # median_error = np.median(errors)


    print(f"overlap between the frames (%): {len(all_uv2)/len(uv)} points")
    print(f"Nomber of correspondances : {len(errors)} points")
    print(
        f"mean matcher error : {np.mean(errors):.2f} pixels, median :{np.median(errors):.2f} pixels"
    )

    draw_matches(wimg0, wimg1,uv1_valid , uv2_gt, match_pairs = None, max_points=30000)
    #draw_matches1(img1, img1, uv2_gt, uv2_pred, match_pairs, max_points=3000, save_path="results/{id1}.png")
    return overlap, n_matches, errors, mse


# ------------------------------ uv1 → uv2 → uv1 (reprojection) ---------------------------------
def uv1_to_uv2_loop_check(id1, id2, images_dir, poses_dir, depths_dir, images_path_list,  K):
    # Load data
    im1, im2, cam1_to_cam2, cam2_to_cam1, depth1, depth2 = load_img(id1, id2, images_dir, poses_dir, depths_dir, images_path_list)


    # Run Glue Factory feature matching
    wimg0, wimg1, aligned_uv1, aligned_uv2, scores, match_pairs = matchLightGlue(im1, im2)
   

    # Segment foreground from first image
    mask1 = seg(im1)[0]
    ys, xs = np.where(mask1 > 0)
    uv_all = np.stack([xs, ys], axis=1)

    # Project all segmented pixels forward: cam1 → cam2
    all_uv2, _, _, uv1_vgt0 = project_to_other(aligned_uv1, aligned_uv1, K, depth1, cam1_to_cam2)

    # Forward projection: cam1 → cam2
    uv2_gt, valid_indices1, uv1_valid, uv1_gt = project_to_other(aligned_uv1, aligned_uv2, K, depth1, cam1_to_cam2)
   
    # Backward projection: cam2 → cam1
    uv1_Rproj, valid_indices2, _, uv1_vgt = project_to_other(uv2_gt, uv1_vgt0, K, depth2, cam2_to_cam1)


    
    
    print(f"[DEBUG] aligned_uv1 shape: {aligned_uv1.shape}")
    print(f"[DEBUG] uv2_gt shape: {uv2_gt.shape}")
    print(f"[DEBUG] uv1_Rproj shape: {uv1_Rproj.shape}")
    print(f"[DEBUG] uv1_gt shape: {uv1_vgt.shape}")
    # Compute errors
    errors = np.linalg.norm(uv1_Rproj - uv1_vgt, axis=1)
    errors1 = np.linalg.norm(uv2_gt - uv1_gt, axis=1)
    overlap_ratio = len(uv2_gt) / len(aligned_uv1) 

       # Stats
    n_matches = len(errors)
    n_matches1 = len(errors1)
    mean_error = np.mean(errors) if n_matches > 0 else 30000
    median_error = np.median(errors) if n_matches > 0 else 30000
    mean_error1 = np.mean(errors1) if n_matches1 > 0 else 30000
    median_error1 = np.median(errors1) if n_matches1 > 0 else 30000
    # mean_error = np.mean(errors) if n_matches > 0 else 0.0
    # median_error = np.median(errors) if n_matches > 0 else 0.0
    # mean_error1 = np.mean(errors1) if n_matches1 > 0 else 0.0
    # median_error1 = np.median(errors1) if n_matches1 > 0 else 0.0
    
    # draw_matches(img1, img2, uv2_gt, uv2_pred, match_pairs = None, max_points=None)
    return overlap_ratio, n_matches, mean_error, median_error,n_matches1, mean_error1, median_error1

if __name__ == "__main__":

    K = np.array([[397.8731, 0, 320], [0, 396.0224, 240], [0, 0, 1]])

    path = Path("/home/chhaileng/Documents/test_model/glue-factory/testing_tree/tree42_1")
    # path = Path("/home/rscherrer/Downloads/scan_1")
    images_dir = path / "images"
    depths_dir = path / "depth"
    poses_dir = path / "pose"
    images_path_list = sorted(os.listdir(images_dir), key=lambda x: int(x.split(".")[0]))
    result_dir = "/home/chhaileng/Documents/test_model/glue-factory/Output_error/white"
    
    
    # Loop over id2 > id1

    i = 20  # fixed
    results = []
    

    
    for id1 in range(1, len(images_path_list) - i):  # -4 so id2 is in range
        id2 = id1 + i  # fixed offset
        try:
            print(f"Matching {id1} -> {id2}")

            overlap, n_kp_21, mean_err_21, median_err_21, n_kp_12, mean_err_12, median_err_12 = uv1_to_uv2_loop_check(
                id1, id2, images_dir, poses_dir, depths_dir, images_path_list, K
            )

            results.append({
                "id1": id1,
                "id2": id2,
                "overlap (%)": round(overlap * 100, 2),
                "matches uv1→uv2": n_kp_12,
                "mean_error uv1→uv2 (px)": round(mean_err_12, 2),
                "median_error uv1→uv2 (px)": round(median_err_12, 2),
                "matches uv2→uv1": n_kp_21,
                "mean_error uv2→uv1 (px)": round(mean_err_21, 2),
                "median_error uv2→uv1 (px)": round(median_err_21, 2),
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