import torch
import numpy as np
import cv2
from gluefactory.utils.image import read_image
#from gluefactory.utils.tensor import numpy_to_torch
from gluefactory.utils.experiments import load_experiment
from gluefactory.eval.utils import eval_matches_homography
import os
import argparse



def numpy_to_torch(array, dtype=torch.float32):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).type(dtype)
    raise TypeError("Input should be a NumPy array.")

# ---------------- Segmentation ----------------
def seg(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)/255.0
    h, s, v = cv2.split(hsv)
    keep_mask = (h < 0.19) | (h > 0.521)
    mrgb = np.ones_like(rgb)*255
    mrgb[keep_mask] = rgb[keep_mask]
    return keep_mask.astype(bool), mrgb

# ---------------- Visualization ----------------
def draw_combined_window(img_left, img_right, kpts0, kpts1, match_array):
    canvas = np.concatenate([img_left, img_right], axis=1)
    offset = img_left.shape[1]

    # --- Draw all keypoints in both images ---
    for pt in kpts0:
        pt1 = tuple(np.round(pt).astype(int))
        cv2.circle(canvas, pt1, 2, (255, 0, 0), -1)  # Blue: image 0

    for pt in kpts1:
        pt2 = tuple(np.round(pt).astype(int) + np.array([offset, 0]))
        cv2.circle(canvas, pt2, 2, (0, 0, 255), -1)  # Red: image 1

    # --- Draw match lines for matched keypoints only ---
    for idx0, idx1 in enumerate(match_array):
        if idx1 == -1:
            continue  # Skip unmatched
        pt1 = tuple(np.round(kpts0[idx0]).astype(int))
        pt2 = tuple(np.round(kpts1[idx1]).astype(int) + np.array([offset, 0]))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)  # Green line

    return canvas


# ---------------- Matching + Drawing ----------------
def match_and_draw(img0_path, img1_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_experiment(model_path).eval().to(device)

    img0 = read_image(img0_path, grayscale=False) / 255.0
    img1 = read_image(img1_path, grayscale=False) / 255.0
    img0_uint8 = (img0 * 255).astype(np.uint8)
    img1_uint8 = (img1 * 255).astype(np.uint8)

    mask0, img0_whitebg = seg(img0_uint8)
    mask1, img1_whitebg = seg(img1_uint8)

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
    match_img = draw_combined_window(img0_whitebg, img1_whitebg, kpts0, kpts1, matches)
    scores = pred['matching_scores0'][0].cpu().numpy()

    cv2.imwrite(output_path, cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR))
    print(f"Saved match image to: {output_path}")

   

## ============================= evaluation ======================================
     # Keep all valid matches
    valid_matches = (matches > -1)
    match_pairs = np.column_stack((np.where(valid_matches)[0], matches[valid_matches]))

    # === Compute Homography ===
    src_pts = kpts0[match_pairs[:, 0]]
    dst_pts = kpts1[match_pairs[:, 1]]

    if len(src_pts) >= 4:  # Minimum 4 points needed
        homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homography = np.array(homography, dtype=np.float32)
        print("Estimated 3x3 Homography matrix:")
        print(homography)

        # Optional: save to file
       # np.savetxt("estimated_homography.txt", H)
    else:
        print("Not enough matches to compute homography.")


        # Evaluation
    pred_data = {
        "keypoints0": torch.from_numpy(kpts0),
        "keypoints1": torch.from_numpy(kpts1),
        "matches0": torch.from_numpy(matches),
        "matching_scores0": torch.from_numpy(scores),
    }

    eval_data = {
        "view0": {"image_size": img0.shape[:2]},
        "view1": {"image_size": img1.shape[:2]},
        "H_0to1": torch.from_numpy(homography),
    }

    eval_result = eval_matches_homography(eval_data, pred_data)
    print("Homography Evaluation Results:")
    for k, v in eval_result.items():
        print(f"  {k}: {v:.4f}")


# ---------------- Main ----------------
if __name__ == "__main__":
       # === HARDCODED paths (edit the base directory only) ===
    base_path = "/home/chhaileng/Documents/test_model/glue-factory/testing_tree/tree38/images"
    img0_path = os.path.join(base_path, "00000.png")  # fixed reference image
    model_path = "/home/chhaileng/Documents/test_model/glue-factory/checkpoint_best.tar"
    output_base = "/home/chhaileng/Documents/test_model/glue-factory/output_img/check_keypoint1Py"

    index = 1  # start from 1 (00001.png)

    while True:
        
        img1_path = os.path.join(base_path, f"{index:05}.png")
        output_path = os.path.join(output_base, f"match_tree_only_{index}.png")

        if not os.path.exists(img1_path):
            print(f"Image not found: {img1_path}")
            break

        match_and_draw(img0_path, img1_path, model_path, output_path)

        # Display the image
        img = cv2.imread(output_path)
        cv2.imshow("Matched Keypoints", img)

        print(f"\nShowing: {index:05}.jpg — Press any key to continue, or ESC to exit.")
        key = cv2.waitKey(0)

        if key == 27:  # ESC key
            print("Exiting.")
            break

        index += 1

    cv2.destroyAllWindows()