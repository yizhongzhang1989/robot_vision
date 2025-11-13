"""
SuperSift Keypoint Tracker Module
=================================
An experimental keypoint tracking module that combines SIFT and SuperPoint features to enhance keypoint tracking performance.
By leveraging the strengths of both algorithms and incorporating epipolar geometry constraints, this module delivers robust tracking across multiple reference images (requires >1 reference image).

Usage:
    from core.supersift_keypoint_tracker import SuperSiftKeypointTracker

    tracker = SuperSiftKeypointTracker()
    tracker.set_reference_image(ref_image, keypoints)
    result = tracker.track_keypoints(target_image)
    tracker.remove_reference_image()  # Clean up when finished

For examples and test cases, see: examples/supersift_keypoint_tracker_example.py

GPU-accelerated SIFT detection is highly recommended. To install pypopsift:
1. Ensure CUDA and the NCC compiler are installed (PyTorch with CUDA support is usually sufficient).
2. Install CMake (version >= 3.24). On Ubuntu: sudo snap install cmake --classic
3. Install pybind11: pip install pybind11[global]
4. Clone the repository: git clone https://github.com/OpenDroneMap/pypopsift
5. Build pypopsift:
    cd pypopsift && mkdir build && cd build && cmake .. && make -j8
6. Install the package:
    cd .. && pip install ..
"""

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
import json
import pickle
import concurrent.futures
# from scipy.optimize import minimize

# Import the base class
from core.keypoint_tracker import KeypointTracker

# Conditional imports to avoid issues when pypopsift is not installed
try:
    from pypopsift import popsift

    USE_PYPOPSIFT = True
except ImportError:
    USE_PYPOPSIFT = False

try:
    from core.utils import get_project_paths
except ImportError:
    # When running as main, imports will be handled in main function
    pass

# Additional imports for ightGlue
from PIL import Image
from transformers import LightGlueImageProcessor, LightGlueForKeypointMatching
from huggingface_hub import snapshot_download
import torch


class SuperSiftKeypointTracker(KeypointTracker):
    """SuperSift Keypoint Tracker combining SIFT and SuperPoint features.

    This keypoint tracker uses SIFT for robust keypoint detection and SuperPoint/SuperGlue
    for enhanced matching and tracking. It supports multiple reference images (>1) and incorporates
    epipolar geometry constraints to improve tracking accuracy.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        """Initialize the SuperSift Keypoint Tracker.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu').
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        if not USE_PYPOPSIFT:
            print("Warning: pypopsift is not installed. For optimal performance, especially GPU-accelerated SIFT detection, please install pypopsift.")
            print("Proceeding with OpenCV's CPU-based SIFT implementation instead.")

        self.device = device
        self.model_loaded = False

        # Get project paths (override base class paths with more complete version)
        try:
            self.paths = get_project_paths()
        except NameError:
            # Fallback when running as main
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.paths = {"project_root": project_root}

        self.__load_models()

        self.thresold_distance = 5.0
        self.downsample_factor = 4  # downsample factor for LightGlue processing
        # Data structures to hold reference images and keypoints, as well as other information
        self.template_color_images = []
        self.template_imagefilelist = []
        self.precomputed_data = []  # list of (keypoints, descriptors) for each reference image
        self.keypoints_list = []  # list of keypoints dict for each reference image
        self.keypoints_dict = {}  # combined keypoints dict across all reference images
        self.F_store = []  # fundamental matrices between consecutive reference images
        self.update_required = False  # flag to indicate if F_store needs to be updated

    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================

    def set_reference_image(self, image: np.ndarray, keypoints: Optional[List[Dict]] = None, image_name: Optional[str] = None) -> Dict:
        """Set reference image for keypoint tracking with optional image key.

        This is the first of two main public methods. Use this to store reference
        images with their associated keypoints for later tracking operations.

        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format.
            keypoints: Optional list of keypoint dictionaries with 'x', 'y' keys.
            image_name: Optional string name to identify this reference image.
                       If None, uses 'default' and sets as default reference.

        Returns:
            Dict with success status and information about the set reference image.
        """
        self.template_color_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self.__detect_and_compute(gray_image)
        self.precomputed_data.append((kp, des))
        self.keypoints_list.append(keypoints if keypoints is not None else [])
        self.template_imagefilelist.append(image_name if image_name is not None else f"template_{len(self.template_imagefilelist)}")

        self.update_required = True

    # ============================================================================

    def remove_reference_image(self, image_name: Optional[str] = None) -> Dict:
        """Remove a stored reference image by name.

        This is the third of three main public methods. Use this to clean up
        stored reference images when they are no longer needed.

        Args:
            image_name: Name of the reference image to remove. If None, removes the last reference image if it exists.

        Returns:
            Dict with removal status. Should include at least:
            {
                'success': bool,
                'removed_key': str,  # The key that was actually removed
                'remaining_count': int,
                'error': str  # Only if success=False
            }
        """
        index = -1
        if image_name in self.template_imagefilelist:
            index = self.template_imagefilelist.index(image_name)
        elif image_name is None and len(self.template_imagefilelist) > 0:
            index = len(self.template_imagefilelist) - 1

        if index != -1:
            removed_key = self.template_imagefilelist[index]
            del self.template_imagefilelist[index]
            del self.template_color_images[index]
            del self.precomputed_data[index]
            del self.keypoints_list[index]

            self.update_required = True

            return {"success": True, "removed_key": removed_key, "remaining_count": len(self.template_imagefilelist)}
        else:
            return {"success": False, "removed_key": None, "remaining_count": len(self.template_imagefilelist), "error": f"Reference image '{image_name}' not found."}

    # ============================================================================
    def track_keypoints(self, target_image: np.ndarray, reference_name: Optional[str] = None, **kwargs) -> Dict:
        """Track keypoints from stored reference image to target image.

        This is the second of three main public methods. Use this to track keypoints
        from a stored reference image to a target image.

        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format.
            reference_name: Name of stored reference image to use. If None, uses default reference.
            **kwargs: Implementation-specific parameters (e.g., bidirectional=True, return_flow=True for FFPPKeypointTracker)

        Returns:
            Dict with tracking results. Should include at least:
            {
                'success': bool,
                'tracked_keypoints': List[Dict],  # List of tracked keypoint dicts with 'x', 'y'
                'keypoints_count': int,
                'processing_time': float,  # Total processing time in seconds
                'reference_name': str,
                'error': str  # Only if success=False
            }
        """
        if not self.model_loaded:
            return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": "LightGlue model not loaded."}
        if len(self.template_color_images) < 2:
            return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": "SuperSiftKeypointTracker requires more than one reference image."}

        if self.update_required:
            self.__update_keypoints_dict()
            try:
                self.F_store = self.__compute_template_keypoint_maping()
            except Exception as e:
                return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": f"Error computing fundamental matrices: {e}"}
            if self.F_store is None or len(self.F_store) != len(self.template_color_images):
                return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": "Failed to compute valid fundamental matrices between reference images."}
            self.update_required = False

        time_start = cv2.getTickCount()
        gray_test_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        estimated_positions = self.__find_point_correspondence(gray_test_image, target_image)
        time_end = cv2.getTickCount()
        processing_time = (time_end - time_start) / cv2.getTickFrequency()
        tracked_keypoints = []
        for name, entry in estimated_positions.items():
            est_x, est_y, squared_dis = entry
            if est_x is not None and est_y is not None:
                tracked_keypoints.append({"name": name, "x": est_x, "y": est_y, "deviation": np.sqrt(squared_dis)})
        return {"success": True, "tracked_keypoints": tracked_keypoints, "keypoints_count": len(tracked_keypoints), "processing_time": processing_time, "reference_name": "Multiple References"}

    # ============================================================================
    # Private METHODS
    # ============================================================================
    def __load_models(self):
        """Load LightGlue model."""

        project_root = self.paths["project_root"]
        model_root = os.path.join(project_root, "models")
        model_id = os.path.join(model_root, "lightglue_superpoint")

        if not os.path.exists(model_id):
            snapshot_download(repo_id="ETH-CVG/lightglue_superpoint", cache_dir=model_id, local_dir=model_id)

        try:
            self.processor = LightGlueImageProcessor.from_pretrained(model_id, use_fast=True)
            self.superpoint_model = LightGlueForKeypointMatching.from_pretrained(model_id)
            self.superpoint_model.to(self.device).eval()

            # create dummy images to warm up the model
            fake_image = Image.new("RGB", (640, 480), color=(255, 255, 255))
            fake_images = [[fake_image, fake_image]]
            with torch.inference_mode():
                inputs = self.processor(fake_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                self.superpoint_model(**inputs)

            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading LightGlue model: {e}")
            raise e

    # ============================================================================
    def __update_keypoints_dict(self):
        """Update the combined keypoints dictionary across all reference images."""
        self.keypoints_dict = {}
        for i, keypoints in enumerate(self.keypoints_list):
            for kp in keypoints:
                name = kp["name"]
                x = kp["x"]
                y = kp["y"]
                dict_entry = {"template_index": i, "coordinates": (x, y)}
                if name not in self.keypoints_dict:
                    self.keypoints_dict[name] = {}
                    self.keypoints_dict[name]["info"] = []
                    self.keypoints_dict[name]["estimated_position"] = (None, None)
                self.keypoints_dict[name]["info"].append(dict_entry)

    # ============================================================================
    def __detect_and_compute(self, cv_image):
        """
        Detect keypoints and compute descriptors using SIFT or pypopsift.
        Args:
            cv_image (np.ndarray): Input image in OpenCV format (BGR or grayscale).
        Returns:
            keypoints (np.ndarray): Detected keypoints as an array of (x, y) coordinates.
            descriptors (np.ndarray): Corresponding descriptors for the keypoints.
        """
        grayimage = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        if USE_PYPOPSIFT:
            # peak_threshold = 0.04 # mimic opencv default
            peak_threshold = 0.03  # more keypoints, Lowe's paper
            keypoints, descriptors = popsift(grayimage, peak_threshold=peak_threshold, edge_threshold=10, target_num_features=0)
            keypoints = keypoints[:, :2]
        else:
            detector = cv2.SIFT_create()
            keypoints, descriptors = detector.detectAndCompute(grayimage, None)
            keypoints = np.array([kp.pt for kp in keypoints])

        return keypoints, descriptors

    # ============================================================================

    def __match_features(self, desc1, desc2):
        """
        Match features between two sets of descriptors using the specified method.
        Args:
            desc1 (np.ndarray): Descriptors from the first image.
            desc2 (np.ndarray): Descriptors from the second image.
        """

        # Use cross-check brute-force matcher (commented out for now)
        # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # good_matches = matcher.match(desc1, desc2)

        # Use FLANN-based matcher for feature matching
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches

    # ============================================================================

    def __compute_template_keypoint_maping(self):
        """
        Compute fundamental matrices between consecutive reference images.
        Returns:
            F_store (list): List of fundamental matrices between consecutive reference images (0->1, 1->2, ..., N-1->0).
        """
        F_store = []
        for i in range(len(self.keypoints_list)):
            ptlist_i = {}
            for entry_i in self.keypoints_list[i]:
                ptlist_i[entry_i["name"]] = (entry_i["x"], entry_i["y"])
            j = (i + 1) % len(self.keypoints_list)
            pts_i = []
            pts_j = []
            for entry_j in self.keypoints_list[j]:
                name_j = entry_j["name"]
                if name_j in ptlist_i:
                    pts_i.append(ptlist_i[name_j])
                    pts_j.append((entry_j["x"], entry_j["y"]))
            if len(pts_i) < 8:
                raise ValueError(f"Not enough common keypoints between template {i} and {j} to compute fundamental matrix.")
            F_ij, _ = cv2.findFundamentalMat(np.array(pts_i), np.array(pts_j), cv2.FM_LMEDS)
            F_store.append(F_ij)  # F from i to j
        return F_store

    # ============================================================================

    def __compute_fundamentalmatrices(self, keypoints1, descriptor1, keypoints2, descriptor2, ransac_thresh=0.5):
        """
        Find the fundamental matrix between two images using feature matching.
        Args:
            keypoints1 (np.ndarray): Keypoints from image1.
            template_descriptor (np.ndarray): Descriptors from image 1.
            keypoints2 (np.ndarray): Keypoints from image 2.
            descriptor2 (np.ndarray): Descriptors from image 2.
            ransac_thresh (float): RANSAC reprojection threshold.

        Returns:
            F (np.ndarray): Estimated fundamental matrix.
        """
        matches = self.__match_features(descriptor1, descriptor2)

        if len(matches) < 8:
            print("Not enough matches ({}) found to compute the fundamental matrix.").format(len(matches))
            return None, None

        pts1 = np.array([keypoints1[m.queryIdx] for m in matches], dtype=np.float32)
        pts2 = np.array([keypoints2[m.trainIdx] for m in matches], dtype=np.float32)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=ransac_thresh)
        return F, mask.sum() / len(matches)

    # ============================================================================

    def __point_to_epipolar_distance(self, F, point1, point2, cross_distance=True):
        """
        Calculate the distance from point2 to the epipolar line at point1 using the fundamental matrix F, and vice versely if cross_distance is True.
        Args:
            F (np.ndarray): Fundamental matrix.
            point1 (tuple): Coordinates of the first point (x1, y1).
            point2 (tuple): Coordinates of the second point (x2, y2).
            cross_distance (bool): Whether to calculate the distance from point1 to the epipolar line at point2.
        Returns:
            distance (float): Distance from point2 to the epipolar line at point1.
            distance_back (float, optional): Distance from point1 to the epipolar line at point2.
        """
        a, b, c = np.ravel(F @ np.array([point1[0], point1[1], 1]).T)
        d = np.sqrt(a**2 + b**2)
        distance = abs(a * point2[0] + b * point2[1] + c) / d
        if cross_distance:
            a, b, c = np.ravel(F.T @ np.array([point2[0], point2[1], 1]).T)
            d = np.sqrt(a**2 + b**2)
            distance_back = abs(a * point1[0] + b * point1[1] + c) / d
            return distance, distance_back
        else:
            return distance

    # ============================================================================

    def __filter_matched_points_via_epipolar_constraint(self, test_keypoints_map_list):
        """
        Filter matched keypoints using epipolar geometry constraints.
        Args:
            test_keypoints_map_list (list): List of dictionaries mapping test keypoints to template keypoints for each reference image.
        Returns:
            F_dict (dict): Dictionary of fundamental matrices for each reference image.
            Ratio_dict (dict): Dictionary of inlier ratios for each reference image.
        """
        F_dict, Ratio_dict = {}, {}
        for idx, matches in enumerate(test_keypoints_map_list):
            filtered_matches = []
            for (tx, ty), (template_x, template_y) in matches.items():
                filtered_matches.append((tx, ty, template_x, template_y))

            next_idx = (idx + 1) % len(self.precomputed_data)
            prev_idx = (idx - 1 + len(self.precomputed_data)) % len(self.precomputed_data)
            F = self.F_store[idx]
            invF = self.F_store[prev_idx].T
            for (tx, ty), (template_x, template_y) in matches.items():
                if (tx, ty) in test_keypoints_map_list[next_idx]:
                    x, y = test_keypoints_map_list[next_idx][(tx, ty)]
                    distance, distance_back = self.__point_to_epipolar_distance(F, (template_x, template_y), (x, y), True)
                    if max(distance, distance_back) < self.thresold_distance:
                        filtered_matches.append((tx, ty, template_x, template_y))
                        continue

                if (tx, ty) in test_keypoints_map_list[prev_idx]:
                    x, y = test_keypoints_map_list[prev_idx][(tx, ty)]
                    distance, distance_back = self.__point_to_epipolar_distance(invF, (template_x, template_y), (x, y), True)
                    if max(distance, distance_back) < self.thresold_distance:
                        filtered_matches.append((tx, ty, template_x, template_y))

            if len(filtered_matches) < 7:
                print(f"Not enough matches ({len(filtered_matches)}) after filtering between test image and template {idx}.")
                continue
            filter_testp = np.array([[tx, ty] for tx, ty, template_x, template_y in filtered_matches], dtype=np.float32)
            filter_templatep = np.array([[template_x, template_y] for tx, ty, template_x, template_y in filtered_matches], dtype=np.float32)

            F, mask = cv2.findFundamentalMat(filter_testp, filter_templatep, cv2.FM_7POINT if len(filtered_matches) == 7 else cv2.FM_RANSAC, ransacReprojThreshold=0.618)
            mask = mask.ravel().astype(bool)
            # print(f"Template {idx}: {mask.sum()} inliers found out of {len(filtered_matches)} matches.")
            F_dict[idx] = F
            Ratio_dict[idx] = mask.sum() / len(filtered_matches) if len(filtered_matches) > 0 else 0.0
        return F_dict, Ratio_dict

    # ============================================================================

    def __compute_findamental_matrices_via_superpoint(self, color_test_image, test_kp, test_des):
        """
        Compute fundamental matrices between the test image and each reference image using SuperPoint features.
        Args:
            color_test_image (np.ndarray): Test image in RGB format.
            test_kp (np.ndarray): Keypoints from the test image.
            test_des (np.ndarray): Descriptors from the test image.
        Returns:
            F_dict (dict): Dictionary of fundamental matrices for each reference image.
            Ratio_dict (dict): Dictionary of inlier ratios for each reference image.
        """
        F_dict = {}
        Ratio_dict = {}

        test_keypoints_map_list = []

        # pil_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        pil_image = Image.fromarray(color_test_image)
        downsample_factor = self.downsample_factor
        pil_image = pil_image.resize((pil_image.width // downsample_factor, pil_image.height // downsample_factor))

        # Split computation into batches to avoid GPU OOM
        batch_size = 1
        images = []
        for id, template_image in enumerate(self.template_color_images):
            pil_template_image = Image.fromarray(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
            pil_template_image = pil_template_image.resize((pil_template_image.width // downsample_factor, pil_template_image.height // downsample_factor))
            images.append([pil_image, pil_template_image])

        processed_outputs = []
        for start_idx in range(0, len(images), batch_size):
            batch_images = images[start_idx : min(start_idx + batch_size, len(images))]
            with torch.inference_mode():
                inputs = self.processor(batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.superpoint_model(**inputs)
                image_sizes = [[(image.height, image.width) for image in pair_image] for pair_image in batch_images]
                batch_processed = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
                processed_outputs.extend(batch_processed)

        for id in range(len(self.template_color_images)):
            test_keypoint_map = {}
            keypoint0 = downsample_factor * processed_outputs[id]["keypoints0"].cpu().numpy()
            keypoint1 = downsample_factor * processed_outputs[id]["keypoints1"].cpu().numpy()
            # print(f"SuperPoint found {len(keypoint0)} matches between test image and template {id}.")
            for p, q in zip(keypoint0, keypoint1):
                test_keypoint_map[(int(p[0]), int(p[1]))] = (int(q[0]), int(q[1]))

            matches = self.__match_features(test_des, self.precomputed_data[id][1])
            for m in matches:
                tx, ty = test_kp[m.queryIdx]
                # if (tx, ty) not in test_keypoint_map:
                template_x, template_y = self.precomputed_data[id][0][m.trainIdx]
                test_keypoint_map[(int(tx), int(ty))] = (int(template_x), int(template_y))

            test_keypoints_map_list.append(test_keypoint_map)

        return self.__filter_matched_points_via_epipolar_constraint(test_keypoints_map_list)

    # ============================================================================
    def __compute_findamental_matrices_via_cross_check(self, test_kp, test_des):
        test_keypoints_map_list = []
        test_keypoint_map = []
        for idx, (template_kp, template_des) in enumerate(self.precomputed_data):
            test_keypoint_map = {}
            matches = self.__match_features(test_des, template_des)
            for m in matches:
                tx, ty = test_kp[m.queryIdx]
                template_x, template_y = template_kp[m.trainIdx]
                test_keypoint_map[(int(tx), int(ty))] = (int(template_x), int(template_y))
            test_keypoints_map_list.append(test_keypoint_map)

        return self.__filter_matched_points_via_epipolar_constraint(test_keypoints_map_list)

    # ============================================================================

    def __compute_findamental_matrices_directly(self, test_kp, test_des):
        # Parallelize the computation of fundamental matrices
        def compute_F_for_template(args):
            idx, (template_kp, template_des), test_kp, test_des = args
            F, ratio = self.__compute_fundamentalmatrices(test_kp, test_des, template_kp, template_des)
            return idx, F, ratio

        F_dict = {}
        Ratio_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args_list = [(idx, data, test_kp, test_des) for idx, data in enumerate(self.precomputed_data)]
            results = list(executor.map(compute_F_for_template, args_list))
            for idx, F, ratio in results:
                if F is not None:
                    F_dict[idx] = F
                    Ratio_dict[idx] = ratio
        return F_dict, Ratio_dict

    # ============================================================================

    def __gather_coefficients(self, px, py, x, y, F, ratio=1.0):
        """
        Gather coefficients for the epipolar constraint quadratic form.
        Args:
            px (float): x-coordinate of the point in the first image.
            py (float): y-coordinate of the point in the first image.
            x (float): x-coordinate of the point in the second image.
            y (float): y-coordinate of the point in the second image.
            F (np.ndarray): Fundamental matrix.
            ratio (float): Scaling ratio for the coefficients.
        Returns:
            tuple: Coefficients (px2_coeff, py2_coeff, pxpy_coeff, px_coeff, py_coeff, constant_term).
        """
        A = F[0, 0] * x + F[1, 0] * y + F[2, 0]
        B = F[0, 1] * x + F[1, 1] * y + F[2, 1]
        C = F[0, 2] * x + F[1, 2] * y + F[2, 2]
        D = (F[0, 0] * px + F[0, 1] * py + F[0, 2]) ** 2 + (F[1, 0] * px + F[1, 1] * py + F[1, 2]) ** 2
        invF = F.T
        a = invF[0, 0] * x + invF[0, 1] * y + invF[0, 2]
        b = invF[1, 0] * x + invF[1, 1] * y + invF[1, 2]
        c = invF[2, 0] * x + invF[2, 1] * y + invF[2, 2]
        d = a**2 + b**2
        px2_coeff = A**2 / D + a**2 / d
        py2_coeff = B**2 / D + b**2 / d
        pxpy_coeff = (2 * A * B) / D + (2 * a * b) / d
        px_coeff = (2 * A * C) / D + (2 * a * c) / d
        py_coeff = (2 * B * C) / D + (2 * b * c) / d
        constant_term = (C**2) / D + (c**2) / d
        return ratio * px2_coeff, ratio * py2_coeff, ratio * pxpy_coeff, ratio * px_coeff, ratio * py_coeff, ratio * constant_term

    # ============================================================================

    def __find_point_correspondence(self, gray_test_image, color_test_image, method="superpoint"):
        """
        Find marked point correspondences between the test image (gray) and reference images.
        Args:
            gray_test_image (np.ndarray): Test image in OpenCV format (gray).
            color_test_image (np.ndarray): Color test image in OpenCV format (RGB).
            method (str): Method to compute fundamental matrices ('superpoint', 'cross_check', 'direct').
        Returns:
            estimated_positions (dict): Estimated positions of marked points in the test image.
        """
        estimated_positions = {}

        test_kp, test_des = self.__detect_and_compute(gray_test_image)
        if method == "direct":
            F_dict, Ratio_dict = self.__compute_findamental_matrices_directly(test_kp, test_des)
        elif method == "cross_check":
            F_dict, Ratio_dict = self.__compute_findamental_matrices_via_cross_check(test_kp, test_des)
        else:  # default to superpoint method
            F_dict, Ratio_dict = self.__compute_findamental_matrices_via_superpoint(color_test_image, test_kp, test_des)

        for name, entries in self.keypoints_dict.items():
            if len(entries["info"]) < 2:
                continue

            # let p = (px, py, 1) be a point in the target image, where px, py are unknowns
            def loss_func(fx, user_data):
                px, py = fx[0], fx[1]
                f = 0
                sum_ratio = 0
                for entry in user_data[0]:
                    template_index = entry["template_index"]
                    x, y = entry["coordinates"]
                    if template_index not in user_data[1]:
                        # print("    No fundamental matrix found for this template index.")
                        continue
                    px2_coeff, py2_coeff, pxpy_coeff, px_coeff, py_coeff, constant_term = self.__gather_coefficients(px, py, x, y, user_data[1][template_index], user_data[2][template_index])
                    f += px2_coeff * px**2 + py2_coeff * py**2 + pxpy_coeff * px * py + px_coeff * px + py_coeff * py + constant_term
                    sum_ratio += user_data[2][template_index]
                return f / sum_ratio if sum_ratio > 0 else f

            iteration = 5  # number of iterations for nonlinear quadratic minimization, should be enough
            prev_loss = float("inf")
            px, py = 0, 0 # initial guess, does not matter
            for it in range(iteration):
                px2_coeff = py2_coeff = pxpy_coeff = px_coeff = py_coeff = constant_term = 0
                for entry in entries["info"]:
                    template_index = entry["template_index"]
                    x, y = entry["coordinates"]

                    if template_index not in F_dict:
                        # print("No fundamental matrix found for this template index.")
                        continue
                    coeffs = self.__gather_coefficients(px, py, x, y, F_dict[template_index], Ratio_dict[template_index])
                    px2_coeff += coeffs[0]
                    py2_coeff += coeffs[1]
                    pxpy_coeff += coeffs[2]
                    px_coeff += coeffs[3]
                    py_coeff += coeffs[4]
                    constant_term += coeffs[5]

                # solve for px and py by minimizing the quadratic form
                # px2_coeff * px^2 + py2_coeff * py^2 + pxpy_coeff * px * py + px_coeff * px + py_coeff * py + constant_term
                # take derivatives and set to zero
                A_matrix = np.array([[2 * px2_coeff, pxpy_coeff], [pxpy_coeff, 2 * py2_coeff]])
                b_vector = np.array([-px_coeff, -py_coeff])
                try:
                    solution = np.linalg.solve(A_matrix, b_vector)
                    px, py = solution[0], solution[1]
                    # print the quadratic form values
                    quad_value = px2_coeff * px**2 + py2_coeff * py**2 + pxpy_coeff * px * py + px_coeff * px + py_coeff * py + constant_term
                    if abs(prev_loss - quad_value) < 1e-6:
                        break
                    prev_loss = quad_value
                except np.linalg.LinAlgError:
                    raise ValueError("Singular matrix encountered while solving for estimated position.")

            user_data = [entries["info"], F_dict, Ratio_dict]

            # the following two lines are optional, using scipy's minimizer to further refine the result. But it should be disable under multithreading as the minimizer is not thread-safe.
            # res = minimize(loss_func, x0=np.array([px, py]), args=(user_data, ), method="BFGS")
            # px, py = res.x[0], res.x[1]
            quad_value = max(loss_func((px, py), user_data), 0)
            estimated_positions[name] = (px, py, quad_value / (2 * len(entries["info"])))

        return estimated_positions
