# =============================================================================
# SuperSift Keypoint Tracker Module
# =============================================================================
# An experimental keypoint tracking module that combines SIFT and SuperPoint features
# to enhance keypoint tracking performance. By leveraging the strengths of both algorithms
# and incorporating epipolar geometry constraints, this module delivers robust tracking
# across multiple reference images (requires >1 reference images).
#
# Usage:
#     from core.supersift_keypoint_tracker import SuperSiftKeypointTracker
#
#     tracker = SuperSiftKeypointTracker()
#
#     # Set reference image(s) with keypoints, the number of reference images should be more than 1
#     tracker.set_reference_image(ref_image1, keypoints1, image_name="ref1")
#     tracker.set_reference_image(ref_image2, keypoints2, image_name="ref2")
#     ...
#
#     # Recommend: you can also load multiple reference images from a folder directly
#     # tracker.load_all_reference_images(reference_image_folder)
#
#     result = tracker.track_keypoints(target_image)
#
#     # Clean up when finished (optional)
#     tracker.remove_reference_image("ref1")
#     tracker.remove_reference_image("ref2")
#     ...
#     # or clear all reference images (recommended)
#     tracker.remove_all_reference_images()
#
# For examples and test cases, see: examples/supersift_keypoint_tracker_example.py
#
# GPU-accelerated SIFT detection is highly recommended. To install pypopsift:
# 1. Ensure CUDA and the NCC compiler are installed (PyTorch with CUDA support is usually sufficient).
# 2. Install CMake (version >= 3.24). On Ubuntu: sudo snap install cmake --classic
# 3. Install pybind11: pip install pybind11[global]
# 4. Clone the repository: git clone https://github.com/OpenDroneMap/pypopsift
# 5. Build pypopsift:
#     cd pypopsift && mkdir build && cd build && cmake .. && make -j8
# 6. Install the package:
#     cd .. && pip install ..
#
# =============================================================================
#
# Algorithm description --- Yang Liu, Nov. 2025
#
# 1. Reference Image Setup
#    - Start with a set of reference images (typically more than one; three is common).
#    - These images share overlapping annotated points, most of them appear in multiple reference images.
#    - Assume there are enough annotated keypoints to compute the fundamental matrix for each adjacent pair of reference images.
#    - Let the reference images be denoted as: \(I_0, I_1, ..., I_{N-1}\)
#    - Their corresponding fundamental matrices: \(F_{0->1}, F_{1->2}, ..., F_{N-1->0}\)
#    - For simplicity, only consider these adjacent pairs and avoid other combinations.
#
# 2. Matching Keypoints Between Target and Reference Images
#    - For a given target image:
#      - Compute keypoint matches with each reference image using:
#        - SuperPoint + LightGlue
#        - SIFT + FLANN
#      - Filter these matches using epipolar geometry constraints derived from the fundamental matrices computed in Step 1.
#    - Filtering process:
#      - Consider two matches:
#        - (X_t, X_r): match between target image and reference image I_r
#        - (X_t, X_{r+1}): match between target image and reference image I_{r+1}
#      - Compute:
#        - Distance from X_r to the epipolar line defined by F_{r->r+1} and X_{r+1}
#      - If the distance is below a threshold, keep the match.
#      - Repeat for the pair (X_t, X_{r-1}).
#    - This step removes outliers and improves matching accuracy.
#
# 3. Compute Fundamental Matrices for Target-Reference Pairs
#    - Using the filtered matches from Step 2, compute the fundamental matrices between the target image and each reference image: F_{t->i}.
#
# 4. Estimate Keypoint Positions in Target Image
#    - For each common keypoint:
#      - Estimate its position in the target image using:
#        - Fundamental matrices from Step 3
#        - Epipolar constraints
#      - The estimation is achieved by solving a minimization problem:
#        - Let X_t be the unknown, and its corresponding positions in reference images be Y_0, Y_2, ..., Y_{N-1}.
#        - Minimize the sum of squared epipolar distances:
#          - Distance from Y_i to the epipolar line defined by F_{t->r_i} and X_t
#          - Distance from X_t to the epipolar line defined by F^T_{t->r_i} and Y_i
#      - This approach provides:
#        - A robust estimate of keypoint positions in the target image
#        - A confidence score based on the distance loss
#
# =============================================================================

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import json
import cv2
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

# install LightGlue via:
# git clone https://github.com/cvg/LightGlue.git && cd LightGlue
# python -m pip install -e .
# Using LightGlue
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import rbd
import torch


# ============================================================================
class ImageFeature:
    def __init__(self, siftfeature=None, superpointfeature=None, width=0, height=0, imagename="", keypoints=None):
        self.data = {
            "sift": siftfeature,  # SIFT features
            "superpoint": superpointfeature,  # SuperPoint features
            "width": width,  # image dimensions
            "height": height,  # image dimensions
            "name": imagename,  # image file name
            "keypoints": keypoints if keypoints is not None else [],  # list of keypoint dicts
        }

    def get(self, key):
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in ImageFeature data.")
        return self.data.get(key, None)

    def set(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in ImageFeature data.")


# ============================================================================


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
        # use kwargs to override default parameters if needed
        if "thresold_distance" in kwargs:
            self.thresold_distance = kwargs["thresold_distance"]
        if "ransac_threshold" in kwargs:
            self.ransac_reproj_threshold = kwargs["ransac_threshold"]
        self.thresold_distance = 5.0  # Epipolar distance threshold for filtering matches
        self.ransac_threshold = 0.618  # RANSAC reprojection threshold
        self.filter_factor = 0.6  # Filtering factor for sift matches

        # Data structures to hold reference images and keypoints, as well as other information
        self.template_image_data = []  # list of ImageFeature objects for each reference image

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
        self.template_image_data.append(
            self.__create_image_data(
                color_image=image,  #
                keypoints=keypoints,  #
                image_name=image_name if image_name is not None else f"template_{len(self.template_image_data)}",  #
                color_order="RGB",  #
            )  #
        )
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
        if image_name is not None:
            index = next((i for i, img in enumerate(self.template_image_data) if img.get("name") == image_name), -1)
        elif image_name is None and len(self.template_image_data) > 0:
            index = len(self.template_image_data) - 1

        if index != -1:
            removed_key = self.template_image_data[index].get("name")
            del self.template_image_data[index]

            self.update_required = True

            return {"success": True, "removed_key": removed_key, "remaining_count": len(self.template_image_data)}
        else:
            return {"success": False, "removed_key": None, "remaining_count": len(self.template_image_data), "error": f"Reference image '{image_name}' not found."}

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
        if len(self.template_image_data) < 2:
            return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": "SuperSiftKeypointTracker requires more than one reference image."}

        if self.update_required:
            self.__update_keypoints_dict()
            try:
                self.F_store = self.__compute_template_keypoint_maping()
            except Exception as e:
                return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": f"Error computing fundamental matrices: {e}"}
            if self.F_store is None or len(self.F_store) != len(self.template_image_data):
                return {"success": False, "tracked_keypoints": [], "keypoints_count": 0, "processing_time": 0.0, "reference_name": "Multiple References", "error": "Failed to compute valid fundamental matrices between reference images."}
            self.update_required = False

        time_start = cv2.getTickCount()
        test_image_data = self.__create_image_data(color_image=target_image, color_order="RGB")
        estimated_positions = self.__find_point_correspondence(test_image_data)
        time_end = cv2.getTickCount()
        processing_time = (time_end - time_start) / cv2.getTickFrequency()
        tracked_keypoints = []
        for name, entry in estimated_positions.items():
            est_x, est_y, squared_dis = entry
            if est_x is not None and est_y is not None:
                tracked_keypoints.append({"name": name, "x": est_x, "y": est_y, "deviation": np.sqrt(squared_dis)})
        return {"success": True, "tracked_keypoints": tracked_keypoints, "keypoints_count": len(tracked_keypoints), "processing_time": processing_time, "reference_name": "Multiple References"}

    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================

    def load_all_reference_images(self, image_folder: str):
        """
        Load all reference images and their keypoints from the specified folder.
        Args:
            image_folder (str): Path to the folder containing template images and JSON files.
        """
        self.__load_templates(image_folder)
        if len(self.template_image_data) == 0:
            raise ValueError("No template images found in the specified folder.")
        self.F_store = self.__compute_template_keypoint_maping()
        self.update_required = False

    # ============================================================================

    def remove_all_reference_images(self):
        """
        Remove all stored reference images and associated data.
        """
        self.template_image_data = []
        self.keypoints_dict = {}
        self.F_store = []
        self.update_required = False

    # ============================================================================
    # Private METHODS
    # ============================================================================

    def __load_models(self):
        """Load LightGlue model."""

        try:
            self.superpoint_extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
            self.superpoint_matcher = LightGlue(features="superpoint").eval().to(self.device)
            # create a dummy run to load the model onto GPU
            dummy_image = torch.rand(1, 3, 240, 320).to(self.device)
            with torch.inference_mode():
                dummy_features = self.superpoint_extractor({"image": dummy_image})
                _ = self.superpoint_matcher({"image0": dummy_features, "image1": dummy_features})
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading LightGlue model: {e}")
            print("Proceeding without LightGlue model functionality. Performance may be degraded.")
            self.model_loaded = False
            # raise e

    # ============================================================================
    def __create_image_data(self, color_image: np.ndarray, gray_image: Optional[np.ndarray] = None, keypoints=None, image_name="", color_order="RGB"):
        """
        Create ImageFeature object from image
        Args:
            color_image (np.ndarray): Color image in RGB or BGR format. (H, W, 3)
            gray_image (np.ndarray, optional): Grayscale version of the image. If None, it will be computed.
            keypoints (list, optional): List of keypoint dictionaries.
            image_name (str, optional): Name of the image.
            color_order (str): Color order of the input color_image ("RGB" or "BGR").
        """
        if gray_image is None:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY) if color_order == "RGB" else cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        sift_kp, sift_des = self.__detect_and_compute(gray_image)
        torch_image = self.__convert2tensor(color_image, order=color_order)
        with torch.inference_mode():
            superpoint_feat = self.superpoint_extractor({"image": torch_image}) if self.model_loaded else None
        return ImageFeature(  #
            siftfeature=(sift_kp, sift_des),  #
            superpointfeature=superpoint_feat,  #
            width=color_image.shape[1],  #
            height=color_image.shape[0],  #
            imagename=image_name,  #
            keypoints=keypoints,
        )  #

    # ============================================================================

    def __convert2tensor(self, image, order="RGB"):
        """Convert OpenCV image to Torch tensor."""
        if order == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        torch_image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)[None].to(self.device) / 255.0
        return torch_image

    # ============================================================================

    def __load_templates(self, input_folder: str):
        """
        Load template images and their keypoints from the specified folder.
        Precompute the neccessary data for each template image.
        Args:
            input_folder (str): Path to the folder containing template images and JSON files.
        """

        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder {input_folder} does not exist.")

        json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

        self.template_image_data = []

        for json_file in json_files:
            json_path = os.path.join(input_folder, json_file)
            image_path = json_path.replace(".json", ".jpg")
            if not os.path.exists(image_path):
                raise ValueError(f"Image file {image_path} not found for JSON file {json_file}")

            with open(json_path, "r") as f:
                data = json.load(f)
            keypoints = data.get("keypoints")
            if keypoints is None:
                raise ValueError(f"No keypoints found in JSON file {json_file}")

            self.template_image_data.append(
                self.__create_image_data(
                    color_image=cv2.imread(image_path, cv2.IMREAD_COLOR),  #
                    gray_image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),  #
                    keypoints=keypoints,  #
                    image_name=os.path.basename(image_path),  #
                    color_order="BGR",
                )  #
            )

        self.__update_keypoints_dict()

    # ============================================================================
    def __update_keypoints_dict(self):
        """Update the combined keypoints dictionary across all reference images."""
        self.keypoints_dict = {}

        for i in range(len(self.template_image_data)):
            for kp in self.template_image_data[i].get("keypoints"):
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

    def __detect_and_compute(self, cv_image: np.ndarray, feature_type="sift"):
        """
        Detect keypoints and compute descriptors using SIFT or pypopsift.
        Args:
            cv_image (np.ndarray): Input image in OpenCV format (BGR or grayscale).
            feature_type (str): Type of feature detector to use ("sift").
        Returns:
            keypoints (np.ndarray): Detected keypoints as an array of (x, y) coordinates.
            descriptors (np.ndarray): Corresponding descriptors for the keypoints.
        """
        grayimage = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        if feature_type == "sift":
            if USE_PYPOPSIFT:
                # peak_threshold = 0.04 # mimic opencv default
                peak_threshold = 0.03  # more keypoints, Lowe's paper
                keypoints, descriptors = popsift(grayimage, peak_threshold=peak_threshold, edge_threshold=10, target_num_features=0)
                keypoints = keypoints[:, :2]
            else:
                detector = cv2.SIFT_create()
                keypoints, descriptors = detector.detectAndCompute(grayimage, None)
                keypoints = np.array([kp.pt for kp in keypoints])
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return keypoints, descriptors

    # ============================================================================

    def __match_sift_features(self, image0_data: ImageFeature, image1_data: ImageFeature):
        """
        Match features between two sets of sift features.
        Args:
            image0_data (ImageFeature): ImageFeature object for the first image.
            image1_data (ImageFeature): ImageFeature object for the second image.
        """
        # Use FLANN-based matcher for feature matching
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 64}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        matches = matcher.knnMatch(image0_data.get("sift")[1], image1_data.get("sift")[1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.filter_factor * n.distance:
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
        for i in range(len(self.template_image_data)):
            ptlist_i = {}
            for entry_i in self.template_image_data[i].get("keypoints"):
                ptlist_i[entry_i["name"]] = (entry_i["x"], entry_i["y"])
            j = (i + 1) % len(self.template_image_data)
            pts_i = []
            pts_j = []
            for entry_j in self.template_image_data[j].get("keypoints"):
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

    def __compute_findamental_matrices(self, image0_data: ImageFeature, image1_data: ImageFeature, ransac_thresh: float = 0.5):
        """
        Find the fundamental matrix between two images using feature matching.
        Args:
            image0_data (ImageFeature): ImageFeature object for the first image.
            image1_data (ImageFeature): ImageFeature object for the second image.
            ransac_thresh (float): RANSAC reprojection threshold.

        Returns:
            F (np.ndarray): Estimated fundamental matrix.
        """
        matches = self.__match_sift_features(image0_data, image1_data)

        if len(matches) < 8:
            print("Not enough matches ({}) found to compute the fundamental matrix.".format(len(matches)))
            return None, None

        pts1 = np.array([image0_data.get("sift")[0][m.queryIdx] for m in matches], dtype=np.float32)
        pts2 = np.array([image1_data.get("sift")[0][m.trainIdx] for m in matches], dtype=np.float32)
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

            # for (tx, ty), (template_x, template_y) in matches.items():
            #     filtered_matches.append((tx, ty, template_x, template_y))

            next_idx = (idx + 1) % len(self.template_image_data)
            prev_idx = (idx - 1 + len(self.template_image_data)) % len(self.template_image_data)

            F = self.F_store[idx]
            invF = self.F_store[prev_idx].T
            for (tx, ty), entries in matches.items():
                find = False
                best_template_x, best_template_y = -1, -1
                best_distance = self.thresold_distance
                for template_x, template_y in entries:
                    if (tx, ty) in test_keypoints_map_list[next_idx]:
                        kp_list = test_keypoints_map_list[next_idx][(tx, ty)]
                        if not isinstance(kp_list, list) or len(kp_list) == 0:
                            print(type(kp_list), len(kp_list))
                            raise ValueError("kp_list must be a non-empty list of template keypoints.")
                        for x, y in kp_list:
                            distance, distance_back = self.__point_to_epipolar_distance(F, (template_x, template_y), (x, y), True)
                            dis = max(distance, distance_back)
                            if dis < best_distance:
                                best_distance = dis
                                best_template_x, best_template_y = template_x, template_y
                                find = True
                            continue

                    if (tx, ty) in test_keypoints_map_list[prev_idx]:
                        kp_list = test_keypoints_map_list[prev_idx][(tx, ty)]
                        for x, y in kp_list:
                            distance, distance_back = self.__point_to_epipolar_distance(invF, (template_x, template_y), (x, y), True)
                            dis = max(distance, distance_back)
                            if dis < best_distance:
                                best_distance = dis
                                best_template_x, best_template_y = template_x, template_y
                                find = True
                if find:
                    filtered_matches.append((tx, ty, best_template_x, best_template_y))

            if len(filtered_matches) < 7:
                print(f"Not enough matches ({len(filtered_matches)}) after filtering between test image and template {idx}.")
                continue
            filter_testp = np.array([[tx, ty] for tx, ty, template_x, template_y in filtered_matches], dtype=np.float32)
            filter_templatep = np.array([[template_x, template_y] for tx, ty, template_x, template_y in filtered_matches], dtype=np.float32)

            F, mask = cv2.findFundamentalMat(filter_testp, filter_templatep, cv2.FM_7POINT if len(filtered_matches) == 7 else cv2.FM_RANSAC, ransacReprojThreshold=self.ransac_threshold)
            mask = mask.ravel().astype(bool)
            # print(f"Template {idx}: {mask.sum()} inliers found out of {len(filtered_matches)} matches.")
            F_dict[idx] = F
            Ratio_dict[idx] = mask.sum() / len(filtered_matches) if len(filtered_matches) > 0 else 0.0
        return F_dict, Ratio_dict

    # ============================================================================

    def __compute_findamental_matrices_via_superpoint(self, test_image_data):
        """
        Compute fundamental matrices between the test image and each reference image using SuperPoint features.
        Args:
            test_image_data (ImageFeature): ImageFeature object for the test image.
        Returns:
            F_dict (dict): Dictionary of fundamental matrices for each reference image.
            Ratio_dict (dict): Dictionary of inlier ratios for each reference image.
        """
        test_keypoints_map_list = []

        for id in range(len(self.template_image_data)):
            with torch.inference_mode():
                matches01 = self.superpoint_matcher({"image0": test_image_data.get("superpoint"), "image1": self.template_image_data[id].get("superpoint")})
            feats0, feats1, matches01 = [rbd(x) for x in [test_image_data.get("superpoint"), self.template_image_data[id].get("superpoint"), matches01]]  # remove batch dimension
            matches = matches01["matches"]  # indices with shape (K,2)
            sp_keypoint0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            sp_keypoint1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
            # print(f"SuperPoint found {len(sp_keypoint0)} matches between test image and template {id}.")

            test_keypoint_map = {}
            for p, q in zip(sp_keypoint0, sp_keypoint1):
                if (int(p[0]), int(p[1])) not in test_keypoint_map:
                    test_keypoint_map[(int(p[0]), int(p[1]))] = []
                test_keypoint_map[(int(p[0]), int(p[1]))].append((int(q[0]), int(q[1])))

            matches = self.__match_sift_features(test_image_data, self.template_image_data[id])
            for m in matches:
                tx, ty = test_image_data.get("sift")[0][m.queryIdx]
                template_x, template_y = self.template_image_data[id].get("sift")[0][m.trainIdx]
                if (int(tx), int(ty)) not in test_keypoint_map:
                    test_keypoint_map[(int(tx), int(ty))] = []
                test_keypoint_map[(int(tx), int(ty))].append((int(template_x), int(template_y)))

            test_keypoints_map_list.append(test_keypoint_map)

        return self.__filter_matched_points_via_epipolar_constraint(test_keypoints_map_list)

    # ============================================================================

    def __compute_findamental_matrices_via_cross_check(self, test_image_data):
        """
        Compute fundamental matrices between the test image and each reference image using sift feature matching and epipolar constraints.
        Args:
            test_image_data (ImageFeature): ImageFeature object for the test image.
        Returns:
            F_dict (dict): Dictionary of fundamental matrices for each reference image.
            Ratio_dict (dict): Dictionary of inlier ratios for each reference image.
        """
        test_keypoints_map_list = []
        test_keypoint_map = []
        for tp_image_data in self.template_image_data:
            test_keypoint_map = {}
            matches = self.__match_sift_features(test_image_data, tp_image_data)
            for m in matches:
                tx, ty = test_image_data.get("sift")[0][m.queryIdx]
                template_x, template_y = tp_image_data.get("sift")[0][m.trainIdx]
                test_keypoint_map[(int(tx), int(ty))] = (int(template_x), int(template_y))
            test_keypoints_map_list.append(test_keypoint_map)

        return self.__filter_matched_points_via_epipolar_constraint(test_keypoints_map_list)

    # ============================================================================

    def __compute_findamental_matrices_directly(self, test_image_data):
        """
        Compute fundamental matrices between the test image and each reference image directly using sift feature matching.
        Args:
            test_image_data (ImageFeature): ImageFeature object for the test image.
        Returns:
            F_dict (dict): Dictionary of fundamental matrices for each reference image.
            Ratio_dict (dict): Dictionary of inlier ratios for each reference image.
        """

        # Parallelize the computation of fundamental matrices
        def compute_F_for_template(args):
            idx, test_image_data, template_image_data = args
            F, ratio = self.__compute_fundamentalmatrices(test_image_data, template_image_data, ransac_thresh=self.ransac_threshold)
            return idx, F, ratio

        F_dict = {}
        Ratio_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args_list = [(idx, test_image_data, self.template_image_data[idx]) for idx, data in enumerate(self.template_image_data)]
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

    def __find_point_correspondence(self, test_image_data):
        """
        Find marked point correspondences between the test image (gray) and reference images.
        Args:
            test_image_data (ImageFeature): ImageFeature object for the test image.
        Returns:
            estimated_positions (dict): Estimated positions of marked points in the test image.
        """
        estimated_positions = {}

        method = "superpoint"
        if self.model_loaded is False and method == "superpoint":
            method = "cross_check"
        F_dict, Ratio_dict = {}, {}
        try:
            if method == "cross_check":
                F_dict, Ratio_dict = self.__compute_findamental_matrices_via_cross_check(test_image_data)
            else:  # default to superpoint method
                F_dict, Ratio_dict = self.__compute_findamental_matrices_via_superpoint(test_image_data)
        except Exception as e:
            print(f"Error computing fundamental matrices: {e}")
            print("Falling back to direct computation method.")
            F_dict, Ratio_dict = self.__compute_findamental_matrices_directly(test_image_data)

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
            px, py = 0, 0  # initial guess, does not matter
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
