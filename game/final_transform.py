import cv2 as cv
import numpy as np
from main import Game, Board, Player
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import sys
from scipy.optimize import least_squares
import os
import logging
from yolo import yolo_predict
from ultralytics import YOLO
import time
from collections import defaultdict
import csv


# THIS GUY TRANFORMS IMAGE TAKEN FROM AN ANGLE TO THE STRAIGHT VIEW, CROPPED BY OUTER CIRCLE


# This beautiful class is tracking the number of executions and the total time spent of each tracked method in selected class.
# The trick is, that there are a few class instances created during processing script, so the class variables are shared between them.
# Why do i need that? Because i want to understand how i can make script faster, which parts take the significant amount of time.
# And where i need to focus on optimization, and where i can just leave it as it is.

# The time, and number of exectutions of selected methods are appended to the CSV file,
# each time the script is run. This will help me to understand the slowest parts of the script.
# I can accumulate data to check when script fails (if long time) analyzing data from the CSV file, like histograms.
# The CSV file is saved in the same directory as the script.
# These are very universal rows, i can use them in any purpose in the future.

execution_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

def track_class_execution(cls):
    """Class decorator to add execution tracking."""
    class WrappedClass(cls):
        def __getattribute__(self, name):
            attr = super(WrappedClass, self).__getattribute__(name)
            if callable(attr) and name in ["dbscan_clustering", "fit_ellipse", "transform_ellipses_to_circle_1", "find_center", "gather_extreme_points"]:
                def new_attr(*args, **kwargs):
                    start_time = time.time()
                    result = attr(*args, **kwargs)
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    execution_stats[name]["count"] += 1
                    execution_stats[name]["total_time"] += elapsed_time

                    return result
                return new_attr
            return attr

    return WrappedClass

def save_execution_stats(img_path, stats_dict):
    csv_filename = "execution_stats.csv"
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    txt_dir = os.path.join(script_dir, 'txt')
    os.makedirs(txt_dir, exist_ok=True)
    csv_file_path = os.path.join(txt_dir, csv_filename)
    file_exists = os.path.isfile(csv_file_path)
    
    previous_totals = {}
    
    # Read the previous totals if the file already exists
    if file_exists:
        with open(csv_file_path, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                method = row['method']
                previous_totals[method] = {
                    'count': int(row['count']),
                    'total_time': float(row['total_time'])
                }
    
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["method", "count", "total_time", "execution_time", "execution_count"])
        
        for method, data in stats_dict.items():
            prev_count = previous_totals.get(method, {}).get('count', 0)
            prev_time = previous_totals.get(method, {}).get('total_time', 0.0)
            
            execution_time = data['total_time'] - prev_time
            execution_count = data['count'] - prev_count
            
            writer.writerow([method, data['count'], data['total_time'], execution_time, execution_count])

    print(f"Execution stats saved")

# That's how it works, just add @track_class_execution before class definition where method is defined.
# I should mention, that all the core methods are within classes, outside classes functions just combine 
# the classes methods, so i don't need to track them, because they are reworked constantly.
# Basically, outside functuions just define algorithm of usage of classes methods

@track_class_execution
class BoardTransform(Board):
    def __init__(self, board):
        self._img = board._img
        self._center = self._default_center()
        self._outer_ellipse = None
        self._inner_ellipse = None
        self._img_size = board._img_size
        

    def _default_center(self):
        return (self._img.shape[1] // 2, self._img.shape[0] // 2)

    def draw_center(self, center=None, color=(0, 255, 0), s=7):
        if center is None:
            center = self._center
        cv.circle(self._img, center, s, color, -1)
        return self._img
    
    def draw_circles(self, circles):
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            cv.circle(self._img, center, 1, (255, 255, 255), 4)
            radius = circle[2]
            cv.circle(self._img, center, radius, (255, 255, 255), 1)
        return self._img
    
    def draw_points(self, points, img=None, title='', color=(0, 255, 0), radius=3, display=True):
        if img is None:
            img = self._img.copy()
        if isinstance(points[0], (list, tuple, np.ndarray)):
            for point in points:
                cv.circle(img, (int(point[0]), int(point[1])), radius, color, -1)
        else:
            cv.circle(img, (int(points[0]), int(points[1])), radius, color, -1)
        if display:
            self.display_image(cv.cvtColor(img, cv.COLOR_BGR2RGB), title, cmap=None)

        return img
    
    def draw_ellipse(self, ellipse, color=(0, 255, 0)):
        center, axes, angle = ellipse
        ellipse = ((center[0], center[1]), (axes[0], axes[1]), angle)

        output_image = self._img.copy()
        cv.ellipse(output_image, ellipse, color=color, thickness=3) 
        ellipse_img = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
        self._img = output_image
        return ellipse_img

    def calculate_geometric_center(self, points):
        x_coords = points[:, 1]
        y_coords = points[:, 0]
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return int(centroid_x), int(centroid_y)

    def calc_circle_center_least_squares(self, points):
        x = points[:, 1]
        y = points[:, 0]
        
        def calc_R(xc, yc):
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = np.array([x_m, y_m])
        center_optimized = least_squares(f, center_estimate).x
        return int(center_optimized[0]), int(center_optimized[1])

    def calculate_median_center(self, points):
        x_median = np.median(points[:, 1])
        y_median = np.median(points[:, 0])
        return int(x_median), int(y_median)

    # CROPS (TODO: SEPARATE CLASS!!!)
    def crop_radial(self, center=None, crop_factor=0.95):
        if center is None:
            center = self._center       
        min_dimension = min(self._img.shape[0], self._img.shape[1])
        radius = int(min_dimension * crop_factor/2)
        mask = np.zeros_like(self._img, dtype=np.uint8)
        cv.circle(mask, center, radius, (255, 255, 255), thickness=-1)

        cropped_img = cv.bitwise_and(self._img, mask)
        self._img = cropped_img

        return cropped_img

    def crop_square(self, center=None, crop_factor=0.95):   
        if center is None:
            center = self._center  
        center_x, center_y = center
        
        x_start = max(center_x - crop_factor // 2, 0)
        y_start = max(center_y - crop_factor // 2, 0)
        
        x_end = min(x_start + crop_factor, self._img.shape[1])
        y_end = min(y_start + crop_factor, self._img.shape[0])
        
        cropped_image = self._img[y_start:y_end, x_start:x_end]
        self._img = cropped_image

        return self._img, (x_start, y_start)

    def crop_ellipse(self, ellipse, outer_padding_factor=0.03, inner_padding_factor=0.0):
        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
        
        padded_major_axis = int(major_axis_length * (1 + outer_padding_factor))
        padded_minor_axis = int(minor_axis_length * (1 + outer_padding_factor))
        
        outer_padded_ellipse = ((center_x, center_y), (padded_major_axis, padded_minor_axis), angle)
        
        mask = np.zeros_like(self._img, dtype=np.uint8)
        cv.ellipse(mask, outer_padded_ellipse, (255, 255, 255), thickness=-1)
        
        inner_major_axis = int(major_axis_length * inner_padding_factor)
        inner_minor_axis = int(minor_axis_length * inner_padding_factor)
        
        inner_padded_ellipse = ((center_x, center_y), (inner_major_axis, inner_minor_axis), angle)

        if inner_padding_factor > 0:
            cv.ellipse(mask, inner_padded_ellipse, (0, 0, 0), thickness=-1)
        masked_img = cv.bitwise_and(self._img, mask)
        self._img = masked_img
        
        return self._img

    def resize_image(self, target_size=None, half_size=False, predictions=None):
        """Resizes img with defined center to selected target_size.
        Recalculates center and predictions after resizing.
        Redefines ._img and ._center.
        Returns img, center, and transformed predictions if provided."""
        
        if target_size is None:
            target_size = self._img_size
        if half_size:
            target_size = (target_size[0] // 2, target_size[1] // 2)

        resize_factor_x = target_size[0] / self._img.shape[1]
        resize_factor_y = target_size[1] / self._img.shape[0]

        resized_image = cv.resize(self._img, target_size, interpolation=cv.INTER_AREA if resize_factor_x < 1 or resize_factor_y < 1 else cv.INTER_CUBIC)

        if len(resized_image.shape) == 2:
            resized_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)

        new_center_x = int(self._center[0] * resize_factor_x)
        new_center_y = int(self._center[1] * resize_factor_y)
        new_center = (new_center_x, new_center_y)

        transformed_predictions = None
        if predictions is not None:
            transformed_predictions = []
            for x, y in predictions:
                new_x = int(x * resize_factor_x)
                new_y = int(y * resize_factor_y)
                transformed_predictions.append((new_x, new_y))

        self._img = resized_image
        self._center = new_center
        
        return transformed_predictions

    def final_crop(self, ellipse, predictions=None, padding=0.03):

        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
        radius = int(max(major_axis_length, minor_axis_length) / 2 * (1 + padding))
        
        mask = np.zeros((self._img.shape[0], self._img.shape[1]), dtype=np.uint8)
        cv.circle(mask, (int(center_x), int(center_y)), radius, 255, thickness=-1)
        
        if len(self._img.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        masked_img = cv.bitwise_and(self._img, mask)
        
        x, y, w, h = cv.boundingRect(cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
        
        cropped_img = masked_img[y:y+h, x:x+w]
        self._img = cropped_img

        new_center_x = int(center_x - x)
        new_center_y = int(center_y - y)
        self._center = (new_center_x, new_center_y)

        self._outer_ellipse = (
            (new_center_x, new_center_y), 
            self._outer_ellipse[1], 
            self._outer_ellipse[2]
        )
        
        self._inner_ellipse = (
            (new_center_x, new_center_y), 
            self._inner_ellipse[1], 
            self._inner_ellipse[2]
        )

        transformed_predictions = None
        if predictions is not None:
            transformed_predictions = []
            for pred_x, pred_y in predictions:
                new_x = pred_x - x
                new_y = pred_y - y
                if 0 <= new_x < w and 0 <= new_y < h:
                    transformed_predictions.append((new_x, new_y))

        return transformed_predictions
    
    def mirror_image(self, axis):
        if axis == 'x':
            mirrored_img = cv.flip(self._img, 0)
        elif axis == 'y':
            mirrored_img = cv.flip(self._img, 1)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        self._img = mirrored_img
        return self._img

    def expand_canvas(self, predictions, target_center, scale_factor=1.5):
        original_height, original_width = self._img.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        new_canvas = np.zeros((new_height, new_width, self._img.shape[2]), dtype=self._img.dtype)

        top_left_x = (new_width // 2) - target_center[0]
        top_left_y = (new_height // 2) - target_center[1]

        if top_left_x < 0:
            top_left_x = 0
        if top_left_y < 0:
            top_left_y = 0

        bottom_right_x = min(top_left_x + original_width, new_width)
        bottom_right_y = min(top_left_y + original_height, new_height)
        new_canvas[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = self._img[:bottom_right_y-top_left_y, :bottom_right_x-top_left_x]

        self._img = new_canvas
        self._center = (target_center[0] + top_left_x, target_center[1] + top_left_y)

        adjusted_predictions = []
        for pred in predictions:
            adjusted_pred_x = pred[0] + top_left_x
            adjusted_pred_y = pred[1] + top_left_y
            adjusted_predictions.append((adjusted_pred_x, adjusted_pred_y))

        return adjusted_predictions


    # SOME COLOR MASK STUFF (TODO: SEPARATE CLASS!!!)
    def _generate_gray_ranges(self, lower_scale=1.0, upper_scale=1.0):
        # base_ranges = [
        #     (np.array([0, 0, 140]), np.array([180, 50, 255])),
        #     (np.array([0, 0, 128]), np.array([180, 50, 138])),
        #     (np.array([0, 0, 160]), np.array([180, 50, 170])),
        #     (np.array([0, 0, 180]), np.array([180, 50, 190]))
        # ]
        base_ranges = [
        (np.array([0, 0, 60]), np.array([180, 50, 200])),  # Dark gray to light gray
        (np.array([0, 0, 200]), np.array([180, 50, 240]))  # Light gray, stopping before white
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 254).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges

    def _generate_green_ranges(self, lower_scale=1.0, upper_scale=1.0):
        base_ranges = [
            (np.array([35, 50, 50]), np.array([85, 255, 255])),   # Main green range
            (np.array([30, 40, 40]), np.array([90, 255, 255])),   # Slightly wider range
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges


    def _generate_red_ranges(self, lower_scale=1.0, upper_scale=1.0):
        lower_red1 = np.clip(np.array([0, 70, 50]) * lower_scale, 0, 255).astype(int)
        upper_red1 = np.clip(np.array([10, 255, 255]) * upper_scale, 0, 255).astype(int)
        lower_red2 = np.clip(np.array([170, 70, 50]) * lower_scale, 0, 255).astype(int)
        upper_red2 = np.clip(np.array([180, 255, 255]) * upper_scale, 0, 255).astype(int)
        adjusted_ranges = [(lower_red1, upper_red1), (lower_red2, upper_red2)]
        return adjusted_ranges


# This paper was very useful in understaning better how to choose the color with HSV over RGB,
# https://handmap.github.io/hsv-vs-rgb/
# despite this, other ranges remained in RGB (for grays, greens, and blues). 
# Actually, It's needed to do with green colors (I mean, more precise choice) in HSV, 
# because dart board has greens and reds, which are my goal.
# Just because I'm lazy, and, which is also important, 
# my board has no green colors on background, where it's placed.
# But i should say that for usability in different conditions, it's better to use HSV for greens.
# For color searching on photos i used Photoshop.

# TODO: WORK HERE WITH HSV BOUNDARIES!!!! Now doesnt work okay

    def _generate_red_ranges_new(self, lower_scale=0.5, upper_scale=3.0, avoid_lower_scale=1.0, avoid_upper_scale=1.0):
        target_colors = [
            [216, 72, 72],  # d84848
            [216, 62, 62],  # d83e3e
            [214, 68, 69],  # d64445
            [151, 40, 47],  # 97282f
            [197, 81, 81],  # c55151
            [212, 74, 72],  # d44a48
            [239, 71, 70],  # ef4746
            [203, 64, 67],  # cb4043
            [221, 72, 76],  # dd484c
            [212, 63, 65],  # d43f41
        ]
        avoid_colors = [
            [254, 115, 112],  # fe7370
            [130, 87, 78],    # 82574e
            [99, 59, 51],     # 633b33
            [231, 70, 85],    # e74655
            [217, 72, 75],    # d9484b
        ]

        adjusted_ranges = []
        for color in target_colors:
            hsv_color = cv.cvtColor(np.uint8([[color]]), cv.COLOR_RGB2HSV)[0][0]
            lower_bound = np.clip(hsv_color * lower_scale, 0, 255).astype(int)
            upper_bound = np.clip(hsv_color * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((lower_bound, upper_bound))

        filtered_ranges = []
        for lower, upper in adjusted_ranges:
            intersects = False
            for avoid_color in avoid_colors:
                avoid_hsv = cv.cvtColor(np.uint8([[avoid_color]]), cv.COLOR_RGB2HSV)[0][0]
                avoid_lower_bound = np.clip(avoid_hsv * avoid_lower_scale, 0, 255).astype(int)
                avoid_upper_bound = np.clip(avoid_hsv * avoid_upper_scale, 0, 255).astype(int)
                if (np.all(lower <= avoid_upper_bound) and np.all(upper >= avoid_lower_bound)):
                    intersects = True
                    break
            if not intersects:
                filtered_ranges.append((lower, upper))

        return filtered_ranges


    def _generate_blue_ranges(self, lower_scale=1.0, upper_scale=1.0):
        base_ranges = [
            (np.array([90, 50, 70]), np.array([130, 255, 255])),  # Main blue range
            (np.array([85, 50, 50]), np.array([135, 255, 255])),  # Slightly wider range
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges
    
    def overlay_mask_on_image(self, image, mask, alpha=0.5, color=(0, 255, 0)):
        mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        mask_colored[:, :, 0] = np.where(mask > 0, color[0], 0)
        mask_colored[:, :, 1] = np.where(mask > 0, color[1], 0)
        mask_colored[:, :, 2] = np.where(mask > 0, color[2], 0)

        overlay = cv.addWeighted(image, 1, mask_colored, alpha, 0)
        overlay_img = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)

        return overlay_img

    def apply_masks(self, colors, lower_scale=1.0, upper_scale=1.0):
        
        if len(self._img.shape) == 2:
            self._img = cv.cvtColor(self._img, cv.COLOR_GRAY2BGR)

        hsv = cv.cvtColor(self._img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color in colors:
            if color == 'gray':
                ranges = self._generate_gray_ranges(lower_scale, upper_scale)
            elif color == 'green':
                ranges = self._generate_green_ranges(lower_scale, upper_scale)

            elif color == 'red':

                # ranges = self._generate_red_ranges(lower_scale, upper_scale)
                # NO RANGES, DONT TOUCH !!!
                ranges = self._generate_red_ranges()

            elif color == 'blue':
                ranges = self._generate_blue_ranges(lower_scale, upper_scale)
            else:
                raise ValueError("Unsupported color. Use 'gray', 'green', 'red', or 'blue'.")

            for lower, upper in ranges:
                mask = cv.inRange(hsv, lower, upper)
                combined_mask = cv.bitwise_or(combined_mask, mask)

        self._img = cv.bitwise_and(self._img, self._img, mask=combined_mask)
        
        return self._img, combined_mask
    
    def thicken_mask(self, mask, kernel_size=(5, 5), iterations=1):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
        thickened_mask = cv.dilate(mask, kernel, iterations=iterations)

        return thickened_mask
    # END OF COLOR MASK STUFF

    # START OF DBSCAN  (IT FITS AN ELLIPSE AROUND THE DARTBOARD) (TODO: SEPARATE CLASS!!!)
    def gray_thresh(self, threshold=30):
        gray_image = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)

        return thresh

    def find_center(self, min_radius=13, max_radius=30, param1=70, param2=20):
        self._img = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        circles = cv.HoughCircles(
            self._img,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius)
        if circles is None:
            return None, None

        circles = np.uint16(np.around(circles))
        centers = [(circle[0], circle[1]) for circle in circles[0, :]]
        sum_x = sum([c[0] for c in centers])
        sum_y = sum([c[1] for c in centers])
        num_circles = len(centers)
        center_mean = (int(sum_x / num_circles), int(sum_y / num_circles))
        radii = [circle[2] for circle in circles[0, :]]
        radius_mean = int(np.mean(radii))
        self._center = center_mean   

        return center_mean, radius_mean, circles
    
    # Just yielding doesnt work across the class instances. Unused method
    def yield_number_of_uses(self):
        for i in range(0, 100000):
            yield i
    
    def dbscan_clustering(self, eps=7, min_samples=10, plot=False, threshold=30):
        # next(n)
        thresh = self.gray_thresh(threshold=threshold)
        coords = cv.findNonZero(thresh)

        # #PROBLEM HERE: COLUMN STACK I GOT YOU!
        # thresh = self.preprocess_image()
        # coords = np.column_stack(np.where(thresh > 0))

        if coords is not None:
            coords = coords.reshape(-1, 2)
        else:
            coords = np.empty((0, 2))
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        if len(unique_labels) == 0:
            raise ValueError("No clusters found by DBSCAN.")
        
        largest_cluster_label = unique_labels[sorted_indices[0]]
        largest_cluster_coords = coords[labels == largest_cluster_label]

        if len(unique_labels) > 1:
            smaller_cluster_label = unique_labels[sorted_indices[1]]
            smaller_cluster_coords = coords[labels == smaller_cluster_label]
        else:
            smaller_cluster_coords = np.empty((0, 2))

        if plot:
            self.display_dbscan_results(coords, labels, unique_labels[sorted_indices[:2]])

        return largest_cluster_coords, smaller_cluster_coords

    def display_dbscan_results(self, coords, labels, largest_cluster_labels):
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20', len(largest_cluster_labels))
        fig = plt.figure(figsize=(8, 8))
        for k in unique_labels:
            if k == -1:
                continue  # Skip noise

            class_member_mask = (labels == k)
            xy = coords[class_member_mask]

            if k == largest_cluster_labels[0]:
                plt.plot(xy[:, 1], xy[:, 0], 'r.', markersize=2)  # Largest cluster in red
            elif k in largest_cluster_labels:
                plt.plot(xy[:, 1], xy[:, 0], '.', markersize=2, color=colors(np.where(largest_cluster_labels == k)[0][0]))

        plt.title('DBSCAN')
        plt.gca().invert_yaxis()
        plt.xlim(0, self._img.shape[1])
        plt.ylim(self._img.shape[0], 0)
        plt.show()
        # fig.savefig('../upload/dbscan.png')

    def gather_extreme_points(self, coords=None, rotations=14):
        if coords is None:
            thresh = self.gray_thresh(threshold=30)
            coords = cv.findNonZero(thresh)
            if coords is not None:
                coords = coords.reshape(-1, 2)
            else:
                coords = np.empty((0, 2))
        all_extreme_points = []
        original_coords = coords.copy()
        for i in range(0,rotations):
            angle = i * 120 / rotations
            rotated_coords = self.rotate_coordinates_around_center(original_coords, angle, self._center)
            extreme_points = self.extract_extreme_points(rotated_coords)
            extreme_points_rotated_back = self.rotate_coordinates_around_center(extreme_points, -angle, self._center)
            all_extreme_points.append(extreme_points_rotated_back)
        all_extreme_points = np.vstack(all_extreme_points)

        return all_extreme_points

    def extract_extreme_points(self, coords):
        if len(coords) < 5:
            raise ValueError("Not enough points to extract extremities.")

        max_x_idx = np.argmax(coords[:, 1])
        min_x_idx = np.argmin(coords[:, 1])
        max_y_idx = np.argmax(coords[:, 0])
        min_y_idx = np.argmin(coords[:, 0])

        extreme_points = np.array([
            coords[max_x_idx],
            coords[min_x_idx],
            coords[max_y_idx],
            coords[min_y_idx]
        ])

        return extreme_points

    def rotate_coordinates_around_center(self, coords, angle, center):
        translated_coords = coords - center
        theta = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_coords = translated_coords @ rotation_matrix.T
        rotated_coords += center

        return rotated_coords
    
    def fit_ellipse(self, coords, outer=True):
        if len(coords) < 5:
            raise ValueError("Not enough points to fit an ellipse.")

        coords = np.array(coords, dtype=np.float32)
        ellipse = cv.fitEllipse(coords)
        if outer:
            self._outer_ellipse = ellipse
        else:
            self._inner_ellipse = ellipse

        return ellipse

@track_class_execution
class PerspectiveTransform(BoardTransform):
    def __init__(self, board):
        self._center = board._center
        self._img = board._img
        self._outer_ellipse = board._outer_ellipse
        self._inner_ellipse = board._inner_ellipse
        self._img_size = board._img_size

    def generate_ellipse_points(self, ellipse, num_points=120):
        center = np.array(ellipse[0], dtype=np.float32)
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        angle = np.radians(ellipse[2])

        points = []
        for t in np.linspace(0, 2*np.pi, num_points, endpoint=False):
            x = a * np.cos(t)
            y = b * np.sin(t)
            
            x_rot = center[0] + x * np.cos(angle) - y * np.sin(angle)
            y_rot = center[1] + x * np.sin(angle) + y * np.cos(angle)
            
            points.append([x_rot, y_rot])

        return np.array(points, dtype=np.float32)

    def transform_ellipses_to_circle_1(self, outer_ellipse, inner_ellipse, center, predictions):

        destination_center = center
        points_src_outer = self.generate_ellipse_points(outer_ellipse)
        points_src_outer = np.vstack([points_src_outer, outer_ellipse[0]])
        points_dst_outer = []
        for point in points_src_outer[:-1]:
            angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
            radius = ((outer_ellipse[1][0] + outer_ellipse[1][1]) / 4) * 1.0
            dst_x = destination_center[0] + radius * np.cos(angle)
            dst_y = destination_center[1] + radius * np.sin(angle)
            points_dst_outer.append([dst_x, dst_y])
        points_dst_outer.append(destination_center)
        points_dst_outer = np.array(points_dst_outer, dtype=np.float32)


        points_src_inner = self.generate_ellipse_points(inner_ellipse)
        points_src_inner = np.vstack([points_src_inner, inner_ellipse[0]])
        points_dst_inner = []
        for point in points_src_inner[:-1]:
            angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
            radius = ((inner_ellipse[1][0] + inner_ellipse[1][1]) / 4) * 1.0
            dst_x = destination_center[0] + radius * np.cos(angle)
            dst_y = destination_center[1] + radius * np.sin(angle)
            points_dst_inner.append([dst_x, dst_y])
        points_dst_inner.append(destination_center)
        points_dst_inner = np.array(points_dst_inner, dtype=np.float32)

        points_src = np.vstack([points_src_outer, points_src_inner])
        points_dst = np.vstack([points_dst_outer, points_dst_inner])

        homography_matrix, _ = cv.findHomography(points_src, points_dst)
        transformed_image = cv.warpPerspective(self._img, homography_matrix, (self._img.shape[1], self._img.shape[0]))

        center_homogeneous = np.array([center[0], center[1], 1.0], dtype=np.float32).reshape(-1, 1)
        transformed_center_homogeneous = homography_matrix @ center_homogeneous
        transformed_center = transformed_center_homogeneous[:2] / transformed_center_homogeneous[2]

        transformed_predictions = []
        for pred in predictions:
            pred_homogeneous = np.array([pred[0], pred[1], 1.0], dtype=np.float32).reshape(-1, 1)
            transformed_pred_homogeneous = homography_matrix @ pred_homogeneous
            transformed_pred = transformed_pred_homogeneous[:2] / transformed_pred_homogeneous[2]
            transformed_predictions.append((int(transformed_pred[0][0]), int(transformed_pred[1][0])))

        self._img = transformed_image
        self._center = (int(transformed_center[0][0]), int(transformed_center[1][0]))
        # print(self._img.shape)
        return transformed_image, transformed_predictions


# TODO: ADD THIS TO CLASS PerspectiveTransform, URGENTLY

def find_bulls_eye(board, crop_eye=0.25, min_radius=8, max_radius=30, param1=70, param2=20, plots=False, centered=False):
    """Should receive class with ._img defined
    Returns center, if found, otherwise None"""
    if centered:
        try:
            initial_center = (board._img.shape[1] // 2, board._img.shape[0] // 2)
            crop_mask = BoardTransform(board)
            crop_mask.crop_radial(center=initial_center, crop_factor=crop_eye)
            crop_mask.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
            precise_center, radius, circles = crop_mask.find_center(min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)
            # print(f"precise_center: {precise_center}")
            if plots:
                crop_mask.draw_circles(circles)
                crop_mask.draw_points(precise_center)

            return precise_center
        except ValueError:
            initial_center = board._center
    else:
        initial_center = board._center
    # This function looks for the bulls eye around 25% of the image center
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for step in np.linspace(0.0, 0.15, 6):
        iter = 0
        for angle in angles:
            radius_step = min(board._img.shape[:2]) * step
            iter += 1
            # print(iter)
            if iter > 8:
                print("I can't find the center. Please take another image.")
                sys.exit()
            dx = radius_step * np.cos(angle)
            dy = radius_step * np.sin(angle)
            new_center = (int(initial_center[0] + dx), int(initial_center[1] + dy))

            crop_mask = BoardTransform(board)
            try:
                crop_mask.crop_radial(center=new_center, crop_factor=crop_eye)
                crop_mask.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
                bulls_eye, _ = crop_mask.dbscan_clustering(plot=False, eps=5, min_samples=10, threshold=10)
                center = crop_mask.calc_circle_center_least_squares(bulls_eye)

                try:
                    # print("HERE")
                    crop_mask_res = BoardTransform(board)
                    crop_mask_res.crop_radial(center=center, crop_factor=crop_eye)
                    crop_mask_res.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
                    precise_center, radius, circles = crop_mask_res.find_center(min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)
 
                    # print(f"Center found...")
                    return precise_center
                except ValueError:
                    # print("HERE HERE")
                    return center
            except ValueError:
                raise ValueError("Center not found.")
    print("Tough to find a center... Let's give another try...")
    return None

def ellipses(board, eps=10, min_samples=7, threshold=10):
    dbscan = BoardTransform(board)
    dbscan.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
    outer_ring, inner_ring = dbscan.dbscan_clustering(eps=eps, min_samples=min_samples, threshold=threshold)
    
    extreme_points = board.gather_extreme_points(coords=outer_ring)
    outer_ellipse = board.fit_ellipse(extreme_points, outer=True)
    extreme_points = board.gather_extreme_points(coords=inner_ring)
    inner_ellipse = board.fit_ellipse(extreme_points, outer=False) 
    return board, outer_ellipse, inner_ellipse

# TODO: Maybe a little bit more work here?
def find_ellipse(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False, padding=0.03):
    """Finds outer and inner ellipse.
    Should receive pre-cropped board class BoardTransform.
    Crops around outer ellipse found.
    Returns board back with ellipses"""
    first_try = None
    second_try = None
    try:
        eps_ratio_semis = []
        ratio_outer_inner = None
        while ratio_outer_inner is None or ratio_outer_inner > 1.015 and eps > 4:
            board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
            ratio_outer_inner = max((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))\
                /min((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))
            ratio_outer = (max(outer_ellipse[1][0],outer_ellipse[1][1]))/(min(outer_ellipse[1][0],outer_ellipse[1][1]))
            eps_ratio_semis.append((eps, ratio_outer_inner, ratio_outer))
            eps -= 0.2                
            # print(f'Semis ratio: {ratio_outer_inner}, eps: {eps}, ratio_outer: {ratio_outer}')
        # print(f"Got the board...")
        # print(f"Ratio: {ratio_outer_inner}")
    except (IndexError, ValueError):
        print(f"Unsure with results...Let's try to adjust parameters little bit...")
        filtered = [i for i in eps_ratio_semis if i[2]<1.5] # Angles limited to 1.5 semis ratio
        if filtered:
            eps = min(filtered, key=lambda x: x[1])[0] # Choosing best eps based on similarity of ellipses
            # print(f"Chosen eps: {eps}, semis ratio: {min(filtered, key=lambda x: x[1])[1]}, ratio_outer: {min(filtered, key=lambda x: x[1])[2]}")
            first_try = [eps, min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
        else:
            filtered = [i for i in eps_ratio_semis if i[2]<2] # Angles limited to 2 semis ratio
            if filtered:
                eps = min(filtered, key=lambda x: x[1])[0] # Choosing best eps based on similarity of ellipses
                # print(f"Chosen eps: {eps}, semis ratio: {min(filtered, key=lambda x: x[1])[1]}, ratio_outer: {min(filtered, key=lambda x: x[1])[2]}")
                first_try = [eps, min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
            else:
                print("One more try...")
       
        # Now iterate over min_samples
        try: 
            min_samples_ratio_semis = []
            eps = first_try[0]
            while ratio_outer_inner > 1.03 and min_samples < 100:
                board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
                ratio_outer_inner = max((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))\
                    /min((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))
                ratio_outer = (max(outer_ellipse[1][0],outer_ellipse[1][1]))/(min(outer_ellipse[1][0],outer_ellipse[1][1]))
                min_samples_ratio_semis.append((min_samples, ratio_outer_inner, ratio_outer, threshold))
                if threshold > 0:
                    threshold -= 1
                min_samples += 0.5
                # print(f'Semis ratio: {ratio_outer_inner}, min_samples: {min_samples}, ratio_outer: {ratio_outer}')
                # print(f"Ratio: {ratio_outer_inner}")

            filtered = [i for i in min_samples_ratio_semis if i[2]<2]
            min_samples = min(filtered, key=lambda x: x[1])[0]
            second_try = [eps, min_samples, min(min_samples_ratio_semis, key=lambda x: x[1])[1], min(min_samples_ratio_semis, key=lambda x: x[1])[2]]
        
        except (IndexError, ValueError):
            filtered = [i for i in min_samples_ratio_semis if i[2]<2]
            print("Hmm...")
            if filtered:
                min_samples = min(filtered, key=lambda x: x[1])[0]
                second_try = [first_try[0], min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
            else:
                print("Cant detect board, please take image from another angle")
                sys.exit()

    if (first_try is not None) and (second_try is not None):
        # print(f"First try: {first_try}, Second try: {second_try}")
        eps = second_try[0]
        min_samples = second_try[1]
        threshold = second_try[3]
    # print(f"Final parameters: eps: {eps}, min_samples: {min_samples}, threshold: {threshold}")
    board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
    board._outer_ellipse = outer_ellipse
    board._inner_ellipse = inner_ellipse
    board.crop_ellipse(outer_ellipse, outer_padding_factor=padding)
    # predictions = board.final_crop(outer_ellipse, predictions=predictions, padding=padding)

    if plot_ellipse:
        board_copy = BoardTransform(board)
        board_copy.draw_ellipse(outer_ellipse, color=(0, 255, 0))
        board_copy.draw_ellipse(inner_ellipse, color=(255, 0, 0))
        board_copy.display_image_self(bgr=False)
    return board

def initial_prepare(board, crop_eye=0.25, crop_scale=1.0, size=None):
    """Receives basic BoardTransform with ._img
    Resizes image if needed
    Finds center, crops around it with circle
    Should be passed to find_ellipse further
    Return cropped board back with defined center, if center found"""

    if size:
        board.resize_image(target_size=size)
    try:
        bulls_eye = find_bulls_eye(board, crop_eye=crop_eye)
    except Exception as e:
        print(f"Center was not found: {e}")
        raise ValueError("Center was not found")
    if bulls_eye:
        board._center = bulls_eye
        board.crop_radial(bulls_eye, crop_factor=0.8*crop_scale)
    else:
        board.crop_radial(crop_factor=0.9*crop_scale)
    
    return board

def transform_perspective(board, predictions, plots=False, crop_eye=0.25, shifts=None):
    """Receives prepared board with defined center, ellipses
    Returns transformed image, with refined center, if found"""
    transform = PerspectiveTransform(board)
    # transform.draw_points(predictions)
    _,predictions = transform.transform_ellipses_to_circle_1(transform._outer_ellipse, transform._inner_ellipse, transform._center, predictions)
    try:
        new_center = find_bulls_eye(transform, crop_eye=crop_eye)
        old_center = transform._center
        transform._center = new_center

        shift_x = new_center[0] - old_center[0]
        shift_y = new_center[1] - old_center[1]
        if shifts is not None:
            shifts.append((shift_x, shift_y))
        compensated_predictions = []
        for pred in predictions:
            compensated_x = pred[0] + shift_x/2
            compensated_y = pred[1] + shift_y/2
            compensated_predictions.append((compensated_x, compensated_y))

        predictions = compensated_predictions
        # transform.draw_points(predictions)

    except (ValueError, TypeError):
        print("Center was not refined")
    if plots:
        transform_copy = BoardTransform(transform)
        transform_copy.draw_center(center=new_center, color=(0, 0, 255), s=5)
        transform_copy.display_image_self(f'Transform', bgr=False)
    return transform, predictions, shifts

def crop_around(board, predictions, square_size=50):
    img = board._img
    mask = np.zeros_like(img)

    padding = square_size // 2
    
    for prediction in predictions:
        x, y = prediction
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(img.shape[1], int(x + padding))
        y2 = min(img.shape[0], int(y + padding))
        
        mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        
    return mask

# THIS METHOD IS NOT USED, SAVED FOR FUTURE WORK
def iterative_transform_todo(board,
                    predictions,
                    init_transform=True,
                    eps=10, min_samples=7, 
                    threshold=10, 
                    iterations=3,
                    crop_eye=0.25,
                    accuracy=0.01,
                    save_logs=False,
                    plot_final=False,
                    plot_steps=False):
    
    original = BoardTransform(board)

    # TODO: rework in while loop. Define conditions for that
    # logs = []
    # outer_ellipse_semis = []
    # inner_ellipse_semis = []
    # outer_ellipse_center = []
    # inner_ellipse_center = []
    # center = []
    
    shifts = []
    for i in range(1,iterations+1):
        # print(f'Iteration {i}')
        # board.draw_points(predictions)
        board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
        # board.draw_points(predictions)

        # outer_ellipse_semis.append(board._outer_ellipse[1])
        # inner_ellipse_semis.append(board._inner_ellipse[1])
        # outer_ellipse_center.append(board._outer_ellipse[0])
        # inner_ellipse_center.append(board._inner_ellipse[0])
        # center.append(board._center)
        # board.draw_points(predictions)
        board, predictions, shifts = transform_perspective(board, predictions, plots=plot_steps, crop_eye=crop_eye, shifts=shifts)
        if i == 3 or i==4 or i == 5:
            # print(f'Iteration {i}')
            # print(f"len predictions: {len(predictions)}")
            # board.draw_points(predictions)
            part2 = YOLO('./yolos/part2.pt')
            cropped = crop_around(board, predictions, square_size=40)
            predictions_new, n_classes_predictions = yolo_predict(cropped, part2)
            # print(f"len predictions: {len(predictions_new)}")
            if len(predictions_new) == len(predictions):
                predictions = predictions_new
                if i==3:
                    print('Hold on, we are getting closer...')
                if i==4:
                    print('Almost there...')
            # predictions = board.resize_image(target_size=(800, 800), predictions=predictions)
            # board.draw_points(predictions_new)
        # board.draw_points(predictions)
    # print('uncompensated')
    # board.draw_points(predictions)

    # first_shift = shifts[0]
    # last_shift = shifts[-1]

    # # Calculate the total shift by comparing the last shift with the first shift
    # total_shift_x = last_shift[0] - first_shift[0]
    # total_shift_y = last_shift[1] - first_shift[1]

    # # Apply this total shift to each prediction
    # compensated_predictions = []
    # for pred in predictions:
    #     compensated_x = pred[0] - total_shift_x/iterations
    #     compensated_y = pred[1] - total_shift_y/iterations
    #     compensated_predictions.append((compensated_x, compensated_y))

    # print('compensated1')
    # board.draw_points(compensated_predictions)

    total_shift_x = 0
    total_shift_y = 0

    # Calculate the total shifts
    for shift in shifts:
        total_shift_x += shift[0]
        total_shift_y += shift[1]

    # Calculate the average shift
    avg_shift_x = total_shift_x / len(shifts)
    avg_shift_y = total_shift_y / len(shifts)

    # Apply this average shift to each prediction
    compensated_predictions = []
    for pred in predictions:
        compensated_x = pred[0] + avg_shift_x
        compensated_y = pred[1] + avg_shift_y
        compensated_predictions.append((compensated_x, compensated_y))
    # print(f'Total compensation: {avg_shift_x}, {avg_shift_y}')
    # print('compensated2')
    # board.draw_points(compensated_predictions)
    predictions = compensated_predictions

    board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
    # outer_ellipse_semis.append(board._outer_ellipse[1])
    # inner_ellipse_semis.append(board._inner_ellipse[1])
    # outer_ellipse_center.append(board._outer_ellipse[0])
    # inner_ellipse_center.append(board._inner_ellipse[0])
    # center.append(board._center)

    # def normalize_list(numbers):
    #     min_val = min(numbers)
    #     max_val = max(numbers)
    #     normalized = [(x - min_val) / (max_val - min_val) for x in numbers]
    #     return normalized
    
    # outer_differences = [abs(outer[0] - outer[1]) for outer in outer_ellipse_semis]
    # inner_differences = [abs(inner[0] - inner[1]) for inner in inner_ellipse_semis]

    # outer_differences_normalized = normalize_list(outer_differences)
    # inner_differences_normalized = normalize_list(inner_differences)

    # delta_outer = []
    # i = 0
    # while (i+1) < len(outer_differences_normalized):
    #     (outer_differences_normalized[i+1] - outer_differences_normalized[i])
    #     delta_outer.append((outer_differences_normalized[i+1] - outer_differences_normalized[i]))
    #     i += 1

    # delta_inner = []
    # i = 0
    # while (i+1) < len(inner_differences_normalized):
    #     (inner_differences_normalized[i+1] - inner_differences_normalized[i])
    #     delta_inner.append((inner_differences_normalized[i+1] - inner_differences_normalized[i]))
    #     i += 1

    # deltas = [x for x in  zip(delta_outer, delta_inner)]

    if save_logs:
        with open('./txt/outer_ellipse_semis.txt', 'w') as f:
            for item in outer_ellipse_semis:
                f.write(f"{item[0]}, {item[1]}\n")
        with open('./txt/inner_ellipse_semis.txt', 'w') as f:
            for item in inner_ellipse_semis:
                f.write(f"{item[0]}, {item[1]}\n")
        with open('./txt/center.txt', 'w') as f:
            for item in center:
                f.write(f"{item[0]}, {item[1]}\n")
        with open('./txt/outer_ellipse_center.txt', 'w') as f:
            for item in outer_ellipse_center:
                f.write(f"{item[0]}, {item[1]}\n")
        with open('./txt/inner_ellipse_center.txt', 'w') as f:
            for item in inner_ellipse_center:
                f.write(f"{item[0]}, {item[1]}\n")
        print('TRANSFORM LOGS SAVED TO TXT FILES')

    if plot_final:
        plot_transformed = BoardTransform(board)
        plot_transformed.crop_ellipse(board._outer_ellipse, outer_padding_factor=0.02)
        plot_transformed.draw_ellipse(board._outer_ellipse, color=(0, 255, 0))
        plot_transformed.draw_ellipse(board._inner_ellipse)
        plot_transformed.draw_center(center=board._center, color=(0, 255, 0), s=7)
        plot_transformed.draw_center(tuple(int(x) for x in board._inner_ellipse[0]), color=(255, 0, 0), s=6)
        plot_transformed.draw_center(tuple(int(x) for x in board._outer_ellipse[0]), color=(0, 0, 255), s=6)
        plot_transformed.display_image_self('After initial iterations', bgr=False)
    
    if init_transform:
        n_iterations = 5
        return board, predictions, n_iterations
    
    return board, predictions

# USED THIS INSTEAD
def iterative_transform(board,
                    predictions,
                    init_transform=True,
                    eps=10, min_samples=7, 
                    threshold=10, 
                    iterations=3,
                    crop_eye=0.25,
                    accuracy=0.01,
                    save_logs=False,
                    plot_final=False,
                    plot_steps=False):
    
    original = BoardTransform(board)

    shifts = []
    for i in range(1,iterations+1):
        board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
        board, predictions, shifts = transform_perspective(board, predictions, plots=plot_steps, crop_eye=crop_eye, shifts=shifts)
        if i == 3 or i==4 or i == 5:
            part2 = YOLO('./yolos/best.pt')
            cropped = crop_around(board, predictions, square_size=40)
            predictions_new, n_classes_predictions = yolo_predict(cropped, part2)
            if len(predictions_new) == len(predictions):
                predictions = predictions_new
                if i==3:
                    print('Hold on, we are getting closer...')
                if i==4:
                    print('Almost there...')
        if i == iterations-1:
            outer_ellipse = board._outer_ellipse
            inner_ellipse = board._inner_ellipse
            outer_ellipse = (
                outer_ellipse[0],
                (min(outer_ellipse[1][0], outer_ellipse[1][1]), min(outer_ellipse[1][0], outer_ellipse[1][1])),
                outer_ellipse[2]
            )
            inner_ellipse = (
                inner_ellipse[0],
                (min(inner_ellipse[1][0], inner_ellipse[1][1]), min(inner_ellipse[1][0], inner_ellipse[1][1])),
                inner_ellipse[2]
            )
            board._outer_ellipse = outer_ellipse
            board._inner_ellipse = inner_ellipse
            predictions = board.final_crop(board._outer_ellipse, predictions=predictions, padding=0.01)
            new_center = find_bulls_eye(board, crop_eye=0.25, plots=False, centered=True, max_radius=20, min_radius=13, param1=50, param2=15)
            old_center = board._center
            board._center = new_center
            
    total_shift_x = 0
    total_shift_y = 0
    for shift in shifts:
        total_shift_x += shift[0]
        total_shift_y += shift[1]
    avg_shift_x = total_shift_x / len(shifts)
    avg_shift_y = total_shift_y / len(shifts)

    compensated_predictions = []
    for pred in predictions:
        compensated_x = pred[0] + avg_shift_x
        compensated_y = pred[1] + avg_shift_y
        compensated_predictions.append((compensated_x, compensated_y))
    predictions = compensated_predictions

    board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
    
    if init_transform:
        n_iterations = 5
        return board, predictions, n_iterations

    return board, predictions


def process_image(img_path, output_subdir, eps=10, min_samples=7, threshold=10, crop_scale=1.0, crop_eye=0.25, size_transform=(1100,1100), accuracy=0.05, iterations=8):

    print(f"Processing {img_path}")
    
    try:
        original = Board(img_path)
        board = BoardTransform(original)

        board = initial_prepare(board, crop_eye=crop_eye, crop_scale=crop_scale, size=size_transform)
        board.expand_canvas(target_center=board._center, scale_factor=1.1)

        transformed, n_iterations = iterative_transform(
            board,
            init_transform=True,
            iterations=iterations,
            eps=eps, min_samples=min_samples, threshold=threshold,
            accuracy=accuracy,
            crop_eye=crop_eye,
            save_logs=False,
            plot_final=False, plot_steps=False
        )

        transformed = iterative_transform(
            board,
            init_transform=False,
            iterations=n_iterations,
            eps=eps, min_samples=min_samples, threshold=threshold,
            crop_eye=crop_eye,
            save_logs=False,
            plot_final=False, plot_steps=False
        )
        transformed
        transformed.final_crop(transformed._outer_ellipse)
        transformed.resize_image(target_size=(800, 800))

        processed_img_path = os.path.join(output_subdir, os.path.basename(img_path))
        cv.imwrite(processed_img_path, transformed._img)
        print(f"Processed image saved to {processed_img_path}")

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

def process_directory(input_dir, output_dir, eps, min_samples, threshold, crop_scale, crop_eye, size_transform):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # TODO:
                # Setup logging for each subdirectory
                # setup_logging(output_subdir)

                process_image(img_path, output_subdir, eps, min_samples, threshold, crop_scale, crop_eye, size_transform)

def process_directory_resize(input_dir, output_dir, eps, min_samples, threshold, crop_scale, crop_eye, size_transform):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                img = cv.imread(img_path)
                resized_img = cv.resize(img, (640, 640), interpolation=cv.INTER_AREA)
                output_path = os.path.join(output_subdir, file)
                cv.imwrite(output_path, resized_img)

def find20_center(board):
    dbscan = BoardTransform(board)
    angles_to_crop = [261-8, 279+8]
    radius_outer = (board._outer_ellipse[1][0] + board._outer_ellipse[1][1])/4
    radius_inner = (board._inner_ellipse[1][0] + board._inner_ellipse[1][1])/4
    radiuses_to_crop = [(radius_outer + radius_inner)/2, radius_outer]
    center = board._center
    
    angles_to_crop_rad = [np.deg2rad(angle) for angle in angles_to_crop]
    
    mask = np.zeros(board._img.shape[:2], dtype=np.uint8)

    cv.ellipse(mask, center, 
               (int(radiuses_to_crop[1]), int(radiuses_to_crop[1])), 
               0, 
               np.rad2deg(angles_to_crop_rad[0]), 
               np.rad2deg(angles_to_crop_rad[1]), 
               255, 
               thickness=-1)
    cv.ellipse(mask, center, 
               (int(radiuses_to_crop[0]), int(radiuses_to_crop[0])), 
               0, 
               np.rad2deg(angles_to_crop_rad[0]), 
               np.rad2deg(angles_to_crop_rad[1]), 
               0, 
               thickness=-1)

    cropped_img = cv.bitwise_and(dbscan._img, dbscan._img, mask=mask)
    dbscan._img = cropped_img
    # board.display_image_self('Cropped', bgr=False)

    dbscan.apply_masks(colors=['red'], lower_scale=1.0, upper_scale=1.0)
    ring20, _ = dbscan.dbscan_clustering(eps=10, min_samples=7, threshold=10, plot=False)

    def find_center_point(coords):
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        center_point = (center_x, center_y)
        
        return center_point

    center20 = find_center_point(ring20)
    
    return center20


def transform(img_path, predictions, eps=10, min_samples=7, threshold=10, crop_scale=1.0, crop_eye=0.25, size_transform=(1100,1100), accuracy=0.05, iterations=8):
    print(f"Processing image...")
    # print(f"number of iterations: {iterations}")
    try:
        original = Board(img_path)
        board = BoardTransform(original)
        board = initial_prepare(board, crop_eye=crop_eye, crop_scale=crop_scale, size=size_transform)
        # predictions = board.expand_canvas(target_center=board._center, predictions=predictions, scale_factor=1.1)
# JUST OKAY HERE
        transformed, predictions, n_iterations = iterative_transform(
            board,
            predictions,
            init_transform=True,
            iterations=iterations,
            eps=eps, min_samples=min_samples, threshold=threshold,
            accuracy=accuracy,
            crop_eye=crop_eye,
            save_logs=False,
            plot_final=False, plot_steps=False
        )
        # transformed.display_image_self('Transformed', bgr=False)

        outer_ellipse = transformed._outer_ellipse
        inner_ellipse = transformed._inner_ellipse
        outer_ellipse = (
            outer_ellipse[0],
            (min(outer_ellipse[1][0], outer_ellipse[1][1]), min(outer_ellipse[1][0], outer_ellipse[1][1])),
            outer_ellipse[2]
        )
        inner_ellipse = (
            inner_ellipse[0],
            (min(inner_ellipse[1][0], inner_ellipse[1][1]), min(inner_ellipse[1][0], inner_ellipse[1][1])),
            inner_ellipse[2]
        )
        transformed._outer_ellipse = outer_ellipse
        transformed._inner_ellipse = inner_ellipse
        
        predictions = transformed.final_crop(transformed._outer_ellipse, predictions=predictions, padding=0)
        new_center = find_bulls_eye(transformed, crop_eye=0.25, plots=False, centered=True, max_radius=20, min_radius=13, param1=50, param2=15)
        old_center = transformed._center
        transformed._center = new_center
        # predictions.append(new_center)

        img = transformed.draw_points(predictions, display=False)
        
        # DO YOU REMEMBER ABOUT EXECUTION STATS?
        save_execution_stats(img_path, execution_stats)

        # THIS IS TO COMPENSATE ROTATION OF THE BOARD FURTHER. WORKS WITHIN 8 DEGREES.
        center20 = find20_center(transformed)

        return transformed, img, predictions, center20
    
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    # THIS BLOCK WAS USED TO PROCESS WHOLE LIBRARY, TO PREPARE IMAGES FOR TRAINING
    input_dir = "../test_transform/"
    output_dir = "../test_all/"
    eps = 10
    min_samples = 7
    threshold = 10
    crop_scale = 1.0
    crop_eye = 0.25
    size_transform = (1200, 1200)
    print("yes")
    process_directory(input_dir, output_dir, eps, min_samples, threshold, crop_scale, crop_eye, size_transform)


    # TODO: I WANT ANOTHER ACCURACY TECHNIQUE HERE
    # if init_transform:
    #     def n_iter(deltas, accuracy):
    #         n_iterations = 0
    #         for i, delta_num in enumerate(deltas):
    #             if (abs(delta_num[0]) < accuracy) and (abs(delta_num[1]) < accuracy):
    #                 n_iterations = i+1
    #                 print("HERE1")
    #                 print(f'Number of iterations needed: {n_iterations}, with accuracy > {int(accuracy*100)}%')
    #                 return board, n_iterations, predictions
    #                 break
    #         return None
    #     n_iterations = n_iter(deltas, accuracy)
    #     if (n_iterations is None):
    #         while (n_iterations is None) and (accuracy < 0.1):
    #             n_iterations = n_iter(deltas, accuracy)
    #             accuracy += 0.01
    #     if n_iterations is None:
    #         print('TRANSFORMATION ERROR, ADJUSTING PARAMETERS...') 
    #         # TODO: HERE RETURN TO TRANSFORMATION WITH NEW PARAMETERS
    #         n_iterations = 5
    #         return board, n_iterations, predictions
    #     else:
    #         print("HERE2")
    #         print(f'Number of iterations needed: {n_iterations}, with accuracy > {int(accuracy*100)}%')
    #         return board, n_iterations, predictions