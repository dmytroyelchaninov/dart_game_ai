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

# TODO:
# TRY TRAIN SVM BASED ON DSCAN RESULTS IN FUTURE, OR NN 
# REWORK THE __INIT__ OF EACH CLASS, ADD SUPER()
# REWORK CLASSES DIVISION!!!
# THIS SCRIPT SHOULD HAVE MORE ACCURATE CENTER DETECTION

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
    
    def draw_points(self, points, img=None, title='', color=(0, 255, 0), radius=5):
        if img is None:
            img = self._img.copy()
        if isinstance(points[0], (list, tuple, np.ndarray)):
            for point in points:
                cv.circle(img, (int(point[0]), int(point[1])), radius, color, -1)
        else:
            cv.circle(img, (int(points[0]), int(points[1])), radius, color, -1)

        # self.display_image(cv.cvtColor(img, cv.COLOR_BGR2RGB), title, cmap=None)

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

    def resize_image(self, target_size=None, half_size=False):
        """Resizes img with defined center to selected target_size
        Recalculates center after
        Redefines ._img and ._center
        Returnes img and center just in case"""
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

        self._img = resized_image
        self._center = new_center

        return resized_image, new_center

    def final_crop(self, ellipse, padding_factor=0.03):
        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
        
        # Determine the radius for the cropping circle based on the largest axis
        radius = int(max(major_axis_length, minor_axis_length) / 2 * (1 + padding_factor))
        
        # Create a single-channel mask with a filled circle using the calculated radius
        mask = np.zeros((self._img.shape[0], self._img.shape[1]), dtype=np.uint8)
        cv.circle(mask, (int(center_x), int(center_y)), radius, 255, thickness=-1)
        
        # Convert the mask to a three-channel mask if the image is not single-channel
        if len(self._img.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        
        # Apply the mask to the image
        masked_img = cv.bitwise_and(self._img, mask)
        
        # Find the bounding rectangle of the non-zero region
        x, y, w, h = cv.boundingRect(cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
        
        # Crop the image to this bounding rectangle
        cropped_img = masked_img[y:y+h, x:x+w]
        
        # Update the image and return
        self._img = cropped_img
        return self._img
    
    def mirror_image(self, axis):
        if axis == 'x':
            mirrored_img = cv.flip(self._img, 0)
        elif axis == 'y':
            mirrored_img = cv.flip(self._img, 1)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        self._img = mirrored_img
        return self._img

    def expand_canvas(self, target_center, scale_factor=1.5):
        original_height, original_width = self._img.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        
        # Create a new blank canvas (larger than the original)
        new_canvas = np.zeros((new_height, new_width, self._img.shape[2]), dtype=self._img.dtype)
        
        # Calculate the top-left position where the image should be placed on the new canvas
        top_left_x = (new_width // 2) - target_center[0]
        top_left_y = (new_height // 2) - target_center[1]
        
        # Place the original image onto the new canvas at the calculated position
        new_canvas[top_left_y:top_left_y+original_height, top_left_x:top_left_x+original_width] = self._img

        # Update the image and center (center stays relative to the image)
        self._img = new_canvas
        self._center = (target_center[0] + top_left_x, target_center[1] + top_left_y)
        
        return self._img, self._center

    # SOME COLOR MASK SHT (TODO: SEPARATE CLASS!!!)
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
        
        if len(self._img.shape) == 2:  # If the image is grayscale (1 channel)
            self._img = cv.cvtColor(self._img, cv.COLOR_GRAY2BGR)  # Convert to 3 channels

        hsv = cv.cvtColor(self._img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color in colors:
            if color == 'gray':
                ranges = self._generate_gray_ranges(lower_scale, upper_scale)
            elif color == 'green':
                ranges = self._generate_green_ranges(lower_scale, upper_scale)
            elif color == 'red':
                ranges = self._generate_red_ranges(lower_scale, upper_scale)
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
    # END OF COLOR MASK SHT

    # START OF DBSCAN SHT (IT FITS AN ELLIPSE AROUND THE DARTBOARD) (TODO: SEPARATE CLASS!!!)
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

    def dbscan_clustering(self, eps=7, min_samples=10, plot=False, threshold=30):
        thresh = self.gray_thresh(threshold=threshold)
        coords = cv.findNonZero(thresh)

        # #PROBLEM HERE COLUMN STACK I GOT YOU!
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
        # Ensure enough points for fitting ellipse
        if len(coords) < 5:
            raise ValueError("Not enough points to fit an ellipse.")

        # Fit ellipse to the combined points
        coords = np.array(coords, dtype=np.float32)  # Ensure the points are in float32 format
        ellipse = cv.fitEllipse(coords)
        if outer:
            self._outer_ellipse = ellipse
        else:
            self._inner_ellipse = ellipse
        # Return the adjusted ellipse and its center
        return ellipse

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
 
    def transform_ellipses_to_circle(self, outer_ellipse, inner_ellipse, center, target_size=None, half_size=False):
        points_src_outer = self.generate_ellipse_points(outer_ellipse)
        points_src_inner = self.generate_ellipse_points(inner_ellipse)
        points_src = np.vstack([points_src_outer, points_src_inner, outer_ellipse[0], inner_ellipse[0]])

        destination_center = center
        points_dst = []
        for point in points_src[:-2]:
            angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
            radius = ((outer_ellipse[1][0] + outer_ellipse[1][1]) / 4) * 1.2
            dst_x = destination_center[0] + radius * np.cos(angle)
            dst_y = destination_center[1] + radius * np.sin(angle)
            points_dst.append([dst_x, dst_y])

        points_dst.append(destination_center)
        points_dst.append(destination_center)
        points_dst = np.array(points_dst, dtype=np.float32)

        # Ensure padding is calculated before being used
        max_dst_radius = np.max(np.linalg.norm(points_dst - np.array([center[0], center[1]]), axis=1))
        padding = int(max_dst_radius * 0.7)  # Calculate padding
        output_radius = int(max_dst_radius + padding)
        output_size = (output_radius * 2, output_radius * 2)

        points_dst += padding

        homography_matrix, _ = cv.findHomography(points_src, points_dst)
        transformed_image = cv.warpPerspective(self._img, homography_matrix, output_size)

        center_homogeneous = np.array([center[0], center[1], 1.0], dtype=np.float32).reshape(-1, 1)
        transformed_center_homogeneous = homography_matrix @ center_homogeneous
        transformed_center = transformed_center_homogeneous[:2] / transformed_center_homogeneous[2]

        translation_x = (output_size[0] // 2) - int(transformed_center[0][0])
        translation_y = (output_size[1] // 2) - int(transformed_center[1][0])

        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        transformed_image = cv.warpAffine(transformed_image, translation_matrix, output_size)

        self._img = transformed_image
        self._center = (output_size[0] // 2, output_size[1] // 2)
        self.resize_image(target_size=target_size, half_size=half_size)

        return transformed_image, self._center

    def transform_ellipses_to_circle_1(self, outer_ellipse, inner_ellipse, center):
        # points_src_outer = self.generate_ellipse_points(outer_ellipse)
        # points_src_inner = self.generate_ellipse_points(inner_ellipse)

        # points_src = np.vstack([points_src_outer, points_src_inner, outer_ellipse[0], inner_ellipse[0]])
        # destination_center = center
        # points_dst = []
        # for point in points_src[:-2]:
        #     angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
        #     radius = ((outer_ellipse[1][0] + outer_ellipse[1][1]) / 4) * 1.2
        #     dst_x = destination_center[0] + radius * np.cos(angle)
        #     dst_y = destination_center[1] + radius * np.sin(angle)
        #     points_dst.append([dst_x, dst_y])

        # points_dst.append(destination_center)
        # points_dst.append(destination_center)
        # points_dst = np.array(points_dst, dtype=np.float32)



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


        # Compute the homography matrix without resizing or padding
        homography_matrix, _ = cv.findHomography(points_src, points_dst)
        transformed_image = cv.warpPerspective(self._img, homography_matrix, (self._img.shape[1], self._img.shape[0]))

        # Recalculate the center based on the homography transformation
        center_homogeneous = np.array([center[0], center[1], 1.0], dtype=np.float32).reshape(-1, 1)
        transformed_center_homogeneous = homography_matrix @ center_homogeneous
        transformed_center = transformed_center_homogeneous[:2] / transformed_center_homogeneous[2]

        self._img = transformed_image
        self._center = (int(transformed_center[0][0]), int(transformed_center[1][0]))
        # print(self._img.shape)
        return transformed_image, self._center

# TODO: ADD THIS TO CLASS PerspectiveTransform
def find_bulls_eye(board, crop_eye=0.25, min_radius=13, max_radius=30, param1=70, param2=20):
    """Should receive class with ._img defined
    Returns center, if found, otherwise None"""
    
    # This function looks for the bulls eye around 25% of the image center
    initial_center = board._center
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for step in np.linspace(0.05, 0.15, 5):
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
                    print("HERE")
                    crop_mask_res = BoardTransform(board)
                    crop_mask_res.crop_radial(center=center, crop_factor=crop_eye)
                    crop_mask_res.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
                    precise_center, radius, circles = crop_mask_res.find_center(min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)

                    # crop_mask_res.draw_center(center=precise_center)
                    # crop_mask_res.display_image_self('center')
                    # print(f"Center found...")
                    return precise_center
                except ValueError:
                    print("HERE HERE")
                    return center
            except ValueError:
                pass
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
def find_ellipse(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False):
    """Finds outer and inner ellipse.
    Should receive pre-cropped board class BoardTransform.
    Crops around outer ellipse found.
    Returns board back with ellipses"""
    first_try = None
    second_try = None
    try:
        eps_ratio_semis = []
        ratio_outer_inner = None
        while ratio_outer_inner is None or ratio_outer_inner > 1.015:
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
            while ratio_outer_inner > 1.015 and min_samples < 100:
                board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
                ratio_outer_inner = max((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))\
                    /min((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))
                ratio_outer = (max(outer_ellipse[1][0],outer_ellipse[1][1]))/(min(outer_ellipse[1][0],outer_ellipse[1][1]))
                min_samples_ratio_semis.append((min_samples, ratio_outer_inner, ratio_outer, threshold))
                if threshold > 0:
                    threshold -= 1
                min_samples += 1
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
    board.crop_ellipse(outer_ellipse, outer_padding_factor=0.02)

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
    bulls_eye = find_bulls_eye(board, crop_eye=crop_eye)
    if bulls_eye:
        board._center = bulls_eye
        board.crop_radial(bulls_eye, crop_factor=0.8*crop_scale)
    else:
        board.crop_radial(crop_factor=0.9*crop_scale)
    
    return board

def transform_perspective(board,plots=False, crop_eye=0.25):
    """Receives prepared board with defined center, ellipses
    Returns transformed image, with refined center, if found"""
    transform = PerspectiveTransform(board)
    transform.transform_ellipses_to_circle_1(transform._outer_ellipse, transform._inner_ellipse, transform._center)
    try:
        new_center = find_bulls_eye(transform, crop_eye=crop_eye)
        transform._center = new_center
    except (ValueError, TypeError):
        print("Center was not refined")
    if plots:
        transform_copy = BoardTransform(transform)
        transform_copy.draw_center(center=new_center, color=(0, 0, 255), s=5)
        transform_copy.display_image_self(f'Transform', bgr=False)
    return transform


def iterative_transform(board,
                    init_transform=True,
                    eps=10, min_samples=7, 
                    threshold=10, 
                    iterations=8,
                    crop_eye=0.25,
                    accuracy=0.01,
                    save_logs=False,
                    plot_final=False,
                    plot_steps=False):
    
    original = BoardTransform(board)

    # TODO: rework in while loop. Define conditions for that
    logs = []
    outer_ellipse_semis = []
    inner_ellipse_semis = []
    outer_ellipse_center = []
    inner_ellipse_center = []
    center = []

    for i in range(1,iterations+1):
        # print(f'Iteration {i}')
        board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)

        outer_ellipse_semis.append(board._outer_ellipse[1])
        inner_ellipse_semis.append(board._inner_ellipse[1])
        outer_ellipse_center.append(board._outer_ellipse[0])
        inner_ellipse_center.append(board._inner_ellipse[0])
        center.append(board._center)



        board = transform_perspective(board, plots=plot_steps, crop_eye=crop_eye)

    board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)

    outer_ellipse_semis.append(board._outer_ellipse[1])
    inner_ellipse_semis.append(board._inner_ellipse[1])
    outer_ellipse_center.append(board._outer_ellipse[0])
    inner_ellipse_center.append(board._inner_ellipse[0])
    center.append(board._center)


    def normalize_list(numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        normalized = [(x - min_val) / (max_val - min_val) for x in numbers]
        return normalized
    
    outer_differences = [abs(outer[0] - outer[1]) for outer in outer_ellipse_semis]
    inner_differences = [abs(inner[0] - inner[1]) for inner in inner_ellipse_semis]

    outer_differences_normalized = normalize_list(outer_differences)
    inner_differences_normalized = normalize_list(inner_differences)

    delta_outer = []
    i = 0
    while (i+1) < len(outer_differences_normalized):
        (outer_differences_normalized[i+1] - outer_differences_normalized[i])
        delta_outer.append((outer_differences_normalized[i+1] - outer_differences_normalized[i]))
        i += 1

    delta_inner = []
    i = 0
    while (i+1) < len(inner_differences_normalized):
        (inner_differences_normalized[i+1] - inner_differences_normalized[i])
        delta_inner.append((inner_differences_normalized[i+1] - inner_differences_normalized[i]))
        i += 1

    deltas = [x for x in  zip(delta_outer, delta_inner)]

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
        def n_iter(deltas, accuracy):
            n_iterations = 0
            for i, delta_num in enumerate(deltas):
                if (abs(delta_num[0]) < accuracy) and (abs(delta_num[1]) < accuracy):
                    n_iterations = i+1
                    return n_iterations
            return None
        n_iterations = n_iter(deltas, accuracy)
        if (n_iterations is None):
            while (n_iterations is None) and (accuracy < 0.1):
                n_iterations = n_iter(deltas, accuracy)
                accuracy += 0.01
        if n_iterations is None:
            print('TRANSFORMATION ERROR, ADJUSTING PARAMETERS...') 
            # TODO: HERE RETURN TO TRANSFORMATION WITH NEW PARAMETERS
            n_iterations = 8
            return board, n_iterations
        else:
            # print(f'Number of iterations needed: {n_iterations}, with accuracy > {int(accuracy*100)}%')
            return board, n_iterations

    return board


if __name__ == '__main__':  

    # TODO:
    # REWORK THIS BEFORE DEPLOYMENT (GET TRANSFORMED)
    # SCRIPT SHOULD NOT BE EXECUTED TWICE, SIZE TRANSFORM SHOULD BE ORIGINAL PIC
    # IT SHOULD STOP WHENEVER CONDITION IN GET TRANSFORMED IS MET
    # NOW IT CALCULATES NUMBER OF ITERATIONS NEEDED, THEN ITERATES AGAIN WITH THIS NUMBER, AND BETTER SIZE
    

    img_path = '../data/unprocessed/angle/DSC_0002.JPG'
    # img_path = '../upload/sc_test.jpg'
    original = Board(img_path)
    board = BoardTransform(original)
    eps = 10
    min_samples = 7
    threshold = 10
    crop_scale = 1.0
    crop_eye = 0.2
    size_transform = (800, 800)

    board = initial_prepare(board, crop_eye=crop_eye, crop_scale=crop_scale, size=size_transform)
    board.expand_canvas(target_center=board._center, scale_factor=1.1)
    transformed, n_iterations = iterative_transform(
        board,
        init_transform=True,
        iterations=10, 

        eps=eps, min_samples=min_samples, threshold=threshold,
        accuracy=0.01, 
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
        plot_final=True, plot_steps=False
    )


    # for i in range(1, 5):
    #     print(f'Iteration {i}')
    #     board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=True)
    #     board = transform_perspective(board, plots=True, crop_eye=crop_eye)

    # board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=True)
    # transform = transform_perspective(board, plots=True, crop_eye=crop_eye)




    # for imgs in os.listdir('../upload/new/new_new'):
    # # for imgs in os.listdir('../data/unprocessed/angle/test/'):

    #     # img_path = f'../upload/new/{imgs}'
    #     img_path = f'../upload/new/new_new/{imgs}'
    #     try:
    #         original = Board(img_path)
    #     except AttributeError:
    #         print(f'Error with {img_path}')
    #         continue
    #     board = BoardTransform(original)
    #     eps = 10
    #     min_samples = 7
    #     threshold = 10
    #     crop_scale = 1.0
    #     crop_eye = 0.2
    #     size_transform = (800, 800)

    #     board = initial_prepare(board, crop_eye=crop_eye, crop_scale=crop_scale, size=size_transform)
    #     board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=True)
        




    # transform.draw_center(center=transform._center, color=(0, 255, 0), s=5)
    # transform.display_image_self('Final pic', bgr=False)




    # board_init = crop_prepare_initial(board, eps=eps, min_samples=min_samples, threshold=threshold, crop_scale=crop_scale, crop_eye=crop_eye)
    # board_init.expand_canvas(target_center=board_init._center, scale_factor=1.2)
    # base = crop_prepare_initial(board_init,
    #                             eps=eps-2, min_samples=min_samples+1, threshold=threshold, crop_scale=crop_scale, crop_eye=crop_eye)


    # # try:   
    # transformed, n_iterations = get_transformed(
    #     original,
    #     init_transform=True,
    #     plot_steps=True,
    #     plot_final=False,
    #     logs=True,
    #     iterations=10,
    #     crop_scale=1.0,
    #     crop_eye=0.25,
    #     accuracy=0.01,
    #     eps=10, min_samples=7, threshold=10
    #     )
        
    # transformed.display_image_self('Final pic', bgr=False)


    # if original._img_size[0] > 3000:
    #     size_transform = (2200, 2200)
    # elif original._img_size[0] > 2000:
    #     size_transform = (1700, 1700)
    # else:
    #     size_transform = (1100, 1100)

    # transformed = get_transformed(
    #     img_path,
    #     init_transform=False,
    #     plot_steps=False,
    #     plot_final=False,
    #     logs=False,
    #     target_size=size_transform,
    #     iterations=n_iterations,
    #     crop_scale=1.0,
    #     crop_eye=0.25,
    #     accuracy=0.01,
    #     eps=10, min_samples=7, threshold=10
    #     )

    # transformed.final_crop(transformed._outer_ellipse, padding_factor=0.02)
    # transformed.resize_image(target_size=(800, 800))

    # final_center = find_bulls_eye(transformed, crop_eye=0.1, min_radius=5, max_radius=20, param1=50, param2=20)
    # transformed._center = final_center
    # print(f'Final center: {final_center}')

    # transformed.draw_center(center=final_center, color=(0, 255, 0), s=5)
    # transformed.display_image_self('Final pic', bgr=False)

        # # print(f'Transformed ellipse: {transformed._outer_ellipse}')
        # # my_list = [float(line.strip()) for line in file]

    # except TypeError:
    #     print('No transformation needed')
    # except ValueError:
    #     print("Image can't be processed, please take another picture")








    
# DEPENDING ON OUTER ELLIPSE SEMIS, INNER ELLIPSE SEMIS, CENTER:
# 1.
# CHOOSE IF CHANGE DBSCAN PARAMETERS, CROP, NUM OF ITERATIONS ADN REAPPLY GET_TRANSFORMED
# WITH NEW PARATERS FOR DBSCAN AND POSSIBLY CROP_EYE
# 2.
# OR FIND ELBOW (FROM CHART?) AND TRANSFORM IN INITIAL SIZE 
# 3.
# OR ADD ITERATIONS=AUTO WITH WHILE LOOP (UNTIL DELTAS ARE SMALL),
# AND INCREASE IF ERRORS






    # gray_mask.crop_ellipse(ellipse, inner_padding_factor=1, outer_padding_factor=0.35)
    # gray_mask._img, mask = gray_mask.apply_masks(colors=['gray'], lower_scale=1.5, upper_scale=1.2)
    # overlay_img = gray_mask.overlay_mask_on_image(base._img, mask, alpha=1, color=(255, 255, 255))
    # base._img = overlay_img