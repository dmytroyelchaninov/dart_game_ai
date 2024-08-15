import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import math
from main import Game, Player, Board

class DartBoard(Game):
    def __init__(self, img_path):
        super().__init__()
        self._img_path = img_path
        self._img = self._load_image()
        self._bulls_eye = None
        self.grid = None

    def _load_image(self):
        img = cv.imread(self._img_path, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image at path {self._img_path} could not be loaded.")
        return img

    def draw_dartboard(self):
        center, radius_mean, centers, circles = self.bulls_eye_main()
        self.grid = DartGrid(radius_mean)
        self._img = self.rotate_image(self._img, center, -9)
        self._img = self.grid.draw_grid(self._img, center)

        
        cv.imshow('Rotated Dartboard', self._img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # self.draw_circles(self._img, circles, center)

        return self._img

    def bulls_eye_main(self):
        gray = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 17)
        dim = gray.shape[0]

        minrad_eye = int(dim / 4.70)
        maxrad_eye = int(minrad_eye * 1.18)
        
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=150,
            param2=50,
            minRadius=minrad_eye,
            maxRadius=maxrad_eye
        )

        if circles is None:
            print("Can't find bulls eye")
            return -1, None, None

        # Drawing center of the bulls eye
        circles = np.uint16(np.around(circles))

        # Extract circle centers
        centers = [(circle[0], circle[1]) for circle in circles[0, :]]

        mean_x = int(np.median([c[0] for c in centers]))
        mean_y = int(np.median([c[1] for c in centers]))
        center_mean = (mean_x, mean_y)

        radii = [circle[2] for circle in circles[0, :]]
        radius_mean = int(np.mean(radii))

        return center_mean, radius_mean, centers, circles

    def rotate_image(self, image, center, angle):
        """
        Rotates the image around the given center by the specified angle.

        Parameters:
        image (ndarray): The image to be rotated.
        center (tuple): The center around which to rotate.
        angle (float): The angle by which to rotate the image.

        Returns:
        rotated_image (ndarray): The rotated image.
        """
        (h, w) = image.shape[:2]
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv.warpAffine(image, M, (w, h))
        return rotated_image



#         self.sectors = 20
#         self.angle_per_sector = 360 / self.sectors
#         self.inner_bulls_eye_radius = 12.7
#         self.outer_bulls_eye_radius = 31.8
#         self.triple_ring_inner_radius = 107
#         self.triple_ring_outer_radius = 115
#         self.double_ring_inner_radius = 162
#         self.double_ring_outer_radius = 170


class DartGrid:
    def __init__(self, triple_ring_outer_radius_pixels):
        self.sectors = 20
        self.angle_per_sector = 360 / self.sectors
        
        # We can use whole dartboard size to calculate the ratio in pixels/mm
        # But this should input striclty whole dartboard without cutted edges
        self.ratio = 800 / 451

        # Convert mm to pixels using fixed ratio
        self.inner_bulls_eye_radius = (6.35/115) * triple_ring_outer_radius_pixels
        self.outer_bulls_eye_radius = (15.9/115) * triple_ring_outer_radius_pixels
        self.triple_ring_inner_radius = (105/115)*triple_ring_outer_radius_pixels
        self.triple_ring_outer_radius = triple_ring_outer_radius_pixels
        self.double_ring_inner_radius = (170/115) * triple_ring_outer_radius_pixels
        self.double_ring_outer_radius = (180/115) * triple_ring_outer_radius_pixels

    def draw_grid(self, img, center, scale=1):
        height, width = img.shape[:2]
        center_x, center_y = center
        
        for i in range(self.sectors):
            angle = i * self.angle_per_sector
            x1 = int(center_x + self.double_ring_outer_radius * scale * np.cos(np.radians(angle)))
            y1 = int(center_y + self.double_ring_outer_radius * scale * np.sin(np.radians(angle)))
            cv.line(img, (center_x, center_y), (x1, y1), (0, 0, 255), 1)
        
        # Draw center of the bulls eye
        cv.circle(img, center, 3, (0, 255, 0), -1)

        cv.circle(img, center, int(self.inner_bulls_eye_radius * scale), (0, 0, 255), 2)
        cv.circle(img, center, int(self.outer_bulls_eye_radius * scale), (0, 255, 0), 2)
        cv.circle(img, center, int(self.triple_ring_inner_radius * scale), (0, 255, 0), 2)
        cv.circle(img, center, int(self.triple_ring_outer_radius * scale), (0, 255, 0), 2)
        cv.circle(img, center, int(self.double_ring_inner_radius * scale), (0, 255, 0), 2)
        cv.circle(img, center, int(self.double_ring_outer_radius * scale), (0, 255, 0), 2)
        
        return img


if __name__ == '__main__':
    path = '../upload/sc_temp.jpg'
    dart_board = DartBoard(path)
    dart_board.draw_dartboard()