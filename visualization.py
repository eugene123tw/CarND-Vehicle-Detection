import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label

from dataset import ImageDataset, DataLoader
from feature_utilities import get_hog_features, bin_spatial, color_hist

class Visualisation(object):
    def __init__(self, pixel_per_cell=4, cell_per_block=2):
        self.pixel_per_cell = pixel_per_cell
        self.cell_per_block = cell_per_block
        self.pixel_per_block = pixel_per_cell * cell_per_block

    def show_cells(self, img, color=(0, 0, 255), show_sliding_block=False):
        H, W, C = img.shape
        nx_cells = W // self.pixel_per_cell
        ny_cells = H // self.pixel_per_cell

        for y in range(ny_cells):
            for x in range(nx_cells):
                img = cv2.rectangle(img, (x * self.pixel_per_cell, y * self.pixel_per_cell),
                              ((x + 1) * self.pixel_per_cell, (y + 1) * self.pixel_per_cell), color, 1)

        cv2.imshow("Cell map", img)
        cv2.waitKey(50)
        cv2.destroyAllWindows()

        if show_sliding_block:
            nx_blocks = nx_cells - self.cell_per_block + 1
            ny_blocks = ny_cells - self.cell_per_block + 1
            for y in range(ny_blocks):
                for x in range(nx_blocks):
                    x1 = x * self.pixel_per_cell
                    y1 = y * self.pixel_per_cell

                    x2 = x1 + self.pixel_per_block
                    y2 = y1 + self.pixel_per_block
                    img_tmp = cv2.rectangle(img.copy(), (x1, y1),
                                        (x2, y2 ), (255, x+1, 0), 2)
                    cv2.imshow("Block map", img_tmp)
                    cv2.waitKey(50)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    visual = Visualisation(pixel_per_cell=32)
    img = cv2.imread("/home/eugene/Dev/udacity_CarND/CarND-Vehicle-Detection/test_images/test1.jpg")
    visual.show_cells(img)