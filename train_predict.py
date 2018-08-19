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

from dataset import ImageDataset, DataLoader
from feature_utilities import get_hog_features, bin_spatial, color_hist

CSPACE = 'HLS'
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
ORIENT = 9
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32

def read_paths_labels(csv_path):
    img_paths = []
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_paths.append(row[0])
            labels.append(int(row[1]))
    return img_paths, labels

def convert_color(img, conv='RGB'):
    if conv == 'RGB':
        return img
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def train(save=False):

    img_paths, labels = read_paths_labels('images_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        img_paths,
        labels,
        test_size=0.2,
        random_state=42)

    train_dataset = ImageDataset(X_train, y_train,
                                 cspace=CSPACE,
                                 orient=ORIENT,
                                 pix_per_cell=PIX_PER_CELL,
                                 cell_per_block=CELL_PER_BLOCK,
                                 spatial_size=SPATIAL_SIZE,
                                 hist_bins=HIST_BINS)

    test_dataset = ImageDataset(X_test, y_test,
                                cspace=CSPACE,
                                orient=ORIENT,
                                pix_per_cell=PIX_PER_CELL,
                                cell_per_block=CELL_PER_BLOCK,
                                spatial_size=SPATIAL_SIZE,
                                hist_bins=HIST_BINS)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    # train_size = 1000
    # test_size = 1000

    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)


    svc = SVC()

    for iter, (features, labels, indices) in enumerate(train_loader, 0):
        X_scaler = StandardScaler().fit(features)
        scaled_X_train = X_scaler.transform(features)
        svc.fit(scaled_X_train, labels)
        accu = svc.score(scaled_X_train, labels)
        print('Iter: %d, Train Accuracy of SVC = %f' % (iter, accu))


    for iter, (features, labels, indices) in enumerate(test_loader, 0):
        scaled_X_test = X_scaler.transform(features)
        print('Test Accuracy of SVC = ', svc.score(scaled_X_test, labels))

    if save:
        joblib.dump(svc, 'svc_HLS.pkl')
        joblib.dump(X_scaler, 'scaler.pkl')

def prediction(img, estimator, scalar, ystart_ystop=[None, None], scale=1):

    # load a pe-trained svc model from a serialized (pickle) file
    svc = estimator
    X_scaler = scalar

    draw_img = np.copy(img)

    if ystart_ystop[0] == None:
        ystart = 0
    else:
        ystart = ystart_ystop[0]

    if ystart_ystop[1] == None:
        ystop = img.shape[0]
    else:
        ystop = ystart_ystop[1]

    img_tosearch = img[ystart:ystop, :, :]
    img_tosearch = convert_color(img_tosearch, conv=CSPACE)
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch,
                                     (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = img_tosearch[:, :, 0]
    ch2 = img_tosearch[:, :, 1]
    ch3 = img_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    nyblocks = (ch1.shape[0] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    nfeat_per_block = ORIENT * CELL_PER_BLOCK ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)
    hog2 = get_hog_features(ch2, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)
    hog3 = get_hog_features(ch3, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)

    for xb in np.arange(nxsteps, step=2):
        for yb in np.arange(nysteps, step=2):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * PIX_PER_CELL
            ytop = ypos * PIX_PER_CELL

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=SPATIAL_SIZE)
            hist_features = color_hist(subimg, nbins=HIST_BINS)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    plt.imshow(draw_img)
    plt.show()


if __name__ == '__main__':


    train(save=True)

    # test_images = [np.array(Image.open(path)) for path in glob.glob('test_images/*.jpg')]
    # svc = joblib.load('svc_HLS.pkl')
    # X_scaler = joblib.load('scaler.pkl')
    # for test_img in test_images:
    #     prediction(test_img, estimator=svc, scalar=X_scaler, ystart_ystop=[350, 550], scale=1.2)
