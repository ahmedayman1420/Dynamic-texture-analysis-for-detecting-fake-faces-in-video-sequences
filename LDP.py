# ========== ----- ========== Import Libraries ========== ----- ========== #

import numpy as np
import pandas as pd

import dlib
import cv2
import os

# ========== ----- ========== End ========== ----- ========== #

# ========== ----- ========== Local Derivative Pattern Services For Image ========== ----- ========== #

def F(a, b):
    if a*b > 0:
        return "0"
    elif a*b <= 0:
        return "1"

def I(picture, h, w, theta):
    if (theta == 0):
        return picture[h,w] - picture[h,w+1]
    elif (theta == 45):
        return picture[h, w] - picture[h-1, w+1]
    elif (theta == 90):
        return picture[h, w] - picture[h-1, w]
    elif (theta == 135):
        return picture[h, w] - picture[h-1, w-1]


def ldp_pixel(picture, h, w):  # calculating local derivative pattern value of a pixel
    eigth_bit_binary = []
    four_decimal_val = []
    decimal_val = 0
    angles = [0, 45, 90, 135]

    for theta in angles:

        # starting from top left,assigning bit to pixels clockwise at theta degree
        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h-1, w-1, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h-1, w, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h-1, w+1, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h, w+1, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h+1, w+1, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h+1, w, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h+1, w-1, theta)))

        eigth_bit_binary.append(
            F(I(picture, h, w, theta), I(picture, h, w-1, theta)))

        l = "".join(eigth_bit_binary)
        decimal_val = int(l, 2)
        four_decimal_val.append(decimal_val)
        eigth_bit_binary = []

    return four_decimal_val


def ldp_img(img):
    m, n = img.shape
    # converting image to grayscale
    # ldp_photo = np.zeros((m, n),np.uint8)
    ldp_img = np.zeros((m, n, 4))
    # converting image to ldp
    for i in range(2, m-2):
        for j in range(2, n-2):
            ldp_pixels = ldp_pixel(img, i, j)
            ldp_img[i, j, 0] = ldp_pixels[0]
            ldp_img[i, j, 1] = ldp_pixels[1]
            ldp_img[i, j, 2] = ldp_pixels[2]
            ldp_img[i, j, 3] = ldp_pixels[3]

    return ldp_img

# ========== ----- ========== End ========== ----- ========== #

# ========== ----- ========== Local Derivative Pattern Main Function For Image ========== ----- ========== #


def extract_hist(gray_scale_img):

    features = ldp_img(gray_scale_img)

    histogram_0, bin_edges = np.histogram(
        features[:, :, 0], bins=256, range=(0, 256))

    histogram_45, bin_edges = np.histogram(
        features[:, :, 1], bins=256, range=(0, 256))

    histogram_90, bin_edges = np.histogram(
        features[:, :, 2], bins=256, range=(0, 256))

    histogram_135, bin_edges = np.histogram(
        features[:, :, 3], bins=256, range=(0, 256))

    hist = np.concatenate(
        (histogram_0, histogram_45, histogram_90, histogram_135))

    return hist

# ========== ----- ========== End ========== ----- ========== #

# ========== ----- ========== Local Derivative Pattern Main Function For TOP ========== ----- ========== #


def LDP_TOP(frames):

    x, y, z = frames.shape
    plane_yz = frames[(x//2), :, :]
    plane_xz = frames[:, (y//2), :]
    plane_xy = frames[:, :, (z//2)]

    hist_yz = extract_hist(plane_yz)
    hist_xz = extract_hist(plane_xz)
    hist_xy = extract_hist(plane_xy)

    hist_TOP = np.concatenate((hist_yz, hist_xz, hist_xy))
    return hist_TOP

# ========== ----- ========== End ========== ----- ========== #

# ========== ----- ========== Training Function ========== ----- ========== #

def train(train_frame_folders):
    # Face detector
    detector = dlib.get_frontal_face_detector()

    list_of_train_LDP = []
    list_of_train_labels = []

    for train_frame_folder in train_frame_folders:
        list_of_train_data = [f for f in os.listdir(
            train_frame_folder) if f.endswith('.mp4')]
        print(len(list_of_train_data))
        for vid in [list_of_train_data[0]]:
            frames = []
            # Video capturing constructor.
            cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))

            # (CAP_PROP_FPS) Returns frame rate of the video (#frames / second).
            frameRate = cap.get(5)
            while cap.isOpened():  # Returns true if video capturing has been initialized already.

                # (CAP_PROP_POS_MSEC) Current position of the video file in milliseconds or video capture timestamp.
                frameId = cap.get(1)
                ret, frame = cap.read()  # The methods/functions combine VideoCapture::grab and VideoCapture::retrieve in one call. This is the most convenient method for reading video files or capturing data from decode and return the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the methods return false and the functions return NULL pointer.
                if ret != True:
                    break

                face_rects, scores, idx = detector.run(frame, 0)
                for i, d in enumerate(face_rects):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()

                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img is not None and crop_img.size > 0:
                        crop_img = cv2.resize(crop_img, (128, 128))
                        gray_scale_img = cv2.cvtColor(
                            crop_img, cv2.COLOR_BGR2GRAY)
                        frames.append(gray_scale_img)

                # d = 3 seconds.
                if frameId % ((int(frameRate)+1)*3) == 0:
                    if (len(frames) > 0):
                        frames_ldp = LDP_TOP(
                            np.array(frames).astype(np.float64))
                        list_of_train_LDP.append(frames_ldp)
                        if train_frame_folder == 'faceforensics_dataset_train/original_sequences':
                            list_of_train_labels.append(0)
                        else:
                            list_of_train_labels.append(1)
                    frames = []

    return list_of_train_LDP, list_of_train_labels
# ========== ----- ========== End ========== ----- ========== #
