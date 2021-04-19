import cv2
import numpy as bb8


# code for this method is taken from first challenge
def image_preparation_for_thresh(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    img_t = 1 - img_gs

    return img_t


# code for resize taken from https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
def resize_photo(img):
    scale_percent = 150
    newWidth = int(img.shape[1] * scale_percent / 100)
    newHeight = int(img.shape[0] * scale_percent / 100)

    resized_img = cv2.resize(img, (int(newWidth), int(newHeight)), interpolation=cv2.INTER_NEAREST)
    return resized_img


# code for brightness taken from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
def brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] += 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


# code for most dominant color taken from: https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def get_most_dominant_color(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)
    a1D = bb8.ravel_multi_index(a2D.T, col_range)
    return bb8.unravel_index(bb8.bincount(a1D).argmax(), col_range)


def get_highest_peak_att_255(peaks, k):
    max_peak = peaks[255]
    return [max_peak, k]


def arrange_histogram_array(max_peaks):
    new_array = []
    for peak in max_peaks:
        color = peak[1]
        pair = peak[0]
        max_hist = pair[0]
        new_array.append([max_hist, color])
    return new_array


def is_color(max_peaks):
    max_peak = [0, ""]
    for peak in max_peaks:
        if peak[0] > max_peak[0]:
            max_peak = [peak[0], peak[1]]

    return max_peak


# histogram explained: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
# code for getting histogram taken from opencvs documenataion: https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html

def determine_color(path):
    if get_most_dominant_color(brightness_up(cv2.imread(path))) != (255, 255, 255):

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = ('r', 'g', 'b')
        max_peaks = []
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            max_peaks.append(get_highest_peak_att_255(histr, color[i]))
            # plt.plot(histr, color=col)
            # plt.xlim([0, 256])
        # plt.show()

        max_hist_array = arrange_histogram_array(max_peaks)
        if is_color(max_hist_array)[1] == 'b':
            return 'BLUE'
        elif is_color(max_hist_array)[1] == 'r':
            return 'RED'
        elif is_color(max_hist_array)[1] == 'g':
            return 'GREEN'
    else:
        return "white"


def image_to_vector(img):
    return img.flatten()


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_image(img):
    return img / 255
