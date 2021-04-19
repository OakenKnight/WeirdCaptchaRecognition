import cv2
import matplotlib.pyplot as plt
import numpy as bb8
from sklearn.cluster import KMeans
import deal_with_water as dww
import deal_with_bricks as dwb
import image_operations as iop
import roi_handler as roi

def morhpological_operations(img_bin):
    # img = bb8.uint8(img_bin)
    img = img_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    erosion = cv2.erode(closing, kernel_e, iterations=4)
    dilation = cv2.dilate(erosion, kernel_c, iterations=2)
    # dilation = cv2.erode(dilation,kernel_c,iterations = 2)
    # img = erosion
    return dilation


def select_roi(image_orig, img_bin):
    img, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            region = img_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    try:
        new_sorted_rectangles = deal_with_hooks_rectangles(sorted_regions, sorted_rectangles)
    except Exception as e:
        new_sorted_rectangles = sorted_rectangles

    new_sorted_regions = []
    img_before_hooks = image_orig.copy()

    for rectangle in sorted_rectangles:
        cv2.rectangle(img_before_hooks, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)
    cv2.imwrite('hukovi.png', img_before_hooks)

    for rectangle in new_sorted_rectangles:
        region = img_bin[rectangle[1]:rectangle[1] + rectangle[3] + 2, rectangle[0]:rectangle[0] + rectangle[2] + 2]
        new_sorted_regions.append(resize_region(region))
        cv2.rectangle(image_orig, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)

    region_distances = []

    for i in range(0, len(new_sorted_rectangles) - 1):
        current = new_sorted_rectangles[i]
        next_rect = new_sorted_rectangles[i + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    # plt.imshow(image_orig)
    # plt.show()
    return image_orig, new_sorted_regions, region_distances, new_sorted_rectangles


def deal_with_hooks_rectangles(sorted_regions, sorted_rectangles):
    new_sorted_rectangles = sorted_rectangles.copy()
    rectangles_2b_removed = []
    for i in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[i]
        next_rect = sorted_rectangles[i + 1]
        if isHook(current[0], next_rect[0], current[2], next_rect[2]):
            rectangles_2b_removed.append(next_rect)
            new_rect = (current[0], next_rect[1], current[2], current[3] + next_rect[3] + 5)
            new_sorted_rectangles[i] = new_rect

    if len(rectangles_2b_removed) > 0:
        for rect in rectangles_2b_removed:
            new_sorted_rectangles.remove(rect)

    return new_sorted_rectangles


def isHook(x0, x1, w0, w1):
    return x0 < x1 and x0 + w0 + 3 > x1 + w1


def split2hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    return h, s, v


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


# code for contrast enhancement taken from https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv

def contrastLab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def make_bin_and_rgb(img):
    img_contrast = contrastLab(img)
    img_rgb = img_contrast.copy()
    img_rgb = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB)
    s_channel = split2hsv(img_contrast)[1]
    img_enhanced_s_channel = cv2.equalizeHist(255 - s_channel)

    # plt.imshow(img_enhanced_s_channel, 'gray')
    # plt.show()

    img_bin = img_gs2bin(img_enhanced_s_channel)
    # plt.imshow(img_bin, 'gray')
    # plt.show()

    return img_bin, img_rgb


def prepare_img_for_roi(img):
    img_bin = make_bin_and_rgb(img)[0]
    img_prepared = morhpological_operations(img_bin)

    return img_prepared




# code for detecting skew taken from: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def detect_rotation_and_rotate(img):

    thresh = prepare_img_for_roi(img)

    coords = bb8.column_stack(bb8.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    #print(angle)
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    #image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_bgr = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #print(angle)
    #plt.imshow(rotated_bgr)
    #plt.show()
    return rotated_bgr



def prepare_train(paths):
    img_stacked = stack_photos(paths)
    # img_stacked = stack_photos(path)

    plt.imshow(img_stacked)
    plt.show()
    img_bin, img_rgb = make_bin_and_rgb(img_stacked)
    img_prepared = prepare_img_for_roi(img_stacked)

    selected_regions, letters, region_distances, new_sorted_rectangles = select_roi(img_rgb.copy(), img_prepared)

    plt.imshow(selected_regions)
    plt.show()
    cv2.imwrite('training.png',selected_regions)
    return letters


# code for most dominant color taken from: https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python

def get_most_dominant_color(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)
    a1D = bb8.ravel_multi_index(a2D.T, col_range)
    return bb8.unravel_index(bb8.bincount(a1D).argmax(), col_range)

def get_highest_peak(peaks,bins):
    max_peak = max(list(peaks))
    max_bin = 0
    for i in range(len(bins)):
        if peaks[i]==max_peak:
            max_bin = i
            return [max(list(peaks)),max_bin]

    return [max(list(peaks)),max(list(bins))]
        
# histogram explained: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
# opencv: https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
def histogram_img(img_rgb):
    max_peaks = []
    for i in range(0,3):
        value_array = img_rgb.mean(axis=i).flatten()
        hist, bin_edges = bb8.histogram(value_array, range(257))
        bin_edges=bin_edges[1:]
        max_peaks.append(get_highest_peak(hist,bin_edges))
        print("Hist")
        print(hist)
        print("Bin_edges")
        print(bin_edges)
        bin_edges=bin_edges[1:]
        #print(bin_edges)

        #peaks, _ = find_peaks(hist)
        print("Peaks")
        print(max_peaks)
        print("-------------------------------------------")
        #plt.plot(bin_edges[peaks], hist[peaks], ls='dotted')
        #plt.show()
        #peaks.append(find_peaks(hist, bin_edges))
    #print(peaks)

    print(get_most_dominant_color(img_rgb))


    return max_peaks


def prepare_test(path):
    img = cv2.imread(path)
    img = brightness_up(img)
    #print("h ",img.shape[0]," i w ",img.shape[1])
    img = resize_photo(img)
   # plt.imshow(img)
   # plt.show()

    img_rotated_bgr = detect_rotation_and_rotate(img)

    img_prepared_rotated = prepare_img_for_roi(img_rotated_bgr)

    img_orig, letters, region_distances, new_sorted_rectangles = select_roi(img_rotated_bgr.copy(), img_prepared_rotated)

    distances = bb8.array(region_distances).reshape(len(region_distances), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(distances)

   # plt.imshow(img_orig)
   # plt.show()
    print(len(letters))

    return letters, k_means




def prepare_test_rotation(path):
    img = cv2.imread(path)
    img = brightness_up(img)
    img_bin, img_rgb = make_bin_and_rgb(img)

    rotated = detect_rotation_and_rotate(img_bin)

    # rotated = cv2.resize(rotated,(8000, 800), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(rotated)
    # plt.show()
    # img_rgb = cv2.resize(img_rgb,(8000, 800), interpolation=cv2.INTER_NEAREST)

    img_prepared = prepare_img_for_roi(rotated)

    image_orig, letters, region_distances, new_sorted_rectangles = select_roi(rotated.copy(), img_prepared)
    # region = img_bin[y:y + h + 1, x:x + w + 1]
    # (x, y, w, h)])
    """
    i=1
    for rect in new_sorted_rectangles:
        cv2.imwrite('test_imwrite/letter'+str(i)+'.png',img[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1])
        i+=1

    """

    distances = bb8.array(region_distances).reshape(len(region_distances), 1)
    try:
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)
    except:
        return letters, None

    # plt.imshow(image_orig,'gray')
    # plt.show()
    print(len(letters))

    return letters, k_means

def testing():
    img = cv2.imread('dataset/validation/train86.png')
    img = brightness_up(img)
    lower_orange = bb8.array([0, 50, 50], bb8.uint8)
    upper_orange = bb8.array([255, 255, 255], bb8.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    test = cv2.inRange(hsv_img, lower_orange, upper_orange)
    plt.imshow(test)
    plt.show()
    return 255-test, img

def test_testing():

    img_bin,img_bgr = testing()

    img_prepared_rotated = morhpological_operations(img_bin)

    img_orig, letters, region_distances, new_sorted_rectangles = select_roi(img_bgr.copy(),
                                                                            img_prepared_rotated)

    plt.imshow(img_orig)
    plt.show()
    distances = bb8.array(region_distances).reshape(len(region_distances), 1)

    try:
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)
    except:
        return letters, None

    # plt.imshow(image_orig,'gray')
    # plt.show()
    print(len(letters))

    return letters, k_means

"""
    orange_paths = []
    orange_paths.append('dataset/validation/train95.png')
    orange_paths.append('dataset/validation/train91.png')
    orange_paths.append('dataset/validation/train86.png')
    orange_paths.append('dataset/validation/train82.png')
    orange_paths.append('dataset/validation/train77.png')
    orange_paths.append('dataset/validation/train75.png')
    orange_paths.append('dataset/validation/train69.png')
    orange_paths.append('dataset/validation/train57.png')
    orange_paths.append('dataset/validation/train56.png')
    dominant_colors=[]
   
     for path in orange_paths:
            img1 = cv2.imread(path)
            img1 = brightness_up(img1)
            dominant_colors.append([get_most_dominant_color(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)),path])
    
    
    
            #plt.imshow(mask1, 'gray')
            #plt.show()
    
    
            #print("-----------")
        #print(orange_paths)
        orange_first_try=[]
        for dominant in dominant_colors:
            colors = dominant[0]
            if colors[1]>140:
                #95,91,69
                img1 = cv2.imread(dominant[1])
                img1 = brightness_up(img1)
                lower_orange_mask = bb8.array([0, 50, 50], bb8.uint8)
                upper_orange_mask = bb8.array([28, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                frame_threshed = cv2.inRange(hsv_img, lower_orange_mask, upper_orange_mask)
                mask1 = 255 - frame_threshed
                lower_orange = bb8.array([0, 50, 50], bb8.uint8)
                upper_orange = bb8.array([19, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                frame_threshed = cv2.inRange(hsv_img, lower_orange, upper_orange)
                test = frame_threshed - (255 - mask1)
                orange_first_try.append([test,dominant[1]])
                #plt.imshow(test, 'gray')
                #plt.show()
            elif colors[1]<12:
                #86
                img1 = cv2.imread(dominant[1])
                img1 = brightness_up(img1)
                lower_orange_mask = bb8.array([0, 50, 50], bb8.uint8)
                upper_orange_mask = bb8.array([28, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower_orange_mask, upper_orange_mask)
                lower_orange = bb8.array([0, 50, 50], bb8.uint8)
                upper_orange = bb8.array([255, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                frame_threshed = cv2.inRange(hsv_img, lower_orange, upper_orange)
                plt.imshow(frame_threshed,'gray')
                plt.show()
                img_bin, img_rgb = make_bin_and_rgb(img1)
    
                #img_rgb_rotated = detect_rotation_and_rotate(frame_threshed, img_rgb)
                #plt.imshow(img_rgb_rotated)
                #plt.show()
                #img_prepared_rotated = prepare_img_for_roi(img_rgb_rotated)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                kernel_c = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    
                frame_threshed = cv2.morphologyEx(frame_threshed, cv2.MORPH_CLOSE, kernel)
    
                erosion = cv2.erode(frame_threshed, kernel_e, iterations=4)
                dilation = cv2.dilate(erosion, kernel_c, iterations=2)
                img_orig, letters, region_distances, new_sorted_rectangles = select_roi(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB).copy(),
                                                                                        dilation)
                print(len(letters))
                distances = bb8.array(region_distances).reshape(len(region_distances), 1)
                k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
                k_means.fit(distances)
    
                plt.imshow(img_orig)
                plt.show()
                print(len(letters))
    
                return letters, k_means
    
    
            elif colors[1]<100 and colors[1]>12:
                img1 = cv2.imread(dominant[1])
                img1 = brightness_up(img1)
                lower_orange_mask = bb8.array([5, 50, 50], bb8.uint8)
                upper_orange_mask = bb8.array([30, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower_orange_mask, upper_orange_mask)
                plt.imshow(255-mask,'gray')
                plt.show()
                lower_orange = bb8.array([0, 50, 50], bb8.uint8)
                upper_orange = bb8.array([50, 255, 255], bb8.uint8)
                hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                frame_threshed = cv2.inRange(hsv_img, lower_orange, upper_orange)
                #plt.imshow(frame_threshed,'gray')
                #plt.show()
    
    
    
        for dominant in dominant_colors:
            print(dominant)
    
    
    """

def detect_rotation_and_rotate_separation(img):
    img = brightness_up(img)
    plt.imshow(img)
    plt.show()
    # img_ctr = hi.contrastLab(img)
    # plt.imshow(img_ctr)
    plt.show()
    # grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    #
    # img_bin = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    # plt.imshow(img_bin,'gray')
    # plt.show()
    # img_res = hi.resize_photo(img_bin)
    # plt.imshow(img_res,'gray')
    # plt.show()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    img_t = 1 - img_gs
    thresh = cv2.adaptiveThreshold(img_t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 3)
    img_bin = 255 - thresh

    ret, img_bin = cv2.threshold(img_bin, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = bb8.column_stack(bb8.where(255-img_bin > 0))
    angle = cv2.minAreaRect(coords)[-1]

    #print(angle)
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    #image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_bgr = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #print(angle)
    #plt.imshow(rotated_bgr)
    #plt.show()
    return rotated_bgr

def select_roi_separation(image_orig, img_bin):
    a = 255-img_bin
    img, contours, hierarchy = cv2.findContours(a.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img,'gray')
    plt.show()

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            region = img_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    try:
        new_sorted_rectangles = deal_with_hooks_rectangles_train(sorted_regions,sorted_rectangles)
    except Exception as e:
        new_sorted_rectangles = sorted_rectangles

    new_sorted_regions = []
    img_before_hooks = image_orig.copy()

    for rectangle in sorted_rectangles:
        cv2.rectangle(img_before_hooks, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)
    cv2.imwrite('hukovi.png', img_before_hooks)

    for rectangle in new_sorted_rectangles:
        region = img_bin[rectangle[1]:rectangle[1] + rectangle[3] + 2, rectangle[0]:rectangle[0] + rectangle[2] + 2]
        new_sorted_regions.append(resize_region(region))
        cv2.rectangle(image_orig, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)

    region_distances = []

    for i in range(0, len(new_sorted_rectangles) - 1):
        current = new_sorted_rectangles[i]
        next_rect = new_sorted_rectangles[i + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    # plt.imshow(image_orig)
    # plt.show()
    cv2.imwrite('rects.png', image_orig)
    print("asda:", len(new_sorted_rectangles))

    return image_orig, new_sorted_regions, region_distances, new_sorted_rectangles


def deal_with_hooks_rectangles_train(sorted_regions, sorted_rectangles):
    new_sorted_rectangles = sorted_rectangles.copy()
    rectangles_2b_removed = []
    for i in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[i]
        next_rect = sorted_rectangles[i + 1]
        if isHook_train(current[0], next_rect[0], current[2], next_rect[2]):
            rectangles_2b_removed.append(next_rect)
            new_rect = (current[0], next_rect[1], current[2], current[3] + next_rect[3] + 5)
            new_sorted_rectangles[i] = new_rect

    if len(rectangles_2b_removed) > 0:
        for rect in rectangles_2b_removed:
            new_sorted_rectangles.remove(rect)

    return new_sorted_rectangles




def isHook_train(x0, x1, w0, w1):
    return x0 < x1 and x0 + w0 + 5 > x1 + w1


def get_letters_k_means_test(path):
    img = cv2.imread(path)
    print(iop.determine_color(path)+" "+path)
    dominant_color = iop.determine_color(path)

    # ako je plava dominantna boja, pozivace se modul dww
    if dominant_color == 'BLUE':
        letters, region_distances = dww.get_letters_and_distances_water(path)

        distances = bb8.array(region_distances).reshape(len(region_distances), 1)

        try:
            k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
            k_means.fit(distances)
        except:
            return letters, None

        print(len(letters))

        return letters, k_means
    #ako je crvena dominantna poziva se modul iz dwb
    elif dominant_color == 'RED':
        letters, region_distances = dwb.get_letters_and_distances_bricks(path)

        distances = bb8.array(region_distances).reshape(len(region_distances), 1)

        try:
            k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
            k_means.fit(distances)
        except:
            return letters, None

        print(len(letters))

        return letters, k_means
    else:
        img = iop.resize_photo(img)
        img = iop.brightness_up(img)

        rotated_img = detect_rotation_and_rotate_separation(img)

        img_t = iop.image_preparation_for_thresh(img)

        ret, img_bin = cv2.threshold(img_t, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # plt.imshow(img_bin)
        # plt.show()
        image_orig, letters, region_distances, new_sorted_rectangles = roi.select_roi(rotated_img, img_bin)
        # plt.imshow(image_orig)
        # plt.show()

        distances = bb8.array(region_distances).reshape(len(region_distances), 1)

        try:
            k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
            k_means.fit(distances)
        except:
            return letters, None
        print(len(letters))

        return letters, k_means

def get_letters_train(path):
    img = cv2.imread(path)
    img = iop.brightness_up(img)

    img_t = iop.image_preparation_for_thresh(img)

    ret, img_bin = cv2.threshold(img_t, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(img_bin)
    # plt.show()

    image_orig, letters, region_distances, new_sorted_rectangles = roi.select_roi(img, img_bin)
    # plt.imshow(image_orig)
    # plt.show()


    return letters

#test_testing()
def paper_test(path):
    img = cv2.imread(path)
    img = brightness_up(img)
    img = resize_photo(img)
    #img = contrastLab(img)
    #plt.imshow(img)
    #plt.show()
    print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    blank_image = bb8.zeros((h, w,3), bb8.uint8)
    blank_image[:, :] = (0, 0,3)
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(blank_image)
    # plt.show()
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = img_gs > 215
    img_bin = 255 - img_bin
    # plt.imshow(img_bin,'gray')
    # plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(bb8.uint8(img_bin), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    mask = cv2.dilate(mask, kernel, iterations=2)

    plt.imshow(mask,'gray')
    plt.show()

    mask = 255 - mask

    img_test = mask + blank_image
    plt.imshow(img_test)
    plt.show()


    #
    #plt.imshow(1-img_gs,'gray')
    #plt.show()
    # ret, mask = cv2.threshold(img_gs, 220, 255, cv2.THRESH_OTSU)
    # #
    # # img1 = cv2.bitwise_not(255-mask,img_gs)
    # mask = 255-mask
    # #plt.imshow(mask,'gray')
    # #plt.show()
    # src1_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # change mask to a 3 channel image
    # plt.imshow(src1_mask)
    # plt.show()
    #
    # dst = cv2.addWeighted(img, 0.9, src1_mask, 0.3,0)
    # plt.imshow(dst)
    # plt.show()
    #
    #
    # img_gs = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gs,'gray')
    # plt.show()
    # ret, mask = cv2.threshold(img_gs, 220, 255, cv2.THRESH_OTSU)
    # plt.imshow(mask)
    # plt.show()
    #
    # imgg = cv2.bitwise_not(img,img1)
    # plt.imshow(imgg)
    # plt.show()
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # mask = cv2.erode(bb8.uint8(mask),kernel,iterations=2)
    # plt.imshow(mask)
    # plt.show()
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(mask)
    # plt.show()
    #
    #h,s,v = split2hsv(img)
    #plt.imshow(v,'gray')
    #plt.show()


