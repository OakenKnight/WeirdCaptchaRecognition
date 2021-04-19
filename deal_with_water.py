import cv2
import matplotlib.pylab as plt
import numpy as bb8
def brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] += 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img
def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def resize_photo(img):
    old_height = img.shape[0]
    old_width = img.shape[1]
    #print(old_width, old_height)
    aspectRatio = (old_height / old_width)
    #print(aspectRatio)
    scale_percent = 150
    newWidth = int(img.shape[1] * scale_percent / 100)
    newHeight = int(img.shape[0] * scale_percent / 100)
    #newHeight = (new_width * aspectRatio)

    #newWidth = (new_width / aspectRatio)
    resized_img = cv2.resize(img, (int(newWidth), int(newHeight)), interpolation=cv2.INTER_NEAREST)
    return resized_img

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
        if area >= 100:
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
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 3)

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

def contrastLAB(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def detect_rotation_and_rotate_separation_water(img_bin, img):

    a = 255 - img_bin
    coords = bb8.column_stack(bb8.where(a> 0))
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

def separate_water(img_bgr):

    img = contrastLAB(img_bgr)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # plt.imshow(img_rgb)
    # plt.show()
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_blue = bb8.array([110, 50, 50])
    upper_blue = bb8.array([130, 255, 255])

    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    blue_lower = bb8.array([70, 10, 20])
    blue_upper = bb8.array([110, 255, 255])
    blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
    blue_mask = 255 - blue_mask
    # plt.imshow(blue_mask, 'gray')
    # plt.show()
    res = cv2.bitwise_and(img_rgb, img_rgb, mask=blue_mask)
    plt.imshow(res)
    plt.show()
    s = 1 - cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    # plt.imshow(s, 'gray')
    # plt.show()
    ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(thresh, 'gray')
    # plt.show()

    return thresh
def get_letters_and_distances_water(path):
    img_bgr = cv2.imread(path)
    img_bgr = brightness_up(img_bgr)
    img_bgr = resize_photo(img_bgr)

    img_bin_before_rotation = separate_water(img_bgr)

    img_bgr_rotated = detect_rotation_and_rotate_separation_water(img_bin_before_rotation,img_bgr)
    img_rgb_rotated = cv2.cvtColor(img_bgr_rotated,cv2.COLOR_BGR2RGB)
    img_bin_after_rotation = separate_water(img_bgr_rotated)
    image_orig, letters, region_distances, new_sorted_rectangles = select_roi_separation(img_rgb_rotated, img_bin_after_rotation)
    return letters, region_distances

