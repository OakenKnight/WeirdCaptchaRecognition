import cv2
import matplotlib.pyplot as plt

import image_operations as iop


def select_roi(image_orig, img_bin):
    a = 255 - img_bin
    img, contours, hierarchy = cv2.findContours(a.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img, 'gray')
    plt.show()

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            region = img_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([iop.resize_region(region), (x, y, w, h)])

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
        new_sorted_regions.append(iop.resize_region(region))
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
    return x0 < x1 and x0 + w0 + 5 > x1 + w1
