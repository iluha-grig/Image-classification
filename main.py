import numpy as np
import cv2 as cv
from skimage import morphology


def show(img):
    cv.imshow('img', img)
    cv.waitKey(0)


def check_in_tri(point, a, b, c):
    x = (a[0] - point[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (a[1] - point[1])
    y = (b[0] - point[0]) * (c[1] - b[1]) - (c[0] - b[0]) * (b[1] - point[1])
    z = (c[0] - point[0]) * (a[1] - c[1]) - (a[0] - c[0]) * (c[1] - point[1])
    return np.sign(x) == np.sign(y) and np.sign(x) == np.sign(z)


def detect(img_file_name):
    img = cv.imread(img_file_name)
    # img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    edges = cv.Canny(img, 200, 250, 3)
    # show(edges)
    edges = (morphology.remove_small_holes(edges) * 255).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    triangle = 0
    all_centers = []
    all_triangles = []
    eps_centroids = 20
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        vertices = cv.approxPolyDP(contour, 0.05 * perimeter, True)
        area = cv.contourArea(vertices)
        if len(vertices) == 3 and area > 1500:
            moment = cv.moments(vertices)
            centroid = np.array([moment['m10'] / moment['m00'], moment['m01'] / moment['m00']])
            for center in all_centers:
                if np.sqrt(np.sum((center - centroid) ** 2)) < eps_centroids:
                    break
            else:
                all_centers.append(centroid)
                all_triangles.append(vertices)
                triangle += 1
                # cv.drawContours(img, [vertices], 0, (255, 255, 255), 2, cv.LINE_AA)

    edges = cv.Canny(img, 100, 200, 3)
    # show(edges)
    edges = (morphology.remove_small_holes(edges) * 255).astype(np.uint8)
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    # edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    # edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    # show(edges)
    contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(triangle)
    for centroid, elem in zip(all_centers, all_triangles):
        img_copy = img.copy()
        cv.drawContours(img_copy, [elem], 0, (255, 255, 255), 1, cv.LINE_AA)
        print(centroid[0], centroid[1], sep=', ', end='; ')
        dots = [0, 0, 0]
        h1, h2, h3 = (elem[0].ravel() + elem[1].ravel()) / 2, \
                     (elem[1].ravel() + elem[2].ravel()) / 2, \
                     (elem[2].ravel() + elem[0].ravel()) / 2
        for contour in contours:
            moment = cv.moments(contour)
            if moment['m00'] == 0:
                continue
            point_center = np.array([moment['m10'] / moment['m00'], moment['m01'] / moment['m00']])
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            (x, y), radius = cv.minEnclosingCircle(contour)
            if area < 50 and perimeter < 60 and radius < 5.5 and\
                    check_in_tri(point_center, elem[0].ravel(), elem[1].ravel(), elem[2].ravel()):
                cv.drawContours(img_copy, [contour], 0, (255, 255, 255), 1, cv.LINE_AA)
                if check_in_tri(point_center, h1, h2, centroid) or check_in_tri(point_center, h1, h2, elem[1].ravel()):
                    dots[0] += 1
                elif check_in_tri(point_center, h2, h3, centroid) or check_in_tri(point_center, h2, h3, elem[2].ravel()):
                    dots[1] += 1
                else:
                    dots[2] += 1
        print(dots[0], dots[1], dots[2], sep=', ')
        show(img_copy)

    return img


if __name__ == '__main__':
    img = detect('Pict_4_1.bmp')
    # show(img)
