import cv2
import numpy as np

def select_roi(image):
    cv2.imshow('Select ROI', image)
    roi = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Select ROI')  
    return roi

def local_color_transfer(source_img, target_img, roi, mean_scale=1.0, std_scale=1.0):
    if source_img is None or target_img is None:
        print("Error: One or both of the images are not loaded.")
        return None

    x, y, w, h = roi
    source_roi = source_img[y:y+h, x:x+w]

    source_lab = cv2.cvtColor(source_roi, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

    mean_source, std_source = cv2.meanStdDev(source_lab)
    mean_target, std_target = cv2.meanStdDev(target_lab)

    scaled_mean_source = mean_source * mean_scale
    scaled_std_source = std_source * std_scale

    std_target_safe = np.where(std_target == 0, 1, std_target)

    result_lab = cv2.merge([
        (target_lab[:, :, 0] - mean_target[0]) * (scaled_std_source[0] / std_target_safe[0]) + scaled_mean_source[0],
        (target_lab[:, :, 1] - mean_target[1]) * (scaled_std_source[1] / std_target_safe[1]) + scaled_mean_source[1],
        (target_lab[:, :, 2] - mean_target[2]) * (scaled_std_source[2] / std_target_safe[2]) + scaled_mean_source[2]
    ])

    result_bgr = cv2.cvtColor(np.uint8(result_lab), cv2.COLOR_LAB2BGR)
    return result_bgr

source_img = cv2.imread('2.jpg')
target_img = cv2.imread('window1.jpg')

if source_img is None or target_img is None:
    print("Error: One or both of the images could not be loaded.")
    exit()

roi = select_roi(source_img)

mean_scale = 1.0
std_scale = 1.0

cv2.namedWindow('Parameter Adjustment')
cv2.createTrackbar('Mean Scale', 'Parameter Adjustment', int(mean_scale * 100), 200, lambda x: None)
cv2.createTrackbar('Std Scale', 'Parameter Adjustment', int(std_scale * 100), 200, lambda x: None)

while True:
    mean_scale = cv2.getTrackbarPos('Mean Scale', 'Parameter Adjustment') / 100.0
    std_scale = cv2.getTrackbarPos('Std Scale', 'Parameter Adjustment') / 100.0

    output_img = local_color_transfer(source_img, target_img, roi, mean_scale, std_scale)

    cv2.imshow('Source Image', source_img)
    cv2.imshow('Target Image', target_img)
    cv2.imshow('Output Image', output_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()