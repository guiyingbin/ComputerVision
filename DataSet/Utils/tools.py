import numpy as np
import cv2


def resize_img(img, img_size):
    h, w = img_size
    img_h, img_w, img_c = img.shape
    new_img = np.ones((h, w, img_c), dtype=np.uint8)*127
    ratio = min(w/img_w, h/img_h)
    n_h, n_w = int(np.round(h*ratio, 0)), int(np.round(w*ratio, 0))
    d_h, d_w = max(h-n_h, 0)//2, max(w-n_w, 0)//2
    img = cv2.resize(img, (n_w, n_h))
    new_img[d_h:d_h+n_h, d_w:d_w+n_w, :] = img
    return new_img


if __name__ == "__main__":
    img = cv2.imread(r"D:\ocr_data\train_images\0.jpg")
    new_img = resize_img(img, (32, 240))
    cv2.imshow("img", new_img)
    cv2.waitKey()
    print(new_img.shape)

