from rembg import remove
import numpy as np
import cv2
from skimage.measure import label
from skimage.color import label2rgb
from matplotlib import pyplot as plt

img = cv2.imread('volume/bocni.jpg')
img_out = remove(img)
img_out = np.array(img_out)
mask = np.moveaxis(img_out, -1, 0)
mask = mask[3]
mask.setflags(write=1)
mask[mask < 100] = 0
mask[mask >= 100] = 1
label_image = label(mask)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('volume/bocni_removed.jpg', masked)
