import cv2
from model.grabcut import GrabCut

if __name__ == '__main__':

    # read image
    img = cv2.imread('camellia.png').astype('float')
    gc = GrabCut(3, 50.0)
    gc.segment(img, [30, 0, 150, 106], 20)
