import cv2
import os
from model.grabcut import GrabCut

if __name__ == '__main__':

    # mkdir
    dname = 'result'
    if not os.path.isdir(dname):
        os.mkdir('result')

    # read image
    image = cv2.imread('camellia.png').astype('float')

    # grabcut
    gc = GrabCut(3, 50.0)
    tl, br = (0, 40), (106, 150)
    gc.segment(image, tl, br, 20, dname)
