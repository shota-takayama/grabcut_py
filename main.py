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
    fore = [30, 0, 150, 106]
    T = 20
    gc.segment(image, fore, T, dname)
