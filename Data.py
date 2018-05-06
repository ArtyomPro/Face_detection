import dlib
import cv2
import numpy as np
from Preprocessing import Preprocessing as pr

def get_target():
    target = []
    m = pr()
    for i in range(14):
        for j in range (14):
            if (i == j):
                continue
            f1 = str(i+1)
            f2 = str(j+1)
            if (i<9):
                f1 = '0'+str(i+1)
            if (j<9):
                f2 = '0'+str(j+1)
            p1 = cv2.imread('faces/2-'+f1+'.jpg',cv2.COLOR_GRAY2BGR)
            p2 = cv2.imread('faces/2-'+f2+'.jpg',cv2.COLOR_GRAY2BGR)
            p1 = m.preprocessing_image2(p1)
            p2 = m.preprocessing_image2(p2)
            target.append(m.compare(p1,p2))
    return target

def get_imagine():
    imagine = []
    m = pr()
    for i in range(14):
        for j in range (14):
            if (i==j):
                continue
            f1 = str(i+1)
            f2 = str(j+1)
            if (i<9):
                f1 = '0'+str(i+1)
            if (j<9):
                f2 = '0'+str(j+1)
            p1 = cv2.imread('faces/2-'+f1+'.jpg',cv2.COLOR_GRAY2BGR)
            p2 = cv2.imread('faces/3-'+f2+'.jpg',cv2.COLOR_GRAY2BGR)
            p1 = m.preprocessing_image2(p1)
            p2 = m.preprocessing_image2(p2)
            imagine.append(m.compare(p1,p2))
    return imagine

def save_file(filename,mass):
    f = open(filename, 'w')
    for i in mass:
        f.write(' '.join(i))
    f.close()

def get_data_from_file(filename):
    f = open(filename, 'r')
    mass = f.readline().split()
    mass = list(map(float,mass))
    f.close()
    return mass