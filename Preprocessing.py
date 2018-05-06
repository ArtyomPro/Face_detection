import dlib
import cv2
from scipy.spatial import distance

class Preprocessing():
    def __init__(self, sp, facerec, detector, img):
        self.sp = sp
        self.facerec = facerec
        self.detector = detector
        self.img = img
        self.predict_vector = None

    def preprocessing_image(self,filename):
        self.img = cv2.imread(filename,cv2.COLOR_BAYER_BG2BGR)
        dets = self.detector( self.img, 1)
        for k, d in enumerate(dets):
            self.shape = self.sp( self.img, d)
        self.predict_vector = self.facerec.compute_face_descriptor( self.img, self.shape)

    def preprocessing_image(self):
        dets = self.detector(self.img, 1)
        for k, d in enumerate(dets):
            self.shape = self.sp(self.img, d)
        self.predict_vector = self.facerec.compute_face_descriptor(self.img, self.shape)

    def get_predict_vector(self):
        return self.predict_vector

    def compare(self,predict_vector2):
        dist = distance.euclidean(self.predict_vector, predict_vector2)
        return dist

    def compare_percent(self,predict_vector2):
        dist = distance.euclidean(self.predict_vector, predict_vector2)
        return (1-dist)*100