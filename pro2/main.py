# STEP 1 : Import modules 
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : Create inference instance
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3 : Load data
img1 = cv2.imread('sectry1.jpg')
img2 = cv2.imread('sectry2.jpg')
# provide sample images

# STEP 4 : inference
face1 = app.get(img1)
assert len(face1)==1
face2 = app.get(img2)
assert len(face2)==1

# STEP 5 : post processing
# 5-1 : draw face bounding box

# 5-2 : calculate face similarity
# then print all-to-all face similarity

feat1 = np.array(face1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(face2[0].normed_embedding, dtype=np.float32)

sims = np.dot(feat1, feat2.T)
print(sims)



