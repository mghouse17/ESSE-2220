"""
Lab 7: Feature Detection in Space Images (SIFT & ORB)
-----------------------------------------------------
Instructions:
1. Fill in the missing parts using the OpenCV documentation.
2. You must look up how to use the functions marked with TODO.
3. Use the official docs: https://docs.opencv.org/
   or Google search like: cv2.SIFT_create site:docs.opencv.org
4. Save all your output images using cv2.imwrite().
"""

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# --- Step 1: Read the image ---
img = cv2.imread('space.jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('space.jpg', cv2.IMREAD_COLOR)

if img is None or img_color is None:
    raise FileNotFoundError(f"space.jpg was not found.")


print("Shape of grayscale image:", img.shape)
print("Shape of color image:", img_color.shape)

# Grayscale images are 2D arrays: (height, width).
# Color images are 3D arrays: (height, width, channels).
# For a color image, the 3 channels are typically B, G, and R.


# --- Step 2: Create SIFT detector ---
sift = cv2.SIFT_create()

# --- Step 3: Detect keypoints and descriptors ---
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)

# --- Step 4: Print some info ---
print("Number of keypoints (SIFT):", len(keypoints_sift))
print("Descriptor shape (SIFT):", descriptors_sift.shape)

# --- Step 5: Draw SIFT keypoints ---
img_sift = cv2.drawKeypoints(img, keypoints_sift, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(os.path.join('sift_keypoints.jpg'), img_sift)

# --- Step 6: Create ORB detector ---
# OpenCV docs show ORB_create accepts parameters such as nfeatures.
# Increasing nfeatures usually increases the number of detected keypoints.
orb = cv2.ORB_create(nfeatures=500)

# --- Step 7: Detect and compute with ORB ---
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)

print("Number of keypoints (ORB):", len(keypoints_orb))
print("Descriptor shape (ORB):", descriptors_orb.shape)

# --- Step 8: Draw ORB keypoints ---
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
cv2.imwrite(os.path.join('orb_keypoints.jpg'), img_orb)

# --- Step 9: Top 50 strongest keypoints ---
sorted_kp_sift = sorted(keypoints_sift, key=lambda kp: kp.response, reverse=True)[:50]
sorted_kp_orb = sorted(keypoints_orb, key=lambda kp: kp.response, reverse=True)[:50]

img_sift_top50 = cv2.drawKeypoints(img, sorted_kp_sift,None, color=(255, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
img_orb_top50 = cv2.drawKeypoints(img,sorted_kp_orb,None,color=(0, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

cv2.imwrite(os.path.join('sift_top50.jpg'), img_sift_top50)
cv2.imwrite(os.path.join('orb_top50.jpg'), img_orb_top50)

# --- Step 10: Print and explain descriptors ---
print("First two SIFT descriptors:\n", descriptors_sift[:2])
print("First two ORB descriptors:\n", descriptors_orb[:2])

# Descriptors are local feature vectors. SIFT produces 128-element float vectors,
# while ORB produces binary descriptors that encode local intensity comparisons.

# --- Step 11: Visualize one descriptor vector ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sift_vector = descriptors_sift[0]
orb_vector = descriptors_orb[0]

axes[0].bar(np.arange(len(sift_vector)), sift_vector, color='blue')
axes[0].set_title('SIFT Descriptor Vector (First Keypoint)')
axes[0].set_xlabel('Descriptor Element')
axes[0].set_ylabel('Value')

axes[1].bar(np.arange(len(orb_vector)), orb_vector, color='green')
axes[1].set_title('ORB Descriptor Vector (First Keypoint)')
axes[1].set_xlabel('Descriptor Element')
axes[1].set_ylabel('Value')

plt.tight_layout()
fig.savefig(os.path.join('sift_descriptor_plot.jpg'))
plt.close(fig)

