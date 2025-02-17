import cv2
import numpy as np

IMG_NO = 2

# Image folders
input_folder = "InputImages/"
output_folder = "OutputImages/"

# Images
if IMG_NO == 1:
    imgs = ["board_1.jpg", "board_2.jpg"]
else:
    imgs = ["flowers_1.jpg", "flowers_2.jpg"]

# Load the images
img1 = cv2.imread(input_folder+imgs[0])  # Query image
img2 = cv2.imread(input_folder+imgs[1])  # Train image


# Detect keypoints and descriptors using ORB
descriptor = cv2.SIFT_create()
kp1, des1 = descriptor.detectAndCompute(img1, None)
kp2, des2 = descriptor.detectAndCompute(img2, None)


get_response = lambda x: abs(x[0].response) # Response of a keypoint

# Sort keypoints in the increasing order of response
sorted_out1 = sorted(list(zip(kp1, des1)), key=get_response, reverse=True)
sorted_out2 = sorted(list(zip(kp2, des2)), key=get_response, reverse=True)


n_points = 10000 # No. of best keyponts to be selected

# Selecting best keypoints and their descriptors
top_kp1 = np.array([i for i, _ in sorted_out1[:n_points]])
top_des1 = np.float32([j for _, j in sorted_out1[:n_points]])

top_kp2 = np.array([i for i, _ in sorted_out2[:n_points]])
top_des2 = np.float32([j for _, j in sorted_out2[:n_points]])

# Matching keypoints
matcher = cv2.BFMatcher()
rawMatches = matcher.knnMatch(top_des1, top_des2, k=2)
good_matches = [] # Filtered matches
for m_n in rawMatches:
    if len(m_n) == 2:
        m, n = m_n
        if m.distance < n.distance * 0.75:
            good_matches.append(m)


# Construct the two sets of points
ptsA = np.float32([top_kp1[m.queryIdx].pt for m in good_matches])
ptsB = np.float32([top_kp2[m.trainIdx].pt for m in good_matches])

# Compute the homography between the two sets of points
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

# Warping second image to the dimension of first image
result = cv2.warpPerspective(img2, np.linalg.inv(H),
			(img1.shape[1] + img2.shape[1], img2.shape[0]))
# Overlapping the first image
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# Drawing matches
img1_keypoints = cv2.drawKeypoints(img1, top_kp1, 0, (255, 0, 0))
img2_keypoints = cv2.drawKeypoints(img2, top_kp2, 0, (255, 0, 0))
img_matches = cv2.drawMatches(img1, top_kp1, img2, top_kp2, good_matches, None, matchColor=(255, 0, 0))

# Display Images
# cv2.imshow("Image", img_matches)
cv2.imshow("Panoroma", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Images
cv2.imwrite(output_folder+"matches.jpg", img_matches)
cv2.imwrite(output_folder+"final_panaroma.jpg", result)
cv2.imwrite(output_folder+"img1_keypoints.jpg", img1_keypoints)
cv2.imwrite(output_folder+"img2_keypoints.jpg", img2_keypoints)