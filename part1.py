import cv2
import numpy as np

# Image folders
input_folder = "InputImages/"
output_folder = "OutputImages/"

# Load Image
img = cv2.imread(input_folder+"indian_coins.png", cv2.IMREAD_COLOR)

# Gaussian Blur
blur_img = cv2.GaussianBlur(img, (9, 9), 0)

# Convert to Greyscale
grey_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

# Define parameters 
t_lower = 50 # Lower Threshold 
t_upper = 100 # Upper threshold 
  
# Applying the Canny Edge filter 
edges = cv2.Canny(grey_img, t_lower, t_upper)

#Threshold Processing
ret, bin_img = cv2.threshold(grey_img,
                             0, 255, 
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

coin_number = 0 # No. of coins
segmented_images = [] # Segmented image for each coin
for contour in contours:
    # Selecting contours with area > Min area 
    if cv2.contourArea(contour) > 150:
        # Finding radius
        (x, y), radius = cv2.minEnclosingCircle(contour) 
        center = (int(x), int(y))
        radius = int(radius)

        black_image = np.zeros(img.shape, dtype = np.uint8) # Black image
        cv2.circle(black_image, center, radius, (255, 255, 255), -1) # Circle filled with white
        segmented_images.append(black_image & img) # Segmented coin

        # Draw bounding circle in the original image
        cv2.circle(img, center, radius, (0,0,255), 2) 
        coin_number += 1

print(f"{coin_number} coins are detected in the image.")

# Display Image
# res = np.hstack((grey_img, edges, bin_img))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Images
cv2.imwrite(output_folder+"indian_coins_greyscale.jpg", grey_img)
cv2.imwrite(output_folder+"indian_coins_edges.jpg", edges)
cv2.imwrite(output_folder+"indian_coins_detected.jpg", img)
cv2.imwrite(output_folder+"indian_coins_blurred.jpg", blur_img)
cv2.imwrite(output_folder+"indian_coins_thresholded.jpg", bin_img)

for i, image in enumerate(segmented_images):
    cv2.imwrite(output_folder+f"segmented_coin_{i}.jpg", image)