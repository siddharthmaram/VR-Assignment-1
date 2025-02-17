part1.py - Segmentation of coins
part2.py - Panoroma creation

InputImages folder
- "indian_coins.png" is the image used for segmenting coins.
- "flowers_1.jpg" and "flowers_2.jpg" are used to create the panaroma. 
- You can change IMG_NO variable at the top of "part2.py" to 1 to create panaroma of images "board_1.jpg" and "board_2.jpg".

OutputImages folder
- "segmented_coins_{coin_no}.jpg" are the segmented outputs of each coin in the image.
- "indian_coins_blurred.jpg" is the output after applying Gaussian Blur.
- "indian_coins_edges.jpg" shows canny edges.
- "indian_coins_greyscale.jpg" is the greyscale image. 
- "indian_coins_thresholded.jpg" is the output after applying Otsu's thresholding.
(Its blurry because I converted to greyscale after applying Gaussian Blur).
- "indian_coins_detected.jpg" shows bounding circles aroung each coin.
- "img1_keypoints.jpg" and "img2_keypoints.jpg" show the detected keypoints in the respective images.
- "matches.jpg" shows the matching keypoints in both images
- "final_panoroma.jpg" is the panoroma created from the two images

To run
- pip install -r requirements.txt
- python part1.py
- python part2.py

NOTE:
- "part2.py" takes a few seconds to run.
