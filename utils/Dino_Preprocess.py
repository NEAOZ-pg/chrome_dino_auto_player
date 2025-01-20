import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy

class Dino_Preprocess():
 
    @staticmethod
    def crop_blue_frame(image, lower_blue_hsv=(105, 5, 120), upper_blue_hsv=(120,255,255)): 

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for the blue color
        lower_blue = lower_blue_hsv
        upper_blue = upper_blue_hsv

        # Create a mask for the blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Use morphological operations to enhance the border
        mask = cv2.dilate(mask, numpy.ones((3, 3), numpy.uint8), iterations=2)
        mask = cv2.erode(mask, numpy.ones((3, 3), numpy.uint8), iterations=2)

        # Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours are found
        if contours:
            # Find the largest contour, which should be the blue border
            contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to get a polygon with fewer vertices
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the approximated contour has 4 corners
            if len(approx) == 4:
                # Sort the points in a consistent order: top-left, top-right, bottom-right, bottom-left
                corners = sorted([tuple(point[0]) for point in approx], key=lambda x: (x[1], x[0]))

                # Rearrange to ensure the order is: top-left, top-right, bottom-right, bottom-left
                top_left, top_right = sorted(corners[:2], key=lambda x: x[0])
                bottom_left, bottom_right = sorted(corners[2:], key=lambda x: x[0])

                # Calculate the offset to move the corners slightly inward
                offset = numpy.sqrt(numpy.sum((numpy.array(top_right) - numpy.array(top_left)) ** 2)) / 200  # Adjust as needed to get the inner edge

                # Calculate inner corners by moving each corner inward
                inner_corners = [
                    (top_left[0] + offset, top_left[1] + offset),
                    (top_right[0] - offset, top_right[1] + offset),
                    (bottom_right[0] - offset, bottom_right[1] - offset),
                    (bottom_left[0] + offset, bottom_left[1] - offset),
                ]

                # Define the width and height for the perspective transform output
                width = max(
                    numpy.linalg.norm(numpy.array(inner_corners[0]) - numpy.array(inner_corners[1])),
                    numpy.linalg.norm(numpy.array(inner_corners[2]) - numpy.array(inner_corners[3]))
                )
                height = max(
                    numpy.linalg.norm(numpy.array(inner_corners[0]) - numpy.array(inner_corners[3])),
                    numpy.linalg.norm(numpy.array(inner_corners[1]) - numpy.array(inner_corners[2]))
                )

                # Define destination points for perspective transform
                dst_pts = numpy.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")

                # Perform perspective transformation
                src_pts = numpy.array(inner_corners, dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image, M, (int(width), int(height)))
                return warped
        else:
            return None

    @staticmethod    
    def convert_Gray2binary(image, lower_threshold=160):
        # change the second parameters below to adjust the threshold
        _, binary_image = cv2.threshold(image, lower_threshold, 255,cv2.THRESH_BINARY)
        return binary_image

    @staticmethod    
    def crop_binaryframe_dino(binary_image, dino_size=320, bias=15):
        dino_image = binary_image[:, bias :dino_size + bias] 
        return dino_image

    @staticmethod    
    def crop_binaryframe_dino_number(binary_image):
        dino_image = binary_image[:, :320] 
        number_image = binary_image[:32, 448:]
        return dino_image, number_image

    @staticmethod    
    def resize_and_gray_frame(img, resize_shape=(512, 128)):
        # 512, 128
        resized_image = cv2.resize(img, resize_shape)
        gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2GRAY)
        return gray_img

    @staticmethod
    def flip_image(image):
        mean_num = numpy.mean(image)
        if mean_num < 126:
            inverted_image = 255 - image
        elif mean_num > 128: 
            inverted_image = image
        else:
            inverted_image = None
        return inverted_image

