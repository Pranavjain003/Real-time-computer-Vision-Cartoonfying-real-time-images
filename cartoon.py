import cv2
import numpy as np

def cartoonify_image(image, sharpness_factor=1.5, bilateral_filter_d=9, bilateral_filter_sigma_color=300, bilateral_filter_sigma_space=300):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, d=bilateral_filter_d, sigmaColor=bilateral_filter_sigma_color, sigmaSpace=bilateral_filter_sigma_space)

    # Apply an edge-preserving filter to highlight edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)

    # Convert the image to the LAB color space
    color = cv2.bilateralFilter(image, d=bilateral_filter_d, sigmaColor=bilateral_filter_sigma_color, sigmaSpace=bilateral_filter_sigma_space)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)

    # Convert the LAB image back to BGR color space
    cartoon = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Combine the edges and color images
    cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)

    # Apply a sharpening filter
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, sharpness_factor + 8, -1],
                                  [-1, -1, -1]])
    cartoon = cv2.filter2D(cartoon, -1, kernel_sharpening)

    return cartoon

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Apply cartoonify function to the frame with a more realistic color tone, sharpened
        cartoon_realistic = cartoonify_image(frame, sharpness_factor=1.5, bilateral_filter_d=9, bilateral_filter_sigma_color=300, bilateral_filter_sigma_space=300)

        # Display the original, and cartoon with a more realistic color tone
        cv2.imshow("Original", frame)
        cv2.imshow("Cartoon (Realistic)", cartoon_realistic)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
