import cv2 as cv
import numpy as np

# Load the image
img_path = '../upload/sc.jpg'
img = cv.imread(img_path, cv.IMREAD_COLOR)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv.GaussianBlur(gray, (9, 9), 2)

# # Detect circles using HoughCircles
# dart = cv.HoughCircles(
#     gray,
#     cv.HOUGH_GRADIENT,
#     dp=1,
#     minDist=gray.shape[0] / 2,
#     param1=100,
#     param2=30,
#     minRadius=100,  # Set based on expected dartboard size
#     maxRadius=0     # 0 means no upper limit; adjust as needed
# )




# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     # Assume the first detected circle is the dartboard's outer edge
#     x, y, r = circles[0][0]

#     # Draw the detected circle for visualization
#     cv.circle(img, (x, y), r, (0, 255, 0), 2)
#     cv.circle(img, (x, y), 2, (0, 0, 255), 3)  # Center point

#     # # Crop the dartboard
#     # cropped = img[y-r:y+r, x-r:x+r]

#     # # Optional: Resize the cropped image
#     # resized = cv.resize(cropped, (512, 512), interpolation=cv.INTER_AREA)

#     # Display the cropped and resized image
#     cv.imshow('Cropped Dartboard', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def bulls_eye_main():
    """
    Detects red circles in an image and draws them.
    """
    default_filename = 'board.jpg'

    # Load image
    img_path = '../upload/sc.jpg'
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    if img is None:
        print(f'Error opening image!')
        return -1



    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray,17)
    
    dim = gray.shape[0]

    # # Detecting dart board
    # dart = cv.HoughCircles(
    #     gray,
    #     cv.HOUGH_GRADIENT,
    #     dp=1,
    #     minDist=gray.shape[0] / 2,
    #     param1=100,
    #     param2=30,
    #     minRadius=int(dim/3.5),  # Set based on expected dartboard size
    #     maxRadius=int(dim/2)    # 0 means no upper limit; adjust as needed
    # )





    # Controlling distance between circles in midDist 
    minrad_eye = int(dim/4.70)
    maxrad_eye = int(minrad*1.68)

    # Detects circles over the bulls eye
    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=150,
        param2=50,
        minRadius=minrad_eye,
        maxRadius=maxrad_eye
    )

    if circles is None:
        print("Can't find bulls eye")
        return -1
    
    # Drawing circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0,:]:
            center = (circle[0], circle[1])
            # Draw the circle center
            cv.circle(img, center, 1, (255,255,255), 4)
            # Draw the circle outline
            radius = circle[2]
            cv.circle(img, center, radius, (255,255,255), 1)

    # Drawing center of the bulls eye
    if circles is not None:

        circles = np.uint16(np.around(circles))

        # Extract circle centers
        centers = [(circle[0], circle[1]) for circle in circles[0, :]]

        # Calculate the mean of the circle centers
        mean_x = int(np.median([c[0] for c in centers]))
        mean_y = int(np.median([c[1] for c in centers]))
        mean_center = (mean_x, mean_y)

        # Draw the mean center
        cv.circle(img, mean_center, 3, (255, 0, 0), -1) 
        cv.circle(gray, mean_center, 3, (255, 0, 0), -1) 
    
    # print(circles)

    # Display image
    cv.imshow('Circles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # # show gray image
    cv.imshow('Gray', gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

bulls_eye_main()

# if __name__ == '__main__':
#     bulls_eye_main()


