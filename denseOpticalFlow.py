import numpy as np
import cv2
import time

def draw_red_circles_with_opacity(img, points, opacity=0.6, radius=5):
    """
    Draw red circles on the image at the specified points with given opacity.
    """
    overlay = img.copy()
    
    # Draw circles on the overlay
    for (x, y) in points:
        cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1)  # Red color (BGR)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    
    return img

def draw_hsv_on_gray(img, flow, threshold=1.5):
    h, w = img.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    # Calculate magnitude and angle
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    # Create HSV image
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 16, 255)
    
    # Convert to BGR
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert grayscale to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Blend images
    blended = cv2.addWeighted(img_bgr, 0.5, bgr_flow, 0.7, 0)

    return blended

def dense_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    max_speed = 0
    max_frame = None
    max_speed_points = []

    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Start time to calculate FPS
        start = time.time()

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        # Calculate speed (magnitude)
        fx, fy = flow[:,:,0], flow[:,:,1]
        speed = np.sqrt(fx**2 + fy**2)

        # Get maximum speed and corresponding points
        max_current_speed = np.max(speed)
        if max_current_speed > max_speed:
            max_speed = max_current_speed
            max_frame = img.copy()  # Store the frame with max speed
            max_speed_points = np.argwhere(speed > 1.5)  # Points where speed exceeds threshold

        # End time
        end = time.time()
        fps = 1 / (end - start)

        # Overlay HSV flow values without rectangles
        hsv_on_gray = draw_hsv_on_gray(gray, flow)

        # Draw red circles on the max speed frame
        if max_frame is not None:
            final_frame = draw_red_circles_with_opacity(max_frame, max_speed_points, opacity=0.6)
        
        # Display the result
        cv2.imshow('Optical Flow', hsv_on_gray)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Show the frame with the highest speed after the video ends
    if max_frame is not None:
        final_frame = draw_red_circles_with_opacity(max_frame, max_speed_points, opacity=0.6)
        cv2.imshow('Frame with Maximum Speed', final_frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
