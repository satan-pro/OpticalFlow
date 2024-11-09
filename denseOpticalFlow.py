import numpy as np
import cv2

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

def calculate_velocity_direction(flow, cumulative_data):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    velocity = np.sqrt(fx**2 + fy**2)
    direction = np.arctan2(fy, fx)
    cumulative_data['total_velocity'] += np.sum(velocity)
    cumulative_data['total_flow_vectors'] += velocity.size
    cumulative_data['sin_sum'] += np.sum(np.sin(direction))
    cumulative_data['cos_sum'] += np.sum(np.cos(direction))
    return cumulative_data

def compute_average_velocity_direction(cumulative_data):
    if cumulative_data['total_flow_vectors'] > 0:
        average_velocity = cumulative_data['total_velocity'] / cumulative_data['total_flow_vectors']
    else:
        average_velocity = 0
    average_direction = np.arctan2(cumulative_data['sin_sum'], cumulative_data['cos_sum'])
    average_direction_degrees = np.degrees(average_direction)
    return average_velocity, average_direction_degrees

def dense_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    prevgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cumulative_data = {'total_velocity': 0, 'total_flow_vectors': 0, 'sin_sum': 0, 'cos_sum': 0}
    last_frame = None  # To hold the last frame processed

    while True:
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cumulative_data = calculate_velocity_direction(flow, cumulative_data)

        hsv_on_gray = draw_hsv_on_gray(gray, flow)
        cv2.imshow('Optical Flow - Farneback', hsv_on_gray)
        last_frame = hsv_on_gray  # Store the last frame

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    average_velocity, average_direction = compute_average_velocity_direction(cumulative_data)
    print(f"Farneback Average Velocity: {average_velocity:.2f} pixels/frame")
    print(f"Farneback Average Direction: {average_direction:.2f} degrees")

    # Display the last frame until 'q' is pressed
    if last_frame is not None:
        cv2.imshow("Last Frame - Optical Flow", last_frame)
        print("Displaying the last frame. Press 'q' to close.")
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dense_optical_flow('aerial_cars.mp4')
