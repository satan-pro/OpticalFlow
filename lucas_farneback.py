import cv2
import numpy as np

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

def calculate_hsv_color(speed, angle):
    # Map angle to Hue (direction), and speed to Value (brightness)
    hue = int((angle + np.pi) * 90 / np.pi)  # Map angle to [0, 180] for Hue
    saturation = 255
    value = min(255, max(300, int(speed * 30)))  # Scale speed for Value
    
    # Create HSV color and convert to BGR for OpenCV display
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr_color)  # Convert to tuple for OpenCV line drawing

def lucas_kanade_optical_flow(video_path):
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)
    cumulative_data = {'total_velocity': 0, 'total_flow_vectors': 0, 'sin_sum': 0, 'cos_sum': 0}
    last_frame = None  # To store the last frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    # Calculate speed and angle
                    dx, dy = a - c, b - d
                    speed = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)
                    
                    # Update cumulative data for average velocity and direction
                    cumulative_data['total_velocity'] += speed
                    cumulative_data['total_flow_vectors'] += 1
                    cumulative_data['sin_sum'] += np.sin(angle)
                    cumulative_data['cos_sum'] += np.cos(angle)
                    
                    # Get dynamic color based on angle and speed
                    color = calculate_hsv_color(speed, angle)
                    
                    # Draw line with dynamic color
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                # Update points for the next frame
                p0 = good_new.reshape(-1, 1, 2)

        img = cv2.add(frame, mask)
        cv2.imshow("Lucas-Kanade Optical Flow with HSV Color Lines", img)
        last_frame = img.copy()  # Update the last frame with the current display frame
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()

    average_velocity, average_direction = compute_average_velocity_direction(cumulative_data)
    print(f"Lucas-Kanade Average Velocity: {average_velocity:.2f} pixels/frame")
    print(f"Lucas-Kanade Average Direction: {average_direction:.2f} degrees")

    # Display the last frame until 'q' is pressed
    if last_frame is not None:
        cv2.imshow("Last Frame - Lucas-Kanade Optical Flow with HSV Color Lines", last_frame)
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the function directly
if __name__ == "__main__":
    lucas_kanade_optical_flow("roundabout.mp4")
