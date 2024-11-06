import cv2
import numpy as np
import time

def hybrid_optical_flow(video_path):
    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Parameters for Farneback Optical Flow
    fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade requires initial points
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create mask for drawing
    mask = np.zeros_like(old_frame)

    # Performance metrics
    frame_count = 0
    total_time_lk = 0
    total_time_fb = 0

    # Dynamic weights initialization
    weight_lk = 0.5  # Start with equal weight for both
    weight_fb = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: Lucas-Kanade Sparse Optical Flow (for sparse features)
        start_time = time.time()
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw sparse flow (Lucas-Kanade)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            # Update weights based on sparse point reliability
            weight_lk = 0.7 if len(good_new) > 50 else 0.3  # More points, rely more on LK
            p0 = good_new.reshape(-1, 1, 2)
        else:
            # If no points are found, reinitialize
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        total_time_lk += time.time() - start_time

        # Step 2: Farneback Dense Optical Flow
        start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **fb_params)
        total_time_fb += time.time() - start_time

        # Convert dense flow to HSV for visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        dense_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Adjust weights based on flow magnitude
        avg_magnitude = np.mean(mag)
        weight_fb = 0.7 if avg_magnitude > 2.0 else 0.3  # Higher motion, rely more on Farneback

        # Combine Lucas-Kanade and Farneback results using weighted average
        hybrid_flow = cv2.addWeighted(frame, weight_lk, dense_flow, weight_fb, 0)
        cv2.imshow('Hybrid Optical Flow (Weighted)', hybrid_flow)

        # Display individual flow results for comparison
        cv2.imshow('Sparse Optical Flow (Lucas-Kanade)', cv2.add(frame, mask))
        cv2.imshow('Dense Optical Flow (Farneback)', dense_flow)

        # Press 'q' to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Calculating Average FPS and Time per Frame for both methods
    if frame_count > 0:
        avg_time_lk = total_time_lk / frame_count
        avg_time_fb = total_time_fb / frame_count
        fps_lk = 1 / avg_time_lk
        fps_fb = 1 / avg_time_fb

        print("\nPerformance Evaluation:")
        print(f"Lucas-Kanade Optical Flow: {fps_lk:.2f} FPS, {avg_time_lk:.4f} seconds per frame")
        print(f"Farneback Optical Flow: {fps_fb:.2f} FPS, {avg_time_fb:.4f} seconds per frame")
        print(f"Hybrid Weighting: Lucas-Kanade Weight = {weight_lk:.2f}, Farneback Weight = {weight_fb:.2f}")
