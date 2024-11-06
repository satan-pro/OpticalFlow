import cv2
import numpy as np

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade Optical Flow
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                p0 = good_new.reshape(-1, 1, 2)

        img = cv2.add(frame, mask)
        cv2.imshow("Lucas-Kanade Optical Flow", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()

    cap.release()
    cv2.destroyAllWindows()

# Run the function directly
if __name__ == "__main__":
    lucas_kanade_optical_flow("roundabout.mp4")
