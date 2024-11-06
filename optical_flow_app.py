import tkinter as tk
from tkinter import filedialog, messagebox
import sparseOpticalFlowLK as lucas_kanade
import denseOpticalFlow as farneback
import hybridOpticalFlow as hybrid

def run_optical_flow(method):
    video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    
    if not video_path:
        messagebox.showwarning("Warning", "No video file selected.")
        return

    if method == "Lucas-Kanade":
        lucas_kanade.lucas_kanade_optical_flow(video_path)
    elif method == "Farneback":
        farneback.dense_optical_flow(video_path)
    elif method == "Hybrid":
        hybrid.hybrid_optical_flow(video_path)
    else:
        messagebox.showerror("Error", "Invalid optical flow method.")

def create_gui():
    root = tk.Tk()
    root.title("Optical Flow Application")

    tk.Label(root, text="Choose Optical Flow Method:", font=("Helvetica", 16)).pack(pady=20)

    btn_lucas_kanade = tk.Button(root, text="Lucas-Kanade", command=lambda: run_optical_flow("Lucas-Kanade"), font=("Helvetica", 14))
    btn_lucas_kanade.pack(pady=10)

    btn_farneback = tk.Button(root, text="Farneback", command=lambda: run_optical_flow("Farneback"), font=("Helvetica", 14))
    btn_farneback.pack(pady=10)

    btn_hybrid = tk.Button(root, text="Hybrid", command=lambda: run_optical_flow("Hybrid"), font=("Helvetica", 14))
    btn_hybrid.pack(pady=10)

    root.geometry("400x300")
    root.mainloop()

if __name__ == "__main__":
    create_gui()
