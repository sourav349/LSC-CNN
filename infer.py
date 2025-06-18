import os
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image
#from model import LSC_CNN  # Ensure your model is defined correctly
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ----------------------------- #
# Configuration
# ----------------------------- #
input_path = "/home/himanshu/outputimage.jpg"  # Change as needed
output_dir = "/home/himanshu/Desktop/NUC_DATA/nuc_dataset/dataset/test_data"
model_path = "checkpoints/final_modelone.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------- #
# Load Model
# ----------------------------- #
model = LSC_CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------- #
# Transforms
# ----------------------------- #
transform = T.ToTensor()
to_pil = T.ToPILImage()

# ----------------------------- #
# Helper: RNS Calculation
# ----------------------------- #
def calculate_rns(tensor_img):
    mean_val = torch.mean(tensor_img)
    std_val = torch.std(tensor_img)
    return (std_val / (mean_val + 1e-8)).item()

# ----------------------------- #
# PSNR and SSIM Calculation
# ----------------------------- #
def calculate_psnr_ssim(img1, img2):
    img1_np = np.array(img1, dtype=np.float32) / 255.0
    img2_np = np.array(img2, dtype=np.float32) / 255.0
    psnr = compare_psnr(img1_np, img2_np, data_range=1.0)
    ssim = compare_ssim(img1_np, img2_np, data_range=1.0)
    return psnr, ssim

# ----------------------------- #
# Denoising Inference
# ----------------------------- #
def denoise_image(img: Image.Image):
    img_gray = img.convert("L")
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(img_tensor)
    input_tensor = img_tensor.squeeze().cpu().clamp(0, 1)
    output_tensor = output_tensor.squeeze().cpu().clamp(0, 1)
    return input_tensor, output_tensor

# ----------------------------- #
# Process Single Image
# ----------------------------- #
def process_single_image(image_path):
    img = Image.open(image_path).convert("L")
    input_tensor, output_tensor = denoise_image(img)

    # Convert to PIL for metric calculation
    input_img_pil = to_pil(input_tensor)
    output_img_pil = to_pil(output_tensor)

    # RNS
    rns_input = calculate_rns(input_tensor)
    rns_output = calculate_rns(output_tensor)

    # PSNR and SSIM
    psnr_input, ssim_input = calculate_psnr_ssim(input_img_pil, input_img_pil)
    psnr_output, ssim_output = calculate_psnr_ssim(input_img_pil, output_img_pil)

    # Save and Show Output Image
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    output_img_pil.save(out_path)
    output_img_pil.show(title="Denoised Image")

    # Display Results in Table Format
    print("\n📊 Results (Image Quality Metrics):")
    print("------------------------------------------------------------")
    print(f"{'Metric':<15} | {'Input Image':<15} | {'Denoised Image':<15}")
    print("------------------------------------------------------------")
    print(f"{'RNS':<15} | {rns_input:<15.6f} | {rns_output:<15.6f}")
    print(f"{'PSNR (dB)':<15} | {psnr_input:<15.2f} | {psnr_output:<15.2f}")
    print(f"{'SSIM':<15} | {ssim_input:<15.4f} | {ssim_output:<15.4f}")
    print("------------------------------------------------------------")
    print(f"✅ Saved denoised image: {out_path}")

# ----------------------------- #
# Folder and Video Logic (Optional)
# ----------------------------- #
def process_folder(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(folder_path, fname)
            process_single_image(img_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(output_dir, "denoised_video.avi")
    out = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        input_tensor, output_tensor = denoise_image(img)

        input_img_pil = to_pil(input_tensor)
        output_img_pil = to_pil(output_tensor)

        rns_input = calculate_rns(input_tensor)
        rns_output = calculate_rns(output_tensor)
        psnr_output, ssim_output = calculate_psnr_ssim(input_img_pil, output_img_pil)

        den_frame = np.array(output_img_pil)

        if out is None:
            h, w = den_frame.shape
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h), isColor=False)

        out.write(den_frame)

        print(f"🎞️ Frame {frame_idx:03d} | RNS_in: {rns_input:.4f} | RNS_out: {rns_output:.4f} | PSNR: {psnr_output:.2f} | SSIM: {ssim_output:.4f}")
        frame_idx += 1

    cap.release()
    if out:
        out.release()
        print(f"\n✅ Denoised video saved to: {out_path}")

# ----------------------------- #
# Main Entry
# ----------------------------- #
if os.path.isdir(input_path):
    print("📁 Detected: Directory of images")
    process_folder(input_path)
elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg")):
    print("🎞️ Detected: Video file")
    process_video(input_path)
elif input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
    print("🖼️ Detected: Single image")
    process_single_image(input_path)
elif input_path.lower() == "realtime":
    print("🟢 Realtime not supported with quality metrics.")
else:
    raise ValueError("❌ Unsupported input type.")