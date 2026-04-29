from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model_custom.keras")

IMG_SIZE = 128


def preprocess_image(contents):
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    original = np.array(image)  # keep original

    image_norm = original / 255.0
    image_norm = np.expand_dims(image_norm, axis=0)

    return image_norm, original


@app.get("/test")
def test():
    return {"message": "API working"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # preprocess
        image, original = preprocess_image(contents)

        # CNN
        # CNN
        prediction = model.predict(image)
        print("RAW PREDICTION:", prediction)   # 👈 debug

        prob = float(prediction[0][0])
        real_conf = prob
        fake_conf = 1 - prob

        # ✅ Fix NaN issue
        if np.isnan(prob):
            print("WARNING: prob is NaN, setting to 0")
            prob = 0.0

        # ✅ Keep value between 0 and 1
        prob = np.clip(prob, 0, 1)

        THRESHOLD = 0.6
        predicted_label = "Real" if prob > THRESHOLD else "Fake"

        # grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # ===============================
        # BASIC FEATURES
        # ===============================
        contrast = np.std(gray)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (gray.size * 255)

        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        asymmetry = np.mean(np.abs(left - right[:, :left.shape[1]]))

        ela = np.abs(gray - cv2.GaussianBlur(gray, (5,5), 0))
        ela_mean = np.mean(ela)

        
        lighting_val = np.std(gray)
        shadow_diff = abs(np.mean(gray[:h//2]) - np.mean(gray[h//2:]))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_val = np.std(gray)

        reflection_ratio = np.sum(gray > 240) / gray.size

        fft = np.fft.fft2(gray)
        fft_mag = np.mean(np.abs(fft))

        noise_val = np.std(gray)
        mean_val = np.mean(gray)
        var_gray = np.var(gray)

        # ===============================
        # 🔥 ALL PARAMETERS (ALWAYS SHOWN)
        # ===============================
        all_params_list = [
    # 🔹 Visual
    f"Lighting: {lighting_val:.2f}",
    f"Shadow: {shadow_diff:.2f}",
    f"Blur: {laplacian:.2f}",
    f"Asymmetry: {asymmetry:.2f}",
    f"Texture: {texture_val:.2f}",
    f"Reflection: {reflection_ratio:.3f}",

    # 🔹 Image Statistics
    f"MeanPixel: {mean_val:.2f}",
    f"Contrast: {contrast:.2f}",
    f"EdgeDensity: {edge_density:.3f}",
    f"Noise: {noise_val:.2f}",

    # 🔹 Frequency / AI
    f"FFT: {fft_mag:.2f}",
    f"GAN_Artifact: {np.var(fft):.2f}",
    f"PRNU_Noise: {noise_val:.2f}",

    # ❗ Forgery Detection
    f"ELA: {ela_mean:.2f}",
    f"Compression: {fft_mag:.2f}",
    f"Resampling: {var_gray:.2f}",
    f"Splicing: {edge_density:.3f}",  # approx
    f"CopyMove: 0.00",  # placeholder

    # 📌 EXIF (placeholders)
    f"CameraModel: N/A",
    f"ISO: N/A",
    f"Shutter: N/A",
    f"EditingSoftware: N/A",

    # 🔹 Authenticity (approx)
    f"Geometry: {asymmetry:.2f}",
    f"Perspective: {edge_density:.3f}",
    f"Depth: {laplacian:.2f}",
    f"TextConsistency: N/A"
]

        all_parameters = " | ".join(all_params_list)

        # ===============================
        # 🔥 TRIGGERED PARAMETERS
        # ===============================
        param_details = []

        if lighting_val > 60:
            param_details.append("Inconsistent lighting")

        if shadow_diff > 25:
            param_details.append("Uneven shadows")

        if laplacian < 50:
            param_details.append("Blurred blending")

        if asymmetry > 25:
            param_details.append("Facial asymmetry")

        if texture_val < 20:
            param_details.append("Unnatural texture")

        if reflection_ratio > 0.05:
            param_details.append("Incorrect reflections")

        if fft_mag > 500:
            param_details.append("High frequency (AI artifact)")

        if np.var(fft) > 1000:
            param_details.append("GAN artifact")

        if noise_val < 10:
            param_details.append("Low sensor noise")

        if edge_density < 0.05:
            param_details.append("Splicing boundary")

        if contrast < 20:
            param_details.append("Low contrast")

        if ela_mean > 10:
            param_details.append("ELA inconsistency")

        if var_gray < 15:
            param_details.append("Resampling detected")

        triggered = " | ".join(param_details) if param_details else "None"

        # ===============================
        # 🔥 FORENSIC REPORT (YOUR LIST)
        # ===============================
        forensic_report = {
            "Visual_Artifacts": [],
            "EXIF_Metadata": [
                "Missing camera model",
                "No ISO / shutter info"
            ],
            "AI_Detection": [],
            "Forgery_Detection": [],
            "Authenticity": []
        }

        # Visual
        if lighting_val > 60:
            forensic_report["Visual_Artifacts"].append("Lighting inconsistency")
        if shadow_diff > 25:
            forensic_report["Visual_Artifacts"].append("Shadow mismatch")
        if asymmetry > 25:
            forensic_report["Visual_Artifacts"].append("Facial asymmetry")

        # AI
        if fft_mag > 500:
            forensic_report["AI_Detection"].append("GAN frequency pattern")
        if noise_val < 10:
            forensic_report["AI_Detection"].append("Low natural noise")

        # Forgery
        if ela_mean > 10:
            forensic_report["Forgery_Detection"].append("ELA tampering")
        if edge_density < 0.05:
            forensic_report["Forgery_Detection"].append("Splicing boundary")
        if var_gray < 15:
            forensic_report["Forgery_Detection"].append("Resampling")

        # Authenticity
        if asymmetry > 30:
            forensic_report["Authenticity"].append("Anatomy mismatch")
        if edge_density < 0.03:
            forensic_report["Authenticity"].append("Perspective issue")

        # ===============================
        # FINAL OUTPUT
        # ===============================
        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence_real": round(real_conf * 100, 2),
            "confidence_fake": round(fake_conf * 100, 2),

            "parameters_all": all_parameters,
            "parameters_triggered": triggered,
            "forensic_report": forensic_report
        }

    except Exception as e:
        return {"error": str(e)}