import json
import os
import numpy as np
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

# Configuration
MODEL_PATH = os.path.join("models", "best_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
IMG_SIZE = (224, 224)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60  # 60%

# Fallback class names if class_names.json is missing
DEFAULT_CLASS_NAMES = ["glass", "metal", "paper", "plastic"]

# Disposal guidance dictionary
DISPOSAL_GUIDANCE = {
    "glass": (
        "♻️ **Glass** — Rinse the item and remove caps or lids. "
        "Place in a glass recycling bin. Avoid mixing with ceramics or "
        "mirrors, which are not recyclable with standard glass. "
        "Local rules may vary — check with your municipality."
    ),
    "metal": (
        "♻️ **Metal** — Rinse cans and tins; crush if possible to save space. "
        "Aluminium and steel are widely recycled. Remove paper labels if "
        "required by your local program. Aerosol cans should be empty before "
        "recycling. Local rules may vary."
    ),
    "paper": (
        "♻️ **Paper** — Place clean, dry paper and cardboard in the paper "
        "recycling bin. Remove plastic windows from envelopes. Shredded paper "
        "may need to be bagged. Soiled or waxed paper usually goes in general "
        "waste. Local rules may vary."
    ),
    "plastic": (
        "♻️ **Plastic** — Check the resin code (number inside the triangle). "
        "Codes 1 (PET) and 2 (HDPE) are most widely accepted. Rinse "
        "containers and replace caps. Soft plastics (bags, wraps) often "
        "require separate drop-off points. Local rules may vary."
    ),
}

# Model loading (cached so it runs only once)
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the trained Keras model and class names once."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"**Model file not found** at `{MODEL_PATH}`.\n\n"
            "Please train the model first:\n```\npython train.py --data_dir dataset/\n```"
        )
        st.stop()

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"**Failed to load model:** {exc}")
        st.stop()

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = DEFAULT_CLASS_NAMES
        st.warning(
            f"`{CLASS_NAMES_PATH}` not found — using default class order: {class_names}. "
            "Re-run training to generate this file."
        )

    return model, class_names

# Preprocessing
def preprocess_image(image: Image.Image, model=None) -> np.ndarray:
    """Resize, convert to RGB, and format for the model."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(image, dtype=np.float32)

    # Check whether the model handles normalization internally so we don't double-normalize.
    has_internal_preprocess = False
    if model is not None:
        ops_and_layers = [
            getattr(op, "name", "").lower() for op in getattr(model, "operations", [])
        ] + [
            getattr(l, "name", "").lower() for l in getattr(model, "layers", [])
        ]
        has_internal_preprocess = any(
            "divide" in name or "subtract" in name or "rescal" in name or "preprocess" in name
            for name in ops_and_layers
        )

    if not has_internal_preprocess:
        arr = keras.applications.mobilenet_v2.preprocess_input(arr)

    return np.expand_dims(arr, axis=0)  # Batch dimension

# Prediction
def predict(model, class_names, image: Image.Image):
    """Return (predicted_class, confidence, all_probabilities)."""
    preprocessed = preprocess_image(image, model=model)
    probabilities = model.predict(preprocessed, verbose=0)[0]
    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    predicted_class = class_names[top_idx]
    return predicted_class, confidence, probabilities

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Smart Waste Classifier",
        page_icon="♻️",
        layout="centered",
    )

    st.title("♻️ Smart Waste Classification")
    st.markdown(
        "Upload an image of a waste item to classify it as **glass**, "
        "**metal**, **paper**, or **plastic** and receive disposal guidance."
    )

    # Load model once
    model, class_names = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image…",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        # Validate and display the image
        try:
            image = Image.open(uploaded_file)
            image.verify()  # Check file integrity
            # Re-open after verify (verify can close the stream)
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
        except Exception:
            st.error("The uploaded file is not a valid image. Please try another file.")
            return

        try:
            st.image(image, caption="Uploaded image", width="stretch")
        except TypeError:
            st.image(image, caption="Uploaded image", use_container_width=True)

        # Predict button
        if st.button("🔍 Classify", type="primary"):
            with st.spinner("Classifying…"):
                predicted_class, confidence, probs = predict(model, class_names, image)

            st.divider()

            is_uncertain = confidence < CONFIDENCE_THRESHOLD

            #Results
            if is_uncertain:
                st.warning(
                    f"⚠️ **Uncertain prediction.** The model's highest confidence "
                    f"is only **{confidence:.1%}**, which is below the "
                    f"{CONFIDENCE_THRESHOLD:.0%} threshold.\n\n"
                    "**Please check the material manually** before following any "
                    "disposal instructions."
                )
                st.info(
                    f"Best guess: **{predicted_class.capitalize()}** "
                    f"({confidence:.1%} confidence)"
                )
            else:
                st.success(
                    f"**Predicted material:** {predicted_class.capitalize()}  \n"
                    f"**Confidence:** {confidence:.1%}"
                )

            # Per-class probabilities
            with st.expander("Class probabilities"):
                for i, name in enumerate(class_names):
                    st.progress(float(probs[i]), text=f"{name.capitalize()}: {probs[i]:.1%}")

            #Disposal guidance
            st.divider()
            if is_uncertain:
                st.markdown(
                    "🚫 **Disposal guidance withheld** — the prediction is not "
                    "confident enough. Verify the material type before disposing."
                )
            else:
                st.subheader("Disposal Guidance")
                guidance = DISPOSAL_GUIDANCE.get(
                    predicted_class,
                    "No guidance available for this category.",
                )
                st.markdown(guidance)

if __name__ == "__main__":
    main()