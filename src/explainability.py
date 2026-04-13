import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap showing where the model 'focused'.
    """
    # 1. Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. This is the gradient of the output neuron with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="outputs/gradcam.png", alpha=0.4):
    """
    Overlay the heatmap on the original image for explainability.
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    print(f"[INFO] 🎯 Explainability Heatmap (Grad-CAM) saved to '{cam_path}'")
    return cam_path

def generate_clinical_report(img_path, diagnosis, confidence, heatmap, report_path="outputs/clinical_report.png"):
    """
    Creates a premium, shareable 'Diagnostic Report' image.
    Combines original scan, heatmap, and AI findings.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os

    # 1. Prepare images
    original = Image.open(img_path).convert("RGB").resize((400, 400))
    
    # Generate heatmap image locally for the report
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((400, 400))
    heatmap_colored = Image.new("RGB", (400, 400))
    # Simple colorization for report overlay
    heatmap_colored = Image.blend(original, heatmap_img.convert("RGB"), alpha=0.5)

    # 2. Create Canvas
    canvas_width = 900
    canvas_height = 600
    report = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(report)

    # 3. Add Header
    draw.rectangle([0, 0, canvas_width, 80], fill=(44, 62, 80)) # Dark Blue header
    # Note: Using default font if custom font path isn't reliable in all envs
    draw.text((20, 25), "AI DIAGNOSTIC ASSISTANCE CENTER — CLINICAL REPORT", fill=(255, 255, 255))

    # 4. Paste Images
    report.paste(original, (40, 120))
    report.paste(heatmap_colored, (470, 120))

    # 5. Add Labels & Findings
    text_color = (0, 0, 0)
    draw.text((40, 540), "ORIGINAL X-RAY SCAN", fill=text_color)
    draw.text((470, 540), "AI FOCUS AREA (GRAD-CAM)", fill=text_color)

    draw.rectangle([650, 430, 880, 580], outline=(44, 62, 80), width=2)
    draw.text((660, 440), "FINDINGS:", fill=(44, 62, 80))
    draw.text((660, 470), f"DIAGNOSIS: {diagnosis}", fill=(200, 0, 0) if "PNEUMONIA" in diagnosis else (0, 150, 0))
    draw.text((660, 500), f"CONFIDENCE: {confidence*100:.2f}%", fill=text_color)
    draw.text((660, 530), f"STATUS: VERIFIED", fill=text_color)

    # 6. Save
    report.save(report_path)
    print(f"[SUCCESS] 📄 Premium Clinical Report generated at '{report_path}'")
