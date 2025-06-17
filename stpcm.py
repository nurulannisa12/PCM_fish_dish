### FINAL PROJECT PCM - NURUL ANNISA - 5023221031
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import streamlit as st
import scipy.ndimage as ndi
from skimage import exposure,filters, morphology, segmentation, feature, measure, color
from skimage.filters import gaussian
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk, binary_closing
from skimage.io import imread
from skimage.color import label2rgb, rgb2gray
from skimage.measure import regionprops
import os
import cv2
import imageio.v2 as imageio

# --- Histogram Implementation ---
def plot_histogram_with_image(image, title="Histogram", colormap="gray"):
    # Buat 2 kolom berdampingan
    col1, col2 = st.columns(2)
    # Pastikan image uint8
    if image.dtype != np.uint8:
        image_disp = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        image_disp = image

    # Histogram
    hist = ndi.histogram(image_disp, min=0, max=255, bins=256)
    # Plot histogram
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, color='black')
    ax.set_title(title)
    ax.set_xlabel("Intensity Value")
    ax.set_ylabel("Number of Pixels")
    ax.grid(True)
    col1.pyplot(fig)
    # Gambar sumber
    col2.image(image_disp, caption="Source Image", use_column_width=True, clamp=True, channels="GRAY")

def plot_rgb_histogram(image, key_prefix=""):
    if st.checkbox("Tampilkan Histogram RGB per Channel", key=f"{key_prefix}_rgbhist"):
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image / 255.0

        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        hist_r, bins_r = exposure.histogram(r)
        hist_g, bins_g = exposure.histogram(g)
        hist_b, bins_b = exposure.histogram(b)

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        axs[0, 0].plot(bins_r, hist_r, color='red')
        axs[0, 1].plot(bins_g, hist_g, color='green')
        axs[0, 2].plot(bins_b, hist_b, color='blue')

        axs[1, 0].imshow(r, cmap='gray')
        axs[1, 1].imshow(g, cmap='gray')
        axs[1, 2].imshow(b, cmap='gray')

        for i in range(3):
            axs[0, i].set_title(f'Histogram {"RGB"[i]} Channel')
            axs[1, i].set_title(f'{"RGB"[i]} Channel Image')
            axs[0, i].grid(True)
            axs[1, i].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        return r, g, b
    else:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        return r, g, b


# --- Image Preprocessing ---
def apply_clahe(image_channel, label="CLAHE Result", key_prefix=""):
    # Slider dengan key unik berdasarkan konteks (FISH/DISH)
    value_cl = st.slider(
        "Choose Clip Limit value",
        min_value=0.001,
        max_value=0.02,
        value=0.007,
        step=0.001,
        key=f"{key_prefix}_clip_limit"
    )

    # Terapkan CLAHE
    im_clahe = exposure.equalize_adapthist(image_channel, clip_limit=value_cl)
    # Histogram
    hist_clahe, bins_clahe = exposure.histogram(im_clahe)

    # Plot hasil CLAHE dan histogram
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(im_clahe, cmap='gray')
    axs[0].set_title(f'{label}')
    axs[0].axis('off')
    axs[1].plot(bins_clahe, hist_clahe, color='black')
    axs[1].set_title('Histogram after CLAHE')
    axs[1].set_xlabel('Intensity')
    axs[1].set_ylabel('Pixel Count')
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    return im_clahe

def apply_otsu_threshold(image, title="Otsu Thresholding", is_dish=False):
    """
    image       : channel citra grayscale (0-1 atau 0-255)
    title       : judul plot
    is_dish     : True jika jenis citra DISH (background putih)
    """
    # Pastikan dalam uint8
    if image.max() <= 1.0:
        image_uint = (image * 255).astype(np.uint8)
    else:
        image_uint = image.astype(np.uint8)

    # Thresholding Otsu
    thresh_val = filters.threshold_otsu(image_uint)
    if is_dish:
        binary_mask = image_uint < thresh_val  # background terang → ambil gelap
    else:
        binary_mask = image_uint > thresh_val  # background gelap → ambil terang

    # Plot hasil
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    axs[0].imshow(image_uint, cmap='Blues')
    axs[0].set_title('Original Channel (uint8)')
    axs[0].axis('off')

    axs[1].hist(image_uint.ravel(), bins=256, color='blue')
    axs[1].axvline(thresh_val, color='red', linestyle='--')
    axs[1].set_title(f'Histogram + Otsu Threshold = {thresh_val:.1f}')

    axs[2].imshow(binary_mask, cmap='gray')
    axs[2].set_title('Thresholded Mask\n(Foreground = White)')
    axs[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    return binary_mask

def filter_and_label_cells(binary_mask, key_prefix="fish"):
    st.markdown("### Parameter Filtering dan Morphology")
    # Slider min size
    min_obj_size = st.number_input(
        "Minimum Object Size (remove_small_objects)",
        min_value=100,
        max_value=20000,
        value=8000 if key_prefix == "fish" else 8000,
        step=100,
        key=f"{key_prefix}_min_obj"
    )
    min_hole_size = st.number_input(
        "Minimum Hole Size (fill_holes)",
        min_value=100,
        max_value=10000,
        value=1200 if key_prefix == "fish" else 2500,
        step=100,
        key=f"{key_prefix}_min_hole"
    )
    smooth_radius = st.slider(
        "Smooth Kontur (Closing Radius)",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        key=f"{key_prefix}_smooth"
    )

    # Step 1: remove small objects
    mask_no_small = morphology.remove_small_objects(binary_mask, min_size=min_obj_size)
    # Step 2: fill holes
    mask_filled = np.logical_not(
        morphology.remove_small_objects(np.logical_not(mask_no_small), min_size=min_hole_size))
    # Step 3: morphological closing (haluskan kontur)
    if smooth_radius > 0:
        mask_smoothed = morphology.binary_closing(mask_filled, morphology.disk(smooth_radius))
    else:
        mask_smoothed = mask_filled.copy()

    # Labeling
    labels, nlabels = ndi.label(mask_smoothed)
    rand_cmap = ListedColormap(np.random.rand(256, 3))
    labels_for_display = np.where(labels > 0, labels, np.nan)

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(binary_mask, cmap='gray')
    axs[0].set_title('Mask Awal (Threshold)')
    axs[0].axis('off')
    axs[1].imshow(mask_no_small, cmap='gray')
    axs[1].set_title('Remove Small Object')
    axs[1].axis('off')
    axs[2].imshow(mask_filled, cmap='gray')
    axs[2].set_title('Fill Holes')
    axs[2].axis('off')
    axs[3].imshow(mask_smoothed, cmap='gray')
    axs[3].set_title('Smoothed (Closing)')
    axs[3].axis('off')
    axs[4].imshow(labels_for_display, cmap=rand_cmap)
    axs[4].set_title(f'Labeled = {nlabels}')
    axs[4].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    st.success(f"✅ Total objek terdeteksi setelah filtering: **{nlabels} objek**")
    return mask_smoothed, labels, nlabels

def watershed_segmentation(image_segmented, image, key_prefix="fish", show_plot=True):
    st.markdown("### Parameter Watershed")

    # Nilai default disesuaikan berdasarkan jenis citra
    sigma_default = 1.5 if key_prefix == "fish" else 1.8
    min_dist_default = 30 if key_prefix == "fish" else 60
    footprint_default = 20 if key_prefix == "fish" else 21

    sigma = st.slider(
        "Gaussian Sigma (Smooth Distance Map)",
        min_value=0.0, max_value=5.0, value=sigma_default, step=0.1,
        key=f"{key_prefix}_sigma"
    )
    min_distance = st.slider(
        "Minimum Distance antar Seed", min_value=1, max_value=50,
        value=min_dist_default, step=1,
        key=f"{key_prefix}_min_distance"
    )
    footprint_size = st.slider(
        "Ukuran Footprint Seed", min_value=1, max_value=30,
        value=footprint_default, step=1,
        key=f"{key_prefix}_footprint"
    )

    # Step 1: Distance transform & Gaussian smoothing
    distance = ndi.distance_transform_edt(image_segmented)
    distance_smooth = gaussian(distance, sigma=sigma)
    # Step 2: Cari koordinat seed (local maxima)
    coordinates = feature.peak_local_max(
        distance_smooth,
        labels=image_segmented,
        min_distance=min_distance,
        footprint=np.ones((footprint_size, footprint_size))
    )
    # Step 3: Buat mask seed
    local_maxi = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(coordinates.T)] = True
    # Step 4: Label markers
    markers = measure.label(local_maxi)
    # Step 5: Watershed segmentation
    labels_ws = segmentation.watershed(-distance, markers, mask=image_segmented)
    # Step 6: Buat boundary dan overlay
    boundaries = find_boundaries(labels_ws, mode='inner')
    thick_boundaries = binary_dilation(boundaries, disk(1))

    image_with_lines = image.copy()
    if image_with_lines.dtype != np.uint8:
        image_with_lines = (image_with_lines * 255).clip(0, 255).astype(np.uint8)
    image_with_lines[thick_boundaries] = [255, 255, 0]

    nlabels = len(np.unique(labels_ws)) - 1  # Kurangi background
    st.success(f"✅ Total objek terdeteksi setelah watershed: {nlabels}")
    if show_plot:
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        axs[0].imshow(distance, cmap='magma')
        axs[0].set_title('Distance Transform')
        axs[0].axis('off')

        axs[1].imshow(color.label2rgb(labels_ws, image=image, bg_label=0))
        axs[1].set_title("Watershed Result (Sel Terpisah)")
        axs[1].axis('off')

        axs[2].imshow(thick_boundaries, cmap='gray')
        axs[2].set_title("Boundary Lines (Binary)")
        axs[2].axis('off')

        axs[3].imshow(image_with_lines)
        axs[3].set_title("Overlay: Watershed Line on Original Image")
        axs[3].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

    return labels_ws, thick_boundaries, image_with_lines

def extract_gt_mask_from_yellow(gt_rgb):
    r = gt_rgb[:, :, 0]
    g = gt_rgb[:, :, 1]
    b = gt_rgb[:, :, 2]

    yellow_mask = (r > 200) & (g > 200) & (b < 100)
    yellow_mask_dilated = binary_dilation(yellow_mask, disk(1))
    closed = binary_closing(yellow_mask_dilated, disk(3))
    gt_mask = ndi.binary_fill_holes(closed)
    return gt_mask

def evaluate_segmentation_result(ground_truth_path, result_mask):
    gt_img = imread(ground_truth_path)
    if gt_img.ndim == 3 and gt_img.shape[-1] == 4:
        gt_img = gt_img[:, :, :3]

    gt_mask = extract_gt_mask_from_yellow(gt_img)

    if result_mask.ndim == 3:
        result_mask = result_mask[:, :, 0]

    gt_resized = resize(gt_mask, result_mask.shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)

    y_true = gt_resized.flatten().astype(bool)
    y_pred = (result_mask > 0).flatten().astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    iou = intersection / union if union > 0 else 0.0
    dice = 2. * intersection / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) > 0 else 0.0

    # st.subheader("7. Evaluasi Hasil Segmentasi")
    st.write(f"IoU  : {iou:.4f}")
    st.write(f"Dice : {dice:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(gt_resized, cmap='gray')
    axs[0].set_title("Ground Truth Mask")
    axs[0].axis('off')

    axs[1].imshow(result_mask, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

## -------- UNTUK ANALISIS SINYAL HER2 DAN CEN17 -----------##
def plot_chan_signal (im):
    # Pastikan image RGB bertipe float 0-1
    if im.dtype != np.float32 and im.dtype != np.float64:
        image_rgb = im / 255.0
    else:
        image_rgb = im.copy()

    # Ekstrak channel merah (HER2) dan hijau (CEN17)
    her2_channel = image_rgb[:, :, 0]
    cen17_channel = image_rgb[:, :, 1]
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(her2_channel, cmap='Reds')
    axs[0].set_title("HER2 Channel (Red)")
    axs[0].axis('off')

    axs[1].imshow(cen17_channel, cmap='Greens')
    axs[1].set_title("CEN17 Channel (Green)")
    axs[1].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    return her2_channel, cen17_channel

def extract_and_plot_dish_signal_channels(im, her2_mask_thresh=0.1):
    # Konversi ke float [0–1] jika perlu
    image_rgb = im / 255.0 if im.dtype != np.float32 and im.dtype != np.float64 else im.copy()
    # HER2: Pink = R - G
    her2_channel = np.clip(image_rgb[:, :, 0] - image_rgb[:, :, 1], 0, 1)
    # Grayscale → sumber CEN17
    grayscale = rgb2gray(image_rgb)
    # Buat masking → titik yang terlalu merah (HER2)
    mask = her2_channel > her2_mask_thresh
    # CEN17 = grayscale + suppress HER2 → diputihkan (1.0)
    cen17_channel = np.copy(grayscale)
    cen17_channel[mask] = 1.0  # diputihkan karena latar belakang putih
    # Inversi agar sinyal CEN17 menjadi terang (1.0 → 0.0 dan sebaliknya)
    cen17_channel = 1.0 - cen17_channel
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(her2_channel, cmap='Reds')
    axs[0].set_title("DISH HER2 Channel (Red - Green)")
    axs[0].axis('off')
    axs[1].imshow(cen17_channel, cmap='gray')
    axs[1].set_title("DISH CEN17 Channel (Grayscale - HER2 masked)")
    axs[1].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    return her2_channel, cen17_channel

def apply_clahe_sig(image_channel, label="Channel", key_prefix=""):
    st.write(f"CLAHE diterapkan pada: **{label}**")

    clip_limit = st.number_input(
        "Clip Limit (manual input)",
        min_value=0.001,
        max_value=0.05,
        value=0.007,
        step=0.001,
        format="%.3f",
        key=f"{key_prefix}_clip_limit"
    )
    im_clahe = exposure.equalize_adapthist(
        image_channel,
        clip_limit=clip_limit,
        kernel_size=(32, 32)
    )
    hist_clahe, bins_clahe = exposure.histogram(im_clahe)
    # Pilih cmap dan warna histogram
    label_lower = label.lower()
    if "her2" in label_lower:
        cmap = "Reds"
        line_color = "red"
    elif "cen17" in label_lower:
        cmap = "Greens"
        line_color = "green"
    else:
        cmap = "gray"
        line_color = "black"

    # Plot citra + histogram
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(im_clahe, cmap=cmap)
    axs[0].set_title(f'CLAHE Result ({label})')
    axs[0].axis('off')
    axs[1].plot(bins_clahe, hist_clahe, color=line_color)
    axs[1].set_title(f'Histogram after CLAHE ({label})')
    axs[1].set_xlabel('Intensity')
    axs[1].set_ylabel('Pixel Count')
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    return im_clahe

def stretch_channel(channel, label="Channel", cmap="gray", key_prefix=""):
    st.subheader(f"{label} – Intensity Stretching")
    stretch_min = st.number_input(
        "Stretch range (manual input)",
        min_value=0.2,
        max_value=0.7,
        value=0.33,
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_stretch_min"
    )
    stretched = rescale_intensity(channel, in_range=(stretch_min, 1.0))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(channel, cmap=cmap)
    axs[0].set_title(f"{label} Original")
    axs[0].axis('off')

    axs[1].imshow(stretched, cmap=cmap)
    axs[1].set_title(f"{label} Stretched (min={stretch_min})")
    axs[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    return stretched

def detect_and_plot_signal_coords(image_rgb, her2_stretched, cen17_stretched,
                                  default_thresh_min=0.33, thresh_max=1.0,
                                  key_prefix="fish"):
    """
    Mendeteksi dan memvisualisasikan koordinat titik sinyal HER2 dan CEN17 dari hasil stretching.
    """
    col1, col2 = st.columns(2)
    with col1:
        her2_min = st.number_input(
            f"Threshold Minimum HER2 ({key_prefix})", 
            min_value=0.0, max_value=1.0,
            value=default_thresh_min, step=0.01, format="%.2f",
            key=f"{key_prefix}_her2_thresh"
        )
    with col2:
        cen17_min = st.number_input(
            f"Threshold Minimum CEN17 ({key_prefix})", 
            min_value=0.0, max_value=1.0,
            value=default_thresh_min, step=0.01, format="%.2f",
            key=f"{key_prefix}_cen17_thresh"
        )

    # Dapatkan koordinat titik sinyal
    her2_coords = np.argwhere((her2_stretched >= her2_min) & (her2_stretched <= thresh_max))
    cen17_coords = np.argwhere((cen17_stretched >= cen17_min) & (cen17_stretched <= thresh_max))

    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb)

    if her2_coords.size > 0:
        ax.scatter(
            her2_coords[:, 1], her2_coords[:, 0],
            s=5, facecolors='none', edgecolors='red', linewidths=0.5, label='HER2'
        )
    if cen17_coords.size > 0:
        ax.scatter(
            cen17_coords[:, 1], cen17_coords[:, 0],
            s=5, facecolors='none', edgecolors='lime', linewidths=0.5, label='CEN17'
        )

    ax.set_title("Titik Sinyal Terdeteksi Berdasarkan Threshold Manual")
    ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    return her2_coords, cen17_coords

def analyze_her2_cen17_ratio(labels_ws, her2_coords, cen17_coords):
    """
    Menganalisis jumlah sinyal HER2 dan CEN17 dalam setiap region dari hasil watershed.
    """
    # Persiapan
    label_map = labels_ws
    regions = regionprops(label_map)

    her2_map = np.zeros_like(label_map, dtype=int)
    cen17_map = np.zeros_like(label_map, dtype=int)

    # Tandai posisi titik sinyal di peta
    her2_map[her2_coords[:, 0], her2_coords[:, 1]] = 1
    cen17_map[cen17_coords[:, 0], cen17_coords[:, 1]] = 1

    # List hasil
    region_ids, her2_counts, cen17_counts, ratios, statuses = [], [], [], [], []

    # Loop per region
    for region in regions:
        region_id = region.label
        coords = region.coords

        h_count = her2_map[coords[:, 0], coords[:, 1]].sum()
        c_count = cen17_map[coords[:, 0], coords[:, 1]].sum()
        ratio = h_count / c_count if c_count > 0 else 0

        # Status HER2
        if ratio > 2.0:
            status = "HER2-Positive"
        elif 1.8 <= ratio <= 2.0:
            status = "Equivocal"
        else:
            status = "HER2-Negative"

        region_ids.append(region_id)
        her2_counts.append(h_count)
        cen17_counts.append(c_count)
        ratios.append(ratio)
        statuses.append(status)

    # Buat DataFrame
    df_ratio = pd.DataFrame({
        "Region": region_ids,
        "HER2_Count": her2_counts,
        "CEN17_Count": cen17_counts,
        "HER2/CEN17_Ratio": ratios,
        "HER2_Status": statuses
    })

    # Tampilkan sebagai scrollable dataframe
    st.dataframe(df_ratio.style.format({'HER2/CEN17_Ratio': '{:.2f}'}), height=400)

    # Opsional: filter positif
    st.markdown("**Region HER2-Positive**")
    st.dataframe(df_ratio[df_ratio['HER2_Status'] == 'HER2-Positive'])
    return df_ratio

def visualize_her2_cen17_classification(im, labels_ws, her2_coords, cen17_coords, df_ratio):
    """
    Menampilkan visualisasi hasil klasifikasi HER2/CEN17 dengan bounding box dan overlay sinyal.
    """
    # Buat boundary dan overlay
    boundaries = find_boundaries(labels_ws, mode='inner')
    overlay_img = label2rgb(labels_ws, image=im, bg_label=0)
    regions = measure.regionprops(labels_ws)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)  # Citra asli
    ax.imshow(boundaries, cmap='gray', alpha=0.5)  # Garis boundary semi-transparan

    # Titik-titik sinyal
    if her2_coords.shape[0] > 0:
        ax.scatter(
            her2_coords[:, 1], her2_coords[:, 0],
            s=5, facecolors='none', edgecolors='red', linewidths=0.5, label='HER2')

    if cen17_coords.shape[0] > 0:
        ax.scatter(
            cen17_coords[:, 1], cen17_coords[:, 0],
            s=5, facecolors='none', edgecolors='lime', linewidths=0.5, label='CEN17')

    # Bounding box + label per sel
    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        cy, cx = region.centroid

        status = df_ratio.loc[df_ratio["Region"] == region.label, "HER2_Status"].values[0]
        color_box = {
            'HER2-Positive': 'red',
            'Equivocal': 'yellow',
            'HER2-Negative': 'green'}[status]

        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor=color_box, linewidth=1.2)
        ax.add_patch(rect)
        ax.plot(cx, cy, 'o', color=color_box, markersize=4)
        ax.text(cx, cy, f'{region.label}', color='white', fontsize=6, ha='center', va='center')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

    ax.set_title("Visualisasi Status HER2/CEN17 + Titik Sinyal")
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

#### ======================== ####
###  STREAMLIT IMPLEMENTATION  ###
#### ======================== ####
st.set_page_config(layout="wide")
st.title("Final Project PCM - FISH nand DISH Analysis and Segmentation")

# --- PILIH JENIS CITRA ---
tab_fish, tab_dish = st.tabs(["A. CITRA FISH", "B. CITRA DISH"])
with tab_fish:
    image_folder = "FISH"
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    selected_file = st.selectbox("Pilih gambar FISH:", image_files)
    image_path = os.path.join(image_folder, selected_file)
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # st.image(im, caption="Citra FISH RGB Asli", use_container_width=True)
    st.image(im, caption="Citra FISH RGB Asli", width=500)
    
    # --- PROSES ---
    st.subheader("1. Histogram Analysis: 3 channel RGB")
    r, g, b = plot_rgb_histogram(im, key_prefix="fish")   # untuk tab FISH
    st.subheader("2. Preprocessing: CLAHE")
    b_clahe = apply_clahe(b, label="CLAHE on Blue Channel (FISH)", key_prefix="fish_b")
    st.subheader("3. Segmentasi: Otsu Thresholding")
    binary_mask = apply_otsu_threshold(b_clahe, title="Threshold FISH", is_dish=False)
    # binary_mask = apply_otsu_threshold(b_clahe)
    st.subheader("4. Filtering dan Pelabelan")
    image_segmented, labels, nlabels = filter_and_label_cells(binary_mask, key_prefix="fish")
    st.subheader("5. Watershed Segmentation")
    labels_ws, boundaries_ws, overlay_img = watershed_segmentation(image_segmented, im, key_prefix="fish")
    st.subheader("6. Evaluasi Citra terhadap Ground Truth")
    gt_folder = "REF_FISH"  # ganti sesuai nama folder ground truth kamu
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.lower().endswith(('.png', '.jpg'))])
    # Pilih file dari daftar
    selected_gt = st.selectbox("Pilih file Ground Truth yang sesuai:", gt_files)
    # Bangun path
    gt_path = os.path.join(gt_folder, selected_gt)
    # Evaluasi hasil segmentasi dibandingkan ground truth
    evaluate_segmentation_result(gt_path, labels_ws > 0)

    st.header("**ANALISIS KLASIFIKASI SINYAL HER2 DAN CEN17**")
    st.subheader("1. Plot channel tiap sinyal")
    her2_channel, cen17_channel = plot_chan_signal(im)
    st.subheader("2. Preprocessing : CLAHE")
    her2_clahe = apply_clahe_sig(her2_channel, label="HER2 (Red)", key_prefix="her2")
    cen17_clahe = apply_clahe_sig(cen17_channel, label="CEN17 (Green)", key_prefix="cen17")
    st.subheader("3. Stretching aim to get the signal")
    her2_stretched = stretch_channel(her2_channel, label="HER2", cmap="Reds", key_prefix="her2")
    cen17_stretched = stretch_channel(cen17_channel, label="CEN17", cmap="Greens", key_prefix="cen17")
    st.subheader("4. Deteksi Titik Sinyal dari Stretching")
    her2_coordsF, cen17_coordsF = detect_and_plot_signal_coords(im, her2_stretched, cen17_stretched, key_prefix="fish")
    st.subheader("5. Analisis Rasio HER2/CEN17 per Sel")
    df_ratio = analyze_her2_cen17_ratio(labels_ws, her2_coordsF, cen17_coordsF)
    st.subheader("6. Visualisasi Hasil Klasifikasi HER2/CEN17")
    visualize_her2_cen17_classification(im, labels_ws, her2_coordsF, cen17_coordsF, df_ratio)


with tab_dish:
    image_folder2 = "DISH"
    image_files2 = sorted([f for f in os.listdir(image_folder2) if f.lower().endswith(('.jpg', '.png'))])
    selected_file2 = st.selectbox("Pilih gambar DISH:", image_files2)
    image_path2 = os.path.join(image_folder2, selected_file2)
    im2 = cv2.imread(image_path2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    st.image(im2, caption="Citra DISH RGB Asli", width=500)
    
    ####--- PROSES --- ####
    st.subheader("1. Histogram Analysis: 3 channel RGB")
    rD, gD, bD = plot_rgb_histogram(im2, key_prefix="dish")   # untuk tab DISH
    st.subheader("2. Preprocessing: CLAHE")
    r_clahe = apply_clahe(rD, label="CLAHE on Red Channel (DISH)", key_prefix="dish_r")
    st.subheader("3. Segmentasi: Otsu Thresholding")
    binary_mask_D = apply_otsu_threshold(r_clahe, title="Threshold DISH", is_dish=True)
    # binary_mask = apply_otsu_threshold(b_clahe)
    st.subheader("4. Filtering dan Pelabelan")
    image_segmented_D, labelsD, nlabelsD = filter_and_label_cells(binary_mask_D, key_prefix="dish")
    st.subheader("5. Watershed Segmentation")
    labels_ws_D, boundaries_ws_D, overlay_img_D = watershed_segmentation(image_segmented_D, im2, key_prefix="dish")
    st.subheader("6. Evaluasi Citra terhadap Ground Truth")
    gt_folderD = "REF_DISH"  # ganti sesuai nama folder ground truth kamu
    gt_filesD = sorted([f for f in os.listdir(gt_folderD) if f.lower().endswith(('.png', '.jpg'))])
    # Pilih file dari daftar
    selected_gtD= st.selectbox("Pilih file Ground Truth DISH yang sesuai:", gt_filesD)
    # Bangun path
    gt_pathD = os.path.join(gt_folderD, selected_gtD)
    # Evaluasi hasil segmentasi dibandingkan ground truth
    evaluate_segmentation_result(gt_pathD, labels_ws_D > 0)

    st.header("**ANALISIS KLASIFIKASI SINYAL HER2 DAN CEN17 PADA CITRA DISH**")
    st.subheader("1. Plot channel tiap sinyal")
    her2_channelD, cen17_channelD = extract_and_plot_dish_signal_channels(im2, her2_mask_thresh=0.1)
    st.subheader("2. Stretching aim to get the signal")
    her2_stretchedD = stretch_channel(her2_channelD, label="HER2 DISH", cmap="Reds", key_prefix="her2D")
    cen17_stretchedD = stretch_channel(cen17_channelD, label="CEN17 DISH", cmap="Greens", key_prefix="cen17D")
    st.subheader("3. Deteksi Titik Sinyal dari Stretching DISH")
    her2_coordsD, cen17_coordsD = detect_and_plot_signal_coords(im2, her2_stretchedD, cen17_stretchedD, key_prefix="dish")
    st.subheader("4. Analisis Rasio HER2/CEN17 per Sel")
    df_ratioD = analyze_her2_cen17_ratio(labels_ws_D, her2_coordsD, cen17_coordsD)
    st.subheader("5. Visualisasi Hasil Klasifikasi HER2/CEN17")
    visualize_her2_cen17_classification(im2, labels_ws_D, her2_coordsD, cen17_coordsD, df_ratioD)


