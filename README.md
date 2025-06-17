# PCM_fish_dish
This repository contains a Streamlit-based biomedical image analysis app developed for final project in Pattern Classification and Matching (PCM). The app provides end-to-end segmentation and classification of HER2/CEN17 signals on FISH and DISH microscopic images, supporting digital pathology workflows.

🔬 Main Features:
✅ CLAHE enhancement and histogram analysis for RGB and grayscale channels

✅ Otsu thresholding with adaptive background handling (FISH vs DISH)

✅ Morphological filtering (small object removal, hole filling, contour smoothing)

✅ Watershed segmentation with interactive parameter tuning

✅ Evaluation metrics using IoU and Dice Similarity against ground truth

✅ HER2 & CEN17 signal detection based on pixel intensity range

✅ Per-cell ratio analysis and HER2 status classification (Positive / Equivocal / Negative)

✅ Interactive visualizations: bounding boxes, centroids, signal overlays

📂 FISH/        ← Raw FISH test images  
📂 DISH/        ← Raw DISH test images  
📂 REF_FISH/    ← Ground truth for evaluation  
📄 stpcm.py     ← Streamlit app  
📄 README.md    ← This file  

```bash
streamlit run stpcm.py
```

