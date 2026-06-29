import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# 1. Konversi Citra
# ─────────────────────────────────────────────

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_binary(img, thresh=127):
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


# ─────────────────────────────────────────────
# 2. Perbaikan Kualitas Citra
# ─────────────────────────────────────────────

def histogram_equalization(img):
    gray = to_grayscale(img)
    return cv2.equalizeHist(gray)


def contrast_stretching(img):
    gray = to_grayscale(img)
    mn, mx = gray.min(), gray.max()
    return ((gray - mn) / (mx - mn) * 255).astype(np.uint8)


def brightness_adjustment(img, beta=50):
    """Naikkan/turunkan kecerahan gambar. beta > 0 lebih terang, < 0 lebih gelap."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def sharpening(img):
    """Perkuat detail tepi menggunakan unsharp mask kernel 3×3."""
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


# ─────────────────────────────────────────────
# 3. Filtering
# ─────────────────────────────────────────────

def mean_filter(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))


def median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def gaussian_filter(img, ksize=5, sigma=1):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ─────────────────────────────────────────────
# 4. Deteksi Tepi
# ─────────────────────────────────────────────

def edge_canny(img, t1=100, t2=200):
    gray = to_grayscale(img)
    return cv2.Canny(gray, t1, t2)


def edge_sobel(img):
    gray = to_grayscale(img).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(np.sqrt(sx**2 + sy**2))


def edge_prewitt(img):
    """Deteksi tepi menggunakan operator Prewitt (kernel 3×3 horizontal & vertikal)."""
    gray = to_grayscale(img).astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float32)
    sx = cv2.filter2D(gray, -1, kx)
    sy = cv2.filter2D(gray, -1, ky)
    return cv2.convertScaleAbs(np.sqrt(sx**2 + sy**2))


# ─────────────────────────────────────────────
# 5. Segmentasi Citra
# ─────────────────────────────────────────────

def segment_kmeans(img, k=3):
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)


def segment_threshold(img, thresh=127):
    return to_binary(img, thresh)


def segment_watershed(img):
    gray = to_grayscale(img)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    result  = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [255, 0, 0]
    return result


# ─────────────────────────────────────────────
# Metrik Kualitas
# ─────────────────────────────────────────────

def compute_ssim(orig, result):
    """SSIM antara gambar original dan hasil (0–1, makin tinggi makin mirip)."""
    g1 = to_grayscale(orig)   if orig.ndim   == 3 else orig
    g2 = to_grayscale(result) if result.ndim == 3 else result
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
    score, _ = ssim_fn(g1, g2, full=True)
    return round(float(score), 4)


def compute_psnr(orig, result):
    """PSNR (dB) — makin tinggi makin baik (≥ 30 dB = kualitas baik)."""
    g1 = to_grayscale(orig)   if orig.ndim   == 3 else orig
    g2 = to_grayscale(result) if result.ndim == 3 else result
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
    val = cv2.PSNR(g1.astype(np.float64), g2.astype(np.float64))
    return round(val, 2)


# ─────────────────────────────────────────────
# PROCESSES — registry untuk GUI
# Semua lambda pastikan output RGB 3-channel
# ─────────────────────────────────────────────

PROCESSES = {
    # Konversi
    "Grayscale":
        lambda img: cv2.cvtColor(to_grayscale(img), cv2.COLOR_GRAY2RGB),
    "Binary":
        lambda img: cv2.cvtColor(to_binary(img), cv2.COLOR_GRAY2RGB),

    # Perbaikan Kualitas
    "Histogram Equalization":
        lambda img: cv2.cvtColor(histogram_equalization(img), cv2.COLOR_GRAY2RGB),
    "Contrast Stretching":
        lambda img: cv2.cvtColor(contrast_stretching(img), cv2.COLOR_GRAY2RGB),
    "Brightness Adjustment":
        brightness_adjustment,                          # output sudah RGB
    "Sharpening":
        sharpening,                                     # output sudah RGB

    # Filtering
    "Mean Filter":    mean_filter,
    "Median Filter":  median_filter,
    "Gaussian Filter": gaussian_filter,

    # Deteksi Tepi
    "Edge Detection (Canny)":
        lambda img: cv2.cvtColor(edge_canny(img), cv2.COLOR_GRAY2RGB),
    "Edge Detection (Sobel)":
        lambda img: cv2.cvtColor(edge_sobel(img), cv2.COLOR_GRAY2RGB),
    "Edge Detection (Prewitt)":
        lambda img: cv2.cvtColor(edge_prewitt(img), cv2.COLOR_GRAY2RGB),

    # Segmentasi
    "Segmentasi K-Means (k=3)": segment_kmeans,
    "Segmentasi K-Means (k=5)": lambda img: segment_kmeans(img, k=5),
    "Segmentasi Threshold":
        lambda img: cv2.cvtColor(segment_threshold(img), cv2.COLOR_GRAY2RGB),
    "Segmentasi Watershed":     segment_watershed,
}