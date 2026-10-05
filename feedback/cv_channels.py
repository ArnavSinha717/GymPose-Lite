"""Classical CV preprocessing channels.

Computes Canny edges, HSV skin segmentation, and Sobel gradient magnitude
from an RGB image. These are used as additional input channels to the model
(6-channel input: RGB + edges + skin + gradient) and displayed as panels
in the demo to showcase image processing concepts.
"""

import numpy as np
import cv2


def compute_cv_channels(img_rgb):
    """Compute classical CV preprocessing channels from an RGB image.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image

    Returns:
        edges: (H, W) float32, Canny edge map normalized [0, 1]
        skin:  (H, W) float32, skin segmentation mask {0, 1}
        grad:  (H, W) float32, Sobel gradient magnitude normalized [0, 1]
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

    # Skin detection via HSV thresholding
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    skin1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
    skin = ((skin1 | skin2) / 255.0).astype(np.float32)

    # Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = grad / (grad.max() + 1e-8)

    return edges, skin, grad


def build_6ch_input(img_norm, edges, skin, grad):
    """Stack normalized RGB with CV channels into 6-channel input.

    Args:
        img_norm: (H, W, 3) float32, ImageNet mean/std normalized
        edges, skin, grad: (H, W) float32 from compute_cv_channels

    Returns:
        (H, W, 6) float32 array ready for torch conversion
    """
    return np.concatenate([img_norm, np.stack([edges, skin, grad], axis=-1)], axis=-1)


def cv_channels_to_display(edges, skin, grad):
    """Convert raw CV channel arrays to displayable RGB images.

    Args:
        edges, skin, grad: (H, W) float32 from compute_cv_channels

    Returns:
        edges_vis: (H, W, 3) uint8 — white edges on black
        skin_vis:  (H, W, 3) uint8 — green-tinted skin mask
        grad_vis:  (H, W, 3) uint8 — JET colormapped gradient
    """
    # Edges: white on black
    edges_vis = (edges * 255).astype(np.uint8)
    edges_vis = cv2.cvtColor(edges_vis, cv2.COLOR_GRAY2RGB)

    # Skin: green-tinted mask
    skin_uint8 = (skin * 255).astype(np.uint8)
    skin_vis = np.zeros((*skin.shape, 3), dtype=np.uint8)
    skin_vis[:, :, 1] = skin_uint8  # Green channel

    # Gradient: JET colormap
    grad_uint8 = (grad * 255).astype(np.uint8)
    grad_vis = cv2.applyColorMap(grad_uint8, cv2.COLORMAP_JET)
    grad_vis = cv2.cvtColor(grad_vis, cv2.COLOR_BGR2RGB)

    return edges_vis, skin_vis, grad_vis
