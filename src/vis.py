import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm

from PIL import Image


def apply_colormap_on_image(
    org_img, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.5
):
    """
    Blend GradCAM heatmap with original image.
    """
    if isinstance(org_img, Image.Image):
        org_img = np.array(org_img)
    org_img = org_img.astype(np.uint8)

    activation_map = np.clip(activation_map, 0, 1)
    activation_map = (activation_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(activation_map, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if heatmap.shape[:2] != org_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (org_img.shape[1], org_img.shape[0]))

    overlay = (heatmap * alpha + org_img * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)


def visualize_gradcam(original_image, cams, prompt, score, save_path=None):
    """
    original_image: HxWx3 tensor or numpy array
    cams: list of 2D tensors (gradcam heatmaps)
    prompt: str
    score: float
    """
    col_labels = [
        "Overlay",
        "GradCAM Heatmap",
    ]
    row_labels = ["Positive Focus", "Negative Focus"]

    # Compute min/max for relevant colorbar
    all_heatmaps = np.stack(cams)
    vmin, vmax = all_heatmaps.min(), all_heatmaps.max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(
        f'Prompt: "{prompt}"  |  Similarity: {score:.2f}',
        fontsize=16,
        fontweight="bold",
        y=1.02,
        x=0.44,
    )

    for i in range(2):
        overlay = apply_colormap_on_image(original_image, cams[i], alpha=0.5)
        axes[i, 0].imshow(overlay, vmin=vmin, vmax=vmax)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cams[i], cmap="jet", vmin=vmin, vmax=vmax)
        axes[i, 1].axis("off")

        # Row labels
        axes[0, i].set_title(col_labels[i], fontsize=14, pad=5)
        # Col labels
        fig.text(
            -0.02,
            1 - (i + 0.5) / 2,
            row_labels[i],
            fontsize=14,
            rotation=90,
            va="center",
        )

    sm = cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Activation Intensity")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
