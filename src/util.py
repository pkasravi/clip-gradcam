from typing import Tuple

import torch
import torch.nn.functional as F

from gradcam import GradCAM


def get_clip_embeddings(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through CLIP and extract image and text embeddings.
    """
    model = model.eval().to(device)
    pixel_values = pixel_values.to(device).requires_grad_(True)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_loss=False,
    )
    return outputs.image_embeds, outputs.text_embeds


def compute_clip_gradcam(
    model,
    inputs,
    device,
    invert_loss=False,
    debug=True,
):
    """
    Compute GradCAM for CLIP ViT given an image and a text prompt.
    Returns a normalized heatmap aligned to the input image size.
    """
    gradcam = GradCAM(model)

    # Forward pass
    image_embeds, text_embeds = get_clip_embeddings(
        model,
        inputs["pixel_values"],
        inputs["input_ids"],
        inputs["attention_mask"],
        device,
    )

    # Sanity checks
    assert image_embeds.shape == text_embeds.shape, "Embed mismatch"
    assert (
        torch.isfinite(image_embeds).all() and torch.isfinite(text_embeds).all()
    ), "NaN in embeddings"

    similarity = F.cosine_similarity(image_embeds, text_embeds, dim=-1)
    print(f"[INFO] Similarity: {similarity.item():.4f}")

    loss = -similarity.mean() if invert_loss else similarity.mean()

    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)

    if debug:
        print("\n[DEBUG] --- GradCAM stats ---")
        print("image_embeds:", image_embeds.shape, "mean:", image_embeds.mean().item())
        print("text_embeds:", text_embeds.shape, "mean:", text_embeds.mean().item())
        print(
            "activations:",
            None if gradcam.activations is None else gradcam.activations.shape,
        )
        print(
            "gradients:", None if gradcam.gradients is None else gradcam.gradients.shape
        )
        if gradcam.gradients is not None:
            print(
                "grad min/max:",
                float(gradcam.gradients.min()),
                float(gradcam.gradients.max()),
            )
        if gradcam.activations is not None:
            act = gradcam.activations.detach()
            print("act min/max:", float(act.min()), float(act.max()))
        print("-----------------------------\n")

    cam = gradcam.generate_cam(inputs["pixel_values"].shape[2:], False)
    gradcam.remove_hooks()
    # NOTE: We assume a single sample hence dropping the batch dimension below
    return cam[0], similarity.detach().cpu().item()
