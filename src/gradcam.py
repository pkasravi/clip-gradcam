import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:

    def __init__(self, clip_model, target_layer=None):
        self.model = clip_model
        self.activations = None
        self.gradients = None
        self.handles = []

        if target_layer is None:
            target_layer = clip_model.vision_model.encoder.layers[-1].layer_norm1

        self.target_layer = target_layer
        print(
            f"[INFO] Target layer for GradCAM: {self.target_layer.__class__.__name__}"
        )
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks to capture intermediate outputs (activations)
        and their gradients (∂similarity/∂activations).
        """

        def forward_hook(module, input, output):
            # Store activations from the forward pass.
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    f"Unexpected output type from {module}: {type(output)}"
                )

            self.activations = output.detach()
            print(f"[HOOK] Recorded activations {tuple(self.activations.shape)}")

        def backward_hook(module, grad_input, grad_output):
            # Store gradients flowing back from the similarity loss.
            grad = (
                grad_output[0]
                if isinstance(grad_output, (tuple, list))
                else grad_output
            )
            if grad is not None:
                self.gradients = grad.detach()
                print(f"[HOOK] Recorded gradients {tuple(self.gradients.shape)}")
            else:
                raise ValueError(f"Nonetype gradients from {module}")

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        try:
            self.handles.append(
                self.target_layer.register_full_backward_hook(backward_hook)
            )
        except Exception:
            print("[WARN] register_full_backward_hook failed, using fallback")
            self.handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _precheck(self, debug=True):
        assert (
            self.activations is not None
        ), "No activations recorded. Run forward first."
        assert self.gradients is not None, "No gradients recorded. Run backward first."
        assert (
            self.activations.shape == self.gradients.shape
        ), f"Shape mismatch: acts {self.activations.shape}, grads {self.gradients.shape}"

        if debug:
            print(f"Mean gradient: {self.gradients.mean().item()}")
            print(f"Min gradient: {self.gradients.min().item()}")
            print(f"Max gradient: {self.gradients.max().item()}")
        # Note:
        #   - If all zeros: backward hook failed or layer didn’t receive grads.
        #   - Extremely large values → gradient explosion.
        #   - Normal range (1e-3 – 1e-1) means signal is flowing correctly.

    def generate_cam(self, input_shape, debug=True):
        """
        Convert recorded activations + gradients into a GradCAM heatmap.
        """
        self._precheck(debug)
        # acts = self.activations.detach()
        # grads = self.gradients.detach()

        # Drop CLS token
        acts, grads = self.activations[:, 1:, :], self.gradients[:, 1:, :]

        # Compute weighted sum of patch activations
        weights = grads.mean(dim=1).unsqueeze(1)
        cam = (weights * acts).sum(dim=2)

        num_patches = cam.shape[1]
        grid = int(np.sqrt(num_patches))
        assert (
            grid * grid == num_patches
        ), f"num_patches {num_patches} not perfect square"

        # Normalize and upscale to image size
        cam = cam.reshape(cam.shape[0], grid, grid)
        cam = F.relu(cam)
        cam_min, cam_max = torch.aminmax(cam)
        assert torch.isfinite(cam_min) and torch.isfinite(cam_max), "CAM has NaNs/Infs"
        cam = (cam - cam_min) / (cam_max - cam_min).clamp(min=1e-8)
        cam = F.interpolate(
            cam.unsqueeze(1), size=input_shape, mode="bilinear", align_corners=False
        )
        cam = cam.squeeze(1)
        return cam.cpu().numpy()

    def __del__(self):
        self.remove_hooks()
