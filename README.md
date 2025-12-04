# Grad-CAM for CLIP

This repository demonstrates how to apply Grad-CAM to CLIP models for visualizing image-text similarity.

For a detailed walkthrough, see the accompanying blog post: [**What do VLMs look at? Visualizing CLIP with GradCAM**](todo)

---

## Setup

1. **Install dependencies using uv**

```bash
uv sync
```

2. **Optional:** Verify that the environment works

```bash
uv run python -c "import torch, PIL, transformers; print('Environment OK')"
```

---

## Running the demo

1. **Register the uv environment as a Jupyter kernel**

```bash
uv run python -m ipykernel install --user --name=clip-gradcam --display-name "Python (clip-gradcam)"
```

This allows you to select the `Python (clip-gradcam)` kernel inside Jupyter notebooks.

2. **Launch the demo notebook**

```bash
uv run jupyter lab notebooks/demo.ipynb
```

* The notebook will use the `clip-gradcam` kernel
* Load images from the `data/` folder and visualize Grad-CAM outputs

**Tip:** If you prefer the classic notebook interface:

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

---

## Repository structure

```
clip-gradcam/
├── src/           # Core code: gradcam.py, vis.py, util.py
├── notebooks/     # Demo notebook(s)
├── images/        # Sample images for visualization
├── pyproject.toml # Project dependencies
├── uv.lock        # Locked dependency versions
└── README.md
```

---

## References

* Blog post: [What do VLMs look at? Visualizing CLIP with GradCAM](todo)
* CLIP model: [Hugging Face CLIP](https://huggingface.co/models?filter=clip)
* Grad-CAM original paper: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
