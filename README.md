Here’s the markdown in one code snippet:

```
# CIFAR-10 JPEG Compression and Adversarial Attack

This project applies JPEG compression and adversarial attacks using a custom ResNet-50 model on CIFAR-10 images.

## Setup

1. **Clone the repo**:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script to process CIFAR-10 images:
```bash
python main.py
```

- Compresses images using JPEG.
- Performs FGSM adversarial attacks.
- Saves altered images in `jpeg_altered/`.

Modify `starting_index` in the script to resume processing from a specific image:
```python
starting_index = 28311
```

## Visualize

Use `show_images()` and `show_difference()` to display and compare original, compressed, and adversarial images.

## Files

- **`main.py`**: Runs the JPEG compression and adversarial attack.
- **`CustomResNet.py`**: Defines the custom ResNet-50 model.
- **`jpeg_compression.py`**: JPEG compression functions.
- **`attack.py`**: Implements FGSM attack.
- **`jpeg_altered/`**: Directory for altered images.

