import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import json
import urllib.request
import torch.nn.functional as F

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.base_model.fc.in_features),
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.base_model(x)

# Instantiate the custom model
custom_model = CustomResNet(num_classes=10)
custom_model.eval()  # Set the model to evaluation mode

# RGB -> YCbCr
def rgb_to_ycbcr(image):
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.], [112., -93.786, -18.214]],
        dtype=np.float32).T / 255
    shift = [16., 128., 128.]

    image = tf.cast(image, tf.float32)
    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result

def rgb_to_ycbcr_jpeg(image):
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T
    shift = [0., 128., 128.]

    image = tf.cast(image, tf.float32)
    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result

# Chroma subsampling
def downsampling_420(image):
    y, cb, cr = tf.split(image, 3, axis=3)
    cb = tf.nn.avg_pool(cb, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    cr = tf.nn.avg_pool(cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return (tf.squeeze(y, axis=-1), tf.squeeze(cb, axis=-1), tf.squeeze(cr, axis=-1))

# Block splitting
def image_to_patches(image):
    k = 8
    height, width = image.shape.as_list()[1:3]
    batch_size = tf.shape(image)[0]
    image_reshaped = tf.reshape(image, [batch_size, height // k, k, -1, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, -1, k, k])

# DCT
def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = scale * tf.tensordot(image, tensor, axes=2)
    result.set_shape(image.shape.as_list())
    return result

# Quantization
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T

def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image

def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image

# Dequantization
def y_dequantize(image, factor=1):
    return image * (y_table * factor)

def c_dequantize(image, factor=1):
    return image * (c_table * factor)

# Inverse DCT
def idct_8x8(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
    result = 0.25 * tf.tensordot(image, tensor, axes=2) + 128
    result.set_shape(image.shape.as_list())
    return result

# Block joining
def patches_to_image(patches, height, width):
    k = 8
    batch_size = tf.shape(patches)[0]
    image_reshaped = tf.reshape(patches, [batch_size, height // k, width // k, k, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, height, width])

# Chroma upsampling
def upsampling_420(y, cb, cr):
    def repeat(x, k=2):
        height, width = x.shape.as_list()[1:3]
        x = tf.expand_dims(x, -1)
        x = tf.tile(x, [1, 1, k, k])
        x = tf.reshape(x, [-1, height * k, width * k])
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return tf.stack((y, cb, cr), axis=-1)

# YCbCr -> RGB
def ycbcr_to_rgb_jpeg(image):
    matrix = np.array(
        [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
        dtype=np.float32).T
    shift = [0, -128, -128]

    result = tf.tensordot(image + shift, matrix, axes=1)
    result.set_shape(image.shape.as_list())
    return result

def diff_round(x):
    return tf.round(x) + (x - tf.round(x))**3

def round_only_at_0(x):
    cond = tf.cast(tf.abs(x) < 0.5, tf.float32)
    return cond * (x ** 3) + (1 - cond) * x

def quality_to_factor(quality):
    return tf.cond(
        tf.less(quality, 50), lambda: 5000. / quality,
        lambda: 200. - quality * 2) / 100

def jpeg_compress_decompress(image_tensor, downsample_c=True, rounding=diff_round, quality=75):
    factor = quality_to_factor(quality)
    image_np = (image_tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)  # Detach before converting to numpy
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)

    height, width = image.shape[0], image.shape[1]
    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left

        image = tf.pad(image, [[top, bottom], [left, right], [0, 0]], 'SYMMETRIC')

    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = rgb_to_ycbcr_jpeg(image)
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = tf.split(image, 3, axis=-1)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(comp, rounding, factor)
        components[k] = comp

    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, height // 2, width // 2)
            else:
                comp = patches_to_image(comp, height, width)
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = tf.stack((y, cb, cr), axis=-1)
    image = ycbcr_to_rgb_jpeg(image)

    image = tf.squeeze(image, axis=0)  # Remove batch dimension

    if orig_height != height or orig_width != width:
        image = image[:orig_height, :orig_width]

    image = tf.clip_by_value(image, 0, 255)
    # Convert back to torch tensor and add batch dimension
    decompressed_image_tensor = torch.from_numpy(image.numpy().astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    return decompressed_image_tensor

def jpeg_compress_decompress_tf(image_tensor, quality):
    image_np = (image_tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)  # Detach before converting to numpy
    image_encoded = tf.image.encode_jpeg(image_np, quality=quality)
    image_decoded = tf.image.decode_jpeg(image_encoded)
    decompressed_image_tensor = torch.from_numpy(image_decoded.numpy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return decompressed_image_tensor

def show_images(original, compressed, index):
    original_np = original.squeeze().permute(1, 2, 0).numpy()
    compressed_np = compressed.squeeze().permute(1, 2, 0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(original_np)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(compressed_np)
    ax2.set_title("Compressed Image")
    ax2.axis('off')

    plt.savefig(f"compressed_image_{index}.png")
    plt.show()

from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)
import torch
import torchvision.transforms as transforms
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import urllib.request
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

def ensemble_loss(input_tensor, model, target, quality_levels):
    losses = []
    gradients = []

    # Original loss without any JPEG compression
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    loss = F.nll_loss(output, target)
    loss.backward()
    original_grad = input_tensor.grad.data.clone()
    losses.append(loss.item())
    gradients.append(original_grad)
    input_tensor.requires_grad_(False)

    # Losses and gradients with JPEG compression at different quality levels
    for q in quality_levels:
        compressed_tensor = jpeg_compress_decompress(input_tensor, quality=q)
        compressed_tensor.requires_grad_(True)
        output = model(compressed_tensor)
        loss = F.nll_loss(output, target)
        loss.backward()
        grad = compressed_tensor.grad.data.clone()
        losses.append(loss.item())
        gradients.append(grad)
        compressed_tensor.requires_grad_(False)

    # Compute combined gradient using the formula
    exp_losses = torch.exp(torch.tensor(losses))
    weights = 1 - exp_losses / torch.sum(exp_losses)
    combined_grad = sum(w * g for w, g in zip(weights, gradients))

    return combined_grad



def quantize_to_8_bits(tensor):
    tensor = torch.clamp(tensor * 255, 0, 255).round() / 255
    return tensor



def attack(tensor, net, eps=3/255, n_iter=10):
    """Run the Fast Sign Gradient Method (FSGM) attack.

    Parameters
    ----------
    tensor : torch.Tensor
        The input image of shape `(1, 3, 224, 224)`.

    net : torch.nn.Module
        Classifier network.

    eps : float
        Determines how much we modify the image in a single iteration.

    n_iter : int
        Maximum number of iterations we run the attack for.

    d : float
        Maximum allowable perturbation.

    Returns
    -------
    new_tensor : torch.Tensor
        New image that is a modification of the input image that "fools"
        the classifier.
    """
    original_image = tensor.detach().clone()
    new_tensor = tensor.detach().clone()
    orig_prediction = net(tensor).argmax()
    target = torch.LongTensor([orig_prediction])

    quality_levels = [25, 50, 75]

    d = eps * n_iter

    for i in range(n_iter):
        net.zero_grad()
        combined_grad = ensemble_loss(new_tensor, net, target, quality_levels)
        new_tensor = torch.clamp(new_tensor + eps * combined_grad.sign(), 0, 1)

        # Apply clipping to ensure the perturbation is within the specified distance d
        perturbation = torch.clamp(new_tensor - original_image, min=-d, max=d)
        adversarial_image = torch.clamp(original_image + perturbation, min=0, max=1)

       

        # Check classification outputs
        output_adv = net(adversarial_image)
        output_orig = net(original_image)

        # Calculate the objective function (difference in classification outputs)
        '''loss_adv = F.cross_entropy(output_adv, target)
        loss_orig = F.cross_entropy(output_orig, target)

        # Optimize the objective function
        if loss_adv.item() < loss_orig.item():
            print(f"We fooled the network after {i+1} iterations!")
            print(f"New prediction: {output_adv.argmax().item()}")
            break

        new_tensor = adversarial_image'''

    return adversarial_image, orig_prediction.item(), output_adv.argmax().item()




def save_image(tensor, path):
    """Save a tensor as an image file."""
    pil_image = transforms.ToPILImage()(tensor.squeeze(0))
    pil_image.save(path)


def show_difference(original, perturbed, index, amplification=80):
    original_np = original.squeeze().permute(1, 2, 0).numpy()
    perturbed_np = perturbed.squeeze().permute(1, 2, 0).numpy()
    difference_np = (perturbed - original).squeeze().permute(1, 2, 0).numpy()
    difference_np_amplified = difference_np * amplification

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(original_np)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(perturbed_np)
    ax2.set_title("Perturbed Image")
    ax2.axis('off')

    ax3.imshow(difference_np_amplified)
    ax3.set_title(f"Difference Image (x{amplification})")
    ax3.axis('off')

    plt.savefig(f"difference_image_{index}.png")
    plt.show()

if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize(256),  # ResNet-50 expects 224x224 input size
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-10 dataset
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(cifar10, batch_size=1, shuffle=False)  # Set batch_size to 1 to process one image at a time

    # Create directory for altered images if it doesn't exist
    os.makedirs('jpeg_altered', exist_ok=True)

    # CIFAR-10 class labels
    cifar10_labels = cifar10.classes

   # Adjust the starting index
starting_index = 28311

# Process all images in the CIFAR-10 dataset starting from the 561st image
for batch_idx, (image_tensor_batch, labels) in enumerate(dataloader):
    for i, (image_tensor, label) in enumerate(zip(image_tensor_batch, labels)):
        # Calculate the current image index
        image_index = batch_idx * len(image_tensor_batch) + i
        
        # Skip images until we reach the starting index
        if image_index < starting_index:
            continue

        print(f"Processing image {image_index}")
        actual_label = cifar10_labels[label]
        print(f"Actual class label: {actual_label}")

        # Compress and decompress the image
        compressed_image_tensor = jpeg_compress_decompress_tf(image_tensor.unsqueeze(0), quality=25)
        
        # Perform FGSM attack
        attacked_image_tensor, orig_pred, new_pred = attack(compressed_image_tensor, custom_model, eps=3/255, n_iter=10)
        print(f"Original class index: {orig_pred}, New class index: {new_pred}")
        print(f"Original class label: {cifar10_labels[orig_pred]}, New class label: {cifar10_labels[new_pred]}")

        # Save the altered image with the new index
        save_image(attacked_image_tensor, f'jpeg_altered/{image_index}.png')
