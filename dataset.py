import os
import random
from PIL import Image, ImageDraw
from torchvision import datasets, transforms

def create_shape_mask(shape, size, margin=16):
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = margin, margin, size - margin, size - margin
    if shape == 'circle':
        draw.ellipse((left, top, right, bottom), fill=255)
    elif shape == 'square':
        draw.rectangle((left, top, right, bottom), fill=255)
    elif shape == 'triangle':
        draw.polygon([(size//2, top), (left, bottom), (right, bottom)], fill=255)
    return mask

def apply_mask(image, mask, padding=0):
    image = image.resize(mask.size)
    background = Image.new('RGB', mask.size, (255, 255, 255))
    masked_image = Image.composite(image, background, mask)
    
    if padding > 0:
        new_size = (mask.size[0] + 2 * padding, mask.size[1] + 2 * padding)
        padded_background = Image.new('RGB', new_size, (255, 255, 255))
        padded_background.paste(masked_image, (padding, padding))
        return padded_background
    else:
        return masked_image

def load_images_from_folder(folder_path, max_images=100):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    sampled_files = random.sample(image_files, min(len(image_files), max_images))
    images = [Image.open(f).convert('RGB') for f in sampled_files]
    return images

def load_dataset(dataset_type, class_labels, max_images=100, image_size=256):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    images_by_class = {}
    
    if dataset_type == 'cifar':
        cifar = datasets.CIFAR10(root='./data', train=True, download=True)
        for label in class_labels:
            images_by_class[label] = [img for img, lbl in cifar if lbl == label][:max_images]
    
    elif dataset_type == 'mnist':
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        for label in class_labels:
            images_by_class[label] = [img.convert('RGB') for img, lbl in mnist if lbl == label][:max_images]
    
    elif dataset_type == 'custom':
        for label, folder_path in class_labels.items():
            images_by_class[label] = load_images_from_folder(folder_path, max_images)
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return images_by_class

def generate_dataset(base_images_by_class, shape_class_map, output_dir, size=256, margin=16, padding=0):
    num_images = min(len(images) for images in base_images_by_class.values())
    
    for shape, class_labels in shape_class_map.items():
        shape_dir = os.path.join(output_dir, shape)
        os.makedirs(shape_dir, exist_ok=True)
        
        all_images = []
        for class_label in class_labels:
            all_images.extend(base_images_by_class[class_label])
        
        for i in range(num_images):
            filler_img = random.choice(all_images)
            mask = create_shape_mask(shape, size, margin)
            filled = apply_mask(filler_img, mask, padding)
            filled.save(os.path.join(shape_dir, f"{shape}_{i}.png"))