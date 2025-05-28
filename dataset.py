import os
import random
from PIL import Image, ImageDraw
from torchvision import datasets, transforms

def create_shape_mask(shape, size, margin=16, shape_scale_range=(0.3, 1.0), position_jitter=0.5):
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    
    # random scale
    scale = random.uniform(*shape_scale_range)
    shape_size = int(size * scale)
    
    # random shift
    max_offset = int(size * position_jitter)
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    
    # centered with offset
    center_x = size // 2 + offset_x
    center_y = size // 2 + offset_y
    
    left = max(center_x - shape_size // 2, 0)
    top = max(center_y - shape_size // 2, 0)
    right = min(center_x + shape_size // 2, size)
    bottom = min(center_y + shape_size // 2, size)
    
    if shape == 'circle':
        draw.ellipse((left, top, right, bottom), fill=255)
    elif shape == 'square':
        draw.rectangle((left, top, right, bottom), fill=255)
    elif shape == 'triangle':
        draw.polygon([
            (center_x, top),
            (left, bottom),
            (right, bottom)
        ], fill=255)
    return mask

def apply_mask(image, mask, padding=0):
    image = image.resize(mask.size)
    
    shape_bbox = mask.getbbox()
    if shape_bbox:
        shape_width = shape_bbox[2] - shape_bbox[0]
        shape_height = shape_bbox[3] - shape_bbox[1]
        
        filler_resized = image.resize((shape_width, shape_height))
    
        background = Image.new('RGB', mask.size, (0, 0, 0))
        background.paste(filler_resized, (shape_bbox[0], shape_bbox[1]))
    else:
        background = image.copy()
    
    masked_image = Image.composite(background, Image.new('RGB', mask.size, (0, 0, 0)), mask)
    
    if padding > 0:
        new_size = (mask.size[0] + 2 * padding, mask.size[1] + 2 * padding)
        padded_background = Image.new('RGB', new_size, (0, 0, 0))
        padded_background.paste(masked_image, (padding, padding))
        return padded_background
    else:
        return masked_image

def create_pattern_background(size, pattern_type='stripes', colors=((200, 200, 200), (255, 255, 255)), stripe_width=10.0, dot_radius=5.0, spacing=20.0):
    background = Image.new('RGB', size, colors[1])
    draw = ImageDraw.Draw(background)
    
    if pattern_type == 'stripes':
        x = 0.0
        while x < size[0]:
            x_end = min(x + stripe_width, size[0])
            draw.rectangle([x, 0, x_end - 1, size[1]], fill=colors[0])
            x += stripe_width * 2
    elif pattern_type == 'horizontal_stripes':
        y = 0.0
        while y < size[1]:
            y_end = min(y + stripe_width, size[1])
            draw.rectangle([0, y, size[0], y_end - 1], fill=colors[0])
            y += stripe_width * 2
    elif pattern_type == 'dots':
        y = 0.0
        while y < size[1]:
            x = 0.0
            while x < size[0]:
                bbox = [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius]
                draw.ellipse(bbox, fill=colors[0])
                x += spacing
            y += spacing
    return background

def load_images_from_folder(folder_path, max_images=100):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    sampled_files = random.sample(image_files, min(len(image_files), max_images))
    images = [Image.open(f).convert('RGB') for f in sampled_files]
    return images

def load_dataset(dataset_type, class_labels, max_images=100, image_size=256, pattern_settings=None):
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
    
    elif dataset_type == 'pattern':
        for label in class_labels:
            pattern_config = pattern_settings[label]
            pattern_img = create_pattern_background((image_size, image_size), **pattern_config)
            images_by_class[label] = [pattern_img] * max_images
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return images_by_class

def generate_dataset(base_images_by_class, shape_class_map, output_dir, size=256, margin=16, padding=0, num_samples=100, shape_scale_range = (0.3, 1.0), position_jitter = 0.5):
    for shape, class_labels in shape_class_map.items():
        shape_dir = os.path.join(output_dir, shape)
        os.makedirs(shape_dir, exist_ok=True)
        
        all_images = []
        for class_label in class_labels:
            all_images.extend(base_images_by_class[class_label])
        
        if not all_images:
            raise ValueError(f"No images found for shape '{shape}' and labels {class_labels}")
        
        for i in range(num_samples):
            filler_img = random.choice(all_images)
            mask = create_shape_mask(shape, size, margin, shape_scale_range=shape_scale_range, position_jitter=position_jitter)
            filled = apply_mask(filler_img, mask, padding)
            filled.save(os.path.join(shape_dir, f"{shape}_{i}.png"))
