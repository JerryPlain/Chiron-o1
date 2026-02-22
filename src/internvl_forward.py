import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Build the image normalization pipeline expected by InternVL vision encoder.
    """
    # [A] Convert to RGB (if needed) -> resize -> tensor -> ImageNet normalization.
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Pick the tiling ratio (w_blocks, h_blocks) closest to the input aspect ratio.
    """
    # [A] Minimize aspect-ratio distance.
    # [B] Tie-break with area-aware heuristic to keep enough visual coverage.
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically split an image into multiple square patches for InternVL.

    Steps:
    - Find best block layout based on aspect ratio
    - Resize image to layout-aligned canvas
    - Crop into image_size x image_size tiles
    - Optionally append one thumbnail tile
    """
    # ==========================================================
    # [A] Compute original aspect ratio and candidate tiling layouts.
    # ==========================================================
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Enumerate valid (w_blocks, h_blocks) where number of tiles is in [min_num, max_num].
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Pick the ratio whose shape is closest to the original image.
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # ==========================================================
    # [B] Resize image to the selected tiling canvas.
    # ==========================================================
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))

    # ==========================================================
    # [C] Crop each tile and return patch list.
    # ==========================================================
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Extract one patch from the resized canvas.
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks

    # Optional global thumbnail gives a full-image view in addition to local tiles.
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    Load one image file and convert it to a stacked tensor of InternVL patches.
    """
    # [A] Read image and build preprocess transform.
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    # [B] Dynamic tiling + per-tile normalization.
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    # [C] Shape: [num_patches, 3, input_size, input_size]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def internvl_forward(model, processor, prompt, img_paths, temperature=0.9):
    """
    Run InternVL chat inference with one or multiple images.

    Behavior:
    - Each image is split into patches and converted to bf16 CUDA tensors
    - Single-image path calls model.chat without num_patches_list
    - Multi-image path concatenates patches and passes num_patches_list
    """
    # ==========================================================
    # [A] Preprocess all images into patch tensors.
    # ==========================================================
    all_pixel_values = []
    num_patches_list = []
    
    for img_path in img_paths:
        # Convert each image to patch tensor and move to GPU/bfloat16 for inference.
        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        all_pixel_values.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))

    # ==========================================================
    # [B] Single-image inference path.
    # ==========================================================
    if len(img_paths) == 1:
        pixel_values = all_pixel_values[0]
        generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=temperature)  
        response = model.chat(processor, pixel_values, prompt, generation_config)
    else:
        # ==========================================================
        # [C] Multi-image inference path.
        # Concatenate all patches and provide per-image patch counts.
        # ==========================================================
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        else:
            return "Error: No images provided"
        
        generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=temperature)
        response = model.chat(processor, pixel_values, prompt, generation_config,
                            num_patches_list=num_patches_list)
    
    return response
