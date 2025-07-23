from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image_paths = {
    'West_Highland_white_terrier': 'n02098286_West_Highland_white_terrier.JPEG',
    'American_coot': 'n02018207_American_coot.JPEG',
    'racer': 'n04037443_racer.JPEG',
    'flamingo': 'n02007558_flamingo.JPEG',
    'kite': 'n01608432_kite.JPEG',
    'goldfish': 'n01443537_goldfish.JPEG',
    'tiger_shark': 'n01491361_tiger_shark.JPEG',
    'vulture': 'n01616318_vulture.JPEG',
    'common_iguana': 'n01677366_common_iguana.JPEG',
    'orange': 'n07747607_orange.JPEG'
}

cam_types = {
    "gradcam": GradCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM
}

# Create output directory
os.makedirs("cam_outputs", exist_ok=True)

for name, img_path in image_paths.items():
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    for cam_name, cam_class in cam_types.items():
        cam = cam_class(model=model, target_layers=[model.layer4[-1]])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]

        # Resize original image to 224x224 to match the CAM heatmap size
        image_resized = image.resize((224, 224))
        image_np = transforms.ToTensor()(image_resized).permute(1, 2, 0).numpy()

        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        Image.fromarray(visualization).save(f"cam_outputs/{name}_{cam_name}.jpg")
