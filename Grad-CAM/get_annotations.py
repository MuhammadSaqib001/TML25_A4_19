from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load ResNet50 model with pre-trained ImageNet weights
# The model is set to evaluation mode to ensure consistent predictions.
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# ImageNet standard preprocessing transformations
# These transformations ensure the input images are in the correct format
# (resized, converted to tensor, and normalized) for the ResNet50 model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize with ImageNet mean
                         std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet standard deviation
])

# Define the paths to the 10 ImageNet images
# These are the images for which CAM visualizations will be generated.
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

# Map CAM type names to their respective classes from pytorch_grad_cam library
cam_types = {
    "gradcam": GradCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM
}

# Create an output directory to save the generated CAM images
# os.makedirs ensures the directory exists; exist_ok=True prevents error if it already does.
os.makedirs("cam_outputs", exist_ok=True)

# Iterate through each image to generate CAM visualizations
for name, img_path in image_paths.items():
    try:
        # Open and convert the image to RGB format
        image = Image.open(img_path).convert('RGB')

        # Apply transformations for the model input (including normalization)
        input_tensor = transform(image).unsqueeze(0)

        # Create a resized version of the original image for overlaying the heatmap
        # This ensures the image_np has the same dimensions as the heatmap (224x224).
        image_resized = transforms.Resize((224, 224))(image)
        # Convert the resized image to a NumPy array for visualization
        # Permute dimensions from (C, H, W) to (H, W, C) for image display libraries.
        image_np = transforms.ToTensor()(image_resized).permute(1, 2, 0).numpy()


        # Iterate through each CAM type (GradCAM, AblationCAM, ScoreCAM)
        for cam_name, cam_class in cam_types.items():
            # Initialize the CAM object
            # target_layers=[model.layer4[-1]] specifies the last convolutional layer
            # which is commonly used for generating CAMs as it captures high-level features.
            cam = cam_class(model=model, target_layers=[model.layer4[-1]])

            # Generate the grayscale heatmap
            # targets=[ClassifierOutputTarget(0)] means we generate CAM for the
            # class with the highest predicted probability (top-1 prediction).
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]

            # Overlay the heatmap on the original image
            # use_rgb=True ensures the image is displayed in RGB format.
            visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

            # Save the visualized image
            # The filename includes the image name and the CAM type for easy identification.
            Image.fromarray(visualization).save(f"cam_outputs/{name}_{cam_name}_annotations.jpg")
            print(f"Generated cam_outputs/{name}_{cam_name}_annotations.jpg")

    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}. Please ensure all images are in the script directory.")
    except Exception as e:
        print(f"An error occurred while processing {img_path} with {cam_name}: {e}")

print("\nCAM visualization generation complete. Check the 'cam_outputs' directory for results.")
