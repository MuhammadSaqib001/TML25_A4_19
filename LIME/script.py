import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pickle

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

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

def batch_predict(images):
    model.eval()
    batch = torch.stack([transform(Image.fromarray(img)).to(torch.float32) for img in images])
    with torch.no_grad():
        logits = model(batch)
    return logits.numpy()

explainer = lime_image.LimeImageExplainer()
param_dict = {}

for name, img_path in image_paths.items():
    img = Image.open(img_path).convert('RGB')
    np_img = np.array(img)

    explanation = explainer.explain_instance(
        np_img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    param_dict[name] = {
        'top_labels': 1,
        'hide_color': 0,
        'num_samples': 1000,
        'positive_only': True,
        'num_features': 5,
        'hide_rest': False
    }

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME: {name}")
    plt.axis("off")
    plt.savefig(f"cam_outputs/{name}_lime.png")
    plt.close()

with open("lime_parameters.pkl", "wb") as f:
    pickle.dump(param_dict, f)
