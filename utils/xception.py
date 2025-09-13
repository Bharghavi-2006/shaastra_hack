import torch
from torchvision import transforms
from PIL import Image

# Define your model architecture (match how weights were trained)
from torchvision.models import xception  # (you may need a custom implementation, since torchvision may not have official Xception)

model = xception(pretrained=False)
# adapt final layer for binary classification (fake vs real)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # or whatever structure was used

model_path = './models/ai_detectors/xception-b5690688.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception often uses 299×299
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

def predict_fake(pil_img):
    x = transform(pil_img).unsqueeze(0)  # batch dim
    with torch.no_grad():
        logits = model(x)
        # depends on how you trained: sigmoid or softmax
        score = torch.sigmoid(logits).item()  # if final layer outputs one logit
        return score  # e.g., score near 1 = fake, near 0 = real
