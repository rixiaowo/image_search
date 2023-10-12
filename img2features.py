import torch
from torchvision import models, transforms
from PIL import Image
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()


def img2feature(img, input_dim=224):

    preprocess = transforms.Compose([
        transforms.Resize(input_dim),
        transforms.CenterCrop(input_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    if img.mode == 'RGBA':
        img = img.convert("RGB")
    if img.mode == 'L':
        img = img.convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        x = model(img)
    x = x.cpu()
    x = x / np.linalg.norm(x)
    return x


def main():
    img_path = '/home/liqj/AI/milvus_test/img2feature.py'
    t0 = time.time()
    res = img2feature(img_path)
    print(time.time() - t0, res.shape)


if __name__ == "__main__":
    main()
