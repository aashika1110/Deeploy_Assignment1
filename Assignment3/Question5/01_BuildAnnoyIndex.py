import os
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

images_folder=r"C:\Users\aashi\Downloads\archive\PetImages\Dog"
images=os.listdir(images_folder)

weights=models.ResNet18_Weights.IMAGENET1K_V1
model=models.resnet18(weights=weights)
model.fc=nn.Identity()

print(model)

model.eval()

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

annoy_index=AnnoyIndex(512, 'angular')

for i in range(len(images)):
    image_path = os.path.join(images_folder, images[i])

    try:
        image = Image.open(image_path)
    except (OSError, Image.UnidentifiedImageError) as e:
        print(f"Skipping invalid image: {image_path} ({e})")
        continue

    input_tensor=transform(image).unsqueeze(0)

    if input_tensor.size()[1]==3:
        output_tensor=model(input_tensor)
        annoy_index.add_item(i, output_tensor[0])

        if i%100==0:
            print(f'Processed { i } images.')

annoy_index.build(10)
annoy_index.save('dog_index.ann')

