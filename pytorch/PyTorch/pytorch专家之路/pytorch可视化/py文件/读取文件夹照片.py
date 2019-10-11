from torchvision import datasets, transforms, models
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

image_path = r'C:\Users\beidongjiedeguang\OneDrive\a机器学习\pytorch专家之路\pytorch可视化\data'

transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Resize
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

images = datasets.ImageFolder(root=image_path,transform=transform)
dataloaders = Data.DataLoader(images, batch_size=1, shuffle= 1)
a,label = next(iter(dataloaders))

plt.imshow(a.numpy().squeeze(0).transpose(1,2,0).clip(0,1))
plt.show()