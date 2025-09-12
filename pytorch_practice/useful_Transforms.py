from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')

img_path = 'dataset/train/pytorch/pytorch.jpg'
img = Image.open(img_path)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor, 1)

#Normalize
trans_norm = transforms.Normalize([1, 2, 4], [5, 10, 20])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm, 2)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image('Resize', img_resize, 1)

# Compose()
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([
    trans_resize_2,
    trans_totensor
])
img_resize_2 = trans_compose(img)
writer.add_image('Resize2', img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)


writer.close()
