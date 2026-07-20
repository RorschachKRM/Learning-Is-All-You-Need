from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter


img_path = "Data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
print(img.size)  # (768, 512)

image_cv = cv2.imread(img_path)
print(type(image_cv)) 


writer = SummaryWriter("logs")

"""
🌟🌟🌟实例化 transforms 模块中的 ToTensor 类
    ToTensor 是一个类(class)
    () 调用它的构造函数(__init__)，创建一个实例
    tensor_trans 是一个变量，保存了这个实例的引用
"""
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
print(img_tensor)
print(type(img_tensor))  # <class 'torch.Tensor'>
print(img_tensor.shape)  # torch.Size([3, 512, 768])

writer.add_image("Tensor_Image", img_tensor)

writer.close()





