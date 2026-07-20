from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "Data/train/ants_image/1693954099_46d4c20605.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)  # (512, 768, 3)，HWC型，需要dataformats参数

writer.add_image("test", img_array, 3, dataformats="HWC")
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i,)

writer.close()

print("STRAT")