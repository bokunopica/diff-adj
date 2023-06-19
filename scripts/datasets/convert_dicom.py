import pydicom
import matplotlib.pyplot as plt
import scipy.misc
 
in_path = './00C1E256-E3F7-431D-BCB6-DD1EF09E7DFE/fd0beb35-2d21-4bb8-a9d8-91effa9b5e6d_00001.dcm'
out_path = './output.jpg'
ds = pydicom.read_file(in_path)  #读取.dcm文件
img = ds.pixel_array  # 提取图像信息
print(img.shape)
plt.imshow(img)
plt.show()
scipy.misc.imsave(out_path,img)  #
