import PIL
from PIL import Image,ImageOps
def pad_image(image, target_size):
 
    """
    :param image: input image
    :param target_size: a tuple (num,num)
    :return: new image
    """
 
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
 
    scale = min(w / iw, h / ih)  # 转换的最小比例
 
    # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
    nw = int(iw * scale+0.5)
    nh = int(ih * scale+0.5)

    w += 128
    h += 128
 

    image = image.resize((nw, nh), PIL.Image.BICUBIC)  # 更改图像尺寸，双立法插值效果很好
    #image.show()
    new_image = PIL.Image.new('RGB', (w, h), (0, 0, 0))  # 生成黑色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式

    return new_image
def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # 宽度大于高度，以宽度为基准进行 resize
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        # 高度大于宽度，以高度为基准进行 resize
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    image = image.resize((new_width, new_height))
    width_diff = target_size[0] - image.size[0]
    height_diff = target_size[1] - image.size[1]
    left_padding = 0
    top_padding = 0
    right_padding = width_diff - left_padding
    bottom_padding = height_diff - top_padding
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=0)
    return padded_image