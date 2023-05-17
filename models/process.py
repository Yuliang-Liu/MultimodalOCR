import PIL
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