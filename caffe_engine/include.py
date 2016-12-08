"""
This file defines some general parameters and some commonly used functions.
"""

img_height  = 100 # output image height
img_width   = 100 # output image width

label_ref = ['NE', 'HA', 'AN', 'AF', 'DI', 'SA', 'SU']


def parse_label(filename):
    """
    Given the file name of an image, this function returns the label from
    0 - 6, which represents the siven facial emotions.
    """
    label = filename[4:6]
    angle = filename[6:8]
    if angle[0] != 'S':
        # Only keep frontal face
        return None
    if not label in label_ref:
        # print filename, label, 'unknown label'
        return None
    return label_ref.index(label)

def special_parse(filename):
    """
    Same as the parse_label, except the file name is in slightly different
    format.
    """
    label = filename.split('.')[-2][-2:]
    if label not in label_ref:
        return None
    return label_ref.index(label)

def crop_image(img, crop_ratio, center_offset=(0, 0)):
    """
    Crop a given image.
    """
    # crop_ratio: the larger the more area will be cropped
    h, w, _ = img.shape
    crop_len = int(np.round(w / 2 * crop_ratio))
    mid_h = int(np.round(h / 2)) + center_offset[0]
    mid_w = int(np.round(w / 2)) + center_offset[1]
    img = img[mid_h - crop_len:mid_h + crop_len + 1,
            mid_w - crop_len:mid_w + crop_len + 1, :]
    return img


def rgb2hsi(img):
    """
    Convert an RGB image to HSI image.
    """
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    theta = np.arccos((1/2*(R-G)+(R-B)) / (((R-G)**2 + (R-B)*(G-B))**(0.5)))
    H = theta * (B <= G) + (360.0 - theta) * (B > G)
    S = 1 - (3/(R+G+B)) * np.amin(img, axis=2)
    I = (R+G+B) / 3
    hsi = np.array([H, S, I])
    return hsi

def hsi2rgb(img):
    """
    Convert an HSI image to RGB image.
    """
    pass
    # H = img[..., 0]
    # S = img[..., 1]
    # I = img[..., 2]
    # R = I + 2 * I * S
    # G = I - I * S
    # B = I - I * S


def equalize_rgb(img):
    """
    Perform histogram equalization on an RGB image, this will convert the image
    to HSI then perform histogram equalization on the I channel, then convert
    the HSI image back to RGB.
    """
    pass
    # HSI_img = rgb2hsi(img)
    # equ_I = cv2.equalizeHist(HSI_img[..., 2])
