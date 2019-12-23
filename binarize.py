import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度化的方法可以都试试看
def gray_avg2(image):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            r,g,b = img_rgb[y,x]
            r = np.uint8(r * 0.299)
            g = np.uint8(g * 0.587)
            b = np.uint8(b * 0.114)

            rgb = np.uint8(r + b + g)
            dist[y,x] = rgb
    return dist


def gray_avg(image):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            avg = sum(image[y,x]) / 3
            dist[y,x] = np.uint8(avg)
    
    return dist


def gray_max(image):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            maximum = max(image[y,x])
            dist[y,x] = np.uint8(maximum)
    
    return dist

def gray_min(image):
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rows,cols,_ = image.shape
    dist = np.zeros((rows,cols),dtype=image.dtype)
    
    for y in range(rows):
        for x in range(cols):
            minimum = min(image[y,x])
            dist[y,x] = np.uint8(minimum)
    
    return dist

def split(img, a, b):
    imgs = []
    h, w = img.shape
    for i in range(a):
        for j in range(b):
            imgs.append(img[int(j*h/b):int((j+1)*h/b),int(i*w/a):int((i+1)*w/a)])
    return imgs


# 进行二值化和形态学变换的主要函数
# c为 adpative_threshold_binarize的参数，c越大越能抑制雪花噪声，但会造成笔画断裂或消失，推荐5-9之间
def binarize(GrayImage, c=5):
    # 中值滤波
    GrayImage= cv2.medianBlur(GrayImage,3)
    # cv2.GussianBlur 进行高斯滤波
    # cv2.boxfilter 进行方框滤波
    # cv2.blur进行均值滤波

    # print(GrayImage.shape)
    ret,th1 = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)

    blocksize = 7
    # c可能要选 7或者9
    # 对于黄底黑字，因为对比度小了，需要调小c，如c=5
    c = 5
    # print("Blocksize = {}, c = {}".format(blocksize, c))

    # thresh_mean 效果好
    # blocksize >=5, blocksize大了可以是笔画更连贯，但噪声也会更大， C 越大会过滤更严格（5左右）
    th2_mean = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                        cv2.THRESH_BINARY,blocksize,c)           
    th3_mean = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                        cv2.THRESH_BINARY,blocksize,c+2)

    th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,blocksize,c-1)
    th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,blocksize,c+1)
    # titles = ['Gray Image', 'Global Thresholding (v = 127)',
    # 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding+2']
    # titles = ['Adaptive Mean Thresholding', 'Adaptive Mean Thresholding+2',
    # 'Adaptive Gaussian Thresholding', 'Adaptive Gaussian Thresholding+2']
    # images = [GrayImage, th1, th2_mean, th3_mean]
    # images = [th2_mean, th3_mean, th2, th3]
    
    # 形态学操作，膨胀和腐蚀都是对白色而言
    kernelsize = 3
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    
    # kernelsize = 3 or 4
    kernel2 = np.ones((3, 3), np.uint8)
    kernel2[0, 0] = 0 
    kernel2[0, -1] = 0 
    kernel2[-1, 0] = 0 
    kernel2[-1, -1] = 0 

    kernel3 = np.zeros((5, 5), np.uint8)
    for k in range(5):
        kernel3[2, k] = 1
        kernel3[k, 2] = 1

    # 去噪，使用 3x3 的正方形kernel
    dilation = cv2.dilate(th2_mean, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # 连接断的笔画，使用5x5的十字kernel以及3x3的正方形kernel
    img = cv2.erode(erosion, kernel3, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.dilate(img, kernel3, iterations=1)
    return img


def test_gray_method(img):
    GrayImage = gray_min(img)
    grayimg1 = gray_avg(img)
    grayimg2 = gray_avg2(img)
    grayimg3 = gray_max(img)
    titles = ['Gray Image min', 'Gray Image avg1',
    'Gray Image avg2', 'Gray Image max']
    images = [GrayImage, grayimg1, grayimg2, grayimg3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == "__main__":
    for j in range(6):
        img=cv2.imread('data/h{}.JPG'.format(j+1), 1)
        # GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        GrayImage = gray_min(img)
        imgs = split(GrayImage, 2, 5)
        print("split complete. ")
        # imgs = []
        # for i in range(6):
        #     img = cv2.imread('data/yellow_{}.jpg'.format(i+1), 1)
        #     gray_img = gray_min(img)
        #     imgs.append(gray_img)

        for i in range(len(imgs)):
            pict = binarize(imgs[i], c=5)
            # print(pict.shape)
            a = np.concatenate((imgs[i], pict), axis=0)
            a = np.flip(a.T, 0)
            # print(a.shape)
            cv2.imwrite('data/single/c=5/{}_{}.jpg'.format(j+1,i+1), a)