import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 
from skimage.util import random_noise

def sp_noise (image, prob):
    '''
        (Function make salt & paper noise on an image)
        :param image: src image
        :param prob: the percentage of salt and paper noise
        :return: Image with salt and paper noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gass_noise(image):
    '''
        (Function make Gaussian Noise on an image)
        :param image: src image
        :return: Image with Gaussian noise
    '''
    noise_img = random_noise(image, mode='gaussian', seed=None, clip=True)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img

def perSPNoise(image, when=""): #Function compute the percentage of salt & paper noise
    '''
    (Function compute the percentage of salt & paper noise)
    :param image: src image
    :return: The percentage of salt & paper noise
    '''
    image = cv2.Laplacian(image, cv2.CV_64F)
    coNoNoise = 0
    COUNT = 0
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            listTest = [image[i-1][j], image[i+1][j], image[i][j-1], image[i][j+1],  image[i-1][j - 1], image[i -1][j + 1], image[i+1][j - 1], image[i + 1][j + 1]]

            if (image[i][j] >= 450) or (image[i][j] <= -450):
                COUNT += 1
            else:
                for k in listTest:
                    if abs(image[i][j] - k) > 25:
                        break
                else:
                    coNoNoise += 1
    t = coNoNoise /  (image.shape[0]*image.shape[1])
    if when == "After":
        return COUNT/(image.shape[0]*image.shape[1])
    if t > 0.04:
        return COUNT/(image.shape[0]*image.shape[1])
    else:
        return 0.0


def perGassNoise(image, when=""):
    '''
    (Function compute the percentage of Gaussian noise )
    :param image: src image
    :return: The percentage of Gaussian noise
    '''
    image = cv2.Laplacian(image, cv2.CV_64F)

    x, y = image.shape
    coNoise = 0
    coNoNoise = 0
    co = 0
    for i in range(1, x - 6, 5):
        for j in range(1, y - 6, 5):
            listTest = []
            for I in range(i, i+5):
                for J in range(j, j+5):
                  listTest.append(image[I][J])

            co += 1
            table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for k in range(25):
                if listTest[k] >= -449 and listTest[k] < -300:
                    table[0] += 1
                elif listTest[k] >= -300 and listTest[k] < -200:
                    table[1] += 1
                elif listTest[k] >= -200 and listTest[k] < -100:
                    table[2] += 1
                elif listTest[k] >= -100 and listTest[k] < -50:
                    table[3] += 1
                elif listTest[k] >= -50 and listTest[k] < -0:
                    table[4] += 1
                elif listTest[k] >= 0 and listTest[k] < 50:
                    table[5] += 1
                elif listTest[k] >= 50 and listTest[k] < 100:
                    table[6] += 1
                elif listTest[k] >= 100 and listTest[k] < 200:
                    table[7] += 1
                elif listTest[k] >= 200 and listTest[k] < 300:
                    table[8] += 1
                elif listTest[k] >= 300 and listTest[k] < 400:
                    table[9] += 1
                elif listTest[k] >= 400 and listTest[k] < 450:
                    table[10] += 1

            mySet = set(table)
            test = len(mySet)
            if test >= 7:
                coNoise += 1
            elif test <= 3:
                coNoNoise += 1

    if when == "After":
        return coNoise / co
    if coNoNoise < coNoise:
        return coNoise / co
    else:
        return 0

def perBlur(image, where=""):

    image = cv2.Laplacian(image, cv2.CV_64F)

    x, y = image.shape
    coNoise = 0
    coNoNoise = 0
    for i in image:
        for j in i:
            if j > 220 or j <= -220:
                coNoise += 1
            elif j > -3 and j < 3:
                coNoNoise += 1

    if coNoise / (x*y) == 0:
        return 1 - (coNoNoise / (x*y))
    elif where == "After":
        return coNoise / (x*y)
    else:
        return 0.0


def detectNoise(type, image):
    '''
    :param result: The percentage noise computed
    :param type: The type of noise need to know
    :return: True or False if this noise on the image True (Salt and Paper noise or Normal noise ) False if no noise
    '''
    if type == "SaltPaper":
        result = perSPNoise(image)
        if result >= 0.0085:
            print("The noise is Salt and Pepper Noise")
            print("Before remove or reduce noise, the percentage of noise is: ", result)
            return True
    elif type == "Gaussian":
        result = perGassNoise(image)
        if result >= 0.0085:
            print("The noise is Gaussian")
            print("Before remove or reduce noise, the percentage of noise is: ", result)
            return True
    elif type == "Blur":
        result = perBlur(image)
        if result != 0.0:
            print("The image is blurred")
            print("Before reduce the blur effect, the percentage of noise is: ", result)
            return True
    return False


if __name__ == "__main__":


    arrGassNoise = ["g2.png", "g1.png"]
    arrSPNoise = ["S&Pn3.png", "S&Pn4.png"]
    arrBlur = ["b2.jpg"]
    arrNormal = ["Original.png"]
    #arr = arrGassNoise + arrSPNoise + arrNormal
    arr = arrGassNoise

    for i in range (len(arr)):
        image = cv2.imread(arr[i], cv2.IMREAD_GRAYSCALE)

        if detectNoise("SaltPaper", image):
            median = cv2.medianBlur(image, 3)
            sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharped_img = cv2.filter2D(median, -1, sharpen_filter)

            images = np.concatenate((image, sharped_img), axis=1)

            res = perSPNoise(median, "After")
            print("After remove or reduce noise, the percentage of noise is: ", res)

            plt.imshow(images, cmap='gray')
            plt.show()

        elif detectNoise("Gaussian", image):

            dst = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            gs = cv2.GaussianBlur(image, (3, 3), 0)
            test = cv2.fastNlMeansDenoising(gs, None, 10, 7, 21)
            test = cv2.fastNlMeansDenoising(test, None, 10, 7, 21)

            sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gg = cv2.filter2D(test, -1, sharpen_filter)

            d = np.concatenate((image, gg), axis=1)

            res = perGassNoise(gg, "After")

            print("After remove or reduce noise, the percentage of noise is: ", res)
            plt.imshow(d, cmap="gray")
            plt.show()
        elif detectNoise("Blur", image):

            sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharped_img = cv2.filter2D(image, -1, sharpen_filter)
            sharped_img = cv2.filter2D(sharped_img, -1, sharpen_filter)

            res = perBlur(sharped_img, "After")
            print("After remove or reduce noise, the percentage of noise is: ", res)
            n = np.concatenate((image, sharped_img), axis=1)
            plt.imshow(n, cmap="gray")
            plt.show()
        else:
            print("No noise")

        print("-"*50)



