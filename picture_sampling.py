import matplotlib.pyplot as plt
from skimage import transform, color, exposure
import os
import numpy as np
from tqdm import tqdm
import random
from scipy.ndimage import gaussian_filter
from PIL import Image

size = (75, 75)

random.seed(42)


def resizing(path, save_path, _size=(100, 100)):
    print('\n> load images')
    for filename in tqdm(os.listdir(path)):
        img = plt.imread(os.path.join(path, filename))
        if img is not None:
            img = color.rgb2gray(img) * 255
            y, x = img.shape
            if x < y:
                idx = int((y - x) / 2)
                img_q = img[idx: -idx, :]
            elif x > y:
                idx = int((x - y) / 2)
                img_q = img[:, idx:-idx]
            # plt.subplot(211)
            # plt.imshow(img, cmap='gray')
            # plt.subplot(212)
            # plt.imshow(img_q, cmap='gray')
            # plt.show()

            img_down = transform.resize(img_q, _size)

            plt.imsave(save_path + filename, img_down, cmap='gray')


def load_img(path):
    images = []
    values = []
    print('\n> load images')
    for filename in tqdm(os.listdir(path)):
        img = plt.imread(os.path.join(path, filename))
        if img is not None:
            img = color.rgb2gray(img) * 255
            img_down = transform.resize(img, size)
            values.append(np.sum(img_down) / (size[0] * size[1]))
            images.append(img_down)


    images = np.asarray(images)
    values = np.asarray(values)

    #print(values)
    #plt.imshow(images[0], cmap='gray')
    #plt.show()

    return images, values


def encode_img(img):
    y_shape, x_shape = img.shape
    y_iter = int(y_shape / size[0])
    x_iter = int(x_shape / size[1])
    values = np.zeros((y_iter, x_iter))
    print('\n> code Image')
    for y in tqdm(range(0, y_iter)):
        for x in range(0, x_iter):
            values[y, x] = np.sum(img[y * size[0]:(y+1) * size[0], x * size[1]:(x+1) * size[1]]) / (size[0] * size[1])

    return values


def create_pics(data, data_values, coded, range_idx, range_val, it):

    images = []
    values = []
    idx_list = np.argwhere(coded != range_idx)
    for i in range(0, it):
        idx = random.choice(idx_list)
        img = data[idx]
        value = data_values[idx]
        #delta = range_val - value
        ratio = range_val / value
        img_new = img[0] * ratio
        images.append(img_new)
        values.append(np.sum(img_new) / (size[0] * size[1]))


        # plt.subplot(211)
        # plt.imshow(img[0])
        # plt.subplot(212)
        # plt.imshow(img_new)
        # plt.show()

    return images, values


def make_coded_img(data, data_values, sample_img, sample_values, num_of_ranges=28):
    range_arr = np.linspace(0, 255, num_of_ranges)
    # data_values = np.array([2, 4, 1, 17, 6, 9, 12, 8])

    coded = np.zeros(data_values.shape)

    for x in range(0, coded.shape[0]):
        for j in range(0, len(range_arr)-1):
            if range_arr[j] < data_values[x] <= range_arr[j + 1]:
                coded[x] = j

    # check for missing values
    for j in range(0, len(range_arr)-1):
        number = len(np.argwhere(coded == j))
        if j not in coded or number < 8:
            print('missing %d - # %d' % (j, number))
            range_val = (range_arr[j] + range_arr[j+1]) / 2
            it = 10
            images_new, values_new = create_pics(data, data_values, coded, j, range_val, it)
            data = np.concatenate((data, images_new), axis=0)
            data_values = np.concatenate((data_values, values_new))
            a = np.ones(it) * j
            coded = np.concatenate((coded, np.ones(it) * j))

    print('Number of final pictures: %d' % data.shape[0])
    res = np.zeros(sample_img.shape)
    print('> create img')
    for y in range(0, sample_values.shape[0]):
        for x in range(0, sample_values.shape[1]):

            for j in range(0, len(range_arr) - 1):
                if range_arr[j] < sample_values[y, x] <= range_arr[j + 1]:
                    code_idx_list = np.argwhere(coded == j)
                    if code_idx_list.any():
                        code_idx = random.choice(code_idx_list)
                        res[y * size[0]:(y+1)*size[0], x * size[1]: (x+1)*size[1]] = data[code_idx]
                    else:
                        print('There is no picture for idx: %d -> Range(%d, %d)' % (j, range_arr[j], range_arr[j + 1]))

    plt.figure()
    # plt.subplot(211)
    # plt.imshow(sample_img, cmap='gray')
    plt.subplot(121)
    plt.imshow(res, cmap='gray')
    plt.subplot(122)
    gamma = exposure.adjust_gamma(res, 0.49) # 0.49
    plt.imshow(gamma, cmap='gray')
    # plt.subplot(224)
    # log = exposure.adjust_log(res, 100)
    # plt.imshow(log, cmap='gray')

    gamma = gaussian_filter(gamma, sigma=1)

    plt.imsave('result.jpg', gamma, cmap='gray')
    #plt.show()

if __name__ == "__main__":

    r = False
    if r:
        resizing('img', 'resized_img_square/')
    else:
        # fit sample to encoding size
        # sample = plt.imread('DSC00738.JPG')
        sample = plt.imread('DSC00887.JPG')

        sample = color.rgb2gray(sample) * 255

        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(sample, cmap='gray')

        sample = transform.rescale(sample, (3, 3))

        y_range = int(sample.shape[0] / size[0]) * size[0]
        x_range = int(sample.shape[1] / size[1]) * size[1]
        sample = transform.resize(sample, (y_range, x_range))
        # plt.subplot(122)
        # plt.imshow(sample, cmap='gray')
        # plt.show()

        coded_values = encode_img(sample)

        data, data_values = load_img('resized_img_square')

        make_coded_img(data, data_values, sample, coded_values)



