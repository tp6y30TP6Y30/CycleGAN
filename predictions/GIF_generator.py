from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm
import time

def generateGIFs(query, target_img_name, save_folder):
	print('Saving GIF...')
	print('query: ', query)
	folder_names = os.listdir(query)
	folder_names.sort(key = lambda folder_name: int(folder_name[folder_name.find('_') + 1:]))
	GIF_length = len(folder_names)
	print('GIF_length: ', GIF_length)

	images = []
	for folder_name in folder_names:
		img_path = os.path.join(query, folder_name, target_img_name)
		images.append(imageio.imread(img_path))
	save_path = os.path.join(save_folder, query)
	os.makedirs(save_path, exist_ok = True)
	target_gif_name = target_img_name.replace('.jpg', '.gif')
	imageio.mimsave(os.path.join(save_path, target_gif_name), images, duration = 0.1)
	print('===== {} saved ====='.format(target_gif_name))

if __name__ == '__main__':
	generateGIFs(query = 'horse2zebra', target_img_name = 'fake_B11.jpg', save_folder = 'GIFs/')
