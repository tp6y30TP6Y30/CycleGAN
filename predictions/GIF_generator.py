from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm
import time

def generateGIFs(query, target_img_name, save_folder, interval = 20):
	print('Saving GIF...')
	print('query: ', query)
	print('img_name: ', target_img_name)
	folder_names = os.listdir(query)
	folder_names.sort(key = lambda folder_name: int(folder_name[folder_name.find('_') + 1:]))
	GIF_length = len(folder_names)
	print('GIF_length: ', GIF_length // interval)

	images = []
	for (index, folder_name) in enumerate(folder_names):
		if index % interval != 0: continue
		img_path = os.path.join(query, folder_name, target_img_name)
		images.append(imageio.imread(img_path))
	save_path = os.path.join(save_folder, query)
	os.makedirs(save_path, exist_ok = True)
	target_gif_name = target_img_name.replace('.jpg', '.gif')
	imageio.mimsave(os.path.join(save_path, target_gif_name), images, duration = 0.2, loop = 1)
	print('===== {} saved ====='.format(target_gif_name))
	print()

def refresh_allGIFs(query, save_folder):
	target_img_names = [img_name.replace('.gif', '.jpg') for img_name in os.listdir(os.path.join(save_folder, query))]
	for target_img_name in target_img_names:
		generateGIFs(query, target_img_name, save_folder)

if __name__ == '__main__':
	# generateGIFs(query = 'horse2zebra', target_img_name = 'fake_A22.jpg', save_folder = 'GIFs/')
	refresh_allGIFs(query = 'horse2zebra', save_folder = 'GIFs/')
