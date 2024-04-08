import os
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pdb

num_of_authors = 8
# max_num_of_words_per_author = 400000

def crop(n, max_n):
    return max(0,n) if n <= max_n else max_n

def merge(l, start, end):
    sub_merged = ""
    for o in l[start:end+1]:
        sub_merged = sub_merged+o
    merged = l[:start] + [sub_merged] + l[end+1:]
    return merged
h_l, w_l = [], []

hor_tile = 4
for author_no in range(num_of_authors):
    source_dir = 'sources\\author'+str(author_no + 1)+'\\skany'
    label_dir = 'data\\'+'a'+str(author_no + 1)
    files = os.listdir(source_dir)
    image_files = [file for file in files if file.endswith('bmp')]
    subimage_no = 0
    for image_file in image_files:
        image_file_name = source_dir + "\\" + image_file
        image = mpimg.imread(str(image_file_name))
        image_h, image_w = len(image), len(image[0])
        ver_tile = int(hor_tile * image_h/image_w)
        tile_size = int(image_w / hor_tile)
        for yp in range(ver_tile):
            for xp in range(hor_tile):
                subimage = image[yp*tile_size:(yp+1)*tile_size,
                                 xp*tile_size:(xp+1)*tile_size]
                subimage_no+=1
                mpimg.imsave(label_dir+'\\'+str(subimage_no)+'.bmp',subimage)
            
    
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)   