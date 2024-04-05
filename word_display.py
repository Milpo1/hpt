# A program for reading and displaying handwritten words downloaded from graphic 
# files based on descriptions from text files
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

for author_no in range(num_of_authors):
    file_desc_name = "author" + str(author_no + 1) + "/word_places.txt"
    file_desc_ptr = open(file_desc_name, 'r')
    text = file_desc_ptr.read()
    lines = text.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    num_of_words = 0
    image_file_name_prev = ""
    subimage_dir = 'data\\'+'a'+str(author_no + 1)
    
    if not os.path.exists(subimage_dir):
        os.makedirs(subimage_dir)   
        
    for i in range(number_of_lines):
        row_values = lines[i].split()
        
        if len(row_values) > 6:
            row_values = merge(row_values,1,len(row_values)-5)
        elif len(row_values) < 6:
            continue

        if row_values[0] != '%':
            num_of_words += 1
            number_of_values = len(row_values)
            
            image_file_name = "author" + str(author_no + 1) + "\\" + row_values[0][1:-1]

            if image_file_name != image_file_name_prev:   
                image = mpimg.imread(str(image_file_name))
                image_file_name_prev = image_file_name
            word = row_values[1]
            
            if word == "<brak>":
                continue
            
            row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                int(row_values[4]), int(row_values[5])

            height, width = len(image), len(image[0])
            row1, row2 =  crop(row1,height), crop(row2,height)
            column1, column2 =  crop(column1,width), crop(column2,width)

            subimage = image[min(row1,row2):max(row1,row2),
                            min(column1,column2):max(column1,column2)] 

            subimage = resize(subimage, (128,128))
            mpimg.imsave(subimage_dir+'\\'+str(num_of_words)+'.bmp',subimage)

        # if num_of_words >= max_num_of_words_per_author: break


    file_desc_ptr.close()