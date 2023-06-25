import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image_dataset(dataset_dir, shuffle= False):

    images_paths= []
    labels= []

    # iterate through each image folder
    for rice in os.listdir(dataset_dir):
        rice_dir = dataset_dir / rice
        # if not a folder, continue
        if os.path.isfile(rice_dir):
            continue
        # list all images inside image folder
        # append to list
        images= os.listdir(rice_dir)
        for image in images:
            image_path= rice_dir / image 
            images_paths.append(image_path)
            labels.append(rice)

    # shuffle dataset
    if shuffle:
        import random
        random.seed(0)
        indices = list(zip(images_paths, labels))
        random.shuffle(indices)
        images_paths, labels = zip(*indices)

    return images_paths, labels

# visualize images
def show_image(image_list, label_list):
    fig = plt.figure(figsize=(10, 5))
    for i, file in enumerate(image_list):
        img = Image.open(file)
        print('Image shape:', np.array(img).shape)
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        ax.set_title(label_list[i])
    plt.tight_layout()
    plt.show()