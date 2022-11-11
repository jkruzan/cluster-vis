## A python script to individually save all images into data/binary_images
## Loading the images takes a few minutes, after that, a progress bar will
## appear and show how many images have been saved

## Be sure to create the directory data/binary_images before running
import mat73
from PIL import Image
from tqdm import tqdm

path = './data/BinaryImages.mat'
print("Loading images... this will take a minute or so")
data = mat73.loadmat(path)
image_data = data['CellImgs']
print("Saving Images:")
for i in tqdm(range(len(image_data))):
    im = image_data[i][0]
    image_file = Image.new('1', im.shape)
    image_file.putdata(im.flatten())
    file_name = './data/binary_images/binary_image_'+str(i)+'.bmp'
    image_file.save(file_name)    
