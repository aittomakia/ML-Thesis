# Create synthetic data that generates 20 skewed and rotated images of the given sample image
# If the folder contains 10 samples this generator will create 200 images to that folder

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import os
import sys
import cv2
from tensorflow.python.platform import gfile

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

prefix = 'button'
output_dir = 'Images/'
label_output_dir = 'Labels/'
data_dir = ''
img_array = []
# Number of samples / class
NUMBER_OF_SAMPLES = 15

parser = argparse.ArgumentParser()
# parser.add_argument("--image", help="image to be processed")
parser.add_argument("--data_dir", help="directory that has the files")
parser.add_argument("--img_output", help="Images output location")
parser.add_argument("--label_output", help="Images output location")
parser.add_argument("--prefix", help="Output file prefix. Also serves as the file name")
parser.add_argument("--class_index", help="Class index as a number")
args = parser.parse_args()

if args.data_dir:
    data_dir = args.data_dir

if args.prefix:
  output_prefix = args.prefix
  img_name = args.prefix
  for i in range(NUMBER_OF_SAMPLES):
      img_array.append(data_dir + img_name + "-"+ str(i+1) +".jpg")
      i += 1
      if i >= NUMBER_OF_SAMPLES:
          break
print(img_array)

if args.class_index:
    index = args.class_index

output_dir = args.data_dir

if args.label_output:
    label_output_dir = args.label_output

for img in img_array:
    print(img)
    if gfile.Exists(img):
        print(img)
        loaded_img = load_img(img)
        x = img_to_array(loaded_img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the location that save_to_dir is pointing
        i = 0
        print("Generating images to " + output_dir)
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_dir, save_prefix=output_prefix, save_format='jpg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

# print("Generating label files for images ... ")

#for img_name in os.listdir(data_dir):
#    print(data_dir + img_name)
#    if '.DS_Store' not in img_name and '.txt' not in img_name:
#        img = cv2.imread(data_dir + img_name,0)
#        height, width = img.shape[:2]
#        name = os.path.splitext(img_name)[0] + ".txt"
#        file = open(label_output_dir + name,'w')
        # class_index xmin ymin xmax ymax
#        file.write(str(index) + " " + 0 + " " + 0 + " " + str(width) + " " + str(height))
        # file.write(str(index) + " " + str(int(width / 10)) + " " + str(int(height / 10)) + " " + str(width) + " " + str(height))
#        file.close()

print("Done!")
