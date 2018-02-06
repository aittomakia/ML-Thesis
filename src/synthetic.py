# Create synthetic data that generates 20 images of the given sample image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
from tensorflow.python.platform import gfile

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

prefix = 'button'
output_dir = 'data/button'

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to be processed")
parser.add_argument("--output", help="Images output location")
parser.add_argument("--prefix", help="Output file prefix")
args = parser.parse_args()

if args.image and gfile.Exists(args.image):
  img = load_img(args.image)

if args.output:
    output_dir = args.output

if args.prefix:
    output_prefix = args.prefix

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the location that save_to_dir is pointing
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=output_dir, save_prefix=output_prefix, save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
