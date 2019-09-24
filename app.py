import cv2
import numpy

# Select any one image path from below and  comment out the remaining paths
#  OR add new path for separate image

image_path = r'images/Fig0309(a)(washed_out_aerial_image).tif'
# image_path = r'images/Fig0316(1)(top_left).tif'
# image_path = r'images/Fig0316(2)(2nd_from_top).tif'
# image_path = r'images/Fig0316(3)(third_from_top).tif'
# image_path = r'images/Fig0316(4)(bottom_left).tif'
# image_path = r'images/Fig0354(a)(einstein_orig).tif'
# image_path = r'images/Fig0308(a)(fractured_spine).tif'
# image_path = r'images/lena.jpg'
# image_path = r'images/animal-photography-daylight-elephant-247431.jpg'
# image_path = r'images/lena512color_pale.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
total_pixels = image.size
shape = image.shape
height = shape[0];
width = shape[1]

print("Original image: type: ", type(image))
print("Original image: dType: ", image.dtype)
print("Original image: dimensions/shape: ", shape)
print("Original image: total pixels size: ", total_pixels)
print("Original image: pixel min: ", image.min())
print("Original image: pixel max: ", image.max())

# Histogram Equalization Algorithm

# init some values for algorithm
L = 256
hist_equalized_pix_list = []
flat_image = image.flatten().tolist()
cumulative_sum = 0

# find replacement for original pixel values
for index in range(L):
    probability_distribution = flat_image.count(index) / total_pixels
    cumulative_sum = cumulative_sum + probability_distribution
    s_k = int(round(cumulative_sum * (L - 1)))
    hist_equalized_pix_list.append(s_k)

#  replace original pixel values in image array with histogram equalized pixel values
#  iteration = total_pixels
for index in range(total_pixels):
    flat_image[index] = hist_equalized_pix_list[flat_image[index]]

# print(original_pix_intensity_list)
# print(len(hist_equalized_pix_list))
# print(hist_equalized_pix_list)
# print(hist_equalized_pix_list[255])

#  display new image
image = numpy.asarray(flat_image)
image = image.reshape(height, width)
print("Final image size: ", image.size)
print("Final image dimension/shape: ", image.shape)
image = numpy.array(image, dtype='uint8')
image_resized = cv2.resize(image, (700, 600))
cv2.imshow("Histogram Equalized image", image_resized)
cv2.waitKey(0)
