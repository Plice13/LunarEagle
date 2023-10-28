from PIL import Image

# Open the image
image = Image.open('230926dr.jpg')

# Create a new image with the same size
new_image = Image.new('RGB', image.size)

# Paste the original image with a 5-pixel offset
new_image.paste(image, (100, 100))

# Save the new image
new_image.save('moved_image.png')
