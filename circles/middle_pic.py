from PIL import Image, ImageDraw

image = Image.open('230926dr.jpg')

width, height = image.size
center = (width / 2, height / 2)

draw = ImageDraw.Draw(image)
draw.ellipse([center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5], fill='red')

image.save('sun\image_with_center.jpg')