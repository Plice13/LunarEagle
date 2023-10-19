from PIL import Image
import get_angle_for_rotation


def main(image_path, angle, output_dir_path='rotate/rotated_images/',output_img_name='rotated_image.jpg'):

    image = Image.open(image_path)

    rotated_image = image.rotate(angle)

    rotated_image.save(output_dir_path+output_img_name)

    image.close()

if __name__ == "__main__":
    image_path = '230926dr.jpg' # path for image
    main(image_path, get_angle_for_rotation.main(image_path))