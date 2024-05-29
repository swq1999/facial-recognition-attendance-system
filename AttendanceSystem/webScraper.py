"""
Script that scrapes images from Bing images.

Author: Syed Wajahat Quadri
ID: w21043564
"""

from bing_image_downloader import downloader

def main():
    """
    Main function to start the application.
    """
    celebrity_name = input("Enter celebrity name: ")
    download_celeb_images(celebrity_name)


def download_celeb_images(celebrity_name, num_images=500):
    """
    Downloads images of a specified celebrity using the Bing image downloader.

    Parameters:
    celebrity_name (str): Celebrity name
    num_images (int): Number of images

    Returns:
    None
    """
    # Create the directory name based on the celebrity name
    directory_name = celebrity_name.replace(" ", "_")

    # Download the images
    downloader.download(celebrity_name, limit=num_images, output_dir='employees', 
                        adult_filter_off=True, force_replace=False, timeout=60)

if __name__ == "__main__":
    main()
