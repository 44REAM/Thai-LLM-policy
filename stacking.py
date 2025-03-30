import os
from PIL import Image
from config import available_models_name


def stack_images_grid(folder_path, layout=(2,2), output_path="stacked_image.png", dpi=None):
    """
    Stacks images from the specified folder in a grid layout.

    Args:
        folder_path (str): The path to the folder containing the images.
        layout (tuple): A tuple representing the grid layout (rows, columns).
                       If columns is 0, it will stack vertically.
                       If rows is 0, it will stack horizontally.
                       For a specific grid, provide (num_rows, num_columns).
                       Example: (2, 2) for a 2x2 grid.
                       Example: (4, 1) for a 4x1 vertical stack.
        output_path (str): The path to save the resulting stacked image.
                           Defaults to "stacked_image.png" in the current directory.
        dpi (tuple, optional): The desired DPI for the output image as a tuple (x, y).
                               If None, the original DPI of the first image is used if available,
                               otherwise a default might be applied by Pillow.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in the folder: {folder_path}")
        return

    images = [Image.open(os.path.join(folder_path, img)) for img in image_files]
    num_images = len(images)

    if not images:
        print("Could not open any of the image files.")
        return

    rows, cols = layout

    if cols == 0:  # Vertical stack
        cols = 1
        rows = num_images
    elif rows == 0:  # Horizontal stack
        rows = 1
        cols = num_images
    elif rows * cols > num_images:
        print(f"Warning: Layout ({rows}x{cols}) has more cells than the number of images ({num_images}). Empty cells will be blank.")

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    max_width = max(widths) if widths else 0
    max_height = max(heights) if heights else 0

    output_width = cols * max_width
    output_height = rows * max_height

    new_image = Image.new('RGB', (output_width, output_height))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        if row >= rows or col >= cols:
            break  # Stop if we exceed the specified layout

        x_offset = col * max_width
        y_offset = row * max_height

        # Resize image to fit the cell if necessary
        resized_img = img.resize((max_width, max_height))
        new_image.paste(resized_img, (x_offset, y_offset))

    save_kwargs = {}
    if dpi:
        save_kwargs['dpi'] = dpi
    elif images and hasattr(images[0], 'info') and 'dpi' in images[0].info:
        save_kwargs['dpi'] = images[0].info['dpi']

    try:
        new_image.save(output_path, **save_kwargs)
        print(f"Successfully stacked images in a ({rows}x{cols}) grid and saved to: {output_path}")
        if dpi:
            print(f"Output image DPI set to: {dpi}")
        elif 'dpi' in save_kwargs:
            print(f"Output image DPI set to: {save_kwargs['dpi']}")
        else:
            print("Output image DPI was determined by Pillow (likely default).")
    except Exception as e:
        print(f"Error saving the stacked image: {e}")

if __name__ == '__main__':
    folder_path = 'results/distribution'


    outpath = f'results/distribution.png'
    stack_images_grid(folder_path, dpi = (800,800), layout = (2,2), output_path = outpath)

    folder_path = 'results/different'
    outpath = f'results/different.png'
    stack_images_grid(folder_path, dpi = (800,800), layout = (2,2), output_path = outpath)