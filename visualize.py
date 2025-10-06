import rasterio
import matplotlib.pyplot as plt
import numpy as np


# Dictionary of image bands
band_dict = {
    1: 'Blue',
    2: 'Green',
    3: 'Red',
    4: 'Red Edge 1',
    5: 'Red Edge 2',
    6: 'Red Edge 3',
    7: 'Near Infrared',
    8: 'Red Edge 4',
    9: 'SWIR 1', 
    10: 'SWIR 2'
}


# Plots a comparison between image, label, and model predictions
def plot_comparisons(image, label, prediction, band):
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    plt.tight_layout()

    # Plots the satellite image
    axs[0].imshow(image, cmap='viridis')
    axs[0].set_title(band_dict[band])
    axs[0].axis('off')

    # Plots the binary ground truth labels
    axs[1].imshow(label, cmap='gray')
    axs[1].set_title('Label')
    axs[1].axis('off')

    # Plots the model's binary predictions
    axs[2].imshow((prediction >= 0.5).astype(int), cmap='gray')         # Binarizes predictions
    axs[2].set_title('Output')
    axs[2].axis('off')

    return fig



# Visualizes satellite images from .tif files
def visualize_band(raster_file, band_number, bands_dict, cmap="viridis", pct_contrast=(0, 100)):
    """
    Visualize a single band from a raster (.tif) image file.

    Parameters:
        raster_file (str): Path to the .tif image file
        band_number (int): Band index (1-indexed)
        bands_dict (dict): Dictionary mapping band numbers to names
        cmap (str): Color map to use (default: viridis)
        pct_contrast (tuple): Percentile stretch for contrast clipping
    """
    if isinstance(pct_contrast, (int, float)):
        pct_contrast = (pct_contrast, 100 - pct_contrast)

    with rasterio.open(raster_file) as src:
        data = src.read(band_number)

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap=cmap,
               vmin=np.percentile(data, pct_contrast[0]),
               vmax=np.percentile(data, pct_contrast[1]))
    plt.colorbar(label="Pixel Value")
    plt.title(f"Raster Visualization: {bands_dict[band_number]}")
    plt.show()

# Define your file and bands
planet_raster_file = '../data/S2_IMAGE_TIF/T03VWJ_20190701_0_0_image_raw_values.tif'


# Visualize one band (e.g., Red)
#visualize_band(planet_raster_file, band_number=1, bands_dict=band_dict)







