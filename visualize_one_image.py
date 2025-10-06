# Visualize all 10 bands for one image
for i in range(10):
    pct_contrast=(0, 100)
    cmap="viridis"
    
    if isinstance(pct_contrast, (int, float)):
        pct_contrast = (pct_contrast, 100 - pct_contrast)

    with rasterio.open(planet_raster_file) as src:
        data = src.read(i+1)

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap=cmap,
               vmin=np.percentile(data, pct_contrast[0]),
               vmax=np.percentile(data, pct_contrast[1]))
    plt.colorbar(label="Pixel Value")
    plt.title(f"Raster Visualization: {band_dict[i+1]}")
    plt.savefig(f'visualizations/band_{i+1}.png', bbox_inches='tight')

