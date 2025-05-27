import xarray as xr
import numpy as np
from PIL import Image
import datetime
import os

# Load datasets
temp_ds = xr.open_dataset("temp.nc")
precip_ds = xr.open_dataset("precipitation.nc")
wind_ds = xr.open_dataset("windspeed.nc")

# Extract variable names
temp_var = "T2M"
precip_var = "PRECTOTCORR"
wind_var = "WS10M"

# Output directory
os.makedirs("rgb_images", exist_ok=True)

# Function to normalize data to 0–255
def normalize(data):
    arr = data.values
    arr = np.nan_to_num(arr, nan=0.0)
    min_val, max_val = np.percentile(arr, 2), np.percentile(arr, 98)
    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)

# Loop over each day of 2024
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 12, 31)
delta = datetime.timedelta(days=1)

current_date = start_date
while current_date <= end_date:
    date_str = current_date.isoformat()
    try:
        date_np = np.datetime64(current_date)

        temp_data = temp_ds[temp_var].sel(time=date_np, method="nearest")
        precip_data = precip_ds[precip_var].sel(time=date_np, method="nearest")
        wind_data = wind_ds[wind_var].sel(time=date_np, method="nearest")

        R = normalize(temp_data)
        G = normalize(wind_data)
        B = normalize(precip_data)

        rgb_image = np.stack([R, G, B], axis=-1)
        img = Image.fromarray(rgb_image)
        img.save(f"rgb_images/rgb_image_{date_str}.png")
        print(f"Saved: rgb_images/rgb_image_{date_str}.png")

    except Exception as e:
        print(f"Skipped {date_str}: {e}")

    current_date += delta
