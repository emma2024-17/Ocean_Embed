import os
import copernicusmarine

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

print("Downloading GLORYS satellite data...")

try:
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so", "zos", "uo", "vo"],
        minimum_longitude=45.0,
        maximum_longitude=105.0,
        minimum_latitude=5.0,
        maximum_latitude=30.0,
        start_datetime="2020-01-01",
        end_datetime="2020-01-10",
        minimum_depth=0.0,
        maximum_depth=1000.0,
        output_directory=output_dir,
        output_filename="glorys_subset.nc"
    )
    print(f"Data saved successfully to {output_dir}/glorys_subset.nc")
except Exception as e:
    print(f"Download error: {e}")