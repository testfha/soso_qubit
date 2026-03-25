"""
To download the data you need access to SQDL (ask the software team).

Then run this script, it will ask you for the scope and uuid of the dataset you want to download, and where to save it.
The default save location is the data folder in this repository.
"""

from pathlib import Path
from xarray import load_dataset

root = Path(__file__).parent.parent


def load_by_uuid(uuid, folder=root / "data"):
    return load_dataset(folder / f"{uuid}.hdf5")


def save_hdf5(ds, fname):
    from core_tools.data.ds.ds2xarray import ds2xarray
    from core_tools.data.ds.ds_hdf5 import save_xr_hdf5

    xds = ds2xarray(ds)
    save_xr_hdf5(xds, fname)


if __name__ == "__main__":
    from core_tools.data.sqdl.sqdl_reader import load_by_uuid, init_sqdl

    scope = str(input("Enter the scope of the dataset: "))
    init_sqdl(scope)

    uuid = int(input("Enter the UUID of the dataset to load: "))

    print(f"Loading data with uuid {uuid}")
    ds = load_by_uuid(uuid)
    save_destination = input(
        r"Enter the directory to save the dataset (default root /data): "
    )
    if not save_destination:
        save_destination = root / "data"
    save_destination = Path(save_destination) / f"{uuid}.hdf5"
    print(f"Data loaded, saving to {save_destination}")
    save_hdf5(ds, str(save_destination))
    print("Data saved to disk")
