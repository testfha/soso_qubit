import xarray as xr


def rethreshold_dataset(ds: xr.Dataset, raw_data_var: str, shot_dim: str = "repetition",
                        rethres_dims: list[str] | None = None, new_axis_name: str | None = None, *,
                        invert: bool = False) -> xr.Dataset:
    ds = ds.copy()

    da = ds[raw_data_var]

    if rethres_dims is None:
        rethres_dims = [dim for dim in da.dims if dim != shot_dim]

    if new_axis_name is None:
        new_axis_name = "rethresholded"

    # local threshold: mean over shots
    new_threshold = da.mean(dim=shot_dim, skipna=True)

    # classify shots
    if invert:
        state = (da < new_threshold).astype(int)
    else:
        state = (da >= new_threshold).astype(int)

    # average over shots to get probability
    ds[new_axis_name] = state.mean(dim=shot_dim)
    ds[f"{new_axis_name}_threshold"] = new_threshold
    ds[f"{new_axis_name}_state"] = state

    return ds
