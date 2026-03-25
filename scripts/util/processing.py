import numpy as np
import xarray as xr


def fft_2d(
    ds: xr.Dataset,
    var_name: str,
    fft_dim: str,
    keep_dim: str,
    remove_dc: bool = True,
) -> xr.Dataset:
    """
    Compute a 1D real FFT of ds[var_name] along `fft_dim`, keep `keep_dim`,
    and average all other dimensions away.

    Output dimensions:
        (keep_dim, f"{fft_dim}_freq")

    Assumes the signal is real-valued, so only non-negative frequencies
    are returned.
    """
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not found in dataset variables: {list(ds.data_vars)}")

    da = ds[var_name]

    if fft_dim not in da.dims:
        raise ValueError(f"{fft_dim!r} is not a dimension of {var_name!r}. Dims: {da.dims}")
    if keep_dim not in da.dims:
        raise ValueError(f"{keep_dim!r} is not a dimension of {var_name!r}. Dims: {da.dims}")
    if fft_dim == keep_dim:
        raise ValueError("fft_dim and keep_dim must be different")

    # Average away all dimensions except keep_dim and fft_dim
    dims_to_avg = [dim for dim in da.dims if dim not in (keep_dim, fft_dim)]
    da_proc = da.mean(dim=dims_to_avg, skipna=True) if dims_to_avg else da

    # Ensure consistent order
    da_proc = da_proc.transpose(keep_dim, fft_dim)

    # Remove DC by subtracting the mean along fft_dim
    if remove_dc:
        da_proc = da_proc - da_proc.mean(dim=fft_dim, skipna=True)

    n = da_proc.sizes[fft_dim]

    # Infer sample spacing from coordinate if possible
    if fft_dim in da_proc.coords and n > 1:
        coord = np.asarray(da_proc[fft_dim].values)
        d = float(coord[1] - coord[0])
    else:
        d = 1.0

    freq_dim = f"{fft_dim}_freq"
    freqs = np.fft.rfftfreq(n, d=d)
    fft_vals = np.fft.rfft(da_proc.values, axis=-1)

    coords = {
        keep_dim: da_proc.coords[keep_dim],
        freq_dim: freqs,
    }

    da_fft = xr.DataArray(
        fft_vals,
        dims=(keep_dim, freq_dim),
        coords=coords,
        name=f"{var_name}_fft",
    )

    da_fft_abs = xr.DataArray(
        np.abs(fft_vals),
        dims=(keep_dim, freq_dim),
        coords=coords,
        name=f"{var_name}_fft_abs",
    )

    da_fft_power = xr.DataArray(
        np.abs(fft_vals) ** 2,
        dims=(keep_dim, freq_dim),
        coords=coords,
        name=f"{var_name}_fft_power",
    )

    return xr.Dataset(
        {
            da_fft.name: da_fft,
            da_fft_abs.name: da_fft_abs,
            da_fft_power.name: da_fft_power,
        }
    )