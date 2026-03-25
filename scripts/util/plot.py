import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import image as mpimg


def add_panel_label(
        ax: plt.Axes,
        label: str,
        x: float = -0.11,
        y: float = -0.01,
        **kwargs,
):
    bbox = ax.get_position()
    ax.figure.text(
        bbox.x0 + x,
        bbox.y1 - y,
        label,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        **kwargs,
    )


def make_image_panel(ax, image_path):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis("off")


def plot_2d_heatmap(
        ds: xr.Dataset,
        x_name: str,
        y_name: str,
        z_name: str,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        ax: plt.Axes | None = None,
        **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if z_name not in ds:
        raise KeyError(f"{z_name!r} not found in dataset variables: {list(ds.data_vars)}")

    da = ds[z_name]

    if x_name not in da.dims:
        raise ValueError(f"{x_name!r} is not a dimension of {z_name!r}. Dims are {da.dims}")
    if y_name not in da.dims:
        raise ValueError(f"{y_name!r} is not a dimension of {z_name!r}. Dims are {da.dims}")

    # average over all dimensions except x and y
    avg_dims = [dim for dim in da.dims if dim not in (x_name, y_name)]
    if avg_dims:
        da = da.mean(dim=avg_dims, skipna=True)

    # enforce consistent plotting order: y rows, x columns
    da = da.transpose(y_name, x_name)

    # use pcolormesh to plot the heatmap
    ax.pcolormesh(da.coords[x_name] * x_scale, da.coords[y_name] * y_scale,
                  da.values, shading="auto", **kwargs)

    return da


def plot_1d_scatter(
        ds: xr.Dataset,
        x_name: str,
        y_name: str,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        ax: plt.Axes | None = None,
        **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if y_name not in ds:
        raise KeyError(f"{y_name!r} not found in dataset variables: {list(ds.data_vars)}")

    da = ds[y_name]

    if x_name not in da.dims:
        raise ValueError(f"{x_name!r} is not a dimension of {y_name!r}. Dims are {da.dims}")

    # average away any extra dimensions, keeping only x_name
    avg_dims = [dim for dim in da.dims if dim != x_name]
    if avg_dims:
        da = da.mean(dim=avg_dims, skipna=True)

    ax.scatter(da.coords[x_name] * x_scale, da.values * y_scale, **kwargs)

    return da
