from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import color
from torch import Tensor


def imagesc(
    ax: matplotlib.axes,
    image: Union[Tensor, np.ndarray],
    title: Optional[str] = None,
    colormap: matplotlib.colormaps = plt.cm.gray,
    clim: Optional[tuple[float, float]] = None,
    show_axis: bool = False,
    show_colorbar: bool = True,
) -> None:
    """Display image with scaled colors. Similar to Matlab's imagesc.

    Args:
        ax: Axis to plot on.
        image: Array to plot.
        title: Title of plotting.
        colormap: Colormap of plotting.
        clim: Colormap limits.
        show_axis: Whether to show axis when plotting.
        show_colorbar: Whether to show colorbar when plotting.

    Example:
        >>> plt.figure("image", (18, 6))
        >>> ax = plt.subplot(1, 2, 1)
        >>> imagesc(ax, np.random.rand(100,100), "image", clim=[-1, 1])
        >>> plt.show()
    """

    if clim is not None and isinstance(clim, (list, tuple)):
        if len(clim) == 2 and (clim[0] < clim[1]):
            clim_args = {"vmin": float(clim[0]), "vmax": float(clim[1])}
        else:
            raise ValueError(
                f"clim should be a list or tuple containing 2 floats with clim[0] < clim[1], "
                f"got {clim} instead.",
            )
    else:
        clim_args = {}

    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()

    im = ax.imshow(image, colormap, **clim_args)
    plt.title(title)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax)

    if not show_axis:
        ax.set_axis_off()


def overlay_mask_on_image(
    image: np.ndarray,
    segmentation: np.ndarray,
    bg_label: int = 0,
    alpha: float = 0.3,
    colors: Optional[Union[list, list[list]]] = None,
) -> np.ndarray:
    """Overlay segmentation mask on given image.

    Args:
        image: Image to overlay.
        segmentation: Segmentation mask.
        bg_label: Label of background.
        alpha: Opacity of colorized labels.
        colors: Colors of overlay of the segmentation labels.

    Returns:
        RGB Numpy array of image overlaid with segmentation mask.

    Raises:
        ValueError: When image and segmentation have different shapes.
    """
    if not np.all(image.shape == segmentation.shape):
        raise ValueError(
            f"image {image.shape} does not have the same dimension as segmentation {segmentation.shape}!"
        )

    return color.label2rgb(segmentation, image, bg_label=bg_label, alpha=alpha, colors=colors)
