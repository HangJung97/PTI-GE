import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from src.utils.file_and_folder_operations import subdirs, subfiles
from src.utils.transforms import resample_image, resample_label


def resize_dataset(data_dir: Union[Path, str], size: int = 64) -> None:
    """Resize CAMUS dataset to 64x64.

    Args:
        data_dir: Path to the dataset.
        size: Size to resample the dataset.
    """
    if not isinstance(size, int):
        size = int(size)
        warnings.warn(f"Size provided is not an integer, rounding to {size}.")

    all_cases = subdirs(data_dir, prefix="patient", join=False)

    for case in tqdm(all_cases, desc="Resizing CAMUS dataset", unit="patients"):
        nifti_files = subfiles(os.path.join(data_dir, case), suffix=".nii.gz", join=True)
        for file in nifti_files:
            image = sitk.ReadImage(file)
            ori_shape = [*image.GetSize(), 1]
            ori_spacing = [*image.GetSpacing(), 1]
            new_shape = [size, size, ori_shape[-1]]
            new_spacing = np.array(ori_spacing) * np.array(ori_shape) / np.array(new_shape)

            image_array = sitk.GetArrayFromImage(image).transpose(1, 0)
            image_array = image_array[None, ..., None]
            if "gt" in file:
                resized_image_array = resample_label(image_array, new_shape, True, lowres_axis=[2])
            else:
                resized_image_array = resample_image(
                    image_array.astype(np.uint8), new_shape, True, lowres_axis=[2]
                )
            image = sitk.GetImageFromArray(resized_image_array[0, ..., 0].transpose(1, 0))
            image.SetSpacing(new_spacing[:-1])

            sitk.WriteImage(image, file)


def main():
    """Run the script."""
    import argparse

    PATH_DATA = Path(__file__).parent.parent.parent / "data" / "camus"
    parser = argparse.ArgumentParser(description="Resizing the CAMUS dataset.")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default=PATH_DATA,
        help="Path to the original CAMUS dataset.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=64,
        help="Size to resample the CAMUS dataset: (size, size).",
    )
    args = parser.parse_args()
    resize_dataset(args.data_dir, args.size)


if __name__ == "__main__":
    main()
