"""
Artwork Tiler & Micro-Patch Slicer.
Divides high-resolution canvases into micro-texture patches preserving native pixel resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator
import numpy as np
from PIL import Image


@dataclass
class PatchMetadata:
    """Metadata tracking the spatial coordinates and characteristics of a patch."""
    patch_id: int
    x: int
    y: int
    width: int
    height: int
    canvas_width: int
    canvas_height: int
    variance: float = 0.0
    is_valid: bool = True
    pixels_per_mm: Optional[float] = None
    physical_mm_size: Optional[Tuple[float, float]] = None

    @property
    def normalized_center(self) -> Tuple[float, float]:
        """Returns normalized (cx, cy) in range [0, 1]."""
        cx = (self.x + self.width / 2.0) / self.canvas_width
        cy = (self.y + self.height / 2.0) / self.canvas_height
        return (cx, cy)


@dataclass
class PatchGrid:
    """Collection of extracted patches and their spatial metadata."""
    patches: List[np.ndarray]
    metadata: List[PatchMetadata]
    canvas_size: Tuple[int, int]  # (width, height)
    patch_size: int
    stride: int

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, PatchMetadata]]:
        return zip(self.patches, self.metadata)


class ArtworkTiler:
    """Slices gigapixel artworks into high-fidelity micro-patches."""

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        min_variance: float = 15.0,
        discard_borders: bool = True,
    ) -> None:
        """
        Args:
            patch_size: Dimensions (width and height) of square patches (e.g. 256 or 512).
            stride: Stride between successive patches. If None, defaults to patch_size (non-overlapping).
            min_variance: Minimum pixel intensity variance required to retain a patch (filters blank margins).
            discard_borders: If True, skips incomplete patches at edges that are smaller than patch_size.
        """
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.min_variance = min_variance
        self.discard_borders = discard_borders

    def slice_image(self, image: Image.Image | np.ndarray) -> PatchGrid:
        """
        Slices a canvas into a PatchGrid.
        Accepts PIL Image or RGB uint8 NumPy array (H, W, 3).
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)

        h, w = arr.shape[:2]
        patches: List[np.ndarray] = []
        meta_list: List[PatchMetadata] = []

        patch_id = 0
        y_max = h - self.patch_size + 1 if self.discard_borders else h
        x_max = w - self.patch_size + 1 if self.discard_borders else w

        for y in range(0, max(1, y_max), self.stride):
            for x in range(0, max(1, x_max), self.stride):
                actual_w = min(self.patch_size, w - x)
                actual_h = min(self.patch_size, h - y)

                if self.discard_borders and (actual_w < self.patch_size or actual_h < self.patch_size):
                    continue

                patch = arr[y : y + actual_h, x : x + actual_w]

                # Compute variance across grayscale version to check for informative texture
                if patch.ndim == 3:
                    gray = (
                        0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
                    )
                else:
                    gray = patch

                var = float(np.var(gray))
                is_valid = var >= self.min_variance

                if is_valid:
                    patches.append(patch)
                    meta_list.append(
                        PatchMetadata(
                            patch_id=patch_id,
                            x=x,
                            y=y,
                            width=actual_w,
                            height=actual_h,
                            canvas_width=w,
                            canvas_height=h,
                            variance=var,
                            is_valid=True,
                        )
                    )
                    patch_id += 1

        return PatchGrid(
            patches=patches,
            metadata=meta_list,
            canvas_size=(w, h),
            patch_size=self.patch_size,
            stride=self.stride,
        )

    def slice_metric_patches(
        self,
        image: Image.Image | np.ndarray,
        pixels_per_mm: float,
        target_patch_mm: float = 25.0,
        output_pixel_size: int = 256,
    ) -> PatchGrid:
        """
        Slices canvas into patches of calibrated physical metric dimensions (in millimeters)
        and resamples each patch to a fixed standard pixel resolution.
        
        Mitigates sensor and digitization resolution discrepancies (e.g. 1200 DPI vs 300 DPI)
        ensuring that every patch corresponds to the exact same physical area of paint.
        """
        if pixels_per_mm <= 0:
            raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")

        # Physical window size in raw canvas pixels
        crop_px = max(8, int(round(target_patch_mm * pixels_per_mm)))
        stride_px = max(4, int(round(crop_px * (self.stride / self.patch_size))))

        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            arr = np.array(pil_img)
        else:
            arr = np.asarray(image)
            pil_img = Image.fromarray(arr)

        h, w = arr.shape[:2]
        patches: List[np.ndarray] = []
        meta_list: List[PatchMetadata] = []

        patch_id = 0
        y_max = h - crop_px + 1 if self.discard_borders else h
        x_max = w - crop_px + 1 if self.discard_borders else w

        for y in range(0, max(1, y_max), stride_px):
            for x in range(0, max(1, x_max), stride_px):
                actual_w = min(crop_px, w - x)
                actual_h = min(crop_px, h - y)

                if self.discard_borders and (actual_w < crop_px or actual_h < crop_px):
                    continue

                patch_crop = arr[y : y + actual_h, x : x + actual_w]

                # Check variance
                if patch_crop.ndim == 3:
                    gray = 0.299 * patch_crop[:, :, 0] + 0.587 * patch_crop[:, :, 1] + 0.114 * patch_crop[:, :, 2]
                else:
                    gray = patch_crop

                var = float(np.var(gray))
                if var < self.min_variance:
                    continue

                # Resample to normalized output pixel size (e.g. 256x256)
                crop_pil = Image.fromarray(patch_crop)
                resampled = crop_pil.resize((output_pixel_size, output_pixel_size), Image.Resampling.LANCZOS)
                patch_arr = np.array(resampled)

                patches.append(patch_arr)
                meta_list.append(
                    PatchMetadata(
                        patch_id=patch_id,
                        x=x,
                        y=y,
                        width=actual_w,
                        height=actual_h,
                        canvas_width=w,
                        canvas_height=h,
                        variance=var,
                        is_valid=True,
                        pixels_per_mm=pixels_per_mm,
                        physical_mm_size=(actual_w / pixels_per_mm, actual_h / pixels_per_mm),
                    )
                )
                patch_id += 1

        return PatchGrid(
            patches=patches,
            metadata=meta_list,
            canvas_size=(w, h),
            patch_size=output_pixel_size,
            stride=self.stride,
        )
