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
    """Slices gigapixel artworks into high-fidelity micro-patches with automatic frame isolation."""

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        min_variance: float = 15.0,
        discard_borders: bool = True,
        auto_crop_frame: bool = True,
    ) -> None:
        """
        Args:
            patch_size: Dimensions (width and height) of square patches (e.g. 256 or 512).
            stride: Stride between successive patches. If None, defaults to patch_size (non-overlapping).
            min_variance: Minimum pixel intensity variance required to retain a patch (filters blank margins).
            discard_borders: If True, skips incomplete patches at edges that are smaller than patch_size.
            auto_crop_frame: If True, automatically detects and crops picture frames, mounts, and text banners.
        """
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.min_variance = min_variance
        self.discard_borders = discard_borders
        self.auto_crop_frame = auto_crop_frame

    @staticmethod
    def detect_canvas_bbox(image: Image.Image | np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detects the true painted canvas bounding box (x0, y0, x1, y1), isolating the artwork from:
        1. Uniform outer studio margins, cardboard mounts, and archive text strips/banners.
        2. Rectilinear gilded/wooden picture frames via directional rabbet gradient projection.
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)

        h, w = arr.shape[:2]
        if h < 64 or w < 64 or arr.ndim < 3:
            return 0, 0, w, h

        rgb = arr[:, :, :3].astype(np.float32)
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Stage 1: Strip uniform margins, cardboard passe-partouts, and archive text banners
        y0, y1, x0, x1 = 0, h, 0, w
        max_strip_y = int(h * 0.26)
        max_strip_x = int(w * 0.26)

        for y in range(max_strip_y):
            row = lum[y, int(w * 0.15) : int(w * 0.85)]
            if np.std(row) < 13.5 or np.percentile(row, 30) > 222 or np.percentile(row, 70) < 18:
                y0 = y + 1
            else:
                break
        else:
            y0 = 0

        for y in range(h - 1, h - max_strip_y, -1):
            row = lum[y, int(w * 0.15) : int(w * 0.85)]
            p75 = np.percentile(row, 75)
            p50 = np.percentile(row, 50)
            if np.std(row) < 13.5 or (p75 > 220 and p50 > 195) or (np.percentile(row, 70) < 20):
                y1 = y
            else:
                break
        else:
            y1 = h

        for x in range(max_strip_x):
            col = lum[y0:y1, x]
            if col.size == 0:
                break
            mid = col[int(col.size * 0.15) : int(col.size * 0.85)]
            if np.std(mid) < 13.5 or np.percentile(mid, 30) > 222 or np.percentile(mid, 70) < 18:
                x0 = x + 1
            else:
                break
        else:
            x0 = 0

        for x in range(w - 1, w - max_strip_x, -1):
            col = lum[y0:y1, x]
            if col.size == 0:
                break
            mid = col[int(col.size * 0.15) : int(col.size * 0.85)]
            if np.std(mid) < 13.5 or np.percentile(mid, 30) > 222 or np.percentile(mid, 70) < 18:
                x1 = x
            else:
                break
        else:
            x1 = w

        # Stage 2: Detect rectilinear picture frame rabbets on [y0:y1, x0:x1]
        sub_rgb = rgb[y0:y1, x0:x1]
        sub_lum = lum[y0:y1, x0:x1]
        sh, sw = sub_lum.shape
        if sh < 80 or sw < 80:
            return x0, y0, x1, y1

        gy = np.abs(np.diff(sub_lum, axis=0))
        gx = np.abs(np.diff(sub_lum, axis=1))

        core_y0, core_y1 = int(sh * 0.26), int(sh * 0.74)
        core_x0, core_x1 = int(sw * 0.26), int(sw * 0.74)
        core_rgb = np.median(sub_rgb[core_y0:core_y1, core_x0:core_x1].reshape(-1, 3), axis=0)

        prof_top = np.mean(gy[: int(sh * 0.24), int(sw * 0.22) : int(sw * 0.78)], axis=1)
        prof_bot = np.mean(gy[sh - 1 - int(sh * 0.24) :, int(sw * 0.22) : int(sw * 0.78)][::-1], axis=1)
        prof_left = np.mean(gx[int(sh * 0.22) : int(sh * 0.78), : int(sw * 0.24)], axis=0)
        prof_right = np.mean(gx[int(sh * 0.22) : int(sh * 0.78), sw - 1 - int(sw * 0.24) :][:, ::-1], axis=0)

        base_y = float(np.median(np.mean(gy[core_y0:core_y1, core_x0:core_x1], axis=1))) + 1.5
        base_x = float(np.median(np.mean(gx[core_y0:core_y1, core_x0:core_x1], axis=0))) + 1.5

        def find_rabbet(prof: np.ndarray, base: float, is_axis_y: bool, from_end: bool) -> Tuple[int, float]:
            n = len(prof)
            min_d = max(3, int(n * 0.05))
            if n <= min_d + 2:
                return 0, 0.0
            kernel = np.array([0.25, 0.5, 0.25])
            sprof = np.convolve(prof, kernel, mode="same")
            ratios = sprof / base
            best_d = 0
            for d in range(n - 2, min_d - 1, -1):
                if ratios[d] >= 2.15 and ratios[d] >= ratios[d - 1] and ratios[d] >= ratios[d + 1]:
                    if is_axis_y:
                        if not from_end:
                            outer_strip = sub_rgb[:d, int(sw * 0.22) : int(sw * 0.78)]
                            inner_strip = sub_rgb[d + 2 : min(sh, d + 2 + max(8, int(sh * 0.06))), int(sw * 0.22) : int(sw * 0.78)]
                        else:
                            outer_strip = sub_rgb[sh - d :, int(sw * 0.22) : int(sw * 0.78)]
                            inner_strip = sub_rgb[max(0, sh - d - 2 - max(8, int(sh * 0.06))) : sh - d - 2, int(sw * 0.22) : int(sw * 0.78)]
                    else:
                        if not from_end:
                            outer_strip = sub_rgb[int(sh * 0.22) : int(sh * 0.78), :d]
                            inner_strip = sub_rgb[int(sh * 0.22) : int(sh * 0.78), d + 2 : min(sw, d + 2 + max(8, int(sw * 0.06)))]
                        else:
                            outer_strip = sub_rgb[int(sh * 0.22) : int(sh * 0.78), sw - d :]
                            inner_strip = sub_rgb[int(sh * 0.22) : int(sh * 0.78), max(0, sw - d - 2 - max(8, int(sw * 0.06))) : sw - d - 2]
                    if outer_strip.size == 0 or inner_strip.size == 0:
                        continue
                    outer_med = np.median(outer_strip.reshape(-1, 3), axis=0)
                    inner_med = np.median(inner_strip.reshape(-1, 3), axis=0)
                    diff_inner = float(np.linalg.norm(outer_med - inner_med))
                    diff_core = float(np.linalg.norm(outer_med - core_rgb))
                    if diff_inner >= 16.0 or diff_core >= 24.0 or ratios[d] >= 3.2:
                        best_d = d + 3
                        break
            return best_d, float(np.max(ratios[min_d:]))

        dt, rt = find_rabbet(prof_top, base_y, True, False)
        db, rb = find_rabbet(prof_bot, base_y, True, True)
        dl, rl = find_rabbet(prof_left, base_x, False, False)
        dr, rr = find_rabbet(prof_right, base_x, False, True)

        active_sides = sum(1 for d in (dt, db, dl, dr) if d > 0)
        if active_sides >= 2 or max(rt, rb, rl, rr) >= 3.6:
            if dt > int(sh * 0.03) and db > int(sh * 0.03):
                ref_d = int(min(dt / sh, db / sh) * sw * 0.85)
                frame_med = np.median(sub_rgb[:dt, int(sw * 0.25) : int(sw * 0.75)].reshape(-1, 3), axis=0)
                if dl < ref_d // 2 and ref_d > 8:
                    l_med = np.median(sub_rgb[int(sh * 0.25) : int(sh * 0.75), :ref_d].reshape(-1, 3), axis=0)
                    if np.linalg.norm(l_med - frame_med) < 22.0:
                        dl = ref_d
                if dr < ref_d // 2 and ref_d > 8:
                    r_med = np.median(sub_rgb[int(sh * 0.25) : int(sh * 0.75), sw - ref_d :].reshape(-1, 3), axis=0)
                    if np.linalg.norm(r_med - frame_med) < 22.0:
                        dr = ref_d
            if dl > int(sw * 0.03) and dr > int(sw * 0.03):
                ref_d = int(min(dl / sw, dr / sw) * sh * 0.85)
                frame_med = np.median(sub_rgb[int(sh * 0.25) : int(sh * 0.75), :dl].reshape(-1, 3), axis=0)
                if dt < ref_d // 2 and ref_d > 8:
                    t_med = np.median(sub_rgb[:ref_d, int(sw * 0.25) : int(sw * 0.75)].reshape(-1, 3), axis=0)
                    if np.linalg.norm(t_med - frame_med) < 22.0:
                        dt = ref_d
                if db < ref_d // 2 and ref_d > 8:
                    b_med = np.median(sub_rgb[sh - ref_d :, int(sw * 0.25) : int(sw * 0.75)].reshape(-1, 3), axis=0)
                    if np.linalg.norm(b_med - frame_med) < 22.0:
                        db = ref_d

            y0 += dt
            y1 -= db
            x0 += dl
            x1 -= dr

        if x1 - x0 < int(w * 0.4) or y1 - y0 < int(h * 0.4):
            return 0, 0, w, h

        return x0, y0, x1, y1

    @classmethod
    def crop_frame(cls, image: Image.Image | np.ndarray) -> np.ndarray:
        """Returns the RGB NumPy array cropped to the isolated canvas bounding box."""
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        x0, y0, x1, y1 = cls.detect_canvas_bbox(arr)
        return arr[y0:y1, x0:x1]

    def slice_image(self, image: Image.Image | np.ndarray) -> PatchGrid:
        """
        Slices a canvas into a PatchGrid.
        Accepts PIL Image or RGB uint8 NumPy array (H, W, 3).
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)

        if self.auto_crop_frame and arr.ndim == 3 and min(arr.shape[:2]) >= 128:
            x0, y0, x1, y1 = self.detect_canvas_bbox(arr)
            arr = arr[y0:y1, x0:x1]

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

        if self.auto_crop_frame and arr.ndim == 3 and min(arr.shape[:2]) >= 128:
            x0, y0, x1, y1 = self.detect_canvas_bbox(arr)
            arr = arr[y0:y1, x0:x1]

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
