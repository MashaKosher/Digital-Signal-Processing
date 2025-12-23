from typing import Tuple, Optional
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow (PIL) is required. Install with: pip install pillow")


def _ensure_numpy_array(img: Image.Image) -> "np.ndarray":
    if np is None:
        raise SystemExit("NumPy is required. Install with: pip install numpy")
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def _to_pil_image(arr: "np.ndarray") -> Image.Image:
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    if arr_clipped.ndim == 2:
        return Image.fromarray(arr_clipped)
    if arr_clipped.shape[2] == 1:
        arr_clipped = arr_clipped[:, :, 0]
        return Image.fromarray(arr_clipped)
    return Image.fromarray(arr_clipped)


def _pad_image(arr: "np.ndarray", pad_y: int, pad_x: int, mode: str = "reflect") -> "np.ndarray":
    if mode not in ("reflect", "edge", "constant"):
        mode = "reflect"
    if mode == "constant":
        return np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="constant")
    if mode == "edge":
        return np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    return np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")


def _convolve(arr: "np.ndarray", kernel: "np.ndarray", padding: str = "reflect") -> "np.ndarray":
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    arr_p = _pad_image(arr, ph, pw, mode=padding)  # (H+2p, W+2p, C)
    c_first = np.transpose(arr_p, (2, 0, 1))  # (C, Hp, Wp)
    windows = sliding_window_view(c_first, (1, kh, kw))  # (C, H, W, 1, kh, kw)
    windows = windows[:, :, :, 0, :, :]  # (C, H, W, kh, kw)
    kflip = np.flipud(np.fliplr(kernel)).astype(np.float32)
    out_c = np.einsum('chwij,ij->chw', windows, kflip)
    out = np.transpose(out_c, (1, 2, 0)).astype(np.float32)
    return out


def _convolve1d(arr: "np.ndarray", k1d: "np.ndarray", axis: int, padding: str = "reflect") -> "np.ndarray":
    from numpy.lib.stride_tricks import sliding_window_view
    k1d = np.asarray(k1d, dtype=np.float32)
    klen = int(k1d.shape[0])
    pad = klen // 2
    if axis == 0:
        arr_p = _pad_image(arr, pad, 0, mode=padding)
        c_first = np.transpose(arr_p, (2, 0, 1))  # (C, Hp, W)
        windows = sliding_window_view(c_first, (1, klen, 1))  # (C, H, W, 1, klen, 1)
        windows = windows[:, :, :, 0, :, 0]  # (C, H, W, klen)
        out_c = np.einsum('chwk,k->chw', windows, k1d)
        out = np.transpose(out_c, (1, 2, 0))
        return out.astype(np.float32)
    elif axis == 1:
        arr_p = _pad_image(arr, 0, pad, mode=padding)
        c_first = np.transpose(arr_p, (2, 0, 1))  # (C, H, Wp)
        windows = sliding_window_view(c_first, (1, 1, klen))  # (C, H, W, 1, 1, klen)
        windows = windows[:, :, :, 0, 0, :]  # (C, H, W, klen)
        out_c = np.einsum('chwk,k->chw', windows, k1d)
        out = np.transpose(out_c, (1, 2, 0))
        return out.astype(np.float32)
    else:
        raise ValueError("axis must be 0 (height) or 1 (width)")


def box_blur(img: Image.Image, size: int = 3, padding: str = "reflect") -> Image.Image:
    if size % 2 == 0 or size < 1:
        raise ValueError("Kernel size must be positive odd integer")
    arr = _ensure_numpy_array(img)
    k1d = np.ones((size,), dtype=np.float32) / float(size)
    out = _convolve1d(arr, k1d, axis=1, padding=padding)
    out = _convolve1d(out, k1d, axis=0, padding=padding)
    return _to_pil_image(out)


def _gaussian_kernel(size: int, sigma: float) -> "np.ndarray":
    if size % 2 == 0 or size < 1:
        raise ValueError("Kernel size must be positive odd integer")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _gaussian_kernel1d(size: int, sigma: float) -> "np.ndarray":
    if size % 2 == 0 or size < 1:
        raise ValueError("Kernel size must be positive odd integer")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    k = np.exp(-(ax**2) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def gaussian_blur(img: Image.Image, size: int = 5, sigma: float = 1.0, padding: str = "reflect") -> Image.Image:
    arr = _ensure_numpy_array(img)
    k1d = _gaussian_kernel1d(size, sigma)
    out = _convolve1d(arr, k1d, axis=1, padding=padding)
    out = _convolve1d(out, k1d, axis=0, padding=padding)
    return _to_pil_image(out)


def median_filter(img: Image.Image, size: int = 3, padding: str = "reflect") -> Image.Image:
    if size % 2 == 0 or size < 1:
        raise ValueError("Window size must be positive odd integer")
    from numpy.lib.stride_tricks import sliding_window_view
    arr = _ensure_numpy_array(img)
    kh, kw = size, size
    ph, pw = kh // 2, kw // 2
    arr_p = _pad_image(arr, ph, pw, mode=padding)
    c_first = np.transpose(arr_p, (2, 0, 1))  # (C, Hp, Wp)
    windows = sliding_window_view(c_first, (1, kh, kw))  # (C, H, W, 1, kh, kw)
    windows = windows[:, :, :, 0, :, :]  # (C, H, W, kh, kw)
    out_c = np.median(windows, axis=(-2, -1))  # (C, H, W)
    out = np.transpose(out_c, (1, 2, 0)).astype(np.float32)
    return _to_pil_image(out)


def sobel_operator(img: Image.Image, padding: str = "reflect", to_grayscale: bool = True) -> Image.Image:
    arr = _ensure_numpy_array(img)
    if arr.shape[2] == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr[:, :, 0]
    gray3 = gray[:, :, None].astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]], dtype=np.float32)
    gx = _convolve(gray3, kx, padding=padding)[:, :, 0]
    gy = _convolve(gray3, ky, padding=padding)[:, :, 0]
    mag = np.sqrt(gx * gx + gy * gy)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax > mmin:
        mag = (mag - mmin) * (255.0 / (mmax - mmin))
    else:
        mag = np.zeros_like(mag)
    if to_grayscale:
        return _to_pil_image(mag.astype(np.float32))
    else:
        rgb = np.repeat(mag[:, :, None], 3, axis=2)
        return _to_pil_image(rgb.astype(np.float32))


def sobel_rgb_channels(img: Image.Image, padding: str = "reflect") -> Image.Image:
    arr = _ensure_numpy_array(img)
    if arr.shape[2] != 3:
        raise ValueError("Error")
    
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]], dtype=np.float32)
    
    result_channels = []
    for c in range(3):
        channel = arr[:, :, c:c+1].astype(np.float32)
        gx = _convolve(channel, kx, padding=padding)[:, :, 0]
        gy = _convolve(channel, ky, padding=padding)[:, :, 0]
        mag = np.sqrt(gx * gx + gy * gy)
        
        mmin, mmax = float(mag.min()), float(mag.max())
        if mmax > mmin:
            mag = (mag - mmin) * (255.0 / (mmax - mmin))
        else:
            mag = np.zeros_like(mag)
        
        result_channels.append(mag)
    
    result_rgb = np.stack(result_channels, axis=2).astype(np.float32)
    return _to_pil_image(result_rgb)


class ImageFilterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image Filters: Box, Gaussian, Median, Sobel")
        self.root.geometry("1100x700")

        self.original_image: Optional[Image.Image] = None
        self.processed_image: Optional[Image.Image] = None
        self.display_max_size: Tuple[int, int] = (480, 480)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Top toolbar
        toolbar = ttk.Frame(main)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Open", command=self.on_open).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Save", command=self.on_save).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="Reset", command=self.on_reset).pack(side=tk.LEFT, padx=4, pady=4)

        # Content area
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: previews
        previews = ttk.Frame(content)
        previews.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.orig_label = ttk.Label(previews, text="Original")
        self.orig_label.pack(pady=(8, 0))
        self.orig_canvas = tk.Label(previews, bd=1, relief=tk.SUNKEN)
        self.orig_canvas.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)

        self.proc_label = ttk.Label(previews, text="Processed")
        self.proc_label.pack(pady=(8, 0))
        self.proc_canvas = tk.Label(previews, bd=1, relief=tk.SUNKEN)
        self.proc_canvas.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)

        # Right: controls (tabs)
        controls = ttk.Notebook(content)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        # Box Blur tab
        tab_box = ttk.Frame(controls)
        self.box_size = tk.IntVar(value=3)
        ttk.Label(tab_box, text="Kernel size (odd)").pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Scale(tab_box, from_=1, to=70, orient=tk.HORIZONTAL, variable=self.box_size, command=lambda e: None).pack(fill=tk.X, padx=8)
        ttk.Button(tab_box, text="Apply Box Blur", command=self.apply_box).pack(padx=8, pady=8, fill=tk.X)
        controls.add(tab_box, text="Box Blur")

        # Gaussian tab
        tab_gauss = ttk.Frame(controls)
        self.gauss_size = tk.IntVar(value=5)
        self.gauss_sigma = tk.DoubleVar(value=1.2)
        ttk.Label(tab_gauss, text="Kernel size (odd)").pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Scale(tab_gauss, from_=3, to=70, orient=tk.HORIZONTAL, variable=self.gauss_size, command=lambda e: None).pack(fill=tk.X, padx=8)
        ttk.Label(tab_gauss, text="Sigma").pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Scale(tab_gauss, from_=0.5, to=20.0, orient=tk.HORIZONTAL, variable=self.gauss_sigma, command=lambda e: None).pack(fill=tk.X, padx=8)
        ttk.Button(tab_gauss, text="Apply Gaussian", command=self.apply_gaussian).pack(padx=8, pady=8, fill=tk.X)
        controls.add(tab_gauss, text="Gaussian")

        # Median tab
        tab_median = ttk.Frame(controls)
        self.median_size = tk.IntVar(value=3)
        ttk.Label(tab_median, text="Window size (odd)").pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Scale(tab_median, from_=1, to=25, orient=tk.HORIZONTAL, variable=self.median_size, command=lambda e: None).pack(fill=tk.X, padx=8)
        ttk.Button(tab_median, text="Apply Median", command=self.apply_median).pack(padx=8, pady=8, fill=tk.X)
        controls.add(tab_median, text="Median")

        # Sobel tab
        tab_sobel = ttk.Frame(controls)
        self.sobel_mode = tk.StringVar(value="grayscale")
        ttk.Label(tab_sobel, text="Sobel Mode:").pack(anchor=tk.W, padx=8, pady=(8, 2))
        ttk.Radiobutton(tab_sobel, text="Grayscale", variable=self.sobel_mode, value="grayscale").pack(anchor=tk.W, padx=8)
        ttk.Radiobutton(tab_sobel, text="RGB Channels", variable=self.sobel_mode, value="rgb_channels").pack(anchor=tk.W, padx=8)
        ttk.Button(tab_sobel, text="Apply Sobel", command=self.apply_sobel).pack(padx=8, pady=8, fill=tk.X)
        controls.add(tab_sobel, text="Sobel")

    def _fit_to_display(self, img: Image.Image) -> Image.Image:
        max_w, max_h = self.display_max_size
        w, h = img.size
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        new_size = (int(w * scale), int(h * scale))
        if new_size != img.size:
            return img.resize(new_size, Image.LANCZOS)
        return img

    def _update_previews(self) -> None:
        if self.original_image is not None:
            disp_orig = self._fit_to_display(self.original_image)
            self._orig_tk = ImageTk.PhotoImage(disp_orig)
            self.orig_canvas.configure(image=self._orig_tk)
        if self.processed_image is not None:
            disp_proc = self._fit_to_display(self.processed_image)
            self._proc_tk = ImageTk.PhotoImage(disp_proc)
            self.proc_canvas.configure(image=self._proc_tk)

    def on_open(self) -> None:
        path = filedialog.askopenfilename(title="Open image", filetypes=[
            ("Image files", ".png .jpg .jpeg .bmp .tiff .tif"),
            ("All files", "*.*"),
        ])
        if not path:
            return
        try:
            img = Image.open(path)
            self.original_image = img.convert("RGB")
            self.processed_image = self.original_image.copy()
            self._update_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def on_save(self) -> None:
        if self.processed_image is None:
            messagebox.showinfo("Info", "No processed image to save.")
            return
        path = filedialog.asksaveasfilename(title="Save image", defaultextension=".png",
                                            filetypes=[("PNG", ".png"), ("JPEG", ".jpg .jpeg"), ("BMP", ".bmp"), ("TIFF", ".tif .tiff")])
        if not path:
            return
        try:
            self.processed_image.save(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def on_reset(self) -> None:
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self._update_previews()

    def _get_base_image(self) -> Optional[Image.Image]:
        if self.processed_image is not None:
            return self.processed_image
        return self.original_image

    def apply_box(self) -> None:
        img = self._get_base_image()
        if img is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        k = int(round(self.box_size.get()))
        if k % 2 == 0:
            k += 1
        try:
            self.processed_image = box_blur(img, size=k)
            self._update_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Box blur failed: {e}")

    def apply_gaussian(self) -> None:
        img = self._get_base_image()
        if img is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        size = int(round(self.gauss_size.get()))
        if size % 2 == 0:
            size += 1
        sigma = float(self.gauss_sigma.get())
        try:
            self.processed_image = gaussian_blur(img, size=size, sigma=sigma)
            self._update_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Gaussian blur failed: {e}")

    def apply_median(self) -> None:
        img = self._get_base_image()
        if img is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        k = int(round(self.median_size.get()))
        if k % 2 == 0:
            k += 1
        try:
            self.processed_image = median_filter(img, size=k)
            self._update_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Median filter failed: {e}")

    def apply_sobel(self) -> None:
        img = self._get_base_image()
        if img is None:
            messagebox.showinfo("Info", "Open an image first.")
            return
        try:
            mode = self.sobel_mode.get()
            if mode == "grayscale":
                self.processed_image = sobel_operator(img, to_grayscale=True)
            elif mode == "rgb_channels":
                self.processed_image = sobel_rgb_channels(img)
            else:
                messagebox.showerror("Error", f"Unknown Sobel mode: {mode}")
                return
            self._update_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Sobel failed: {e}")


def main() -> None:
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


