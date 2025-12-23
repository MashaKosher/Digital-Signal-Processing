"""Tkinter app for visualizing cross-correlation (NCC) and auto-correlation (ACF)."""
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageTk


MAX_CROSS_DIM = 256
MAX_AUTO_DIM = 256


@dataclass
class PreparedImage:
    pil: Image.Image
    scale: float
    matrix: np.ndarray


@dataclass
class MatchCandidate:
    value: float
    map_x: int
    map_y: int
    rect: Tuple[int, int, int, int]


@dataclass
class AutoPeak:
    dx: int
    dy: int
    value: float


def pil_to_gray_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


class CorrelationMath:
    @staticmethod
    def normalized_cross_correlation(field: np.ndarray, templ: np.ndarray) -> np.ndarray:
        if field.ndim != 2 or templ.ndim != 2:
            raise ValueError("field и templ должны быть 2D")

        F = field.astype(np.float32, copy=False)
        T = templ.astype(np.float32, copy=False)
        h, w = T.shape
        H, W = F.shape
        if h > H or w > W:
            raise ValueError("Шаблон не помещается в изображение")

        area = h * w
        tol = 1e-6

        T0 = T - T.mean()
        T_l2 = float(np.sqrt(np.sum(T0 * T0)))
        if T_l2 < tol:
            return np.zeros((H - h + 1, W - w + 1), dtype=np.float32)

        def integral(a: np.ndarray) -> np.ndarray:
            return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")

        IF = integral(F)
        IF2 = integral(F * F)

        y0 = np.arange(0, H - h + 1)[:, None]
        x0 = np.arange(0, W - w + 1)[None, :]
        y1, x1 = y0 + h, x0 + w

        sumF = IF[y1, x1] - IF[y0, x1] - IF[y1, x0] + IF[y0, x0]
        sumF2 = IF2[y1, x1] - IF2[y0, x1] - IF2[y1, x0] + IF2[y0, x0]
        meanF = sumF / area

        sum_centered_sq = sumF2 - area * (meanF ** 2)
        sum_centered_sq = np.maximum(sum_centered_sq, 0.0)
        F_l2 = np.sqrt(sum_centered_sq)

        patches = sliding_window_view(F, (h, w))
        dot = np.tensordot(patches - meanF[..., None, None], T0, axes=([2, 3], [0, 1]))

        ncc = np.zeros_like(dot, dtype=np.float32)
        mask = F_l2 > tol
        ncc[mask] = (dot[mask] / (T_l2 * F_l2[mask])).astype(np.float32)

        np.clip(ncc, -1.0, 1.0, out=ncc)
        return ncc

    @staticmethod
    def autocorr_fft(gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32, copy=False)
        g = g - g.mean()
        h, w = g.shape

        H, W = 2 * h - 1, 2 * w - 1
        F = np.fft.fft2(g, s=(2 * h, 2 * w))
        acf_full = np.fft.ifft2(np.abs(F) ** 2).real

        ones = np.ones_like(g, dtype=np.float32)
        O = np.fft.fft2(ones, s=(2 * h, 2 * w))
        overlap_full = np.fft.ifft2(np.abs(O) ** 2).real

        acf = np.fft.fftshift(acf_full)
        overlap = np.fft.fftshift(overlap_full)
        cy, cx = acf.shape[0] // 2, acf.shape[1] // 2
        acf = acf[cy - (H // 2): cy + (H // 2) + 1, cx - (W // 2): cx + (W // 2) + 1]
        overlap = overlap[cy - (H // 2): cy + (H // 2) + 1, cx - (W // 2): cx + (W // 2) + 1]

        var = float(g.var()) if g.size else 1.0
        acf = acf / (np.maximum(overlap, 1.0) * max(var, 1e-12))
        max_overlap = float(overlap.max()) if overlap.size else 1.0
        if max_overlap > 0:
            acf *= overlap / max_overlap

        acf[H // 2, W // 2] = 1.0
        return acf.astype(np.float32, copy=False)

    @staticmethod
    def correlation_to_image(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("Correlation map is empty")
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        span = max(arr_max - arr_min, 1e-12)
        norm = (arr - arr_min) / span
        img = Image.fromarray(np.uint8(np.clip(norm * 255.0, 0, 255)), mode="L")
        return img.convert("RGB")

    @staticmethod
    def top_peaks_from_acf(acf: np.ndarray, k: int = 20, exclude_r: int = 7, nms_r: Optional[int] = None) -> List[
        AutoPeak]:
        h, w = acf.shape
        cy, cx = h // 2, w // 2
        if nms_r is None:
            nms_r = max(3, min(h, w) // 40)

        work = acf.copy()
        work[cy - exclude_r: cy + exclude_r + 1, cx - exclude_r: cx + exclude_r + 1] = -np.inf
        peaks: List[AutoPeak] = []
        for _ in range(k):
            y, x = np.unravel_index(np.argmax(work), work.shape)
            val = float(work[y, x])
            if not np.isfinite(val) or val <= 0:
                break
            peaks.append(AutoPeak(dx=x - cx, dy=y - cy, value=val))
            y0, y1 = max(0, y - nms_r), min(h, y + nms_r + 1)
            x0, x1 = max(0, x - nms_r), min(w, x + nms_r + 1)
            work[y0:y1, x0:x1] = -np.inf
        return peaks


def nms_peaks_from_map(corr: np.ndarray, max_peaks: int, radius: int, min_score: float) -> List[Tuple[int, int, float]]:
    work = corr.copy()
    peaks: List[Tuple[int, int, float]] = []
    for _ in range(max_peaks):
        idx = np.unravel_index(np.argmax(work), work.shape)
        score = float(work[idx])
        if score < min_score:
            break
        y, x = int(idx[0]), int(idx[1])
        peaks.append((x, y, score))
        y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
        work[y0:y1, x0:x1] = -np.inf
    return peaks


def prepare_image(image: Image.Image, max_dim: int) -> PreparedImage:
    width, height = image.size
    max_side = max(width, height)
    if max_side > max_dim:
        scale = max_dim / max_side
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
    else:
        scale = 1.0
        resized = image.copy()
    return PreparedImage(pil=resized, scale=resized.width / width, matrix=pil_to_gray_np(resized))


class ManualCorrelationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Корреляционный анализ изображений")
        self.root.geometry("1400x900")

        self.original_image: Optional[Image.Image] = None
        self.cross_image: Optional[PreparedImage] = None
        self.auto_image: Optional[PreparedImage] = None
        self.fragment_original: Optional[Image.Image] = None
        self.fragment_prepared: Optional[PreparedImage] = None
        self.fragment_coords: Optional[Tuple[int, int, int, int]] = None

        self.match_threshold = tk.DoubleVar(value=0.8)
        self.auto_match_threshold = tk.DoubleVar(value=0.5)

        self.match_candidates: List[MatchCandidate] = []
        self.visible_match_candidates: List[MatchCandidate] = []
        self.auto_peaks: List[AutoPeak] = []
        self.auto_visible_peaks: List[AutoPeak] = []
        self.auto_acf: Optional[np.ndarray] = None
        self.auto_max_value: float = 1.0

        self.mode = tk.StringVar(value="cross")
        self.status_var = tk.StringVar(value="Загрузите изображение")

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(control_frame, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Взаимная корреляция", variable=self.mode, value="cross", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Автокорреляция", variable=self.mode, value="auto", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)

        self.cross_controls = tk.Frame(control_frame)
        self.cross_controls.pack(side=tk.LEFT, padx=10)
        tk.Button(self.cross_controls, text="Случайный фрагмент", command=self.select_random_fragment).pack(side=tk.LEFT, padx=5)
        tk.Button(self.cross_controls, text="Вычислить NCC", command=self.calculate_cross_correlation).pack(side=tk.LEFT, padx=5)

        self.auto_button = tk.Button(control_frame, text="Вычислить ACF", command=self.calculate_autocorrelation)

        tk.Label(self.root, textvariable=self.status_var).pack(fill=tk.X, padx=10)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Cross-correlation widgets
        self.cross_frame = tk.Frame(self.main_frame)
        self.cross_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.cross_frame, text="Исходное изображение").grid(row=0, column=0)
        self.original_canvas = tk.Canvas(self.cross_frame, width=500, height=400, bg="gray")
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.cross_frame, text="Фрагмент").grid(row=0, column=1)
        self.fragment_canvas = tk.Canvas(self.cross_frame, width=250, height=200, bg="gray")
        self.fragment_canvas.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.cross_frame, text="Карта NCC").grid(row=0, column=2)
        self.correlation_canvas = tk.Canvas(self.cross_frame, width=500, height=400, bg="gray")
        self.correlation_canvas.grid(row=1, column=2, padx=5, pady=5)

        match_panel = tk.Frame(self.cross_frame)
        match_panel.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        tk.Label(match_panel, text="Минимальный NCC").grid(row=0, column=0, sticky="w")
        tk.Scale(match_panel, from_=0.2, to=0.95, orient=tk.HORIZONTAL, resolution=0.01, variable=self.match_threshold, command=lambda _:
                 self._update_match_list()).grid(row=0, column=1, sticky="ew", padx=5)
        match_panel.columnconfigure(1, weight=1)

        self.match_listbox = tk.Listbox(match_panel, height=6, exportselection=False)
        self.match_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
        scrollbar = tk.Scrollbar(match_panel, orient=tk.VERTICAL, command=self.match_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky="ns")
        self.match_listbox.config(yscrollcommand=scrollbar.set)
        self.match_listbox.bind("<<ListboxSelect>>", self._on_match_selected)
        match_panel.rowconfigure(1, weight=1)

        # Auto-correlation widgets
        self.auto_frame = tk.Frame(self.main_frame)
        tk.Label(self.auto_frame, text="Исходное изображение").grid(row=0, column=0)
        self.auto_original_canvas = tk.Canvas(self.auto_frame, width=500, height=400, bg="gray")
        self.auto_original_canvas.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.auto_frame, text="Автокорреляционная функция").grid(row=0, column=1)
        self.auto_corr_canvas = tk.Canvas(self.auto_frame, width=400, height=400, bg="gray")
        self.auto_corr_canvas.grid(row=1, column=1, padx=5, pady=5)

        auto_panel = tk.Frame(self.auto_frame)
        auto_panel.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        tk.Label(auto_panel, text="Минимальная относительная сила").grid(row=0, column=0, sticky="w")
        tk.Scale(auto_panel, from_=0.1, to=0.9, orient=tk.HORIZONTAL, resolution=0.01, variable=self.auto_match_threshold, command=lambda _:
                 self._update_auto_match_list()).grid(row=0, column=1, sticky="ew", padx=5)
        auto_panel.columnconfigure(1, weight=1)

        self.auto_listbox = tk.Listbox(auto_panel, height=6, exportselection=False)
        self.auto_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=5)
        auto_scroll = tk.Scrollbar(auto_panel, orient=tk.VERTICAL, command=self.auto_listbox.yview)
        auto_scroll.grid(row=1, column=2, sticky="ns")
        self.auto_listbox.config(yscrollcommand=auto_scroll.set)
        self.auto_listbox.bind("<<ListboxSelect>>", self._on_auto_match_selected)
        auto_panel.rowconfigure(1, weight=1)

        self.auto_info_label = tk.Label(self.auto_frame, text="")
        self.auto_info_label.grid(row=3, column=0, columnspan=2, pady=5)

        self._on_mode_change()

    # ------------------------------------------------------------------ helpers
    def _on_mode_change(self) -> None:
        if self.mode.get() == "cross":
            self.cross_frame.pack(fill=tk.BOTH, expand=True)
            self.auto_frame.pack_forget()
            self.cross_controls.pack(side=tk.LEFT, padx=10)
            self.auto_button.pack_forget()
        else:
            self.cross_frame.pack_forget()
            self.auto_frame.pack(fill=tk.BOTH, expand=True)
            self.cross_controls.pack_forget()
            self.auto_button.pack(side=tk.LEFT, padx=5)

    def load_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            image = Image.open(filename).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл: {exc}")
            return
        self.original_image = image
        self.cross_image = prepare_image(image, MAX_CROSS_DIM)
        self.auto_image = prepare_image(image, MAX_AUTO_DIM)
        self.fragment_original = None
        self.fragment_prepared = None
        self.fragment_coords = None
        self.match_candidates = []
        self.auto_peaks = []
        self.auto_visible_peaks = []
        self.auto_acf = None
        self._display_on_canvas(image, self.original_canvas)
        self._display_on_canvas(image, self.auto_original_canvas)
        self._clear_match_list()
        self._clear_auto_match_list("Выполните автокорреляцию")
        self.status_var.set("Изображение загружено")

    def select_random_fragment(self) -> None:
        if not self.original_image or not self.cross_image:
            messagebox.showwarning("Предупреждение", "Загрузите изображение")
            return
        width, height = self.original_image.size
        if width < 40 or height < 40:
            messagebox.showerror("Ошибка", "Изображение слишком маленькое")
            return
        frag_w = random.randint(max(20, width // 6), max(30, width // 4))
        frag_h = random.randint(max(20, height // 6), max(30, height // 4))
        x = random.randint(0, width - frag_w)
        y = random.randint(0, height - frag_h)
        fragment = self.original_image.crop((x, y, x + frag_w, y + frag_h))
        if self.cross_image:
            scale = self.cross_image.scale
            scaled_box = (
                int(round(x * scale)),
                int(round(y * scale)),
                int(round((x + frag_w) * scale)),
                int(round((y + frag_h) * scale)),
            )
            processed_fragment = self.cross_image.pil.crop(scaled_box)
        else:
            processed_fragment = fragment.copy()
        self.fragment_original = fragment
        self.fragment_prepared = PreparedImage(pil=processed_fragment, scale=1.0, matrix=pil_to_gray_np(processed_fragment))
        self.fragment_coords = (x, y, x + frag_w, y + frag_h)
        self._display_on_canvas(fragment, self.fragment_canvas)
        self._clear_match_list()
        self.status_var.set(f"Фрагмент {frag_w}x{frag_h} выбран. Запустите NCC.")

    # ------------------------------------------------------------------ NCC
    def calculate_cross_correlation(self) -> None:
        if not self.cross_image or not self.fragment_prepared:
            messagebox.showwarning("Предупреждение", "Нужны изображение и фрагмент")
            return
        try:
            corr_map = CorrelationMath.normalized_cross_correlation(self.cross_image.matrix, self.fragment_prepared.matrix)
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        corr_img = CorrelationMath.correlation_to_image(corr_map)
        if self.cross_image:
            corr_img = corr_img.resize(self.cross_image.pil.size, Image.LANCZOS)
        self._display_on_canvas(corr_img, self.correlation_canvas)
        best_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        best_y, best_x = int(best_idx[0]), int(best_idx[1])
        best_value = float(corr_map[best_y, best_x])
        rect = self._rect_from_corr_coords(best_x, best_y, self.fragment_prepared.matrix.shape)
        self._highlight_fragment(rect)
        self.status_var.set(f"Фрагмент найден в ({rect[0]}, {rect[1]}), NCC={best_value:.3f}")

        self.match_candidates = self._build_match_candidates(corr_map, self.fragment_prepared.matrix.shape)
        self._update_match_list()

    def _rect_from_corr_coords(self, map_x: int, map_y: int, tmpl_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if not self.cross_image:
            return (0, 0, 0, 0)
        scale = 1.0 / (self.cross_image.scale or 1.0)
        rect_x = int(round(map_x * scale))
        rect_y = int(round(map_y * scale))
        rect_w = int(round(tmpl_shape[1] * scale))
        rect_h = int(round(tmpl_shape[0] * scale))
        return (rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)

    def _build_match_candidates(self, corr_map: np.ndarray, tmpl_shape: Tuple[int, int]) -> List[MatchCandidate]:
        peaks = nms_peaks_from_map(corr_map, max_peaks=40, radius=6, min_score=self.match_threshold.get())
        candidates: List[MatchCandidate] = []
        for x, y, score in peaks:
            rect = self._rect_from_corr_coords(x, y, tmpl_shape)
            candidates.append(MatchCandidate(value=score, map_x=x, map_y=y, rect=rect))
        return candidates

    def _clear_match_list(self, placeholder: str = "") -> None:
        self.match_candidates = []
        self.visible_match_candidates = []
        self.match_listbox.delete(0, tk.END)
        self.match_listbox.config(state=tk.DISABLED)

    def _update_match_list(self) -> None:
        self.match_listbox.delete(0, tk.END)
        threshold = self.match_threshold.get()
        filtered = [cand for cand in self.match_candidates if cand.value >= threshold]
        if not filtered:
            self.match_listbox.insert(tk.END, "Нет совпадений — увеличьте NCC")
            self.match_listbox.config(state=tk.DISABLED)
            self.visible_match_candidates = []
            return
        self.match_listbox.config(state=tk.NORMAL)
        self.visible_match_candidates = filtered
        for idx, cand in enumerate(filtered, start=1):
            self.match_listbox.insert(tk.END, f"{idx}. NCC={cand.value:.3f} @ ({cand.rect[0]}, {cand.rect[1]})")
        self.match_listbox.selection_clear(0, tk.END)

    def _on_match_selected(self, _event: Optional[tk.Event]) -> None:
        selection = self.match_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx >= len(self.visible_match_candidates):
            return
        cand = self.visible_match_candidates[idx]
        self._highlight_fragment(cand.rect)
        self.status_var.set(f"Выбран фрагмент NCC={cand.value:.3f}")

    # ------------------------------------------------------------------ Auto
    def calculate_autocorrelation(self) -> None:
        if not self.auto_image or not self.original_image:
            messagebox.showwarning("Предупреждение", "Нужно загрузить изображение")
            return
        gray = self.auto_image.matrix
        acf = CorrelationMath.autocorr_fft(gray)
        self.auto_acf = acf
        self._display_on_canvas(CorrelationMath.correlation_to_image(acf), self.auto_corr_canvas)
        peaks = CorrelationMath.top_peaks_from_acf(acf, k=200)
        self.auto_peaks = peaks
        self.auto_max_value = peaks[0].value if peaks else 1.0
        if peaks:
            preview = ", ".join(f"({p.dx},{p.dy})" for p in peaks[:3])
            self.auto_info_label.config(text=f"Топ смещения: {preview}")
        else:
            self.auto_info_label.config(text="Повторы не обнаружены")
        self._update_auto_match_list()
        if self.auto_visible_peaks:
            self.auto_listbox.selection_set(0)
            self._on_auto_match_selected(None)

    def _clear_auto_match_list(self, placeholder: str) -> None:
        self.auto_peaks = []
        self.auto_visible_peaks = []
        self.auto_listbox.delete(0, tk.END)
        self.auto_listbox.insert(tk.END, placeholder)
        self.auto_listbox.config(state=tk.DISABLED)

    def _update_auto_match_list(self) -> None:
        self.auto_listbox.delete(0, tk.END)
        if not self.auto_peaks:
            self.auto_listbox.insert(tk.END, "Нет данных — выполните автокорреляцию")
            self.auto_listbox.config(state=tk.DISABLED)
            return
        threshold = self.auto_match_threshold.get()
        max_val = self.auto_max_value if self.auto_max_value > 0 else 1.0
        filtered = [peak for peak in self.auto_peaks if (peak.value / max_val) >= threshold]
        if not filtered:
            self.auto_listbox.insert(tk.END, "Пики ниже порога")
            self.auto_listbox.config(state=tk.DISABLED)
            self.auto_visible_peaks = []
            return
        self.auto_visible_peaks = filtered
        self.auto_listbox.config(state=tk.NORMAL)
        for idx, peak in enumerate(filtered, start=1):
            rel = peak.value / max_val if max_val else 0.0
            self.auto_listbox.insert(tk.END, f"{idx}. score={rel:.2f} shift=({peak.dx},{peak.dy})")
        self.auto_listbox.selection_clear(0, tk.END)

    def _on_auto_match_selected(self, _event: Optional[tk.Event]) -> None:
        if not self.auto_visible_peaks:
            return
        selection = self.auto_listbox.curselection()
        if not selection:
            return
        peak = self.auto_visible_peaks[selection[0]]
        self._highlight_autocorr_peak(peak)
        max_val = self.auto_max_value if self.auto_max_value > 0 else 1.0
        self.status_var.set(f"Смещение ({peak.dx}, {peak.dy}), score={peak.value / max_val:.2f}")

    def _highlight_autocorr_peak(self, peak: AutoPeak) -> None:
        if not self.original_image or not self.auto_image:
            return

        scale = 1.0 / (self.auto_image.scale or 1.0)
        dx = int(round(peak.dx * scale))
        dy = int(round(peak.dy * scale))

        width, height = self.original_image.size
        base = np.asarray(self.original_image, dtype=np.float32) / 255.0

        min_x = min(0, dx)
        min_y = min(0, dy)
        max_x = max(width, width + dx)
        max_y = max(height, height + dy)
        canvas_w = max_x - min_x
        canvas_h = max_y - min_y

        base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        shift_canvas = np.zeros_like(base_canvas)

        base_x = -min_x
        base_y = -min_y
        base_canvas[base_y:base_y + height, base_x:base_x + width] = base

        shift_x = base_x + dx
        shift_y = base_y + dy

        src_x0 = max(0, -shift_x)
        src_y0 = max(0, -shift_y)
        src_x1 = min(width, canvas_w - shift_x)
        src_y1 = min(height, canvas_h - shift_y)
        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0 = shift_x + src_x0
            dst_y0 = shift_y + src_y0
            shift_canvas[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = base[src_y0:src_y1, src_x0:src_x1]

        ghost = 0.5 * base_canvas + 0.5 * shift_canvas
        ghost_img = Image.fromarray(np.uint8(np.clip(ghost, 0.0, 1.0) * 255.0))
        self._display_on_canvas(ghost_img, self.auto_original_canvas)

    # ------------------------------------------------------------------ drawing
    def _highlight_fragment(self, rect: Tuple[int, int, int, int]) -> None:
        if not self.original_image:
            return
        base = self.original_image.copy()
        draw = ImageDraw.Draw(base)
        self._draw_outline(draw, rect, "red")
        self._display_on_canvas(base, self.original_canvas)

    def _draw_outline(self, draw: ImageDraw.ImageDraw, rect: Tuple[int, int, int, int], color: str) -> None:
        thickness =  max(2, (rect[2] - rect[0]) // 40)
        for offset in range(thickness):
            draw.rectangle((rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset), outline=color)

    def _clip_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if not self.original_image:
            return rect
        w, h = self.original_image.size
        x1 = max(0, min(w, rect[0]))
        y1 = max(0, min(h, rect[1]))
        x2 = max(0, min(w, rect[2]))
        y2 = max(0, min(h, rect[3]))
        return (x1, y1, x2, y2)

    def _display_on_canvas(self, image: Image.Image, canvas: tk.Canvas) -> None:
        canvas.update_idletasks()
        canvas_width = int(canvas.winfo_width() or canvas.cget("width"))
        canvas_height = int(canvas.winfo_height() or canvas.cget("height"))
        canvas_width = max(canvas_width, 10)
        canvas_height = max(canvas_height, 10)
        scale = min(canvas_width / image.width, canvas_height / image.height, 1.0)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        canvas.image = photo


def main() -> None:
    root = tk.Tk()
    ManualCorrelationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
