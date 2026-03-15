#!/usr/bin/env python3


import sys
import traceback
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QScrollArea, QGroupBox,
    QSpinBox, QDoubleSpinBox, QSlider, QFileDialog, QProgressBar,
    QMessageBox, QSizePolicy, QFrame, QSplitter, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auto_contrast(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-based contrast stretch → [0, 1]."""
    p_low, p_high = np.percentile(img, [low, high])
    return (np.clip(img, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)


def _get_slice(img: np.ndarray, z: int) -> np.ndarray:
    """Return 2-D slice from a 2-D or 3-D array."""
    if img.ndim == 3:
        z = max(0, min(z, img.shape[0] - 1))
        return img[z]
    return img


# ---------------------------------------------------------------------------
# Image canvas
# ---------------------------------------------------------------------------

class ImageCanvas(FigureCanvas):
    """Matplotlib canvas embedded in a Qt widget."""

    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 4), tight_layout=True)
        fig.patch.set_facecolor("#1e1e1e")
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#141414")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._base: Optional[np.ndarray] = None
        self._overlay: Optional[np.ndarray] = None

    # -- public API ----------------------------------------------------------

    def display(self, base: np.ndarray, z: int = 0,
                overlay: Optional[np.ndarray] = None):
        """
        Render image(s).

        Parameters
        ----------
        base    : 2-D or 3-D array (reference / moving channel)
        z       : Z slice index (used when arrays are 3-D)
        overlay : optional 2-D or 3-D array shown in green (registered result)
        """
        self._base = base
        self._overlay = overlay
        self._z = z
        self._redraw()

    def clear_canvas(self):
        self._base = None
        self._overlay = None
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#141414")
        self.draw()

    # -- internal ------------------------------------------------------------

    def _redraw(self):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("#141414")

        if self._base is None:
            self.draw()
            return

        base_2d = auto_contrast(_get_slice(self._base, self._z).astype(np.float32))

        if self._overlay is not None:
            ov_2d = auto_contrast(_get_slice(self._overlay, self._z).astype(np.float32))
            h, w = base_2d.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = base_2d   # red   ⟩
            rgb[:, :, 2] = base_2d   # blue  ⟩ → magenta = reference
            rgb[:, :, 1] = ov_2d     # green → registered result
            self.ax.imshow(rgb, vmin=0, vmax=1, aspect="auto")
        else:
            self.ax.imshow(base_2d, cmap="gray", vmin=0, vmax=1, aspect="auto")

        self.draw()


# ---------------------------------------------------------------------------
# Image panel (canvas + Z slider)
# ---------------------------------------------------------------------------

class ImagePanel(QWidget):
    """Canvas plus a Z-slice slider below it."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: bold; color: #aaa; padding: 2px;")
        layout.addWidget(lbl)

        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)

        # Z slider row
        z_row = QWidget()
        z_layout = QHBoxLayout(z_row)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(4)

        z_layout.addWidget(QLabel("Z:"))

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)
        self.z_slider.setEnabled(False)
        z_layout.addWidget(self.z_slider)

        self.z_lbl = QLabel("0 / 0")
        self.z_lbl.setFixedWidth(48)
        z_layout.addWidget(self.z_lbl)

        layout.addWidget(z_row)

        self._base: Optional[np.ndarray] = None
        self._overlay: Optional[np.ndarray] = None

        self.z_slider.valueChanged.connect(self._on_z)

    # -- public --------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        """Display a single (no overlay) image."""
        self._base = image
        self._overlay = None
        self._configure_slider(image)
        self.canvas.display(image, z=self.z_slider.value())

    def set_overlay(self, base: np.ndarray, overlay: np.ndarray):
        """Display base (magenta) + overlay (green) composite."""
        self._base = base
        self._overlay = overlay
        self._configure_slider(base)
        self.canvas.display(base, z=self.z_slider.value(), overlay=overlay)

    def clear(self):
        self._base = None
        self._overlay = None
        self.z_slider.setEnabled(False)
        self.z_slider.setMaximum(0)
        self.z_lbl.setText("0 / 0")
        self.canvas.clear_canvas()

    # -- internal ------------------------------------------------------------

    def _configure_slider(self, image: np.ndarray):
        if image is not None and image.ndim == 3:
            n = image.shape[0]
            self.z_slider.setMaximum(n - 1)
            mid = n // 2
            self.z_slider.setValue(mid)
            self.z_slider.setEnabled(True)
            self.z_lbl.setText(f"{mid} / {n - 1}")
        else:
            self.z_slider.setMaximum(0)
            self.z_slider.setValue(0)
            self.z_slider.setEnabled(False)
            self.z_lbl.setText("0 / 0")

    def _on_z(self, value: int):
        n = self.z_slider.maximum()
        self.z_lbl.setText(f"{value} / {n}")
        if self._base is not None:
            self.canvas.display(self._base, z=value, overlay=self._overlay)


# ---------------------------------------------------------------------------
# Config panel
# ---------------------------------------------------------------------------

class ConfigPanel(QScrollArea):
    """Scrollable sidebar with all algorithm parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(240)
        self.setMaximumWidth(300)
        self.setStyleSheet("QScrollArea { border: none; }")

        root = QWidget()
        self.setWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(8)
        vbox.setContentsMargins(8, 8, 8, 8)

        # ---- Pyramid -------------------------------------------------------
        pg = self._group("Pyramid Settings")
        gl = QGridLayout(pg)
        gl.setColumnStretch(1, 1)

        self.layers     = self._spin(gl, 0, "Layers:",        0, 5,     1)
        self.radius     = self._spin(gl, 1, "Patch Radius:",  1, 20,    5)
        self.iters      = self._spin(gl, 2, "Iterations:",    1, 200,  10)
        self.smooth     = self._dspin(gl, 3, "Smooth Penalty:", 0.0, 10.0, 0.08,  0.001, 4)
        self.mov_range  = self._dspin(gl, 4, "Move Range:",   0.1, 100.0,  5.0,  0.5,   1)
        vbox.addWidget(pg)

        # ---- Channel -------------------------------------------------------
        cg = self._group("Channel Settings")
        cl = QGridLayout(cg)
        cl.setColumnStretch(1, 1)

        self.dual_ch = QCheckBox("Dual Channel (membrane + calcium)")
        self.dual_ch.setChecked(True)
        cl.addWidget(self.dual_ch, 0, 0, 1, 2)

        self.transform = QComboBox()
        self.transform.addItems(["log10", "sqrt", "log2", "raw"])
        cl.addWidget(QLabel("Ca Transform:"), 1, 0)
        cl.addWidget(self.transform, 1, 1)

        self.k_weight  = self._dspin(cl, 2, "k (Ca weight):", 0.0, 1000.0, 50.0, 5.0, 1)
        self.mem_ch    = self._spin(cl,  3, "Membrane Ch:",   0,   9,  1)
        self.ca_ch     = self._spin(cl,  4, "Calcium Ch:",    0,   9,  0)
        vbox.addWidget(cg)

        # ---- Mask ----------------------------------------------------------
        mg = self._group("Mask Settings")
        ml = QGridLayout(mg)
        ml.setColumnStretch(1, 1)

        self.threshold = self._dspin(ml, 0, "Threshold σ:",  0.1, 20.0, 5.0,  0.5, 1)
        self.int_min   = self._spin(ml,  1, "Intensity Min:", 0,  10000,   5)
        self.int_max   = self._spin(ml,  2, "Intensity Max:", 0,  65535, 4000)
        self.int_max.setMaximum(65535)
        vbox.addWidget(mg)

        # ---- Backend -------------------------------------------------------
        bg = self._group("Backend")
        bl = QGridLayout(bg)
        bl.setColumnStretch(1, 1)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        bl.addWidget(QLabel("Device:"), 0, 0)
        bl.addWidget(self.device, 0, 1)
        vbox.addWidget(bg)

        vbox.addStretch()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox { font-weight: bold; }")
        return g

    @staticmethod
    def _spin(grid: QGridLayout, row: int, label: str,
              lo: int, hi: int, default: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(w, row, 1)
        return w

    @staticmethod
    def _dspin(grid: QGridLayout, row: int, label: str,
               lo: float, hi: float, default: float,
               step: float = 0.1, decimals: int = 2) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(default)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(w, row, 1)
        return w

    # -- public --------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "layers":           self.layers.value(),
            "patch_radius":     self.radius.value(),
            "iterations":       self.iters.value(),
            "smooth_penalty":   self.smooth.value(),
            "movement_range":   self.mov_range.value(),
            "dual_channel":     self.dual_ch.isChecked(),
            "transform":        self.transform.currentText(),
            "k":                self.k_weight.value(),
            "membrane_channel": self.mem_ch.value(),
            "calcium_channel":  self.ca_ch.value(),
            "threshold_factor": self.threshold.value(),
            "int_min":          self.int_min.value(),
            "int_max":          self.int_max.value(),
            "device":           self.device.currentText(),
        }


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class RegistrationWorker(QThread):
    """Runs image registration off the main thread."""

    finished = pyqtSignal(object)   # RegistrationResult
    errored  = pyqtSignal(str)
    status   = pyqtSignal(str)

    def __init__(self, moving: np.ndarray, reference: np.ndarray, cfg: dict):
        super().__init__()
        self.moving    = moving
        self.reference = reference
        self.cfg       = cfg

    def run(self):
        try:
            from wholistic_registration.v2.config.settings import (
                PyramidConfig, MaskConfig, ChannelConfig,
            )
            from wholistic_registration.v2.core.registration import FrameRegistrar

            c = self.cfg
            pyramid  = PyramidConfig(
                layers=c["layers"], patch_radius=c["patch_radius"],
                iterations=c["iterations"], smooth_penalty=c["smooth_penalty"],
                movement_range=c["movement_range"],
            )
            mask     = MaskConfig(
                threshold_factor=c["threshold_factor"],
                intensity_range=(c["int_min"], c["int_max"]),
            )
            channels = ChannelConfig(
                dual_channel=c["dual_channel"], transform=c["transform"],
                k=c["k"], membrane_channel=c["membrane_channel"],
                calcium_channel=c["calcium_channel"],
            )

            self.status.emit(f"Initialising registrar on {c['device']}…")
            reg = FrameRegistrar(pyramid=pyramid, mask=mask, channels=channels,
                                 device=c["device"])

            # --- parse channel layout from moving image shape ---------------
            mem_frame, ca_frame = self._split_channels(self.moving, c)
            ref_frame           = self._extract_ref(self.reference, c)

            self.status.emit("Running registration…")
            result = reg.register_single(
                membrane_frame=mem_frame,
                calcium_frame=ca_frame,
                reference=ref_frame,
                return_motion=True,
            )
            self.status.emit("Done.")
            self.finished.emit(result)

        except Exception as e:
            self.errored.emit(f"{e}\n\n{traceback.format_exc()}")

    # -- helpers -------------------------------------------------------------

    def _split_channels(self, img: np.ndarray, c: dict):
        """Extract membrane and calcium frames from the moving image array."""
        mem_idx, ca_idx = c["membrane_channel"], c["calcium_channel"]
        ndim = img.ndim

        # (C, Y, X)  or  (C, Z, Y, X)
        if ndim in (3, 4) and img.shape[0] == 2:
            if ndim == 4:
                return img[mem_idx, 0], img[ca_idx, 0]   # take first Z slab
            return img[mem_idx], img[ca_idx]

        # (Z, Y, X) or (Y, X) → single-channel input
        mem_frame = img if ndim <= 3 else img[0]
        ca_frame  = np.zeros_like(mem_frame)
        return mem_frame, ca_frame

    def _extract_ref(self, img: np.ndarray, c: dict):
        """Extract a single channel from the reference image."""
        mem_idx = c["membrane_channel"]
        ndim = img.ndim

        if ndim in (3, 4) and img.shape[0] == 2:
            if ndim == 4:
                return img[mem_idx, 0]
            return img[mem_idx]

        return img if ndim <= 3 else img[0]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wholistic Registration")
        self.resize(1280, 740)

        self._moving_image:    Optional[np.ndarray] = None
        self._reference_image: Optional[np.ndarray] = None
        self._worker:          Optional[RegistrationWorker] = None

        self._build_ui()
        self._apply_theme()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(8, 8, 8, 4)
        vbox.setSpacing(6)

        # Toolbar
        toolbar = self._build_toolbar()
        vbox.addWidget(toolbar)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        vbox.addWidget(sep)

        # Three-column splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # — Config column —
        config_wrap = QWidget()
        cw_layout = QVBoxLayout(config_wrap)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.setSpacing(3)
        hdr = QLabel("Configuration")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #ccc;"
            "padding: 4px; background: #2d2d2d; border-radius: 3px;"
        )
        cw_layout.addWidget(hdr)
        self.config_panel = ConfigPanel()
        cw_layout.addWidget(self.config_panel)
        splitter.addWidget(config_wrap)

        # — Moving image column —
        self.moving_panel = ImagePanel("Moving Image  (Image 1)")
        splitter.addWidget(self.moving_panel)

        # — Reference / result column —
        self.result_panel = ImagePanel(
            "Reference Image  (Image 2)   |   magenta = ref · green = registered"
        )
        splitter.addWidget(self.result_panel)

        splitter.setSizes([270, 490, 490])
        vbox.addWidget(splitter)

        # Status bar
        sb = self.statusBar()
        self._status_lbl = QLabel("Ready.  Load two images and click ▶ Run Registration.")
        sb.addWidget(self._status_lbl, 1)
        self._pbar = QProgressBar()
        self._pbar.setFixedWidth(180)
        self._pbar.setVisible(False)
        sb.addPermanentWidget(self._pbar)

    def _build_toolbar(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.load_moving_btn = QPushButton("Load Moving Image")
        self.load_moving_btn.setFixedHeight(32)
        self.load_moving_btn.clicked.connect(self._on_load_moving)
        row.addWidget(self.load_moving_btn)

        self.load_ref_btn = QPushButton("Load Reference Image")
        self.load_ref_btn.setFixedHeight(32)
        self.load_ref_btn.clicked.connect(self._on_load_reference)
        row.addWidget(self.load_ref_btn)

        row.addStretch()

        self.run_btn = QPushButton("▶   Run Registration")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setMinimumWidth(190)
        self.run_btn.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold;"
            "              border-radius:4px; font-size:13px; }"
            "QPushButton:hover    { background:#388e3c; }"
            "QPushButton:disabled { background:#555; color:#999; }"
        )
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)

        return w

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
            QPushButton {
                background:#3c3c3c; border:1px solid #555; border-radius:3px;
                padding:4px 12px; color:#ddd;
            }
            QPushButton:hover   { background:#4a4a4a; }
            QPushButton:pressed { background:#2a2a2a; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background:#2a2a2a; border:1px solid #555;
                border-radius:3px; padding:2px 4px; color:#ddd;
            }
            QSlider::groove:horizontal {
                height:4px; background:#555; border-radius:2px;
            }
            QSlider::handle:horizontal {
                background:#888; width:12px; height:12px;
                border-radius:6px; margin:-4px 0;
            }
            QLabel        { color:#ccc; }
            QScrollArea   { background:transparent; }
            QStatusBar    { background:#252525; }
            QProgressBar  {
                border:1px solid #555; border-radius:3px;
                text-align:center; background:#2a2a2a;
            }
            QProgressBar::chunk { background:#2e7d32; }
        """)

    # -- event handlers ------------------------------------------------------

    def _on_load_moving(self):
        img = self._load_file("Load Moving Image (Image 1)")
        if img is not None:
            self._moving_image = img
            self.moving_panel.set_image(img)
            self._set_status(f"Moving image loaded — shape {img.shape}, dtype {img.dtype}")

    def _on_load_reference(self):
        img = self._load_file("Load Reference Image (Image 2)")
        if img is not None:
            self._reference_image = img
            self.result_panel.set_image(img)
            self._set_status(f"Reference image loaded — shape {img.shape}, dtype {img.dtype}")

    def _on_run(self):
        if self._moving_image is None:
            QMessageBox.warning(self, "Missing", "Please load the moving image first.")
            return
        if self._reference_image is None:
            QMessageBox.warning(self, "Missing", "Please load the reference image first.")
            return

        cfg = self.config_panel.get_config()

        self.run_btn.setEnabled(False)
        self._pbar.setRange(0, 0)
        self._pbar.setVisible(True)
        self._set_status("Running registration…")

        self._worker = RegistrationWorker(self._moving_image, self._reference_image, cfg)
        self._worker.finished.connect(self._on_done)
        self._worker.errored.connect(self._on_error)
        self._worker.status.connect(self._set_status)
        self._worker.start()

    def _on_done(self, result):
        self.run_btn.setEnabled(True)
        self._pbar.setVisible(False)
        self._set_status(
            "Registration complete.  "
            "Right panel: magenta = reference · green = registered result."
        )
        # Show reference (magenta) vs registered membrane channel (green)
        self.result_panel.set_overlay(result.reference, result.membrane_registered)

    def _on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self._pbar.setVisible(False)
        self._set_status("Error — see dialog for details.")
        QMessageBox.critical(self, "Registration Error", msg)

    # -- helpers -------------------------------------------------------------

    def _load_file(self, title: str) -> Optional[np.ndarray]:
        path, _ = QFileDialog.getOpenFileName(
            self, title, "",
            "Images (*.tif *.tiff *.nd2 *.npy);;All Files (*)"
        )
        if not path:
            return None

        try:
            p = Path(path)
            ext = p.suffix.lower()
            if ext in (".tif", ".tiff"):
                import tifffile
                img = tifffile.imread(str(p))
            elif ext == ".nd2":
                import nd2
                with nd2.ND2File(str(p)) as f:
                    img = f.asarray()
            elif ext == ".npy":
                img = np.load(str(p))
            else:
                QMessageBox.warning(self, "Unsupported", f"Unsupported format: {ext}")
                return None
            return img.astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return None

    def _set_status(self, msg: str):
        self._status_lbl.setText(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Wholistic Registration")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
