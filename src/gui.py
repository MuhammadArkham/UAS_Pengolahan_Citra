import os, sys, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np
from PIL import Image, ImageTk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS   = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import processing, cnn as C

# ── Palet warna dark theme
BG, SURF, CARD             = "#0d1117", "#161b22", "#21262d"
BLUE, GREEN, AMBER, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#bc8cff"
TEXT, SUB, BORDER          = "#e6edf3", "#8b949e", "#30363d"
FONT = "Segoe UI"


# ── Helper fungsi
def to_ph(img, s=320):
    p = Image.fromarray(img)
    p.thumbnail((s, s), Image.LANCZOS)
    return ImageTk.PhotoImage(p)


def hist_img(a, b):
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.2), facecolor=CARD)
    for ax, im, t, c in zip(axes, [a, b], ["Original", "Hasil"], [BLUE, GREEN]):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
        ax.hist(g.ravel(), bins=64, color=c, alpha=.85, edgecolor="none")
        ax.set_title(t, color=TEXT, fontsize=8, pad=4)
        ax.set_facecolor(BG)
        ax.tick_params(colors=SUB, labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
    fig.tight_layout(pad=1)
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return buf[:, :, :3]


def gstats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return g.mean(), g.std(), int(g.min()), int(g.max())


def safe_bgr(img):
    """Konversi RGB→BGR untuk cv2.imwrite, aman untuk grayscale maupun RGB."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Kelas utama GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CitraLab  —  Pengolahan Citra Digital")
        self.geometry("1020x660")
        self.configure(bg=BG)
        self.img = self.res = self.cnn_img = self.hph = None
        self._style()
        self._build()

    # ── Style
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=SURF, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab", background=SURF, foreground=SUB,
                    padding=[18, 8], font=(FONT, 9), focuscolor=SURF, borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", BG), ("active", CARD)],
              foreground=[("selected", TEXT), ("active", TEXT)])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD,
                    foreground=TEXT, selectbackground=CARD, selectforeground=TEXT,
                    bordercolor=BORDER, arrowcolor=SUB, relief="flat", padding=[8, 4])
        s.map("TCombobox",
              fieldbackground=[("readonly", CARD)],
              foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", CARD)],
              selectforeground=[("readonly", TEXT)],
              bordercolor=[("focus", BLUE), ("!focus", BORDER)])
        s.configure("pb.Horizontal.TProgressbar",
                    troughcolor=SURF, background=BLUE, bordercolor=BORDER)

    # ── Layout utama
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=SURF, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="CitraLab", bg=SURF, fg=BLUE,
                 font=(FONT, 15, "bold")).pack(side="left", padx=(20, 8), pady=14)
        tk.Label(hdr, text="Pengolahan Citra Digital", bg=SURF, fg=SUB,
                 font=(FONT, 9)).pack(side="left")
        tk.Label(hdr, text="Muhammad Arkhamullah Rifai Asshidiq",
                 bg=SURF, fg=SUB, font=(FONT, 8)).pack(side="right", padx=20)
        tk.Frame(hdr, bg=BORDER, height=1).pack(side="bottom", fill="x")

        # Tabs
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        t1, t2, t3 = [tk.Frame(nb, bg=BG) for _ in range(3)]
        nb.add(t1, text="  Pengolahan Citra  ")
        nb.add(t2, text="  Analisis Hasil  ")
        nb.add(t3, text="  Klasifikasi CNN  ")
        self._t1(t1)
        self._t2(t2)
        self._t3(t3)

        # Status bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self.sb = tk.Label(self, text="  Siap.", bg=SURF, fg=SUB,
                           font=(FONT, 8), anchor="w", pady=5)
        self.sb.pack(fill="x")

    # ── Widget helpers
    def _btn(self, p, txt, cmd, bg=BLUE, fg=BG):
        return tk.Button(p, text=txt, command=cmd, bg=bg, fg=fg,
                         font=(FONT, 9, "bold"), relief="flat",
                         padx=14, pady=7, cursor="hand2",
                         activebackground=bg, activeforeground=fg, bd=0)

    def _card(self, p, **kw):
        return tk.Frame(p, bg=CARD, **kw)

    def _panel(self, p, title):
        f   = tk.Frame(p, bg=CARD)
        top = tk.Frame(f, bg=CARD)
        top.pack(fill="x", padx=12, pady=8)
        tk.Label(top, text=title, bg=CARD, fg=BLUE,
                 font=(FONT, 9, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x")
        l = tk.Label(f, bg=BG, text="—", fg=SUB, font=(FONT, 9))
        l.pack(fill="both", expand=True)
        f._l = l
        return f

    def _show(self, panel, img):
        if img is None:
            panel._l.config(image="", text="—")
            panel._l.image = None
        else:
            ph = to_ph(img)
            panel._l.config(image=ph, text="")
            panel._l.image = ph

    def _status(self, msg, color=SUB):
        self.sb.config(text=f"  {msg}", fg=color)

    # ── Tab 1: Pengolahan Citra
    def _t1(self, f):
        bar = self._card(f, pady=10, padx=14)
        bar.pack(fill="x", padx=14, pady=(14, 8))

        self._btn(bar, "Buka Gambar", self._load).pack(side="left", padx=(0, 10))
        tk.Frame(bar, bg=BORDER, width=1).pack(side="left", fill="y", pady=2, padx=6)

        self.pv = tk.StringVar(value=list(processing.PROCESSES)[0])
        ttk.Combobox(bar, textvariable=self.pv, values=list(processing.PROCESSES),
                     width=26, state="readonly", font=(FONT, 9)).pack(
                         side="left", padx=8, ipady=3)

        self._btn(bar, "Proses", self._run, GREEN, BG).pack(side="left", padx=4)
        self._btn(bar, "Simpan", self._save, AMBER, BG).pack(side="left", padx=4)

        self.lf = tk.Label(bar, text="", bg=CARD, fg=SUB, font=(FONT, 8))
        self.lf.pack(side="right", padx=6)

        row = tk.Frame(f, bg=BG)
        row.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.po = self._panel(row, "Original")
        self.po.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.pr = self._panel(row, "Hasil Proses")
        self.pr.pack(side="left", fill="both", expand=True, padx=(6, 0))

    def _load(self):
        p = filedialog.askopenfilename(
            initialdir=DS, filetypes=[("Image", "*.jpg *.jpeg *.png")])
        if not p:
            return
        self.img = processing.load_image(p)
        self.res = None
        self._show(self.po, self.img)
        self._show(self.pr, None)
        self.lf.config(text=os.path.basename(p))
        self._status(f"Gambar dimuat: {os.path.basename(p)}", BLUE)

    def _run(self):
        if self.img is None:
            messagebox.showwarning("", "Load gambar dulu!")
            return
        name     = self.pv.get()
        self.res = processing.PROCESSES[name](self.img)
        self._show(self.pr, self.res)
        self._status(f"✔  {name} selesai.", GREEN)
        self._upd()

    def _save(self):
        if self.res is None:
            messagebox.showwarning("", "Proses gambar dulu!")
            return
        name = (self.pv.get()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", ""))
        out_path = os.path.join(OUT, f"{name}.jpg")
        cv2.imwrite(out_path, safe_bgr(self.res))   # ← fix: pakai safe_bgr
        self._status(f"💾  Tersimpan → output/{name}.jpg", AMBER)

    # ── Tab 2: Analisis Hasil
    def _t2(self, f):
        tk.Label(f, text="Analisis Hasil Pengolahan Citra", bg=BG, fg=TEXT,
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=16, pady=(14, 6))

        mr = tk.Frame(f, bg=BG)
        mr.pack(fill="x", padx=16, pady=(0, 10))
        self.mc = {}
        for k, c in [("Mean", TEXT), ("Std Dev", TEXT), ("Min", TEXT),
                     ("Max", TEXT), ("SSIM", BLUE), ("PSNR", PURPLE)]:
            fc = self._card(mr, padx=12, pady=10)
            fc.pack(side="left", fill="x", expand=True, padx=(0, 6))
            tk.Label(fc, text=k, bg=CARD, fg=SUB, font=(FONT, 7, "bold")).pack(anchor="w")
            v = tk.Label(fc, text="—", bg=CARD, fg=c, font=(FONT, 13, "bold"))
            v.pack(anchor="w", pady=(2, 0))
            d = tk.Label(fc, text="", bg=CARD, fg=SUB, font=(FONT, 7))
            d.pack(anchor="w")
            self.mc[k] = (v, d)

        hf = self._card(f)
        hf.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        tk.Label(hf, text="Histogram Distribusi Piksel", bg=CARD, fg=BLUE,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        tk.Frame(hf, bg=BORDER, height=1).pack(fill="x")
        self.hl = tk.Label(hf, bg=BG, text="Jalankan proses terlebih dahulu.",
                           fg=SUB, font=(FONT, 9))
        self.hl.pack(fill="both", expand=True)

    def _upd(self):
        if self.img is None or self.res is None:
            return
        om, os_, omi, oma = gstats(self.img)
        rm, rs,  rmi, rma = gstats(self.res)
        for k, (ov, rv) in zip(
            ["Mean", "Std Dev", "Min", "Max"],
            [(om, rm), (os_, rs), (omi, rmi), (oma, rma)]
        ):
            d   = rv - ov
            col = GREEN if d >= 0 else "#f85149"
            self.mc[k][0].config(text=f"{ov:.1f}")
            self.mc[k][1].config(text=f"hasil {rv:.1f}  ({d:+.1f})", fg=col)

        ssim = processing.compute_ssim(self.img, self.res)
        psnr = processing.compute_psnr(self.img, self.res)
        self.mc["SSIM"][0].config(text=f"{ssim:.4f}")
        self.mc["SSIM"][1].config(text="1.0 = identik asli")
        self.mc["PSNR"][0].config(text=f"{psnr:.1f} dB")
        self.mc["PSNR"][1].config(text="makin tinggi makin baik")

        try:
            buf      = hist_img(self.img, self.res)
            self.hph = ImageTk.PhotoImage(Image.fromarray(buf))
            self.hl.config(image=self.hph, text="")
        except Exception:
            pass

    # ── Tab 3: Klasifikasi CNN
    def _t3(self, f):
        tc = self._card(f, padx=14, pady=14)
        tc.pack(fill="x", padx=14, pady=(14, 8))
        tk.Label(tc, text="Training Model CNN", bg=CARD, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(anchor="w")
        self.ts = tk.Label(tc, text="Model belum ditraining.", bg=CARD,
                           fg=SUB, font=(FONT, 9))
        self.ts.pack(anchor="w", pady=(4, 8))

        pr = tk.Frame(tc, bg=CARD)
        pr.pack(fill="x", pady=(0, 10))
        self.tp = ttk.Progressbar(pr, length=460, mode="determinate",
                                  style="pb.Horizontal.TProgressbar")
        self.tp.pack(side="left")
        self.tpct = tk.Label(pr, text="0%", bg=CARD, fg=SUB, font=(FONT, 8), width=5)
        self.tpct.pack(side="left", padx=8)
        self._btn(tc, "Mulai Training", self._train, PURPLE, BG).pack(anchor="w")

        if C.model_exists():
            self.ts.config(text="✅  Model tersedia — siap prediksi.", fg=GREEN)

        pc = self._card(f, padx=14, pady=14)
        pc.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        tk.Label(pc, text="Prediksi Gambar", bg=CARD, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(0, 10))
        body = tk.Frame(pc, bg=CARD)
        body.pack(fill="both", expand=True)

        lf = tk.Frame(body, bg=BG, width=260)
        lf.pack(side="left", fill="both", padx=(0, 14))
        lf.pack_propagate(False)
        self.cl = tk.Label(lf, bg=BG, text="—", fg=SUB, font=(FONT, 9))
        self.cl.pack(fill="both", expand=True)
        br = tk.Frame(lf, bg=BG)
        br.pack(pady=8)
        self._btn(br, "Buka",    self._lp).pack(side="left", padx=(0, 6))
        self._btn(br, "Predict", self._predict, GREEN, BG).pack(side="left")

        rf = tk.Frame(body, bg=CARD)
        rf.pack(side="left", fill="both", expand=True)
        tk.Label(rf, text="Hasil Prediksi", bg=CARD, fg=SUB,
                 font=(FONT, 8)).pack(anchor="w")
        self.pc2 = tk.Label(rf, text="—", bg=CARD, fg=GREEN,
                             font=(FONT, 30, "bold"))
        self.pc2.pack(anchor="w", pady=(8, 4))
        self.pconf = tk.Label(rf, text="Confidence: —", bg=CARD, fg=SUB,
                              font=(FONT, 10))
        self.pconf.pack(anchor="w")
        self.pbar = ttk.Progressbar(rf, length=200, mode="determinate",
                                    style="pb.Horizontal.TProgressbar")
        self.pbar.pack(anchor="w", pady=8)

    def _train(self):
        self.ts.config(text="⏳  Training dimulai...", fg=AMBER)
        self.tp["value"] = 0

        def run():
            def cb(ep, tot, acc, vacc):
                pct = int(ep / tot * 100)
                self.tp["value"] = pct
                self.tpct.config(text=f"{pct}%")
                self.ts.config(
                    text=f"Epoch {ep}/{tot}  —  acc: {acc:.1%}  val: {vacc:.1%}",
                    fg=TEXT)
                self.update_idletasks()
            try:
                C.train(DS, callback=cb)
                self.ts.config(text="✅  Training selesai! Model tersimpan.", fg=GREEN)
                self.tp["value"]  = 100
                self.tpct.config(text="100%")
                self._status("Training CNN selesai.", GREEN)
            except Exception as e:
                self.ts.config(text=f"❌  {e}", fg="#f85149")

        threading.Thread(target=run, daemon=True).start()

    def _lp(self):
        p = filedialog.askopenfilename(
            initialdir=DS, filetypes=[("Image", "*.jpg *.jpeg *.png")])
        if not p:
            return
        self.cnn_img = processing.load_image(p)
        ph = to_ph(self.cnn_img, 240)
        self.cl.config(image=ph, text="")
        self.cl.image = ph
        self.pc2.config(text="—")
        self.pconf.config(text="Confidence: —")
        self.pbar["value"] = 0

    def _predict(self):
        if self.cnn_img is None:
            messagebox.showwarning("", "Load gambar dulu!")
            return
        if not C.model_exists():
            messagebox.showwarning("", "Training dulu!")
            return
        lbl, conf = C.predict(self.cnn_img, C.get_labels(DS))
        self.pc2.config(text=lbl)
        self.pconf.config(text=f"Confidence: {conf:.1%}", fg=TEXT)
        self.pbar["value"] = conf * 100
        self._status(f"Prediksi: {lbl}  ({conf:.1%})", GREEN)
