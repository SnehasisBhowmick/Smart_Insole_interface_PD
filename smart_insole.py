import asyncio
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from bleak import BleakClient
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import traceback

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BLE_DEVICES = {
    "Right":  {"address": "9A:58:3C:E3:44:0A", "uuid": "f716a8e8-fa94-4c6a-9f8f-66f536a546f4"},
    "Left": {"address": "53:CD:C2:56:EF:76", "uuid": "19b20001-e8f2-537e-4f6c-d104768a1214"}
}

FSR_COORDINATES = {
    "Left": [
        (163, 190), (105, 170), (175, 280), (150, 340),
        (110, 350), (135, 255), (100, 284), (70, 313),
        (55, 355), (60, 480), (125, 480), (85, 540),
        (125, 617), (165, 619), (125, 658), (165, 660)
    ],
    "Right": [
        (333, 190), (390, 170), (322, 280), (348, 340),
        (390, 350), (365, 255), (400, 284), (430, 313),
        (455, 355), (435, 480), (375, 480), (413, 540),
        (305, 617), (345, 619), (305, 658), (345, 660)
    ]
}

class InsoleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Insole Pressure Visualization (Dual)")
        self.root.configure(bg="#f6f7fa")
        self.canvas_width = 500
        self.canvas_height = 790
        self.color_gradient = self.precompute_colors()
        self.latest_fsr_values = {"Left": [0]*16, "Right": [0]*16}
        self.fsr_data_records = {"Left": [], "Right": []}
        self.heatmap_img_id = None
        self.tk_heatmap_img = None
        self.cop_point = {"Left": None, "Right": None}
        self.ble_address = {"Left": None, "Right": None}
        self.characteristic_uuid = {"Left": None, "Right": None}
        self.stop_event = {"Left": threading.Event(), "Right": threading.Event()}
        self.ble_thread = {"Left": None, "Right": None}
        self.grf_y = {"Left": np.zeros(100), "Right": np.zeros(100)}
        self.status_var = {"Left": ctk.StringVar(value="Left - Ready"), "Right": ctk.StringVar(value="Right - Ready")}
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._update_gui)

    def on_close(self):
        for foot in ["Left", "Right"]:
            self.stop_event[foot].set()
            if self.ble_thread[foot] and self.ble_thread[foot].is_alive():
                self.ble_thread[foot].join(timeout=1.0)
        self.root.destroy()

    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg="#f6f7fa")
        main_frame.pack(expand=1, fill="both", padx=10, pady=10)
        left = tk.Frame(main_frame, bg="#f6f7fa")
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(main_frame, bg="#fcfdff", bd=1, relief="solid")
        right.grid(row=0, column=1, sticky="ns", padx=(12,0))
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self.tk_canvas = tk.Canvas(left, width=self.canvas_width, height=self.canvas_height, bg="#0d47a1", highlightthickness=0)
        self.tk_canvas.pack(padx=10, pady=10)

        # Background image
        if not os.path.exists("insole_pic_gui.png"):
            messagebox.showerror("Missing File", "insole_pic_gui.png not found.")
            self.root.destroy()
            return
        try:
            img = Image.open("insole_pic_gui.png")
            img_ratio = img.width / img.height
            can_ratio = self.canvas_width / self.canvas_height
            if img_ratio > can_ratio:
                new_w = self.canvas_width
                new_h = int(new_w / img_ratio)
            else:
                new_h = self.canvas_height
                new_w = int(new_h * img_ratio)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            self.tk_photo = ImageTk.PhotoImage(img_resized)
            self.image_x = (self.canvas_width - new_w) // 2
            self.image_y = (self.canvas_height - new_h) // 2
            self.bg_img_id = self.tk_canvas.create_image(self.image_x, self.image_y, anchor=tk.NW, image=self.tk_photo)
        except Exception as e:
            messagebox.showerror("Image Load Error", str(e))
            self.root.destroy()
            return

        for x in range(0, self.canvas_width, 42):
            self.tk_canvas.create_line(x, 0, x, self.canvas_height, fill="#e8eaf0", width=1)
        for y in range(0, self.canvas_height, 48):
            self.tk_canvas.create_line(0, y, self.canvas_width, y, fill="#e8eaf0", width=1)

        self.cop_point["Left"] = self.tk_canvas.create_oval(0, 0, 0, 0, fill="#1976d2", outline="#222", width=2)
        self.cop_point["Right"] = self.tk_canvas.create_oval(0, 0, 0, 0, fill="#ff7f50", outline="#444", width=2)

        group = ttk.LabelFrame(right, text="Connection & Actions")
        group.pack(fill="x", padx=16, pady=(24,8))
        ctk.CTkLabel(group, text="Select BLE Device:", font=("Arial", 14)).grid(row=0, column=0, sticky="w", padx=10, pady=8, columnspan=2)
        ctk.CTkButton(group, text="Connect Left", width=90, command=lambda: self._start_ble("Left")).grid(row=1, column=0, padx=5, pady=3)
        ctk.CTkButton(group, text="Connect Right", width=90, command=lambda: self._start_ble("Right")).grid(row=1, column=1, padx=5, pady=3)
        ctk.CTkButton(group, text="Stop Left", width=90, command=lambda: self._stop_ble("Left")).grid(row=2, column=0, padx=5, pady=3)
        ctk.CTkButton(group, text="Stop Right", width=90, command=lambda: self._stop_ble("Right")).grid(row=2, column=1, padx=5, pady=3)
        ctk.CTkLabel(group, textvariable=self.status_var["Left"], text_color="blue").grid(row=3, column=0, padx=4)
        ctk.CTkLabel(group, textvariable=self.status_var["Right"], text_color="red").grid(row=3, column=1, padx=4)

        run_frame = ttk.Frame(right)
        run_frame.pack(fill="x", padx=16, pady=(0,8))
        ctk.CTkButton(run_frame, text="Save to CSV", width=128, command=self._save_data).grid(row=0, column=0, padx=2)
        ctk.CTkButton(run_frame, text="Show Live Graph", width=128, command=self._open_graph).grid(row=0, column=1, padx=2)
        ctk.CTkButton(run_frame, text="Show Sensor Readings", width=128, command=self._open_readings_window).grid(row=0, column=2, padx=2)

        # GRF graphs for both feet
        grf_frame = ttk.LabelFrame(right, text="GRF Graphs")
        grf_frame.pack(fill='both', padx=16, pady=(8,12))
        self.grf_fig, self.grf_ax = plt.subplots(figsize=(3.8,2), dpi=100)
        self.grf_ax.set_title("Total GRF (N)")
        self.grf_ax.set_ylim(0,48000)
        self.grf_ax.set_xlim(0,100)
        self.grf_ax.set_facecolor("white")
        self.grf_x = np.arange(100)
        self.grf_line_left, = self.grf_ax.plot(self.grf_x, self.grf_y["Left"], color='blue', label="Left foot")
        self.grf_line_right, = self.grf_ax.plot(self.grf_x, self.grf_y["Right"], color='red', label="Right foot")
        self.grf_ax.legend(loc="upper right", fontsize=9, frameon=False)
        self.grf_canvas = FigureCanvasTkAgg(self.grf_fig, master=grf_frame)
        self.grf_canvas.get_tk_widget().pack(fill="both", expand=True)

    def precompute_colors(self, min_val=0, max_val=2000):
        gradient = []
        for value in range(min_val, max_val+1):
            ratio = (value - min_val) / (max_val - min_val)
            if ratio <= 0.33:
                blue = int(255 * (1 - ratio / 0.33))
                green = int(255 * (ratio / 0.33))
                gradient.append(f"#{0:02x}{green:02x}{blue:02x}")
            elif ratio <= 0.66:
                green = 255
                red = int(255 * ((ratio-0.33)/0.33))
                gradient.append(f"#{red:02x}{green:02x}{0:02x}")
            else:
                red = 255
                green = int(255 * (1 - (ratio-0.66)/0.34))
                gradient.append(f"#{red:02x}{green:02x}{0:02x}")
        return gradient

    def _start_ble(self, foot):
        self.ble_address[foot] = BLE_DEVICES[foot]["address"]
        self.characteristic_uuid[foot] = BLE_DEVICES[foot]["uuid"]
        self.stop_event[foot].clear()
        if not self.ble_thread[foot] or not self.ble_thread[foot].is_alive():
            self.ble_thread[foot] = threading.Thread(target=self._ble_thread_func, args=(foot,), daemon=True)
            self.ble_thread[foot].start()
        self.status_var[foot].set(f"{foot}: Reading BLE...")

    def _stop_ble(self, foot):
        self.stop_event[foot].set()
        self.status_var[foot].set(f"{foot}: Stopped")

    def _ble_thread_func(self, foot):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_ble(foot))
        except Exception as e:
            print(traceback.format_exc())
            self.root.after(0, lambda: self.status_var[foot].set(f"{foot}: BLE thread error"))

    async def _connect_ble(self, foot):
        if self.ble_address[foot] and self.characteristic_uuid[foot]:
            try:
                async with BleakClient(self.ble_address[foot]) as client:
                    await client.start_notify(self.characteristic_uuid[foot],
                                              lambda sender, data: self._notification_handler(foot, data))
                    while not self.stop_event[foot].is_set():
                        await asyncio.sleep(0.05)
                    await client.stop_notify(self.characteristic_uuid[foot])
            except Exception as e:
                self.root.after(0, lambda: self.status_var[foot].set(f"{foot}: BLE error: {e}"))

    def _notification_handler(self, foot, data):
        try:
            values = list(map(int, data.decode("utf-8").strip().split("_")))
            if len(values) == len(FSR_COORDINATES[foot]):
                self.root.after(0, lambda values=values, foot=foot: self._update_fsr_values(foot, values))
        except Exception as e:
            self.root.after(0, lambda: self.status_var[foot].set(f"{foot}: Data error"))

    def _update_fsr_values(self, foot, values):
        self.latest_fsr_values[foot] = values
        self.fsr_data_records[foot].append(values)

    def _draw_heatmap(self):
        base = np.full((self.canvas_height, self.canvas_width, 3), (0,0,127), dtype=np.uint8)
        for foot, color_map in [("Left", 'jet'), ("Right", 'jet')]:
            heatmap = np.zeros((self.canvas_height, self.canvas_width), dtype=np.float32)
            for (x, y), val in zip(FSR_COORDINATES[foot], self.latest_fsr_values[foot]):
                if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                    heatmap[int(y), int(x)] = val
            sigma = 20
            blurred = gaussian_filter(heatmap, sigma=sigma)
            norm = (blurred / blurred.max() * 255) if blurred.max() > 0 else blurred
            norm = norm.astype(np.uint8)
            rgba = cm.get_cmap(color_map)(norm)
            rgb = (rgba[..., :3] * 255).astype(np.uint8)
            mask = norm > 15
            base[mask] = rgb[mask]
        img = Image.fromarray(base)
        comb = ImageTk.PhotoImage(img)
        if self.heatmap_img_id is None:
            self.heatmap_img_id = self.tk_canvas.create_image(0, 0, anchor=tk.NW, image=comb)
            self.tk_canvas.tag_lower(self.bg_img_id)
            self.tk_canvas.tag_lower(self.heatmap_img_id)
        else:
            self.tk_canvas.itemconfig(self.heatmap_img_id, image=comb)
        self.tk_heatmap_img = comb  # keep ref!

    def _update_gui(self):
        if not self.root.winfo_exists():
            return
        try:
            self._draw_heatmap()
            for foot in ["Left", "Right"]:
                vals = self.latest_fsr_values[foot]
                total = sum(vals)
                if self.cop_point[foot] is not None:
                    if total:
                        x_cop = sum(x * p for (x, y), p in zip(FSR_COORDINATES[foot], vals)) / total
                        y_cop = sum(y * p for (x, y), p in zip(FSR_COORDINATES[foot], vals)) / total
                        self.tk_canvas.coords(self.cop_point[foot], x_cop-8, y_cop-8, x_cop+8, y_cop+8)
                    else:
                        self.tk_canvas.coords(self.cop_point[foot], 0, 0, 0, 0)
                self.grf_y[foot] = np.roll(self.grf_y[foot], -1)
                self.grf_y[foot][-1] = total
            self.grf_line_left.set_ydata(self.grf_y["Left"])
            self.grf_line_right.set_ydata(self.grf_y["Right"])
            self.grf_canvas.draw()
        except Exception as e:
            print(f"Update GUI error: {e}")
        self.root.after(100, self._update_gui)

    def _save_data(self):
        count = 0
        for foot in ["Left", "Right"]:
            if self.fsr_data_records[foot]:
                df = pd.DataFrame(self.fsr_data_records[foot], columns=[f"{foot}_FSR{i+1}" for i in range(16)])
                df.to_csv(f"pressure_data_{foot}.csv", index=False)
                self.status_var[foot].set(f"{foot}: Saved pressure_data_{foot}.csv")
                count += 1
        if count == 0:
            messagebox.showinfo("No Data", "No data to save.")

    def _open_graph(self):
        win = ctk.CTkToplevel(self.root)
        win.geometry("700x440")
        win.title("Live Graph - Dual foot")
        fig, ax = plt.subplots(figsize=(7,3.7))
        ax.set_ylim(0, 3300)
        ax.set_xlim(0, 100)
        ax.set_title("Live Pressure Data (Both Feet, Overlaid)")
        x_data = np.arange(100)
        all_coords = FSR_COORDINATES["Left"] + FSR_COORDINATES["Right"]
        y_datas = [np.zeros(100) for _ in range(32)]
        lines = [ax.plot(x_data, y_datas[i], label=f"{'L' if i<16 else 'R'}FSR {i%16+1}",
                         color='blue' if i<16 else 'red', alpha=0.6)[0] for i in range(32)]
        ax.legend(ncol=4, fontsize=7, frameon=False)
        ax.grid(True, which='both', color="#eee", linewidth=0.8)
        ax.set_facecolor("#f9fafd")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        def update_plot():
            if not win.winfo_exists():
                return
            for i, foot in enumerate(["Left", "Right"]):
                records = self.fsr_data_records[foot][-100:]
                for s in range(16):
                    ys = [row[s] for row in records] if records else [0] * 100
                    ys_padded = np.pad(ys, (100-len(ys), 0), 'constant')
                    lines[s+16*i].set_ydata(ys_padded)
            canvas.draw()
            win.after(200, update_plot)
        update_plot()

    def _open_readings_window(self):
        win = ctk.CTkToplevel(self.root)
        win.geometry("420x600")
        win.title("Live Sensor Readings")
        label_sets = {"Left": [], "Right": []}
        box = tk.Frame(win)
        box.pack()
        for fi, foot in enumerate(["Left", "Right"]):
            tk.Label(box, text=f"{foot} Foot", font=("Arial", 15, "bold")).grid(row=0, column=fi, padx=16)
            for s in range(16):
                lbl = ctk.CTkLabel(box, text=f"FSR{s+1}: 0", font=("Arial", 13))
                lbl.grid(row=s+1, column=fi, padx=12, pady=3)
                label_sets[foot].append(lbl)
        def update_labels():
            if not win.winfo_exists():
                return
            for foot in ["Left", "Right"]:
                for i, lbl in enumerate(label_sets[foot]):
                    lbl.configure(text=f"FSR{i+1}: {self.latest_fsr_values[foot][i]}")
            win.after(100, update_labels)
        update_labels()

def main():
    root = ctk.CTk()
    root.resizable(False, False)
    app = InsoleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()