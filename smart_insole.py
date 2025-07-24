import asyncio
import ctypes
import sys
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
    "Device 1": {"address": "9A:58:3C:E3:44:0A", "uuid": "f716a8e8-fa94-4c6a-9f8f-66f536a546f4"},
    "Device 2": {"address": "53:CD:C2:56:EF:76", "uuid": "19b20001-e8f2-537e-4f6c-d104768a1214"}
}

class InsoleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Insole Pressure Visualization (16 Sensors)")
        self.root.configure(bg="#f6f7fa")
        self.canvas_width = 500
        self.canvas_height = 790
        self.fsr_coordinates = [
            (333, 190), (390, 170), (322, 280), (348, 340),
            (390, 350), (365, 255), (400, 284), (430, 313),
            (455, 355), (435, 480), (375, 480), (413, 540),
            (305, 617), (345, 619), (305, 658), (345, 660)
        ]
        self.latest_fsr_values = [0]*len(self.fsr_coordinates)
        self.fsr_data_records = []
        self.color_gradient = self.precompute_colors()
        self.sensor_points = []
        self.heatmap_img_id = None
        self.tk_heatmap_img = None
        self.cop_point = None
        self.ble_address = None
        self.characteristic_uuid = None
        self.stop_event = threading.Event()
        self.ble_thread = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._update_gui)

    def on_close(self):
        self.stop_event.set()
        if self.ble_thread and self.ble_thread.is_alive():
            self.ble_thread.join(timeout=1.0)
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

        self.tk_canvas = tk.Canvas(left, width=self.canvas_width, height=self.canvas_height, bg="white", highlightthickness=0)
        self.tk_canvas.pack(padx=10, pady=10)

        # Background image
        if not os.path.exists("insole pic.png"):
            messagebox.showerror("Missing File", "insole pic.png not found.")
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
            img_resized = img.resize((new_w, new_h), Image.LANCZOS) # type: ignore
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

        '''self.sensor_points = [
            self.tk_canvas.create_oval(x-11, y-11, x+11, y+11, fill="blue", outline="#555", width=1)
            for x, y in self.fsr_coordinates
        ]'''
        self.cop_point = self.tk_canvas.create_oval(0, 0, 0, 0, fill="#ff7f50", outline="#444", width=2)

        group = ttk.LabelFrame(right, text="Connection & Actions")
        group.pack(fill="x", padx=16, pady=(24,8))
        ctk.CTkLabel(group, text="Select BLE Device:", font=("Arial", 14)).grid(row=0, column=0, sticky="w", padx=10, pady=8)
        ctk.CTkButton(group, text="Device 1", width=80, command=lambda: self._select_device("Device 1")).grid(row=1, column=0, padx=10, pady=3, sticky="w")
        ctk.CTkButton(group, text="Device 2", width=80, command=lambda: self._select_device("Device 2")).grid(row=1, column=1, padx=12, pady=3, sticky="w")
        self.status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(group, textvariable=self.status_var, text_color="green").grid(row=2, column=0, padx=10, pady=8, sticky="w")

        run_frame = ttk.Frame(right)
        run_frame.pack(fill="x", padx=16, pady=(0,8))
        self.start_button = ctk.CTkButton(run_frame, text="Start", width=80, command=self._start_ble)
        self.start_button.grid(row=0, column=0, padx=3)
        self.stop_button = ctk.CTkButton(run_frame, text="Stop", width=80, state="disabled", command=self._stop_ble)
        self.stop_button.grid(row=0, column=1, padx=3)

        action_frame = ttk.Frame(right)
        action_frame.pack(fill="x", padx=16, pady=(0,8))
        ctk.CTkButton(action_frame, text="Save to CSV", width=128, command=self._save_data).pack(side="left", padx=(0,6))
        ctk.CTkButton(action_frame, text="Show Live Graph", width=128, command=self._open_graph).pack(side="left", padx=(2,6))
        ctk.CTkButton(action_frame, text="Show Sensor Readings", width=128, command=self._open_readings_window).pack(side="left", padx=(2,0))

        grf_frame = ttk.LabelFrame(right, text="GRF Graph")
        grf_frame.pack(fill='both',padx=16, pady=(12,17))
        self.grf_fig, self.grf_ax = plt.subplots(figsize=(3.5,2.3), dpi=100)
        self.grf_ax.set_title("Total GRF (N)")
        self.grf_ax.set_ylim(0,48000)
        self.grf_ax.set_xlim(0,100)
        self.grf_ax.set_facecolor("white")
        self.grf_x = np.arange(100)
        self.grf_y = np.zeros(100)
        self.grf_line, = self.grf_ax.plot(self.grf_x, self.grf_y, color='red')
        self.grf_canvas = FigureCanvasTkAgg(self.grf_fig, master=grf_frame)
        self.grf_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_heatmap(self):
        canvas_width, canvas_height = self.canvas_width, self.canvas_height
        heatmap = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        for (x, y), val in zip(self.fsr_coordinates, self.latest_fsr_values):
            if 0 <= x < canvas_width and 0 <= y < canvas_height:
                heatmap[int(y), int(x)] = val

        sigma = 20
        blurred = gaussian_filter(heatmap, sigma=sigma)
        norm = (blurred / blurred.max() * 255) if blurred.max() > 0 else blurred
        norm = norm.astype(np.uint8)
        rgba = cm.get_cmap('jet')(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        self.tk_heatmap_img = ImageTk.PhotoImage(img)
        if self.heatmap_img_id is None:
            self.heatmap_img_id = self.tk_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_heatmap_img)
            self.tk_canvas.tag_lower(self.bg_img_id)
            self.tk_canvas.tag_lower(self.heatmap_img_id)
        else:
            self.tk_canvas.itemconfig(self.heatmap_img_id, image=self.tk_heatmap_img)

    def _select_device(self, name):
        device = BLE_DEVICES[name]
        self.ble_address = device["address"]
        self.characteristic_uuid = device["uuid"]
        self.status_var.set(f"Selected {name}")

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

    def scale_to_color(self, value, min_val=0, max_val=100):
        value = max(min_val, min(value, max_val))
        idx = min(int((value - min_val) / (max_val - min_val) * 100), 100)
        return self.color_gradient[idx]

    def _start_ble(self):
        if not self.ble_address or not self.characteristic_uuid:
            messagebox.showwarning("BLE", "Please select a BLE device")
            return
        self.stop_event.clear()
        self.ble_thread = threading.Thread(target=self._ble_thread_func, daemon=True)
        self.ble_thread.start()
        self.status_var.set("Reading BLE...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def _ble_thread_func(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_ble())
        except Exception as e:
            print(traceback.format_exc())
            self.root.after(0, lambda e=e: self.status_var.set(f"BLE thread error: {e}"))

    def _stop_ble(self):
        self.stop_event.set()
        self.status_var.set("Stopped")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    async def _connect_ble(self):
        if self.ble_address is not None and self.characteristic_uuid is not None:
            try:
                async with BleakClient(self.ble_address) as client:
                    await client.start_notify(self.characteristic_uuid, self._notification_handler)
                    while not self.stop_event.is_set():
                        await asyncio.sleep(0.05)
                    await client.stop_notify(self.characteristic_uuid)
            except Exception as e:
                self.root.after(0, lambda e=e: self.status_var.set(f"BLE error: {e}"))

    def _notification_handler(self, sender, data):
        try:
            values = list(map(int, data.decode("utf-8").strip().split("_")))
            if len(values) == len(self.fsr_coordinates):
                self.root.after(0, lambda values=values: self._update_fsr_values(values))
        except Exception as e:
            self.root.after(0, lambda e=e: self.status_var.set(f"Data error: {e}"))

    def _update_fsr_values(self, values):
        self.latest_fsr_values = values
        self.fsr_data_records.append(values)

    def _update_gui(self):
        if not self.root.winfo_exists():
            return
        try:
            self._draw_heatmap()
            total = sum(self.latest_fsr_values)
            if self.cop_point is not None:
                if total:
                    x_cop = sum(x * p for (x, y), p in zip(self.fsr_coordinates, self.latest_fsr_values)) / total
                    y_cop = sum(y * p for (x, y), p in zip(self.fsr_coordinates, self.latest_fsr_values)) / total
                    self.tk_canvas.coords(self.cop_point, x_cop-8, y_cop-8, x_cop+8, y_cop+8)
                else:
                    self.tk_canvas.coords(self.cop_point, 0, 0, 0, 0)
        except Exception as e:
            print(f"Update GUI error: {e}")
        # Update GRF graph
        self.grf_y = np.roll(self.grf_y, -1)
        self.grf_y[-1] = total
        self.grf_line.set_ydata(self.grf_y)
        self.grf_canvas.draw()
        self.root.after(100, self._update_gui)

    def _save_data(self):
        if not self.fsr_data_records:
            messagebox.showinfo("No Data", "No data to save")
            return
        df = pd.DataFrame(self.fsr_data_records, columns=[f"FSR{i+1}" for i in range(len(self.fsr_coordinates))])
        df.to_csv("pressure_data.csv", index=False)
        self.status_var.set("Saved pressure_data.csv")

    def _open_graph(self):
        win = ctk.CTkToplevel(self.root)
        win.geometry("660x340")
        win.title("Live Graph")
        fig, ax = plt.subplots(figsize=(6,2.8))
        ax.set_ylim(0, 3300)
        ax.set_xlim(0, 100)
        ax.set_title("Live Pressure Data")
        x_data = np.arange(100)
        y_datas = [np.zeros(100) for _ in self.fsr_coordinates]
        lines = [ax.plot(x_data, y_datas[i], label=f"FSR {i+1}")[0] for i in range(len(self.fsr_coordinates))]
        ax.grid(True, which='both', color="#eee", linewidth=0.8)
        ax.set_facecolor("#f9fafd")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        def update_plot():
            if not win.winfo_exists():
                return
            data_len = len(self.fsr_data_records)
            if data_len:
                for i in range(len(lines)):
                    y = [row[i] for row in self.fsr_data_records[-100:]]
                    y_padded = np.pad(y, (100-len(y), 0), 'constant')
                    lines[i].set_ydata(y_padded)
            canvas.draw()
            win.after(200, update_plot)
        update_plot()

    def _open_readings_window(self):
        win = ctk.CTkToplevel(self.root)
        win.geometry("300x540")
        win.title("Live Sensor Readings")
        labels = []
        for i in range(len(self.fsr_coordinates)):
            lbl = ctk.CTkLabel(win, text=f"FSR{i+1}: 0", font=("Arial", 14))
            lbl.pack(anchor="w", padx=20, pady=7)
            labels.append(lbl)
        def update_labels():
            if not win.winfo_exists():
                return
            for i, lbl in enumerate(labels):
                lbl.configure(text=f"FSR{i+1}: {self.latest_fsr_values[i]}")
            win.after(100, update_labels)
        update_labels()

def main():
    root = ctk.CTk()
    root.resizable(False, False)
    app = InsoleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()