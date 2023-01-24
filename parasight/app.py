import os
import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import ImageTk, Image

class App(TkinterDnD.Tk):
    def __init__(self):
        # Initialize
        TkinterDnD.Tk.__init__(self)
        self.title("PARASIGHT")

        # Setup
        self.setup()

    def setup(self):
        # Theme
        theme = os.path.join(os.path.dirname(__file__), "azure.tcl")
        self.tk.call("source", theme)
        self.tk.call("set_theme", "light")

        # Configure
        self.columnconfigure(index=0, weight=1)
        self.rowconfigure(index=0, weight=1)
        self.style = ttk.Style()
        self.style.configure("Treeview", rowheight=30)

        # Master Frame
        master = Master(self, self)
        master.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Sizegrip
        self.sizegrip = ttk.Sizegrip(self)
        self.sizegrip.grid(row=1, column=1, padx=5, pady=5)

        # Set Minimum Size
        self.minsize(int(self.winfo_screenwidth() * 5 / 6), int(self.winfo_screenheight() * 5 / 6))
        x_coordinate = int(self.winfo_screenwidth() / 12)
        y_coordinate = int(self.winfo_screenheight() / 12)
        self.geometry("+{}+{}".format(x_coordinate, y_coordinate))

class Master(ttk.Frame):
    def __init__(self, parent, master):
        # Initialize
        ttk.Frame.__init__(self, parent)
        self.master = master

        # Setup
        self.setup() 

    def setup(self):
        # Model
        self.model = torch.hub.load('.', 'custom', 'weights.pt', source='local')

        # Drag and Drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', lambda e: self.open(e.data))

        # Configure
        self.rowconfigure(index=1, weight=1)
        self.columnconfigure(index=0, weight=1, uniform="fred")
        self.columnconfigure(index=1, weight=1, uniform="fred")

        # Open Frame
        self.open_frame = ttk.Frame(self)
        self.open_frame.columnconfigure(index=1, weight=1)
        self.open_frame.grid(row=0, column=0, sticky="nsew")

        # Open Button
        self.open_button = ttk.Menubutton(self.open_frame, text="Open")
        self.open_button.menu_0 = tk.Menu(self.open_button)
        self.open_button.config(menu=self.open_button.menu_0)
        self.open_button.menu_0.add_command(label="Image", command=lambda: self.open("image"))
        self.open_button.menu_0.add_command(label="Folder", command=lambda: self.open("folder"))
        self.open_button.grid(row=0, column=0, padx=5, pady=5)

        # Open Name
        self.open_label = ttk.Label(self.open_frame)
        self.open_label.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Analyze Frame
        self.analyze_frame = ttk.Frame(self)
        self.analyze_frame.columnconfigure(index=4, weight=1)
        self.analyze_frame.grid(row=0, column=1, sticky="nsew")

        # Analyze Depth
        self.depth_label = ttk.Label(self.analyze_frame, text="Depth")
        self.depth_label.grid(row=0, column=0, padx=5, pady=5)
        self.depth_spin = ttk.Spinbox(self.analyze_frame, from_=1, to=5, increment=1, width=4)
        self.depth_spin.set(3)
        self.depth_spin.grid(row=0, column=1, padx=5, pady=5)

        # Analyze Confidence
        self.confidence_label = ttk.Label(self.analyze_frame, text="Confidence")
        self.confidence_label.grid(row=0, column=2, padx=5, pady=5)
        self.confidence_spin = ttk.Spinbox(self.analyze_frame, from_=0.01, to=0.99, increment=0.01, width=6)
        self.confidence_spin.set(0.90)
        self.confidence_spin.grid(row=0, column=3, padx=5, pady=5)

        # Analyze Button
        self.analyze_progress_1 = ttk.Frame(self.analyze_frame)
        self.analyze_progress_1.grid(row=0, column=5, sticky="nsew")
        self.analyze_button = ttk.Button(self.analyze_progress_1, text="Anaylze", command=self.analyze)
        self.analyze_button.grid(row=0, column=0, padx=5, pady=5)

        # Progress bar
        self.analyze_progress_2 = ttk.Frame(self.analyze_frame)
        self.analyze_progress_2.grid(row=0, column=5, sticky="nsew")
        self.progress_bar = ttk.Progressbar(self.analyze_progress_2, mode="determinate")
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.analyze_progress_1.tkraise()

        # Image Frame
        self.image_frame = ttk.Frame(self)
        self.image_frame.columnconfigure(index=0, weight=1)
        self.image_frame.rowconfigure(index=0, weight=1)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Image Label
        self.image_label = tk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Result Frame
        self.result_frame = ttk.Frame(self)
        self.result_frame.columnconfigure(index=0, weight=1)
        self.result_frame.rowconfigure(index=0, weight=1)
        self.result_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Result Tree
        self.result_tree = ttk.Treeview(self.result_frame, height=10, show="headings")
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        self.result_scroll = ttk.Scrollbar(self.result_frame, orient="vertical")
        self.result_scroll.grid(row=0, column=1, sticky="nsew")
        self.result_tree.config(yscrollcommand=self.result_scroll.set)
        self.result_scroll.config(command=self.result_tree.yview)
        self.result_tree.bind("<<TreeviewSelect>>", lambda e: self.tree_select())
        
        columns = ["Parasite", "Confidence", "Position", "Size", "Image"]
        widths = [100, 10, 10, 10, 10]
        self.result_tree.config(columns=columns)
        for i, col in enumerate(columns):
            self.result_tree.column(col, anchor="center", minwidth=widths[i], width=widths[i])
            self.result_tree.heading(col, text=col, anchor="center")

        # Variable
        self.img_pth = None
        self.parasite_name = [
            "Ascaris lumbricoides",
            "Capillaria philippinensis",
            "Enterobius vermicularis",
            "Fasciolopsis buski",
            "Hookworm",
            "Hymenolepis diminuta",
            "Hymenolepis nana",
            "Opisthorchis viverrine",
            "Paragonimus spp",
            "Taenia spp",
            "Trichuris trichiura"
        ]

    def open(self, kind):
        if kind == "image":
            pth = filedialog.askopenfilename(
                title="Open Image"
            )
        elif kind == "folder":
            pth = filedialog.askdirectory(
                title="Open Folder"
            )
        if pth:
            self.img_pth = pth
            self.open_label.config(text=pth)
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.update()
    
    def analyze(self):
        # Image List
        if not self.img_pth:
            messagebox.showerror(
                title="Error",
                message="Please open an image or folder to analyze."
            )
            return
        if os.path.isdir(self.img_pth):
            img_lst = [f for f in os.listdir(self.img_pth)]
            img_lst.sort()
            img_lst = [os.path.join(self.img_pth, f) for f in img_lst]
        else:
            img_lst = [self.img_pth]

        # Start Detection
        self.analyze_progress_2.tkraise()
        img_cnt = 0

        # Detect
        depth = int(self.depth_spin.get())
        self.egg_imgs = []
        egg_cnt = 0
        for img_name in img_lst:
            # Padding
            img = cv2.imread(img_name)[:,:,::-1]
            if img.shape[0] > img.shape[1]:
                size = img.shape[0]
                displace = (0, (img.shape[0] - img.shape[1]) // 2)
            else:
                size = img.shape[1]
                displace = ((img.shape[1] - img.shape[0]) // 2, 0)
            img_new = np.zeros((size, size, 3))
            img_new[displace[0]:displace[0]+img.shape[0], displace[1]:displace[1]+img.shape[1], :] = img[:, :, :]

            # Cropping
            input_lst = [[img_new]]
            place_lst = [[(0, 0)]]
            for d in range(1, depth):
                temp_img = []
                temp_pos = []
                for img, pos in zip(input_lst[d-1], place_lst[d-1]):
                    size = img.shape[0]
                    temp_img.append(img[:size//2, :size//2, :])
                    temp_pos.append((pos[0], pos[1]))
                    temp_img.append(img[:size//2, size//2:, :])
                    temp_pos.append((pos[0], pos[1]+size//2))
                    temp_img.append(img[size//2:, :size//2, :])
                    temp_pos.append((pos[0]+size//2, pos[1]))
                    temp_img.append(img[size//2:, size//2:, :])
                    temp_pos.append((pos[0]+size//2, pos[1]+size//2))
                input_lst.append(temp_img)
                place_lst.append(temp_pos)
            input_lst = [j for i in input_lst for j in i]
            place_lst = [j for i in place_lst for j in i]

            # Inference
            results = self.model(input_lst)
            boxes = []
            for i, (pred, place) in enumerate(zip(results.pred, place_lst)):
                if pred.numel() != 0:
                    pred_ = pred.clone()
                    pred_[:,0] += place[1]
                    pred_[:,1] += place[0]
                    pred_[:,2] += place[1]
                    pred_[:,3] += place[0]
                    img_tag = torch.full((len(pred), 1), i, device='cuda')
                    pred_ = torch.cat([pred_, img_tag], dim=1)
                    boxes.append(pred_)
            if boxes: boxes = torch.cat(boxes)
            else: boxes = torch.zeros(1, 6)

            # Size NMS
            confidence = float(self.confidence_spin.get())
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            area = wh[:, 0] * wh[:, 1]
            _ , order = torch.sort(area, descending=True)
            order = order[boxes[order,4] > confidence]
            keep = []
            while order.shape[0] > 0:
                keep.append(order[0])
                idx_a = order[0:1].repeat(order.shape[0])
                idx_b = order
                inter_1 = torch.maximum(boxes[idx_a, 0:2], boxes[idx_b, 0:2])
                inter_2 = torch.maximum(boxes[idx_a, 2:4], boxes[idx_b, 2:4])
                inter = torch.clamp(inter_2 - inter_1, 0)
                area_i = inter[:, 0] * inter[:, 1]
                area_s = torch.minimum(area[idx_a], area[idx_b])
                piou = area_i / area_s
                order = order[piou < 0.8]

            # Add Result
            if keep:
                results.render()
                for k in keep:
                    info = boxes[k].tolist()
                    parasite = self.parasite_name[int(info[5])]
                    confidence = '{0:.0%}'.format(info[4])
                    position = "({}, {})".format(int((info[0]+info[2])/2), int((info[1]+info[3])/2))
                    size = "({}, {})".format(int(info[2]-info[0]), int(info[3]-info[1]))
                    image = os.path.basename(img_name)
                    detect = [parasite, confidence, position, size, image]
                    self.result_tree.insert("", index="end", values=detect, tags=egg_cnt)
                    egg_cnt += 1
                    self.egg_imgs.append(results.ims[boxes[k, 6].int()])
                    self.update()

            # Progress
            img_cnt += 1
            self.progress_bar['value'] = img_cnt * 100 / len(img_lst)
            self.update()
        
        self.analyze_progress_1.tkraise()

    def tree_select(self):
        geometry = self.master.geometry()
        img_tag = self.result_tree.item(self.result_tree.selection())["tags"][0]
        image = Image.fromarray(self.egg_imgs[img_tag])
        size = min(self.image_label.winfo_width(), self.image_label.winfo_height())
        image = image.resize((size, size))
        self.image_show = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_show)
        self.master.geometry(geometry)

if __name__ == "__main__":
    root = App()
    root.mainloop()