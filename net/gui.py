import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
from PIL import Image, ImageTk

import sys
import gc
import os
import numpy as np
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils

class ClearVision:
    def __init__(self, root):
        self.root = root
        self.root.title('Clear Vision - FFA-NET Dehazing')
        self.root.geometry('1050x600')
        self.root['bg'] = 'white'
        self.clear_path = None
        self.model_path = None
        self.swinir_path = None

        # Header Frame (Title + Memeber's name horizontally arrayed)
        header_frame = tk.Frame(self.root, bg='#4a90e2')
        header_frame.pack(fill='x',pady=(10, 0))

        self.title_label = tk.Label(header_frame, text='🔬 ClearVision',
                                    font=('Segoe UI', 18, 'bold'), bg='#4a90e2', fg='white')
        self.title_label.pack(side='left', padx=10, pady=10)

        self.members_label = tk.Label(header_frame,
                                      text='Chaebin Park, Jieun Seo, Sujin Lee, Hyojae Lee, Maro Yoon, Wonseok Heo',
                                      font=('Segoe UI', 11), bg='#4a90e2', fg='#e3f2fd')
        self.members_label.pack(side='right', padx=10, pady=10)

        # Original paper link frame
        link_frame = tk.Frame(self.root, bg='#e3f2fd')
        link_frame.pack(fill='x', pady=(0, 20))

        self.link_label = tk.Label(link_frame, text="📄 FFA-NET Link", 
                                   font=('Segoe UI', 11, 'underline'),
                                   bg='#e3f2fd', fg='#1976d2', cursor='hand2', pady=10)
        self.link_label.pack(pady=10)
        self.link_label.bind("<Button-1>", self.open_paper_link)

        # Button frame
        button_frame = tk.LabelFrame(self.root, bg='white',fg='#424242', relief='groove', bd=2)
        button_frame.pack(fill='x',padx=10,pady=10)

        self.upload_btn = tk.Button(button_frame, text='🔗 Load FFA-Net Model', font=('Segoe UI', 11, 'bold'), command=self.select_model)
        self.upload_btn.pack(side='left',padx=5)

        self.image_btn = tk.Button(button_frame, text='🖼️ Select the Image', font=('Segoe UI', 11, 'bold'),command=self.open_file)
        self.image_btn.pack(side='left',padx=5)

        self.result_btn = tk.Button(button_frame, text='🔍 View results', font=('Segoe UI', 11, 'bold'), command=self.run_model)
        self.result_btn.pack(side='left',padx=5)
        
        self.model_label = tk.Label(button_frame, text='No model has been chosen', font=('Segoe UI', 10), 
                           bg='white', fg='#424242', justify='left', wraplength=800)
        self.model_label.pack(side='left',padx=5)
        
        images_frame = tk.Frame(self.root, bg='white')
        images_frame.pack(fill='x', padx=10)
        
        frame_out = tk.Frame(images_frame, bg="#ffffff")
        self.hazy_frame = tk.Frame(frame_out, width=500, height=300, bg="#eeeeee")
        self.hazy_frame.pack()
        frame_label = tk.Label(frame_out, text='Hazy Image', font=('Segoe UI', 10), bg="#ffffff")
        frame_label.pack()
        frame_out.pack(side='left',padx=10, pady=10)
        
        frame_out = tk.Frame(images_frame, bg="#ffffff")
        self.clear_frame = tk.Frame(frame_out, width=500, height=300, bg="#eeeeee")
        self.clear_frame.pack()
        frame_label = tk.Label(frame_out, text='FFA-NET', font=('Segoe UI', 10), bg="#ffffff")
        frame_label.pack()
        frame_out.pack(side='right',padx=10, pady=10)
    
    def select_model(self):
        file_path = filedialog.askopenfilename(
            title="Load FFA-Net Model",
            filetypes=[("Trained model", "*.pk")],
            initialdir='trained_models'
        )
    
        if not file_path:
            return
        
        self.model_path = file_path
        self.model_label.config(text = 'FFA-NET: ' + os.path.basename(self.model_path))
    
    def run_model(self):
        if not self.model_path or not self.clear_path:
            tk.messagebox.showerror(title='Error', message='Select an image and a model')
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckp=torch.load(self.model_path,map_location=device,weights_only=False)
        net=FFA(gps=3,blocks=19)
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
        net.eval()
        
        # Image resize to get reasonable performance for large images
        haze = Image.open(self.clear_path).convert('RGB')
        re_size = (self.hazy_frame.winfo_width() * 3, self.hazy_frame.winfo_height() * 3)
        print("Resize Image:", re_size)
        haze.thumbnail(re_size)
        haze1= tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
        ])(haze)[None,::]
        
        print("FFA-Net inference started")
        with torch.no_grad():
            pred = net(haze1)
        print("FFA-Net inference finished")
        
        pred = pred.clamp(0,1)
        ts=torch.squeeze(pred.cpu())
        
        clear_img = tfs.ToPILImage()(ts)
        swinir_img = None
        
        del ts
        del ckp
        del net
        del haze
        del haze1
        
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            #clear_img.save("ffa-net.png")
            image = clear_img.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
    
            # Remove the previous image
            for widget in self.clear_frame.winfo_children():
                widget.destroy()
    
            # Display the new image
            img_label = tk.Label(self.clear_frame, image=photo)
            img_label.image = photo
            img_label.pack()
        except Exception as e:
            messagebox.showerror("Image Load Failed", str(e))
        
       
    def open_paper_link(self, event):
        """Open the FFA-Net paper link"""
        webbrowser.open("https://arxiv.org/abs/1911.07559")

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select the Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
            initialdir='test_imgs'
        )
    
        if not file_path:
            return
    
        try:
            image = Image.open(file_path)
            image = image.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
    
            # Remove the previous image
            for widget in self.hazy_frame.winfo_children():
                widget.destroy()
    
            # Display the new image
            img_label = tk.Label(self.hazy_frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
            self.clear_path = file_path
    
        except Exception as e:
            messagebox.showerror("Failed to load the image", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = ClearVision(root)
    root.mainloop()