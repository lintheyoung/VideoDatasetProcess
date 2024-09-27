import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import shutil
import zipfile
import tempfile
import re
import xml.etree.ElementTree as ET
from moviepy.editor import VideoFileClip

import webbrowser

import time
import random
import cv2
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import xml.dom.minidom as DOC

import datetime

def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(float(box.find('xmin').text))
        y_min = int(float(box.find('ymin').text))
        x_max = int(float(box.find('xmax').text))
        y_max = int(float(box.find('ymax').text))
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

ctk.set_appearance_mode("System")  # System theme
ctk.set_default_color_theme("blue")  # Blue theme

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('DeDeMaker视频数据处理-V1.0')
        self.geometry('600x700')  # Increased height to accommodate all sections

        # 添加DeDeMaker图标和点击事件
        self.label_dedemaker = ctk.CTkLabel(self, text="DeDeMaker.com", font=("微软雅黑", 15), cursor="hand2")
        self.label_dedemaker.place(x=10, y=10)  # 根据需要调整位置
        self.label_dedemaker.bind("<Button-1>", lambda e: webbrowser.open("http://dedemaker.com"))

        # 视频处理部分
        self.label_video_processing = ctk.CTkLabel(self, text="1、视频处理", font=("微软雅黑", 12))
        self.label_video_processing.pack(pady=(10, 0))
        
        self.frame_video_processing = ctk.CTkFrame(self)
        self.frame_video_processing.pack(pady=5)
        
        self.btn_select_input_video = ctk.CTkButton(self.frame_video_processing, text="选择输入视频", font=("微软雅黑", 12), command=self.select_input_video)
        self.btn_select_input_video.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_select_output_folder_video = ctk.CTkButton(self.frame_video_processing, text="选择输出文件夹", font=("微软雅黑", 12), command=self.select_output_folder_video)
        self.btn_select_output_folder_video.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_start_video = ctk.CTkButton(self.frame_video_processing, text="开始", font=("微软雅黑", 12), command=self.start_video_compression)
        self.btn_start_video.grid(row=0, column=2, padx=10, pady=10)

        # 数据集合并部分
        self.label_dataset_merge = ctk.CTkLabel(self, text="2、合并数据集", font=("微软雅黑", 12))
        self.label_dataset_merge.pack(pady=(10, 0))
        
        self.frame_dataset_merge = ctk.CTkFrame(self)
        self.frame_dataset_merge.pack(pady=5)
        
        self.btn_select_files = ctk.CTkButton(self.frame_dataset_merge, text="选择文件夹", font=("微软雅黑", 12), command=self.select_files)
        self.btn_select_files.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_select_output_folder_dataset = ctk.CTkButton(self.frame_dataset_merge, text="选择输出文件夹", font=("微软雅黑", 12), command=self.select_output_folder_dataset)
        self.btn_select_output_folder_dataset.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_start_dataset = ctk.CTkButton(self.frame_dataset_merge, text="开始", font=("微软雅黑", 12), command=self.start_dataset_merge)
        self.btn_start_dataset.grid(row=0, column=2, padx=10, pady=10)

        # 数据集增强部分
        self.label_dataset_aug = ctk.CTkLabel(self, text="3、数据集增强", font=("微软雅黑", 12))
        self.label_dataset_aug.pack(pady=(10, 0))
        
        self.frame_dataset_aug = ctk.CTkFrame(self)
        self.frame_dataset_aug.pack(pady=5)
        
        self.btn_select_files_aug = ctk.CTkButton(self.frame_dataset_aug, text="选择zip文件", font=("微软雅黑", 12), command=self.select_files_aug)
        self.btn_select_files_aug.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_select_output_folder_dataset_aug = ctk.CTkButton(self.frame_dataset_aug, text="选择输出文件夹", font=("微软雅黑", 12), command=self.select_output_folder_dataset_aug)
        self.btn_select_output_folder_dataset_aug.grid(row=0, column=1, padx=10, pady=10)
        
        self.btn_start_dataset_aug = ctk.CTkButton(self.frame_dataset_aug, text="开始", font=("微软雅黑", 12), command=self.start_dataset_aug)
        self.btn_start_dataset_aug.grid(row=0, column=2, padx=10, pady=10)

        # 进度条显示（可选）
        self.progress_label = ctk.CTkLabel(self, text="", font=("微软雅黑", 10))
        self.progress_label.pack(pady=(10, 0))

        # Attributes for video processing
        self.input_video_path = ''
        self.output_video_folder = ''

        # Attributes for dataset merging
        self.files_to_merge = []
        self.output_dataset_folder = ''

        # Attributes for dataset enhancement
        self.files_to_aug = []
        self.output_dataset_folder_aug = ''

    def select_input_video(self):
        self.input_video_path = filedialog.askopenfilename(title="选择输入视频文件", filetypes=(("MP4 files", "*.mp4"),))

    def select_output_folder_video(self):
        self.output_video_folder = filedialog.askdirectory(title="选择输出文件夹")

    def start_video_compression(self):
        if self.input_video_path and self.output_video_folder:
            timestamp = generate_timestamp()  # 获取时间戳
            output_video_path = os.path.join(self.output_video_folder, f"compressed_video_{timestamp}.mp4")  # 文件名中加入时间戳

            compress_video(self.input_video_path, output_video_path)
            messagebox.showinfo("完成", "视频处理完成！")
        else:
            messagebox.showwarning("警告", "请选择输入视频和输出文件夹！")

    def select_files(self):
        self.files_to_merge = filedialog.askdirectory(title="选择文件夹")  # 让用户选择一个文件夹
        if self.files_to_merge:
            print(f"Selected folder: {self.files_to_merge}")  # 打印所选文件夹路径，可根据需要删除此行

    def select_output_folder_dataset(self):
        self.output_dataset_folder = filedialog.askdirectory(title="选择输出文件夹")

    def start_dataset_merge(self):
        if self.files_to_merge and self.output_dataset_folder:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_path = self.files_to_merge
                unzip_datasets(base_path, temp_dir)
                custom_tag = 'merged'
                merge_voc_datasets(temp_dir, custom_tag, self.output_dataset_folder)

                # 指定合并后数据集的目录路径
                merged_path = os.path.join(self.output_dataset_folder, 'merged')
                
                # 构造输出ZIP文件的路径
                timestamp = generate_timestamp()  # 获取时间戳
                output_zip = os.path.join(self.output_dataset_folder, f"{custom_tag}_dataset_{timestamp}.zip")
                
                # 调用zip_merged_dataset函数压缩合并后的文件夹
                zip_merged_dataset(merged_path, output_zip)

                # 删除merged文件夹及其所有内容
                shutil.rmtree(merged_path)

                print(output_zip)

                messagebox.showinfo("完成", "数据集合并并压缩完成！")
        else:
            messagebox.showwarning("警告", "请选择数据集文件夹和输出文件夹！")

    def select_files_aug(self):
        # 使用filedialog.askopenfilename替换askdirectory
        self.files_to_aug = filedialog.askopenfilename(title="选择ZIP文件", filetypes=(("ZIP files", "*.zip"),))
        if self.files_to_aug:
            print(f"Selected ZIP file: {self.files_to_aug}")  # 打印所选ZIP文件路径

    def select_output_folder_dataset_aug(self):
        self.output_dataset_folder_aug = filedialog.askdirectory(title="选择输出文件夹")

    def start_dataset_aug(self):
        if self.files_to_aug and self.output_dataset_folder_aug:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_path = self.files_to_aug
                # unzip_datasets_and_process now handles augmentation
                augmented_zip_path = unzip_datasets_and_process(base_path, self.output_dataset_folder_aug)

                messagebox.showinfo("完成", "数据增强完成！")
        else:
            messagebox.showwarning("警告", "请选择数据集文件夹和输出文件夹！")

class DataAugmentForObjectDetection():
    def __init__(self, 
                 rotation_rate=1.0,  # Set to 1.0 to ensure rotation is always selected when chosen
                 max_rotation_angle=90,  # Using fixed angles for predictability
                 crop_rate=0.5,  # Adjusted to match user-provided example
                 shift_rate=1.0,
                 change_light_rate=1.0,
                 add_noise_rate=1.0,
                 flip_rate=1.0,
                 cutout_rate=1.0,
                 cut_out_length=50,
                 cut_out_holes=1,
                 cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
    
    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array, 由于输出的像素是在[0,1]之间, 所以得乘以255
        '''
        noisy_img = random_noise(img, mode='gaussian', clip=True) * 255
        return noisy_img.astype(np.uint8)

    # 调整亮度
    def _changeLight(self, img):
        flag = random.uniform(0.5, 1.5) # flag >1 为调亮, <1 为调暗
        adjusted_img = exposure.adjust_gamma(img, flag)
        return adjusted_img.astype(np.uint8)
    
    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bounding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)

            # compute the area of boxB
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

            # compute the intersection over union
            iou = interArea / float(boxBArea)

            return iou

        h, w = img.shape[:2]
        mask = np.ones((h, w, 3), np.float32)

        for n in range(n_holes):
            overlap = True
            attempts = 0
            max_attempts = 10  # To prevent infinite loop

            while overlap and attempts < max_attempts:
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)

                y1 = max(y - length // 2, 0)
                y2 = min(y + length // 2, h)
                x1 = max(x - length // 2, 0)
                x2 = min(x + length // 2, w)

                # Check overlap with any bounding box
                overlap = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box[:4]) > threshold:
                        overlap = True
                        break

                attempts += 1

            if not overlap:
                mask[y1: y2, x1: x2, :] = 0.

        img = (img * mask).astype(np.uint8)

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=90, scale=1.0):
        '''
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有bounding boxes, 一个list, 每个元素为 [x_min, y_min, x_max, y_max, name]
            angle:旋转角度
            scale:缩放比例
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的bounding boxes list
        '''
        h, w = img.shape[:2]
        # 计算旋转矩阵
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        # 旋转图像，使用BORDER_CONSTANT避免反射导致的图像重复
        rot_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # 旋转bounding boxes
        rot_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, name = bbox
            # 定义bbox的四个角
            points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
            # 添加一个维度
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            # 应用旋转矩阵
            transformed_points = rot_mat.dot(points_ones.T).T
            # 获取新的bbox
            x_coords = transformed_points[:,0]
            y_coords = transformed_points[:,1]
            rx_min, ry_min = x_coords.min(), y_coords.min()
            rx_max, ry_max = x_coords.max(), y_coords.max()
            # Clamp to image boundaries
            rx_min = max(0, min(w, rx_min))
            ry_min = max(0, min(h, ry_min))
            rx_max = max(0, min(w, rx_max))
            ry_max = max(0, min(h, ry_max))
            # 确保bbox有效
            if rx_max > rx_min and ry_max > ry_min:
                rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, name])
        
        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含原始图像的一部分，并根据裁剪调整bounding boxes
        输入:
            img:图像array
            bboxes:该图像包含的所有bounding boxes, 一个 list, 每个元素为 [x_min, y_min, x_max, y_max, name]
        输出:
            resized_img:裁剪并调整尺寸后的图像array
            new_bboxes:裁剪并调整后的bounding boxes list
        '''
        h, w = img.shape[:2]
        scale = self.crop_rate  # 例如，0.5 表示裁剪50%的区域

        # 计算裁剪区域的大小
        crop_h, crop_w = int(h * scale), int(w * scale)

        if crop_h >= h or crop_w >= w:
            # 如果裁剪比例过大，返回原图
            return img.copy(), bboxes.copy()

        # 随机选择裁剪区域的位置
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)

        # 裁剪图像
        cropped_img = img[y:y+crop_h, x:x+crop_w].copy()

        # 调整裁剪后的图像大小回原始尺寸
        resized_img = cv2.resize(cropped_img, (w, h))

        # 调整bounding boxes
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2, name = bbox
            # 调整bbox坐标相对于裁剪区域
            new_x1 = x1 - x
            new_y1 = y1 - y
            new_x2 = x2 - x
            new_y2 = y2 - y

            # 只保留与裁剪区域有交集的bbox
            # 如果bbox完全在裁剪区域外，则不保留
            if new_x2 <= 0 or new_y2 <= 0 or new_x1 >= crop_w or new_y1 >= crop_h:
                continue

            # 截取bbox与裁剪区域的交集
            new_x1 = max(new_x1, 0)
            new_y1 = max(new_y1, 0)
            new_x2 = min(new_x2, crop_w)
            new_y2 = min(new_y2, crop_h)

            # 确保bbox有效
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue

            # 缩放bbox坐标回原始图像尺寸
            new_x1 = new_x1 / scale
            new_y1 = new_y1 / scale
            new_x2 = new_x2 / scale
            new_y2 = new_y2 / scale

            # Clamp to image boundaries
            new_x1 = max(0, min(w, new_x1))
            new_y1 = max(0, min(h, new_y1))
            new_x2 = max(0, min(w, new_x2))
            new_y2 = max(0, min(h, new_y2))

            # 确保bbox有效 after scaling
            if new_x2 > new_x1 and new_y2 > new_y1:
                new_bboxes.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2), name])

        return resized_img, new_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有bounding boxes, 一个 list, 每个元素为 [x_min, y_min, x_max, y_max, name]
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding boxes list
        '''
        h, w = img.shape[:2]
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[2] for bbox in bboxes])
        y_max = max([bbox[3] for bbox in bboxes])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # 随机平移范围
        # 为了防止平移过大，限制平移的比例
        max_shift_ratio = 0.3
        max_shift_x = min(d_to_left, d_to_right, int(w * max_shift_ratio))
        max_shift_y = min(d_to_top, d_to_bottom, int(h * max_shift_ratio))

        x_shift = random.uniform(-max_shift_x, max_shift_x)
        y_shift = random.uniform(-max_shift_y, max_shift_y)
        
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])  # x_shift: right positive, left negative; y_shift: down positive, up negative
        shift_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 平移bounding boxes
        shift_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, name = bbox
            sx_min = x_min + x_shift
            sy_min = y_min + y_shift
            sx_max = x_max + x_shift
            sy_max = y_max + y_shift

            # Clamp to image boundaries
            sx_min = max(0, sx_min)
            sy_min = max(0, sy_min)
            sx_max = min(w, sx_max)
            sy_max = min(h, sy_max)

            # Discard invalid boxes
            if sx_max > sx_min and sy_max > sy_min:
                shift_bboxes.append([int(sx_min), int(sy_min), int(sx_max), int(sy_max), name])
        
        return shift_img, shift_bboxes

    # 镜像
    def _flip_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            翻转后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有bounding boxes, 一个 list, 每个元素为 [x_min, y_min, x_max, y_max, name]
            输出:
                flip_img:翻转后的图像array
                flip_bboxes:翻转后的bounding boxes list
        '''
        flip_img = img.copy()
        h, w = img.shape[:2]
        flip_direction = random.choice(['horizontal', 'vertical'])
        if flip_direction == 'horizontal':
            flip_img = cv2.flip(flip_img, 1)
        else:
            flip_img = cv2.flip(flip_img, 0)

        flip_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, name = bbox
            if flip_direction == 'horizontal':
                fx_min = w - x_max
                fx_max = w - x_min
                fy_min = y_min
                fy_max = y_max
            else:
                fx_min = x_min
                fx_max = x_max
                fy_min = h - y_max
                fy_max = h - y_min
            flip_bboxes.append([int(fx_min), int(fy_min), int(fx_max), int(fy_max), name])
        
        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标, 格式为 [ [x_min, y_min, x_max, y_max, name], ... ]
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
            augmentation: 增强操作名称
        '''
        augmentations = ['Crop', 'Rotate', 'Shift', 'Change Lightness', 'Add Noise', 'Cutout', 'Flip']
        
        # 随机决定应用多少种增强方法（例如，1到3种）
        num_augmentations = random.randint(1, min(3, len(augmentations)))
        selected_augmentations = random.sample(augmentations, num_augmentations)

        applied_augmentations = []

        # Make a copy to avoid modifying the original bboxes
        bboxes = [bbox.copy() for bbox in bboxes]

        for augmentation in selected_augmentations:
            if augmentation == 'Crop':
                img, bboxes = self._crop_img_bboxes(img, bboxes)
                applied_augmentations.append('Crop')
            elif augmentation == 'Rotate':
                angle = random.choice([90, 180, 270])  # 使用固定角度以确保更 predictable bbox transformations
                scale = random.uniform(0.7, 1.0)  # 保持 scale >=0.7 to prevent excessive scaling
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
                applied_augmentations.append(f'Rotate_{angle}')
            elif augmentation == 'Shift':
                img, bboxes = self._shift_pic_bboxes(img, bboxes)
                applied_augmentations.append('Shift')
            elif augmentation == 'Change Lightness':
                img = self._changeLight(img)
                applied_augmentations.append('Change_Lightness')
            elif augmentation == 'Add Noise':
                img = self._addNoise(img)
                applied_augmentations.append('Add_Noise')
            elif augmentation == 'Cutout':
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)
                applied_augmentations.append('Cutout')
            elif augmentation == 'Flip':
                img, bboxes = self._flip_pic_bboxes(img, bboxes)
                applied_augmentations.append('Flip')
        
        # Ensure bboxes are within image boundaries
        h, w = img.shape[:2]
        for bbox in bboxes:
            bbox[0] = max(0, min(w, bbox[0]))  # x_min
            bbox[1] = max(0, min(h, bbox[1]))  # y_min
            bbox[2] = max(0, min(w, bbox[2]))  # x_max
            bbox[3] = max(0, min(h, bbox[3]))  # y_max

        # Draw augmentation labels on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green color for text
        thickness = 1
        augmentation_text = ', '.join(applied_augmentations)
        cv2.putText(img, augmentation_text, (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)

        return img, bboxes, augmentation_text

def data_augmentation(extract_folder, custom_tag):
    # 设置源文件夹路径（已从上传的ZIP解压）
    source_pic_root_path = os.path.join(extract_folder, 'images')  # 图片文件夹路径
    source_xml_root_path = os.path.join(extract_folder, 'xml')  # XML文件夹路径

    # 设置保存增强后的图片和XML文件的文件夹路径
    saved_pic_root_path = os.path.join(extract_folder, 'augmented', 'images')
    saved_xml_root_path = os.path.join(extract_folder, 'augmented', 'xml')
    if not os.path.exists(saved_pic_root_path):
        os.makedirs(saved_pic_root_path)
    if not os.path.exists(saved_xml_root_path):
        os.makedirs(saved_xml_root_path)

    dataAug = DataAugmentForObjectDetection()  # 使用你的数据增强类

    for file in os.listdir(source_pic_root_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            pic_path = os.path.join(source_pic_root_path, file)
            xml_path = os.path.join(source_xml_root_path, os.path.splitext(file)[0] + '.xml')

            if not os.path.exists(xml_path):
                print(f"Warning: XML file for {file} does not exist. Skipping.")
                continue

            # 解析XML文件获取bboxes
            coords = parse_xml(xml_path)  # 格式为[[x_min, y_min, x_max, y_max, name]]
            coords = [coord[:5] for coord in coords]  # 确保coords包含名称

            # 读取图片
            img = cv2.imread(pic_path)
            if img is None:
                print(f"Warning: Could not read image {pic_path}. Skipping.")
                continue

            # 应用数据增强
            aug_img, aug_bboxes, augmentation = dataAug.dataAugment(img, coords)

            # 保存增强后的图片
            new_file_name = f"{custom_tag}_aug_{file}"
            cv2.imwrite(os.path.join(saved_pic_root_path, new_file_name), aug_img)

            # 生成和保存增强后的XML
            new_xml_name = os.path.splitext(new_file_name)[0] + '.xml'
            generate_new_xml(aug_bboxes, saved_xml_root_path, new_xml_name, xml_path)

    # 打包处理过的文件夹，准备下载
    timestamp = generate_timestamp()  # 获取时间戳
    augmented_dir = os.path.join(extract_folder, 'augmented')
    aug_zip_path = shutil.make_archive(os.path.join(extract_folder, f'{custom_tag}_dataset_{timestamp}'), 'zip', augmented_dir)
    return aug_zip_path  # 返回ZIP文件的路径，可用于下载链接

def generate_new_xml(bboxes, save_dir, file_name, template_xml):
    """
    生成和保存新的XML文件。
    参数:
    - bboxes: 增强后的bounding boxes，格式为[[x_min, y_min, x_max, y_max, name], ...]
    - save_dir: 保存新XML文件的目录
    - file_name: 新XML文件的名称
    - template_xml: 模板XML文件路径
    """
    # 解析模板XML文件
    tree = ET.parse(template_xml)
    root = tree.getroot()

    # 更新文件名
    for path in root.iter('filename'):
        path.text = file_name

    # 清空现有的object节点
    for obj in root.findall('object'):
        root.remove(obj)

    # 添加新的object节点
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, name = bbox
        obj = ET.SubElement(root, 'object')
        name_tag = ET.SubElement(obj, 'name')
        name_tag.text = name  # obj_name
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(x_min))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y_min))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x_max))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y_max))

    # 将更新后的XML保存到新文件
    tree.write(os.path.join(save_dir, file_name))

def unzip_datasets_and_process(zip_file_path, output_dir):
    import shutil  # 导入shutil库用于文件操作

    # 创建临时目录用于解压
    with tempfile.TemporaryDirectory() as temp_dir:
        # 解压ZIP文件
        print(f"Processing ZIP file: {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extracted {zip_file_path} to {temp_dir}")

        # 调用数据增强函数，注意确保data_augmentation函数返回增强后数据的路径
        augmented_zip_path = data_augmentation(temp_dir, 'augmented')

        # 移动增强后的ZIP文件到指定的持久存储路径
        if os.path.exists(augmented_zip_path):
            # 确保目标目录存在
            os.makedirs(output_dir, exist_ok=True)
            # 移动文件
            target_zip_path = os.path.join(output_dir, os.path.basename(augmented_zip_path))
            shutil.move(augmented_zip_path, target_zip_path)
            print(f"Moved augmented data to {target_zip_path}")
            return target_zip_path  # 返回增强数据的新路径或其他处理结果

    return None

# 定义压缩视频的函数
def compress_video(input_video_path, output_video_path, max_height=320, fps=10):
    # 加载视频
    video = VideoFileClip(input_video_path)
    
    # 计算新的宽度保持纵横比
    aspect_ratio = video.w / video.h
    new_height = max_height
    new_width = int(new_height * aspect_ratio)
    
    # 如果计算出的新宽度大于原视频宽度，使用原视频的尺寸
    if new_width > video.w:
        new_width = video.w
        new_height = video.h
    
    # 调整视频分辨率和帧率
    video = video.resize(height=new_height, width=new_width).set_fps(fps)
    
    # 保存视频
    video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# 步骤1：解压ZIP文件到临时文件夹
def unzip_datasets(base_path, temp_dir):
    zip_files = [f for f in os.listdir(base_path) if f.endswith('.zip')]
    for zip_file in zip_files:
        # 创建以ZIP文件命名的目录
        extract_path = os.path.join(temp_dir, os.path.splitext(zip_file)[0])
        os.makedirs(extract_path, exist_ok=True)
        # 解压ZIP文件到这个目录
        with zipfile.ZipFile(os.path.join(base_path, zip_file), 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {zip_file} to {extract_path}")

# 修改merge_voc_datasets函数，增加output_path参数
def merge_voc_datasets(base_path, custom_tag, output_path):
    global progress
    progress = 0
    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(categories)
    
    # 使用output_path参数来指定合并后的输出目录
    merged_xml_dir = os.path.join(output_path, 'merged', 'xml')
    merged_images_dir = os.path.join(output_path, 'merged', 'images')
    os.makedirs(merged_xml_dir, exist_ok=True)
    os.makedirs(merged_images_dir, exist_ok=True)

    # 定义一个新的排序函数
    def extract_number(filename):
        """
        从文件名中提取帧编号。
        假设文件名格式为：XXXX_XXXX_XXXX_XXXX_0000_coke.xml
        其中第五个部分（索引4）为帧编号。
        """
        parts = filename.split('_')
        if len(parts) < 5:
            return None  # 文件名格式不正确
        frame_str = parts[4]
        # 提取帧编号中的数字部分，防止有其他字符
        match = re.match(r'(\d+)', frame_str)
        if match:
            return int(match.group(1))
        else:
            return None

    def merge_xml(files):
        base_tree = ET.parse(files[0])
        base_root = base_tree.getroot()
        for f in files[1:]:
            tree = ET.parse(f)
            root = tree.getroot()
            for obj in root.findall('object'):
                base_root.append(obj)
        return base_tree

    # 在排序文件之前打印每个类别的文件名
    category_files = []
    for category in categories:
        annotation_dir = os.path.join(base_path, category, 'xml')
        if not os.path.exists(annotation_dir):
            print(f"Warning: Annotation directory {annotation_dir} does not exist. Skipping category.")
            continue
        files = os.listdir(annotation_dir)
        for file in files:
            print(f"Processing file: {file}")  # 打印文件名
        # 过滤掉无法提取帧编号的文件
        sorted_files = sorted(files, key=lambda x: extract_number(x) if extract_number(x) is not None else -1)
        category_files.append(sorted_files)

    if not category_files:
        print("No valid categories found for merging.")
        return

    # 确保所有类别的文件数相同
    if not all(len(category_files[0]) == len(files) for files in category_files):
        print("Warning: Categories have different number of files! Proceeding with minimum common files.")
        min_files = min(len(files) for files in category_files)
        category_files = [files[:min_files] for files in category_files]

    total_files = len(category_files[0])
    processed_files = 0

    for i in range(len(category_files[0])):
        files_to_merge = [os.path.join(base_path, categories[ci], 'xml', category_files[ci][i]) for ci in range(len(categories))]
        merged_tree = merge_xml(files_to_merge)
        frame_number = extract_number(category_files[0][i])
        if frame_number is None:
            print(f"Warning: Could not extract frame number from {category_files[0][i]}. Skipping file.")
            continue  # 跳过无法提取帧编号的文件
        merged_xml_filename = f'{custom_tag}_merged_{frame_number:04d}.xml'  # 使用四位数表示帧编号
        merged_xml_path = os.path.join(merged_xml_dir, merged_xml_filename)
        merged_tree.write(merged_xml_path)
        
        # 复制图片
        original_image_path = os.path.join(base_path, categories[0], 'images', category_files[0][i].replace('.xml', '.jpg'))
        if not os.path.exists(original_image_path):
            original_image_path = os.path.join(base_path, categories[0], 'images', category_files[0][i].replace('.xml', '.png'))
            if not os.path.exists(original_image_path):
                print(f"Warning: Original image for {category_files[0][i]} not found. Skipping image copy.")
                continue
        merged_image_filename = merged_xml_filename.replace('.xml', '.jpg')
        merged_image_path = os.path.join(merged_images_dir, merged_image_filename)
        shutil.copy2(original_image_path, merged_image_path)

        processed_files += 1
        progress = (processed_files / total_files) * 100
        print(f"Progress: {progress:.2f}%")

    print("Merging Complete!")

# 步骤3：将合并后的文件夹压缩为ZIP文件
def zip_merged_dataset(merged_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(merged_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算file_path相对于merged_path的相对路径
                relative_path = os.path.relpath(file_path, merged_path)
                zipf.write(file_path, relative_path)

def print_dir_tree(startpath, max_depth=1):
    for root, dirs, files in os.walk(startpath):
        # 计算当前深度
        level = root.replace(startpath, '').count(os.sep)
        # 如果当前深度超过最大深度，则跳过当前循环的剩余部分
        if level >= max_depth:
            continue
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == '__main__':
    app = App()
    app.mainloop()
