import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import torch.nn.functional as F
import gdown
import os

# Кастомные стили (дизайн)
st.markdown("""
    <style>
    .title { font-size: 32px; color: #2E86C1; text-align: center; }
    .result { font-size: 24px; font-weight: bold; color: #E74C3C; }
    .instructions { font-size: 16px; color: #34495E; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Распознавание рака кожи</div>', unsafe_allow_html=True)
st.markdown('<div class="instructions">Загрузите фото кожи для анализа. Поддерживаются форматы JPG и PNG.</div>', unsafe_allow_html=True)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Класс ResNetFeatMask
class ResNetFeatMask(nn.Module):
    def __init__(self, num_classes=2, pretrained_backbone=True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained_backbone)
        self.features = nn.Sequential(*(list(backbone.children())[:-2]))
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, seg_mask=None):
        feat = self.features(x)
        if seg_mask is not None:
            mask_small = F.interpolate(seg_mask, size=feat.shape[2:], mode='nearest')
            feat = feat * mask_small
        out = self.avgpool(feat)
        out = torch.flatten(out, 1)
        return self.fc(out)

# Функции
def detection(image, yolo_model):
    results = yolo_model(image, imgsz=640, iou=0.8, conf=0.4, verbose=False)
    if len(results[0].boxes) == 0:
        return None, image
    box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    return box, image

def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    return cropped

def dullrazor(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayScale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), 0, borderType=cv2.BORDER_DEFAULT)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
    return dst

def segmentation(cropped_image, seg_model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(input_tensor)['out']
        output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        preds = torch.sigmoid(output) > 0.5

    mask = preds.squeeze().cpu().numpy().astype(np.uint8)
    mask_resized = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
    return mask_resized

# Загрузка моделей
try:
    # YOLO
    if not os.path.exists('best.pt'):
        st.write("Загрузка модели YOLO...")
        gdown.download('https://drive.google.com/file/d/1UjMSJR6f-PrfToiM5jJ_fSawLZoD33oX/view?usp=sharing', 'best.pt', quiet=False)
    yolo_model = YOLO('best.pt')

    # DeepLabV3
    if not os.path.exists('best_model_deeplabv3_26.04.25.pth'):
        st.write("Загрузка модели DeepLabV3...")
        gdown.download('https://drive.google.com/uc?id=148g9Qeax_j2mfKxnALZTTF_umBJL586U', 'best_model_deeplabv3_26.04.25.pth', quiet=False)
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    seg_model.load_state_dict(torch.load('best_model_deeplabv3_26.04.25.pth', map_location=device))
    seg_model.to(device)

    # ResNet
    if not os.path.exists('best_f1_resnet_3.05.25.pth'):
        st.write("Загрузка модели ResNet...")
        gdown.download('https://drive.google.com/file/d/1IaehbxC40PF4UgxJtRDMc0EEhMZXYygf/view?usp=sharing', 'best_f1_resnet_3.05.25.pth', quiet=False)
    class_model = ResNetFeatMask(num_classes=2, pretrained_backbone=True).to(device)
    class_model.load_state_dict(torch.load('best_f1_resnet_3.05.25.pth', map_location=device))
    class_model.to(device)
except Exception as e:
    st.error(f"Ошибка загрузки моделей: {e}")
    st.stop()

# Трансформации для классификации
class_image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class_mask_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t > 0.5).float())
])

# Обработка загруженного файла
uploaded_file = st.file_uploader("Выберите фото", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        # Чтение изображения
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        if image_np.shape[2] == 4:  # Если есть альфа-канал
            image_np = image_np[:, :, :3]
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        st.image(image, caption="Загруженное фото", use_column_width=True)

        # Детекция
        box, _ = detection(image_np, yolo_model)
        if box is None:
            st.warning("Родинка не обнаружена. Пожалуйста, загрузите другое фото.")
            st.stop()

        # Обрезка
        cropped = crop_image(image_np, box)
        st.image(cropped, caption="Обрезанное изображение", use_column_width=True)

        # Удаление волос
        hair_removed = dullrazor(cropped)
        hair_removed_rgb = cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)
        st.image(hair_removed_rgb, caption="Изображение без волос", use_column_width=True)

        # Сегментация
        mask = segmentation(hair_removed, seg_model, device)
        st.image(mask * 255, caption="Маска родинки", use_column_width=True)

        # Классификация
        pil_img = Image.fromarray(hair_removed_rgb)
        img_tensor = class_image_transform(pil_img).unsqueeze(0).to(device)
        mask_uint8 = (mask * 255).astype('uint8')
        pil_mask = Image.fromarray(mask_uint8, mode='L')
        mask_tensor = class_mask_transform(pil_mask).unsqueeze(0).to(device)

        class_model.eval()
        with torch.no_grad():
            outputs = class_model(img_tensor, seg_mask=mask_tensor)
            probs = torch.softmax(outputs, dim=1)
            prob_malign = probs[0, 1].item()
            prob_benign = probs[0, 0].item()

        threshold = 0.3
        pred = 1 if prob_malign >= threshold else 0
        label_map = {0: 'Доброкачественное', 1: 'Злокачественное'}
        result = label_map[pred]

        # Вывод результата
        st.markdown(f'<div class="result">Результат: {result}</div>', unsafe_allow_html=True)
        st.write(f"Вероятность злокачественного новообразования: {prob_malign:.2%}")
        st.write(f"Вероятность доброкачественного новообразования: {prob_benign:.2%}")

        # График вероятностей
        st.bar_chart({"Доброкачественное": prob_benign, "Злокачественное": prob_malign})

    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")
