import cv2
import torch
from src.face_detect import face_detector
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
face_det = face_detector(device=device)
img_path = "/workspace/huangniu_demo/ori.jpg"
out_path = "face_drawed.jpg"
img = cv2.imread(img_path)
dets = face_det.inference_single(img)
face_det.draw_results(img, dets, out_path)