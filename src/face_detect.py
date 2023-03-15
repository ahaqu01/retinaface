from __future__ import print_function
import os
import cv2
from PIL import Image
import numpy as np
from skimage import transform

import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F

from .configs.config import cfg_mnet, cfg_re50
from .models.layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.nms.py_cpu_nms import py_cpu_nms
from .utils.box_utils import decode, decode_landm
from .utils.utils import TensorrtBase


# from configs.config import cfg_mnet, cfg_re50
# from models.layers.functions.prior_box import PriorBox
# from models.retinaface import RetinaFace
# from utils.nms.py_cpu_nms import py_cpu_nms
# from utils.box_utils import decode, decode_landm

class face_detector(object):
    def __init__(self,
                 backbone_name="mobile0.25",
                 model_weights="/workspace/huangniu_demo/retinaface/src/weights/mobilenet0.25_Final.pth",
                 keep_size=False,
                 confidence_threshold=0.3,
                 top_k=5000,
                 nms_threshold=0.4,
                 keep_top_k=100,
                 vis_thres=0.7,
                 device=None,
                 speed_up=False,
                 speed_up_weights="",
                 rebuild_engine=False
                 ):
        if backbone_name == "mobile0.25":
            cfg = cfg_mnet
        elif backbone_name == "resnet50":
            cfg = cfg_re50
        else:
            cfg = None
        self.cfg = cfg
        self.longer_size = self.cfg["image_size"]
        self.keep_size = keep_size
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.speed_up = speed_up
        self.speed_up_weights = speed_up_weights
        self.rebuild_engine = rebuild_engine

        # create retinaface and load weights
        print("create retinaface and load weights")
        self.model = RetinaFace(cfg=cfg, phase='test')
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == 'cuda':
            load_to_cpu = False
        else:
            load_to_cpu = True
        self.model = self.load_model(self.model, model_weights, load_to_cpu)
        print("created retinaface and loaded weights done!")

        # for face align
        self.trans = transform.SimilarityTransform()
        self.REFERENCE_FACIAL_POINTS = [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 87],
            [62.72990036, 87]
        ]
        self.DEFAULT_CROP_SIZE = (96, 112)
        self.ref_pts = self._get_reference_facial_points()

        priorbox = PriorBox(self.cfg, image_size=(self.longer_size, self.longer_size))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

        # speed up by tensorrt
        if self.speed_up:
            os.makedirs(self.speed_up_weights, exist_ok=True)
            onnx_filename = os.path.join(self.speed_up_weights, model_weights.split("/")[-1].replace(".pth", ".onnx"))
            trt_filename = os.path.join(self.speed_up_weights, model_weights.split("/")[-1].replace(".pth", ".engine"))
            self.face_det_trt = TensorrtBase(
                model=self.model,
                speed_up_weights_root=self.speed_up_weights,
                onnx_filename=onnx_filename,
                trt_filename=trt_filename,
                gpu_id=str(self.device.index),
                longer_size=self.longer_size,
                rebuild_engine=self.rebuild_engine,
            )

    def _get_reference_facial_points(self, output_size=(112, 112)):
        tmp_5pts = np.array(self.REFERENCE_FACIAL_POINTS)
        tmp_crop_size = np.array(self.DEFAULT_CROP_SIZE)
        x_scale = output_size[0] / tmp_crop_size[0]
        y_scale = output_size[1] / tmp_crop_size[1]
        tmp_5pts[:, 0] *= x_scale
        tmp_5pts[:, 1] *= y_scale
        return tmp_5pts

    def get_align_processed_faces(self, img, landmarks, output_size=(160, 160)):
        # img: BGR, (H, W, 3), ndarray
        aligned_faces = []
        for src_pts in landmarks:
            self.trans.estimate(src_pts.reshape((5, 2)), self.ref_pts)
            face_img = cv2.warpAffine(img, self.trans.params[0:2, :], output_size)
            face_img = F.to_tensor(np.float32(face_img[:, :, ::-1]))
            face_img = self.fixed_image_standardization(face_img)
            aligned_faces.append(face_img)
        aligned_faces = torch.stack(face_img)
        return aligned_faces

    def get_size(self, img):
        if isinstance(img, (np.ndarray, torch.Tensor)):
            return img.shape[1::-1]
        else:
            return img.size

    def imresample(self, img, sz):
        im_data = interpolate(img, size=sz, mode="area")
        return im_data

    def crop_resize(self, img, box, image_size):
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = cv2.resize(
                img,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        elif isinstance(img, torch.Tensor):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = self.imresample(img.permute(2, 0, 1).unsqueeze(0).float(), (image_size, image_size)).byte().squeeze(
                0).permute(1, 2, 0)
        else:
            out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
        return out

    def extract_face(self, img, box, image_size=160, margin=0):
        """Extract face + margin from PIL Image given bounding box.

        Arguments:
            img {PIL.Image} -- A PIL Image.
            box {numpy.ndarray} -- Four-element bounding box.
            image_size {int} -- Output image size in pixels. The image will be square.
            margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
                Note that the application of the margin differs slightly from the davidsandberg/facenet
                repo, which applies the margin to the original image before resizing, making the margin
                dependent on the original image size.
            save_path {str} -- Save path for extracted face image. (default: {None})

        Returns:
            torch.tensor -- tensor representing the extracted face.
        """

        margin = [
            margin * (box[2] - box[0]) / (image_size - margin),
            margin * (box[3] - box[1]) / (image_size - margin),
        ]
        raw_image_size = self.get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]  # margin==0，无外扩
        face = self.crop_resize(img, box, image_size)
        face = F.to_tensor(np.float32(face))
        return face

    def fixed_image_standardization(self, image_tensor):
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor

    def get_crop_processed_faces(self, img, boxes, image_size=160, margin=0):
        # img: shape=(H,W,3), BGR, ndarray
        # boxes: shape=(N, 4), ndarray
        im = Image.fromarray(img[:, :, ::-1])  # RGB {PIL.Image}
        faces = []
        for i, box in enumerate(boxes):
            face = self.extract_face(im, box, image_size, margin)
            face = self.fixed_image_standardization(face)
            faces.append(face)
        faces = torch.stack(faces)
        return faces

    def pad(self, image):
        height, width, _ = image.shape
        long_side = max(width, height)
        image_padded = np.empty((long_side, long_side, 3), dtype=image.dtype)
        image_padded[:height, :width, :] = image
        return image_padded

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = self.device
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    @torch.no_grad()
    def inference_single(self, img_raw):
        # img_raw, shape:(H,W,3), BGR
        # get resize scale
        img = img_raw.copy()
        height_raw, width_raw = img.shape[:2]
        img = self.pad(img)
        if self.keep_size:
            resize = 1
        else:
            size_max = max(img.shape[:2])
            resize = float(self.longer_size) / float(size_max)

        # resize images
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        img = np.float32(img)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = scale.to(self.device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)

        if not self.speed_up:
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            loc, conf, landms = self.model(img)  # forward pass
            boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()
        else:
            img = np.expand_dims(np.ascontiguousarray(img), 0)
            loc, conf, landms = self.face_det_trt.do_inference(img)

            loc = torch.from_numpy(loc).to(self.device)
            boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()

            scores = conf.squeeze(0)[:, 1]

            landms = torch.from_numpy(landms).to(self.device)
            landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        # ignore low scores
        dets = np.concatenate((dets, landms), axis=1)
        inds_vis = np.where(dets[:, 4] > self.vis_thres)[0]
        dets = dets[inds_vis]

        # Boundary treatment
        dets[:, 2][dets[:, 2] > width_raw] = width_raw
        dets[:, 3][dets[:, 3] > height_raw] = height_raw
        return dets

    def del_irregular_bboxes(self, retinaface_pred, raw_img):
        retinaface_pred_cp = retinaface_pred.copy()
        h, w = raw_img.shape[:2]
        retinaface_pred_cp[:, 0][retinaface_pred_cp[:, 0] < 0] = 0
        retinaface_pred_cp[:, 1][retinaface_pred_cp[:, 1] < 0] = 0
        retinaface_pred_cp[:, 2][retinaface_pred_cp[:, 2] > w] = w
        retinaface_pred_cp[:, 3][retinaface_pred_cp[:, 3] > h] = h
        face_wh = (retinaface_pred_cp[:, 2:4] - retinaface_pred_cp[:, :2]).astype(np.int32)
        face_area = face_wh[:, 0] * face_wh[:, 1]
        bboxes = retinaface_pred_cp[:, :4][face_area > 0].astype(np.int32)
        return bboxes, retinaface_pred[face_area > 0]

    def draw_results(self, img_raw, dets, out_path=""):
        img = img_raw.copy()
        super_folder = os.path.abspath(os.path.dirname(out_path))
        if not os.path.exists(super_folder):
            os.makedirs(super_folder)
        for b in dets:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image
        cv2.imwrite(out_path, img)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    face_det = face_detector(device=device)
    img_path = "/workspace/huangniu_demo/ori.jpg"
    out_path = "face_drawed.jpg"
    img = cv2.imread(img_path)
    dets = face_det.inference_single(img)
    face_det.draw_results(img, dets, out_path)
    # 可以利用人体头部pose关键点, 来确认facedet结果对应的person
