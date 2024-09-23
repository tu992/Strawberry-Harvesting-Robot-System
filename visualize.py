# coding=utf-8
import cv2
import numpy as np
import onnxruntime
import torch
import torchvision
import time
import random
import sys
import io

yG=0
zG=0
count=0
productivity=0
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class YOLOV5_ONNX(object):
    def __init__(self,onnx_path):
        self.onnx_session=onnxruntime.InferenceSession(onnx_path)
        print(onnxruntime.get_device())
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
        self.classes=['unripe', 'unripe', 'ripe']

    def non_max_suppression(self,prediction,
                            conf_thres=0.1,
                            iou_thres=0.1,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.3 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                box_tensor = torch.tensor(box)   
                conf_tensor = torch.tensor(conf)

                x = torch.cat((box_tensor.clone().detach().requires_grad_(True), conf_tensor.clone().detach().requires_grad_(True), j.float()), 1)[conf.view(-1) > conf_thres]
        
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def box_area(self,box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self,box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)

    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,image_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)

        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def nms(self,prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
        print("nms")
        if prediction.dtype is torch.float16:
            prediction = prediction.float()  # to FP32
        xc = prediction[..., 4] > conf_thres  # candidates
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])

            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]

        return output

    def clip_coords(self,boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        # print("scale_coords")
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding 
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain  
        self.clip_coords(coords, img0_shape) 
        return coords

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def infer(self,input_img):
        # print("infer")
  
        img_size=(640,640)
        src_img=input_img
  
        #src_img=cv2.imread(img_path)
        start=time.time()
        src_size=src_img.shape[:2]


        img=self.letterbox(src_img,img_size,stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img=img.astype(dtype=np.float32)
        img/=255.0

        img=np.expand_dims(img,axis=0)
        # print('img resuming: ',time.time()-start)

        # start=time.time()
        input_feed=self.get_input_feed(img)
        # ort_inputs = {self.onnx_session.get_inputs()[0].name: input_feed[None].numpy()}
        pred = torch.tensor(self.onnx_session.run(None, input_feed)[0])
        results = self.non_max_suppression(pred, 0.5,0.5)
        # print('onnx resuming: ',time.time()-start)
        # pred=self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)


        img_shape=img.shape[2:]
        # print(img_size)
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = self.scale_coords(img_shape, det[:, :4],src_size).round()
        # print(time.time()-start)
        #if det is not None and len(det):
        #    self.draw(src_img, det)
        self.draw(src_img, det)

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
        global yG, zG
        # print("plot_one_box")
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        # color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, (255,0,255), thickness=tl, lineType=cv2.LINE_AA)
      
        yG = int((c1[0] + c2[0])/2)
        zG = int((c1[1] + c2[1])/2)
        cv2.circle(img, (yG, zG), 5, (255,0,0), -1)
        
        #Translate
        f = 623.3
        C_x = 640
        C_y = 360
        z = 285
        T_Cam2Robot = np.array([[0,   0,  1,  203],
                                [-1,  0,  0, -110],
                                [0,  -1,  0,  326],
                                [0,   0,  0,    1]])
        
        x_Cam = (yG - C_x)*z/f
        y_Cam = (zG - C_y)*z/f
        P_Cam = np.array([[x_Cam], [y_Cam], [z], [1]])
        P_Robot = np.dot(T_Cam2Robot, P_Cam)
        yG = int(P_Robot[1,:] - 1)
        zG = int(P_Robot[2,:] - 7)
        # print(yG, zG)
                           
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2,(255,0,255), -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            
    def draw(self,img, boxinfo):
        # print("draw")
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        for *xyxy, conf, cls in boxinfo:
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            # print('xyxy: ', xyxy)
            self.plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

        cv2.namedWindow("dst",0)
        cv2.resizeWindow("dst", 1280,720)
        cv2.imshow("dst", img)
        cv2.waitKey(50)
        cv2.imwrite("res1.jpg",img)
        # print("saved")
        # cv2.waitKey(0)
        # cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id + ".jpg"))
        return 0


from robodk import robolink
from robodk import robomath    # Robot toolbox

# Robot Target Define
pHome = [0, -60, -120, 0, 90, 0]
pPre_Place = robomath.transl(34.115,229.057,376.307) * robomath.rotx(-90*robomath.pi/180) * robomath.roty(-20*robomath.pi/180) * robomath.rotz(0)
pPlace = robomath.transl(9.905,295.573,294.601) * robomath.rotx(-127.671*robomath.pi/180) * robomath.roty(-16.071*robomath.pi/180) * robomath.rotz(-12.065*robomath.pi/180)


if __name__=="__main__":
    model=YOLOV5_ONNX(onnx_path="best.onnx")
    print("Chương trình Robot thu hoạch dâu tây - Start")
    start_time = time.time()
    
    RDK = robolink.Robolink()
    # Lấy đối tượng robot UR3
    robot = RDK.Item('UR3')
    Strawberry_Red_1 = RDK.Item('Strawberry_Red_1')
    Strawberry_Red_2 = RDK.Item('Strawberry_Red_2')
    Strawberry_Red_3 = RDK.Item('Strawberry_Red_3')
    Strawberry_Red_4 = RDK.Item('Strawberry_Red_4')
    Strawberry_Red_5 = RDK.Item('Strawberry_Red_5')
    Strawberry_Red_6 = RDK.Item('Strawberry_Red_6')
    
    # Thiết lập tốc độ và gia tốc
    # robot.setSpeed(30)  # Tốc độ (mm/s)
    # robot.setAcceleration(30)  # Gia tốc (mm/s^2)
    
    # Set camera
    CAM_NAME = 'Camera 1'
    CAM_PARAMS = 'SIZE=1280x720' # For more options, see https://robodk.com/doc/en/PythonAPI/robodk.html#robodk.robolink.Robolink.Cam2D_Add

    # Get the camera item
    cam_item = RDK.Item(CAM_NAME, robolink.ITEM_TYPE_CAMERA)
    if not cam_item.Valid():
        cam_item = RDK.Cam2D_Add(RDK.AddFrame(CAM_NAME + ' Frame'), CAM_PARAMS)
        cam_item.setName(CAM_NAME)
    cam_item.setParam('Open', 1)
    
    
    while(True):
        # Đưa robot về vị trí home
        robot.MoveJ(pHome)
        time.sleep(1)
        img_socket = None
        bytes_img = RDK.Cam2D_Snapshot('', cam_item)
        if isinstance(bytes_img, bytes) and bytes_img != b'':
            nparr = np.frombuffer(bytes_img, np.uint8)
            img_socket = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            time.sleep(1)
        if img_socket is None:
            break
        
        model.infer(input_img=img_socket)
              
        if zG != 0:
            target_pose = robomath.Mat([[0.000000,    -0.000000,     1.000000,   421.747 ],
                                            [-1.000000,    -0.000000,     0.000000,  yG ],
                                            [0.000000,    -1.000000,    -0.000000,   zG ],
                                            [0.000000,     0.000000,     0.000000,     1.000000 ]])
            # print(target_pose)
            out_pose = robomath.RelTool(target_pose, 0, 0, -100, rx=0, ry=0, rz=0)
            robot.MoveL(target_pose)

            if zG > 345:
                Strawberry_Red_1.setVisible(False)
                time.sleep(2)              
            elif 305 < zG < 315:
                Strawberry_Red_2.setVisible(False)
                time.sleep(2)
                robot.MoveL(out_pose)
                robot.MoveJ(pPre_Place)
                robot.MoveJ(pPlace)
                time.sleep(2)
                robot.MoveJ(pHome)
                count+=1
                Strawberry_Red_1.setVisible(True)
                Strawberry_Red_2.setVisible(True)
                Strawberry_Red_3.setVisible(True)
                Strawberry_Red_4.setVisible(True)
                Strawberry_Red_5.setVisible(True)
                Strawberry_Red_6.setVisible(True)
                break
            elif 315 < zG < 340:
                Strawberry_Red_4.setVisible(False)
                time.sleep(2)
                
            elif 295 < zG < 305:
                Strawberry_Red_5.setVisible(False)
                time.sleep(2)
            elif 285 < zG < 295:
                Strawberry_Red_6.setVisible(False)
                time.sleep(2)
            else:
                Strawberry_Red_3.setVisible(False)
                time.sleep(2)
            robot.MoveL(out_pose)
            robot.MoveJ(pPre_Place)
            robot.MoveJ(pPlace)
            time.sleep(2)
            count+=1
        # else :break
    end_time = time.time()
    elapsed_time = int(end_time - start_time)
    productivity = int(3600*count/elapsed_time)
    print(f"- Thời gian thu hoạch {elapsed_time} giây")
    print(f"- Sản lượng: {count} quả")
    print(f"- Ước lượng năng suất: {productivity} quả/giờ")
    print("finish")

