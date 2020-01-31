from detection.mtcnn_detector import MtcnnDetector
from detection.detector import Detector, PNetDetector
from network.model import p_net, r_net, o_net
import argparse
import cv2 as cv
import os 
import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)

root_dir = os.path.dirname(__file__).split('MTCNN')[0]
project_dir = os.path.dirname(__file__).split('MTCNN')[1]

def camera(test_mode, input_mode, test_dir, out_path):
    thresh = config.thresh
    min_face_size = config.min_face
    stride = config.stride
    batch_size = config.batches

    detectors = [None, None, None]
    # 模型放置位置
    model_path = ['checkpoint/pnet/pnet-30', 'checkpoint/rnet/rnet-22', 'checkpoint/onet/onet-30']
    pnet = PNetDetector(p_net, model_path[0])
    detectors[0] = pnet

    if test_mode in ["rnet", "onet"]:
        rnet = Detector(r_net, 24, batch_size[1], model_path[1])
        detectors[1] = rnet
    if test_mode == "onet":
        onet = Detector(o_net, 48, batch_size[2], model_path[2])
        detectors[2] = onet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh)

    if input_mode == '1':
        # 选用图片
        for item in os.listdir(test_dir):
            img_path = os.path.join(test_dir, item)
            img = cv.imread(img_path)
            # img = cv.imread("picture/2007_000346.jpg")
            boxes_c, landmarks = mtcnn_detector.detect(img)
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的置信度
                cv.putText(img, '{:.2f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 画关键点
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i]) // 2):
                    cv.circle(img, (int(landmarks[i][2*j]), int(int(landmarks[i][2*j+1]))), 3, (0, 0, 255), -1)
            cv.imshow('im', img)
            k = cv.waitKey(0) & 0xFF
            if k == 27:
                cv.imwrite(out_path + item, img)
        cv.destroyAllWindows()

    if input_mode == '2':
        cap = cv.VideoCapture(0)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(out_path + '/out.mp4', fourcc, 10, (640, 480))
        while True:
            t1 = cv.getTickCount()
            ret, frame = cap.read()
            if ret:
                boxes_c, landmarks = mtcnn_detector.detect(frame)
                t2 = cv.getTickCount()
                t = (t2 - t1) / cv.getTickFrequency()
                fps = 1.0/t
                for i in range(boxes_c.shape[0]):
                    bbox = boxes_c[i, :4]
                    score = boxes_c[i, 4]
                    corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    # 画人脸框
                    cv.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                    # 画置信度
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cv.putText(frame, 'score:{:.2f}  w:{:}   h:{}'.format(score, int(w), int(h)),  (corpbbox[0], corpbbox[1] - 2),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # 画fps值
                cv.putText(frame, 't:{:.4f}(s)'.format(t) + " " + 'fps:{:.3f}'.format(fps), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                # 画关键点
                for i in range(landmarks.shape[0]):
                    for j in range(len(landmarks[i])//2):
                        cv.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 3, (0, 0, 255), -1)
                a = out.write(frame)
                cv.imshow("result", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    parse = argparse.ArgumentParser('camera')
    parse.add_argument('--input_mode', type=str, default='1',
                        help='当前demo的模式：1：输入为单个图片，2：摄像头')
    parse.add_argument('--test_mode', type=str, default='onet',
                        help='MTCNN三个子网络，pnet, rnet, onet')
    parse.add_argument('--test_dir', type=str, default=root_dir + '/MTCNN/picture',
                        help='当input_mode=1时，输入图像的路径')
    parse.add_argument('--out_path', type=str, default=root_dir + '/MTCNN/output',
                        help='输出文件保存路径')
    args = parse.parse_args()
    camera(args.test_mode, args.input_mode, args.test_dir, args.out_path)
    

