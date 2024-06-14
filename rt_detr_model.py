from ultralytics import RTDETR

class RtDetr():
    def __init__(self, model_path, conf=0.7):
        self.conf = conf
        self.model = RTDETR(model_path)

    def predict(self, img):
        res = list(self.model.predict(img, stream=False, conf = self.conf, verbose=False))[0]
        return [{'label' : res.names[int(res.boxes.cls.tolist()[i])], \
                'conf' : round(float(res.boxes.conf[i]), 2), \
                'xywh': [float(x) for x in list(res.boxes.xywh[i])]} \
                        for i in range(len(res.boxes.cls.tolist()))]

if __name__ == '__main__':
    import cv2
    model = RtDetr('/home/duhanin/Документы/work/databriz/veb/train_detr/rtdetr-l.pt')
    img = cv2.imread("/home/duhanin/Изображения/Snapshot_2024-04-04_20-58-36.png")
    #model.predict(img)
    print(model.predict(img))

        