import torch
import network
import numpy as np
from datasets import Cityscapes
from utils import ext_transforms as et
from PIL import Image
from metrics import StreamSegMetrics
import cv2
import os

#---------------------------------------------------#
#   单张图片预测
#---------------------------------------------------#

classes = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
           'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle',
           'bicycle']

def demo(img_path,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_transform = et.ExtCompose([
            et.ExtResize(512),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    label_path = img_path
    image = Image.open(img_path).convert('RGB')
    draw = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    draw = cv2.resize(draw,(1024,512))
    label = Image.open(label_path).convert('L')
    image,target = val_transform(image,label)
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float32)


    metrics = StreamSegMetrics(19)
    model.eval()
    with torch.no_grad():
        metrics.reset()
        outputs = model(image)
        pred = outputs.detach().max(dim=1)[1].cpu().numpy()
        pred = np.squeeze(pred)

        for i in range(19):
            pred_ = pred.copy()
            pred_[pred_==i] = 255
            pred_[pred_!=255] = 0
            pred_ = pred_.astype(np.uint8)

            contours,_ = cv2.findContours(pred_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pred_ = cv2.cvtColor(pred_, cv2.COLOR_GRAY2BGR)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w*h < 1000:continue
                # cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                x_,y_ = int(x+w/2),int(y+h/2)
                imgzi = cv2.putText(draw, classes[i], (x_,y_), font, 0.6, (0,255,0), 2)


        pred = Cityscapes.decode_target(pred).astype(np.uint8)
        # return pred

        new = cv2.addWeighted(imgzi,1.,pred,0.7,0)
        #new = cv2.addWeighted(imgzi, 0.5, pred, 1., 0)
        return pred
        # print(pred.shape)
        # img = Image.fromarray(pred)
        # img.save('pred1.png')
        # img.show()
        



if __name__ == '__main__':
    net = network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16).cuda()
    checkpoint = torch.load('weights/best_deeplabv3_mobilenet_cityscapes_os16.pth')
    net.load_state_dict(checkpoint["model_state"])

    imgs_path = r'test_imgs'
    saves_path = r'out1'

    for img_name in os.listdir(imgs_path):
        print(img_name)
        img_path = os.path.join(imgs_path,img_name)
        out = demo(img_path,net)
        save_path = os.path.join(saves_path,img_name)
        cv2.imwrite(save_path,out)


