from tqdm import tqdm
import network
from torch.utils import data
from datasets import  Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch


#---------------------------------------------------#
#   评估网络模型
#---------------------------------------------------#

def validate(model, loader, device, metrics):
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16).cuda()

    checkpoint = torch.load('weights/my_net.pth')
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    data_root = r'/disk_d/workspace/personalSpace/like_project/1500_deeplabv3+/cityscapes'
    val_dst = Cityscapes(root=data_root,split='val', transform=val_transform)
    val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=True, num_workers=2)

    metrics = StreamSegMetrics(19)
    val_score = validate(
        model=model, loader=val_loader, device=device, metrics=metrics)
    print(metrics.to_str(val_score))