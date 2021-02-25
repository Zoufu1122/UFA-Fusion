import torch
from data_loader import DataTest
import argparse
from model.UFA.model import AF
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np

def test(test_dataloader,args):
    device=args.device
    net = AF(args).to(device)
    net.load_state_dict(torch.load(args.saveModel_dir))
    net.eval()
    print(net)
    t1=time.time()
    for i_test, (image1,image2) in enumerate(test_dataloader):
        image1=image1.to(device)
        image2=image2.to(device)
        out,F1_ori,F2_ori,F1_a,F2_a=net(image1,image2)
        out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))
        out_image.save("result/UFA/"+"color_lytro_"+str(i_test+1).zfill(2) + ".png")
        print("color_lytro_"+str(i_test+1).zfill(2) + ".png")
    t2=time.time()
    print(t2-t1)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testData_dir', type=str, default="DataSet/testData/lytro")
    parser.add_argument('--saveModel_dir', type=str, default='experment/best.pth')
    parser.add_argument('--result', type=str, default='result')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir,transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)



