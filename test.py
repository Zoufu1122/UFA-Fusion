import torch
from data_loader import DataTest
import argparse
from model.addition.model_csa import AF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


def test(test_dataloader,args):
    device=args.device
    net = AF(args).to(device)
    net.load_state_dict(torch.load(args.saveModel_dir,map_location='cpu'))
    net.eval()
    for i_test, (image1,image2) in enumerate(test_dataloader):
        image1=image1.to(device)
        image2=image2.to(device)
        out,F1_ori,F2_ori,F1_a,F2_a=net(image1,image2)
        out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))
        out_image.save("result/IV_CSA/"+str(i_test+1).zfill(2) + ".png")
        #base_path="result/"+"lytro_"+str(i_test+1).zfill(2)
        # for i in range(64):
        #     temp_F1=torch.squeeze(F1_ori.data.cpu(), 0)
        #     temp_F1 = transforms.ToPILImage()(temp_F1[i])
        #     temp_F1.save(base_path+"/F1/" + str(i+1) + ".png")
        #     temp_F1_h=cv2.imread(base_path+"/F1/" + str(i+1) + ".png")
        #     temp_F1_h= cv2.applyColorMap(temp_F1_h, cv2.COLORMAP_JET)
        #     cv2.imwrite(base_path+"/F1/" + str(i+1) + ".png",temp_F1_h)
        #
        #     temp_F2=torch.squeeze(F2_ori.data.cpu(), 0)
        #     temp_F2 = transforms.ToPILImage()(temp_F2[i])
        #     temp_F2.save(base_path+"/F2/" + str(i+1) + ".png")
        #     temp_F2_h=cv2.imread(base_path+"/F2/" + str(i+1) + ".png")
        #     temp_F2_h= cv2.applyColorMap(temp_F1_h, cv2.COLORMAP_JET)
        #     cv2.imwrite(base_path+"/F2/" + str(i+1) + ".png",temp_F2_h)
        #
        #     temp_fusion = torch.squeeze(F1_a.data.cpu(), 0)
        #     temp_fusion = transforms.ToPILImage()(temp_fusion[i])
        #     temp_fusion.save(base_path + "/F1_a/" + str(i+1) + ".png")
        #
        #     temp_F1a_h=cv2.imread(base_path+"/F1_a/" + str(i+1) + ".png")
        #     temp_F1a_h= cv2.applyColorMap(temp_F1a_h, cv2.COLORMAP_JET)
        #     cv2.imwrite(base_path+"/F1_a/" + str(i+1) + ".png",temp_F1a_h)
        #
        #     temp_attfusion = torch.squeeze(F2_a.data.cpu(), 0)
        #     temp_attfusion = transforms.ToPILImage()(temp_attfusion[i])
        #     temp_attfusion.save(base_path+ "/F2_a/" + str(i+1) + ".png")
        #     temp_F2a_h = cv2.imread(base_path + "/F2_a/" + str(i + 1) + ".png")
        #     temp_F2a_h = cv2.applyColorMap(temp_F2a_h, cv2.COLORMAP_JET)
        #     cv2.imwrite(base_path + "/F2_a/" + str(i + 1) + ".png", temp_F2a_h)
        #


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testData_dir', type=str, default="DataSet/IV_RGB")
    parser.add_argument('--saveModel_dir', type=str, default='experment_addition/csa/best.pth')
    parser.add_argument('--result', type=str, default='result')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir,transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)

