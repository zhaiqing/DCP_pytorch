"""Evaluate RPMNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Visualize RPMNet
        python vis.py --noise_type crop --resume [path-to-model.pth] --dataset_path [your_path]/modelnet40_ply_hdf5_2048
        python vis.py --noise_type crop --resume ./pretrained_models/partial-trained.pth

"""
import argparse
import os
import open3d as o3d
import random
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from data import ModelNet40
from model import DCP
from util import transform_point_cloud



def vis(npys):
    pcds = []
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    for ind, npy in enumerate(npys):
        color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
        pcd = o3d.geometry.PointCloud()
        # npy = npy.reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(npy)
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
    return pcds


def inference_vis(data_loader, model: torch.nn.Module):
    # _logger.info('Starting inference...')
    model.eval()

    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(
                data_loader):
            src = src.cuda()
            target = target.cuda()
            rotation_ab = rotation_ab.cuda()
            translation_ab = translation_ab.cuda()
            rotation_ba = rotation_ba.cuda()
            translation_ba = translation_ba.cuda()
            rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = model(src, target)
            src_transformed = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

            src_np = torch.squeeze(src).permute(1, 0).cpu().detach()
            src_transformed_np = torch.squeeze(src_transformed).permute(1, 0).cpu().detach()
            ref_np = torch.squeeze(target).permute(1, 0).cpu().detach()
            # print(src_np.size())
            # print(src_transformed_np.size())
            # print(ref_np.size())
            pcds = vis([src_np, src_transformed_np, ref_np])
            # print(pcds)
            o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    # Arguments and logging
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'modelnet40':
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'dcp':
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
            net.load_state_dict(torch.load(model_path), strict=False)
    inference_vis(test_loader, net)  # Feedforward transforms
