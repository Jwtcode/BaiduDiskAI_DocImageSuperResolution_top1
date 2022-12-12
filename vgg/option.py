import argparse


parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume from specific checkpoint')
# Data specifications
parser.add_argument('--dir_data', type=str, default='./',
                    help='dataset directory')

args = parser.parse_args()


