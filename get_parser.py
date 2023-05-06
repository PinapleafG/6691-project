import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='data', help='model name')

    parser.add_argument('--train_img_path', type=str,
                        default='data', help='train img path')
    parser.add_argument('--train_mask_path', type=str,
                        default='data', help='train mask path')
    parser.add_argument('--valid_img_path', type=str,
                    default='data', help='train img path')
    parser.add_argument('--valid_mask_path', type=str,
                        default='data', help='train mask path')
    parser.add_argument('--description_path', type=str,
                        default='data', help='description path')
    
    parser.add_argument('--img_size', type=int,
                        default=224, help='input image size')
    parser.add_argument('--num_classes', type=int,
                        default=20, help='output channel of network')
    parser.add_argument('--embed_dim', type=int,
                        default=96, help='embed dim of transformer')
    parser.add_argument('--epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--depths_encoder', 
                        default=[2,2,2,2], type=int, nargs="+", metavar='N', 
                        help='depths of encoder')
    parser.add_argument('--depths_decoder',
                        default=[1,2,2,2], type=int, nargs="+", metavar='N',
                        help='depths of decoder')
    parser.add_argument('--num_heads',
                        default=[3,6,12,24], type=int, nargs="+", metavar='N',
                        help='num_heads of transformer')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--contrast', action='store_true', 
                        help='use contrastive learning')

    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    return parser