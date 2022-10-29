import argparse

def parse_args():

    # Model loading/saving
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='./saved_model/',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1024,
                        help='random seed')
    parser.add_argument('--lr', default=0.001, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='epoch')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')


    # Choices of mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                        help='mode')


    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    # parser.add_argument('--RAD_dir', type=str, default='data/open/CLEF/',
    #                     help='RAD dir')
    parser.add_argument('--RAD_dir', type=str, default='data_pathVQA/',
                        help='RAD dir')
    parser.add_argument('--maml', type=bool, default=False,
                        )
    parser.add_argument('--autoencoder', type=bool, default=False,
                        )
    parser.add_argument('--use_RAD', type=bool, default=True,
                        help='')

    # Network Setting
    parser.add_argument('--lstm_out_size', type=int, default='768',
                        help='lstm')
    parser.add_argument('--word_embedding_size', type=int, default='300',
                        help='word_embedding_dim')
    parser.add_argument('--drop_rate', type=float, default='0.1',
                        help='drop_rate')
    parser.add_argument('--MFB_O', type=int, default='1000',
                        help='MFB_O')
    parser.add_argument('--MFB_K', type=int, default='3',
                        help='MFB_K')
    parser.add_argument('--q_glimse', type=int, default='2',
                        help='glimse_q')
    parser.add_argument('--i_glimse', type=int, default='2',
                        help='glimse_i')
    parser.add_argument('--hidden_size', type=int, default='1024',
                        help='if mima 768 else 512')
    parser.add_argument('--HIGH_ORDER', type=bool, default=False,
                        help='high_order')
    parser.add_argument('--activation', type=str, default='relu',
                        help='activation')
    parser.add_argument('--v_dim', type=int, default=128,
                        help='dim of embedding image')
    parser.add_argument('--eps_cnn', type=float, default=1e-5,
                        help='pass')
    parser.add_argument('--momentum_cnn', type=float, default=0.05,
                        help='pass')
    parser.add_argument('--num_stacks', type=int, default=2,
                        help='')



    parser.add_argument('--model', type=str, default='RCAN')
    parser.add_argument('--save', type=str, default='RCAN_BIX4_G10R20P48')
    parser.add_argument('--scale',type=str, default='4+6+8')
    parser.add_argument('--n_resgroups', type=int, default=10)
    parser.add_argument('--n_resblocks', type=int, default=20)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=192,
                        help='output patch size')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--n_colors', type=int, default=1,
                        help='number of color channels to use')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')

    # BAN params
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=128)
    parser.add_argument('--GLIMPSE', type=int, default=4)
    parser.add_argument('--HIDDEN_SIZE', type=int, default=1024)
    parser.add_argument('--K_TIMES', type=int, default=3)
    parser.add_argument('--BA_HIDDEN_SIZE', type=int, default=3072) # HIDDEN_SIZE x K_TIMES
    parser.add_argument('--DROPOUT_R', type=float, default=0.1)
    parser.add_argument('--CLASSIFER_DROPOUT_R', type=float, default=0.1)
    parser.add_argument('--FLAT_OUT_SIZE', type=int, default=1024)

    parser.add_argument('--img_root', type=str, default='F:/PathVQA_official/split/images/')

    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))



    return args