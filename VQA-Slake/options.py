import argparse

def parse_args():

    # Model loading/saving
    parser = argparse.ArgumentParser()

    # common params
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--output', type=str, default='saved_model')
    parser.add_argument('--RAD_dir', type=str, default='data_Slake/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--question_len', default=12, type=int, metavar='N')

    # XXX params
    parser.add_argument('--n_ctx', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--train_csc', type=bool, default=False)
    parser.add_argument('--class_token_position', type=str, default='end')
    parser.add_argument('--maml', type=bool, default=False)
    parser.add_argument('--autoencoder', type=bool, default=False)

    # MFB params
    parser.add_argument('--img_feat_size', type=int, default=512)
    parser.add_argument('--i_glimse', type=int, default=2)
    parser.add_argument('--q_glimse', type=int, default=2)
    parser.add_argument('--ques_feat_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--MFB_O', type=int, default=1000)
    parser.add_argument('--MFB_K', type=int, default=1)
    parser.add_argument('--drop_rate', type=float, default=0.1)


    # BAN params
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=128)
    parser.add_argument('--GLIMPSE', type=int, default=4)
    parser.add_argument('--HIDDEN_SIZE', type=int, default=1024)
    parser.add_argument('--K_TIMES', type=int, default=3)
    parser.add_argument('--BA_HIDDEN_SIZE', type=int, default=3072) # HIDDEN_SIZE x K_TIMES
    parser.add_argument('--DROPOUT_R', type=float, default=0.1)
    parser.add_argument('--CLASSIFER_DROPOUT_R', type=float, default=0.1)
    parser.add_argument('--FLAT_OUT_SIZE', type=int, default=1024)

    parser.add_argument('--mlm_prob', type=float, default=0.15)
    parser.add_argument('--max_position_embeddings', type=int, default=77)





    args = parser.parse_args()

    return args