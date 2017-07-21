from core.solver_e2e import CaptioningSolver
from core.model import CaptionGenerator
#from core.rl_solve_debug import CaptioningSolver
#from core.rl_model import CaptionGenerator
from core.dataset import *
#from core.utils import load_coco_data
import os,pickle
import mxnet as mx

def main():
    # load train dataset
    '''
    data = load_coco_data(data_path='./data', split='train')
    with open(os.path.join("data/val", 'word_to_idx.pkl'), 'rb') as f:
            word_to_idx = pickle.load(f)
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')
    #data = None
    #val_data = None
    #data = val_data
    '''
    max_len = 15
    batch_size = 24
    with open(os.path.join("data", 'word_to_idx.pkl'), 'rb') as f:
            word_to_idx = pickle.load(f)

    data = DataSet(data_path="./image/", json_path="./data/singlecap_train.json", batchsize=batch_size, word_to_idx=word_to_idx, max_total=max_len+2)
    val_data = DataSet(data_path="./image/", json_path="./data/singlecap_val.json", batchsize=batch_size, word_to_idx=word_to_idx, max_total=max_len+2)

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=256, batch_size=batch_size,
                                       dim_hidden=256, n_time_step=max_len+1, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, ctx=mx.gpu(0))

    #data_mp = MpDataSet(10, data)
    #val_data_mp = MpDataSet(5, val_data)
    solver = CaptioningSolver(model, data, val_data, n_epochs=50000, batch_size=batch_size, update_rule='adam',
                                          learning_rate=0.001, print_every=50, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='./model/', test_model='model/lstm-19',
                                     print_bleu=True, log_path='./log/')

    solver.train(017)

if __name__ == "__main__":
    main()
