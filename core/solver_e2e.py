import mxnet as mx
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os,json
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate
from vggnet import *

def _promulticaps(caps):
    data = {}
    for v in caps:
        data[v["image_path"]] = v["caption"]
    return data
class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        self.train_caption_gts = _promulticaps(json.load(open('data/multicaps_train.json')))
        self.val_caption_gts = _promulticaps(json.load(open('data/multicaps_val.json')))
        # set an optimizer by update rule
        if self.update_rule == 'sgd':
            self.optimizer = mx.optimizer.SGD
        if self.update_rule == 'adam':
            self.optimizer = mx.optimizer.Adam
        elif self.update_rule == 'momentum':
            self.optimizer = mx.optimizer.NAG
        elif self.update_rule == 'rmsprop':
            self.optimizer = mx.optimizer.RMSProp

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self, ckptnum):
        # train/val dataset
        
        n_examples = self.data.totalnum
        n_iters_per_epoch = int(np.floor(float(n_examples) / self.batch_size))
        n_iters_val = int(np.floor(self.val_data.totalnum) / self.batch_size)
        
        # build graphs for training model and sampling captions
        self.model.build_input_e2e()
        #self.model.load("model/lstmwithoutcnn", ckptnum)
        loss, exe, input_names = self.model.build_model()
        _, _, generated_captions, gen_exe = self.model.build_sampler(max_len=20)
        #return
        #self.vgg19 = vggnet19(self.model.ctx, self.batch_size)
        #self.vgg19.load("model/vgg/vgg19",0)
        
        # train op
        self.opt = self.optimizer(sym=loss, learning_rate=self.learning_rate)
        args = exe.arg_arrays
        grads = exe.grad_arrays
        states = []
        for i,argn in enumerate(loss.list_arguments()):
            if argn in input_names:
                states.append(None)
            else:
                states.append(self.opt.create_state(i,args[i]))

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        if self.pretrained_model is not None:
            print "Start training with pretrained Model.."
            saver.restore(sess, self.pretrained_model)

        prev_loss = -1
        curr_loss = 0
        start_t = time.time()

        for e in range(self.n_epochs):
            self.data.reset()
            for i in range(n_iters_per_epoch):
                captions_batch, images_batch, image_paths = self.data.next_batch_for_all()
                #features_batch = self.vgg19.getfeatures(images_batch)
                #mx.nd.array(features_batch).copyto(self.model.features_arr)
                mx.nd.array(images_batch).copyto(self.model.image_arr)
                mx.nd.array(captions_batch).copyto(self.model.captions_arr)
                l = exe.forward(is_train=True)
                lloss = l[0].asnumpy().mean()
                if lloss == np.nan or lloss == -np.inf or lloss == np.inf:
                    print lloss
                    continue
                curr_loss += lloss
                '''
                print "loss:",lloss
                print np.sum(np.not_equal(captions_batch[1:,:], self.model._null))/self.batch_size
                print "null:",self.model._null
                continue
                '''
                exe.backward()
                for j,argn in enumerate(loss.list_arguments()):
                    if argn in input_names:
                        continue
                    #if argn in self.model.cnn_params:
                    #    continue
                    #print args[i], grads[i], states[i]
                    self.opt.update(j, args[j], grads[j], states[j])
                    #print "updated weights:",argn
                if (i + 1) % self.print_every == 0:
                    print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, lloss)
                    
                    ground_truths = self.train_caption_gts[image_paths[0]]
                    decoded = ground_truths #decode_captions(ground_truths, self.model.idx_to_word)
                    for j, gt in enumerate(decoded):
                        print "Ground truth %d: %s" % (j + 1, gt)
                    #mx.nd.array(features_batch).copyto(self.model.features_arr)
                    mx.nd.array(images_batch).copyto(self.model.image_arr)
                    #captions_batch.copyto(self.model.captions)
                    gen_caps = gen_exe.forward(is_train=False)
                    gen_caps = gen_caps[0].asnumpy()
                    decoded = decode_captions(gen_caps, self.model.idx_to_word)
                    print "Generated caption: %s\n" % decoded[0]
                    
            print "Previous epoch loss: ", prev_loss/(n_iters_per_epoch*self.batch_size)
            print "Current epoch loss: ", curr_loss/(n_iters_per_epoch*self.batch_size)
            print "Elapsed time(h): ", (time.time() - start_t)/360
            prev_loss = curr_loss
            curr_loss = 0
            '''
            # print out BLEU scores and file write
            if self.print_bleu:
                all_gen_cap = np.zeros((self.batch_size*n_iters_val, 20))
                self.val_data.reset()
                for i in range(n_iters_val):
                    captions_batch, images_batch = self.val_data.next_batch_for_all()
                    features_batch = self.vgg19.getfeatures(images_batch)
                    mx.nd.array(features_batch).copyto(self.model.features_arr)
                    #captions_batch.copyto(self.model.captions)
                    gen_cap = gen_exe.forward(is_train=False)
                    gens = gen_cap[0].asnumpy()
                    all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gens

                np.savetxt("gens.txt",all_gen_cap,fmt=['%s']*all_gen_cap.shape[1],newline='\n')
                all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                scores = evaluate(data_path='./data', split='val', get_scores=True)
                write_bleu(scores=scores, path=self.model_path, epoch=e)
            '''
            # save model's parameters
            if (e + 1) % self.save_every == 0:
                self.model.save("model/lstmwithcnn", ckptnum)
                print "model-%s saved." % (e + 1)