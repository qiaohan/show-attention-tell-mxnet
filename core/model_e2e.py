# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import mxnet as mx
from vggnet import *

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, batch_size=50, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, ctx=mx.cpu()):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            prev2out: (optional) previously generated word to hidden state. (see Eq (2) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (2) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        self.ctx = ctx
        self.batch_size = batch_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = mx.initializer.Xavier()
        self.const_initializer = self.weight_initializer #mx.initializer.Constant(0.0)
        self.emb_initializer = self.weight_initializer #mx.initializer.Uniform(1.0)
        self.build_input_e2e()
    def build_input(self):
        # Place holder for features and captions
        self.raw_features = mx.symbol.Variable(name="features", shape=(self.batch_size, self.L, self.D), dtype="float32") # [batch_size, self.L, self.D]
        self.captions = mx.symbol.Variable(name="captions", shape=(self.batch_size, self.T+1), dtype="float32") # [batch_size, self.T+1]
        self.features_arr = mx.nd.ones([self.batch_size, self.L, self.D]).as_in_context(self.ctx)
        self.captions_arr = mx.nd.ones([self.batch_size, self.T+1]).as_in_context(self.ctx)

        self.input_names=["features","captions"]
        self.input_shapes = {"features":(self.batch_size, self.L, self.D), "captions":(self.batch_size, self.T+1)}
        self.input_types = {"features":"float32", "captions":"int32"}
        self.auxs={}
        self.arguments = {}
        self.initializers = {}
        self.cnn_params = {}
    def build_input_e2e(self):
        # Place holder for features and captions
        vgg19 = vggnet19(self.ctx, self.batch_size)
        vgg19.load("model/vgg/vgg19",0)
        self.raw_features, self.arguments, self.auxs = vgg19.getnet()
        #self.raw_features [batch_size, self.L, self.D]
        self.captions = mx.symbol.Variable(name="captions", shape=(self.batch_size, self.T+1), dtype="float32") # [batch_size, self.T+1]
        self.image_arr = mx.nd.ones([self.batch_size, 3, 224, 224]).as_in_context(self.ctx)
        self.captions_arr = mx.nd.ones([self.batch_size, self.T+1]).as_in_context(self.ctx)

        self.input_names=["data","captions"]
        self.input_shapes = {"data":(self.batch_size, 3, 224, 224)}#, "captions":(self.batch_size, self.T+1)}
        self.input_types = {"data":"float32"}#, "captions":"int32"}
        self.initializers = {}
        self.cnn_params = [n for n in self.arguments if n not in self.input_names]
    def _get_initial_lstm(self, features):
        features_mean = mx.sym.mean(features, axis=1)
        #w_h = mx.sym.Variable(name='w_inith', shape=[self.batch_size, self.D], init=self.weight_initializer)
        #w_h = mx.sym.Variable(name='w_inith', shape=[self.D, self.H], init=self.weight_initializer)
        #b_h = mx.sym.Variable(name='b_inith', shape=[self.H], init=self.const_initializer)
        h = mx.sym.tanh( mx.sym.broadcast_plus(mx.sym.dot(features_mean, self.w_feat2h), self.b_feat2h) )

        #w_c = mx.sym.Variable(name='w_initc', shape=[self.D, self.H], init=self.weight_initializer)
        #b_c = mx.sym.Variable(name='b_initc', shape=[self.H], init=self.const_initializer)
        c = mx.sym.tanh( mx.sym.broadcast_plus(mx.sym.dot(features_mean, self.w_feat2c), self.b_feat2c) )
        return c, h

    def _word_embedding(self, inputs, w):
        x = []
        for inp in inputs:
            x.append(mx.sym.Embedding(data=inp, weight=w, input_dim=self.V, output_dim=self.M, name='word_vector'))  # (N, T, M) or (N, M)
        return x

    def _project_features(self, features):
        #w = mx.sym.Variable(name='w_featureproj', shape=[self.D, self.D], init=self.weight_initializer)
        #features_flat = mx.sym.reshape(features, [-1, self.D])
        #features_proj = features_flat*w
        #features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
        #return features_proj
        return mx.sym.dot(features, self.w_proj)

    def _attention_layer(self, features, features_proj, h, w, b, w_att):

        h_att =  mx.sym.Activation(data=mx.sym.broadcast_plus(features_proj, b) + mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.dot(h,w), axis=1), shape=(self.batch_size, self.L, self.D)), act_type='relu')  # (N, L, D)
        #out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
        tt = mx.sym.dot(h_att, w_att)
        out_att = mx.sym.reshape(data=tt, shape=[-1, self.L])
        alpha = mx.sym.softmax(out_att) # (N, L)
        context = mx.sym.sum(features*mx.sym.broadcast_to(mx.sym.expand_dims(alpha, axis=2), shape=(self.batch_size, self.L, self.D)), axis=1, name='context')  # (N, D)
        return context, alpha

    def _selector(self, context, h, w, b):         
        beta = mx.sym.sigmoid(data=mx.sym.broadcast_plus(mx.sym.dot(h, w), b), name='beta')  # (N, 1)
        context = mx.sym.broadcast_to(beta, shape=(self.batch_size, self.D)) * context
        return context, beta

    def _decode_lstm(self, x, h, context, w_h, b_h, w_out, b_out, w_ctx2out, dropout=False,):
        if dropout:
            h = mx.sym.Dropout(h, p=0.5)
        h_logits = mx.sym.broadcast_plus(mx.sym.dot(h, w_h), b_h)

        if self.ctx2out:
            #w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
            h_logits += mx.sym.dot(context, w_ctx2out)

        if self.prev2out:
            h_logits += x
        h_logits = mx.sym.tanh(h_logits)

        if dropout:
            h_logits = mx.sym.Dropout(h_logits, p=0.5)
        out_logits = mx.sym.broadcast_plus(mx.sym.dot(h_logits, w_out), b_out)
        return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return mx.sym.BatchNorm(data=x, fix_gamma=False, momentum=0.9, eps=2e-5, name=name)
    
    def build_variables(self):
        
        self.embedding_w = mx.sym.Variable(name='w_embed_weight', shape=[self.V, self.M], init=self.emb_initializer)
        self.arguments['w_embed_weight'] = mx.nd.zeros([self.V, self.M], dtype='float32')
        self.initializers['w_embed_weight'] = self.emb_initializer

        self.att_w = mx.sym.Variable(name='w_att_weight', shape=[self.H, self.D], init=self.weight_initializer)
        self.arguments['w_att_weight'] = mx.nd.zeros([self.H, self.D], dtype='float32')
        self.initializers['w_att_weight'] = self.weight_initializer

        self.att_b = mx.sym.Variable(name = 'b_att_bias', shape=[self.D], init=self.const_initializer)
        self.arguments['b_att_bias'] = mx.nd.zeros([self.D], dtype='float32')
        self.initializers['b_att_bias'] = self.const_initializer

        self.att_w_att = mx.sym.Variable(name = 'w_att_att_weight', shape=[self.D, 1], init=self.weight_initializer)
        self.arguments['w_att_att_weight'] = mx.nd.zeros([self.D, 1], dtype='float32')
        self.initializers['w_att_att_weight'] = self.weight_initializer        

        if self.selector:
            self.sele_w = mx.sym.Variable(name = 'w_select_weight', shape=[self.H, 1], init=self.weight_initializer)
            self.arguments['w_select_weight'] = mx.nd.zeros([self.H, 1], dtype='float32')
            self.initializers['w_select_weight'] = self.weight_initializer
            self.sele_b = mx.sym.Variable(name = 'b_select_bias', shape=[1], init=self.const_initializer)
            self.arguments['b_select_bias'] = mx.nd.zeros([1], dtype='float32')
            self.initializers['b_select_bias'] = self.const_initializer
        self.decode_w_h = mx.sym.Variable(name = 'w_h_decode_weight', shape=[self.H, self.M], init=self.weight_initializer)
        self.arguments['w_h_decode_weight'] = mx.nd.zeros([self.H, self.M], dtype='float32')
        self.initializers['w_h_decode_weight'] = self.weight_initializer
        
        self.decode_b_h = mx.sym.Variable(name = 'b_h_decode_bias', shape=[self.M], init=self.const_initializer)
        self.arguments['b_h_decode_bias'] = mx.nd.zeros([self.M], dtype='float32')
        self.initializers['b_h_decode_bias'] = self.const_initializer

        self.decode_w_out = mx.sym.Variable(name = 'w_out_decode_weight', shape=[self.M, self.V], init=self.weight_initializer)
        self.arguments['w_out_decode_weight'] = mx.nd.zeros([self.M, self.V], dtype='float32')
        self.initializers['w_out_decode_weight'] = self.weight_initializer

        self.decode_b_out = mx.sym.Variable(name = 'b_out_decode_bias', shape=[self.V], init=self.const_initializer)
        self.arguments['b_out_decode_bias'] = mx.nd.zeros([self.V], dtype='float32')
        self.initializers['b_out_decode_bias'] = self.const_initializer

        self.decode_w_ctx2out = None
        if self.ctx2out:
            self.decode_w_ctx2out = mx.sym.Variable(name = 'w_ctx2out_weight', shape=[self.D, self.M], init=self.weight_initializer)
            self.arguments['w_ctx2out_weight'] = mx.nd.zeros([self.D, self.M], dtype='float32')
            self.initializers['w_ctx2out_weight'] = self.weight_initializer

        self.w_proj = mx.sym.Variable(name='w_featureproj_weight', shape=[self.D, self.D], init=self.weight_initializer)
        self.arguments['w_featureproj_weight'] = mx.nd.zeros([self.D, self.D], dtype='float32')
        self.initializers['w_featureproj_weight'] = self.weight_initializer

        self.w_feat2h = mx.sym.Variable(name='w_inith_weight', shape=[self.D, self.H], init=self.weight_initializer)
        self.arguments['w_inith_weight'] = mx.nd.zeros([self.D, self.H], dtype='float32')
        self.initializers['w_inith_weight'] = self.weight_initializer

        self.b_feat2h = mx.sym.Variable(name='b_inith_bias', shape=[self.H], init=self.const_initializer)
        self.arguments['b_inith_bias'] = mx.nd.zeros([self.H], dtype='float32')
        self.initializers['b_inith_bias'] = self.const_initializer

        self.w_feat2c = mx.sym.Variable(name='w_initc_weight', shape=[self.D, self.H], init=self.weight_initializer)
        self.arguments['w_initc_weight'] = mx.nd.zeros([self.D, self.H], dtype='float32')
        self.initializers['w_initc_weight'] = self.weight_initializer

        self.b_feat2c = mx.sym.Variable(name='b_initc_bias', shape=[self.H], init=self.const_initializer)
        self.arguments['b_initc_bias'] = mx.nd.zeros([self.H], dtype='float32')
        self.initializers['b_initc_bias'] = self.const_initializer

        #self.lstmparam = mx.rnn.RNNParams(prefix='lstm_')
        self.lstm_cell = mx.rnn.LSTMCell(num_hidden=self.H)# params=self.lstmparam)
        
        features = self.raw_features
        captions = mx.sym.split(self.captions, num_outputs=self.T+1, squeeze_axis=True)

        self.captions_in = [captions[x] for x in range(0,self.T)] #(self.T, batch_size)
        self.captions_out = [captions[x] for x in range(1,self.T+1)]
        self.mask = [mx.sym.cast(mx.sym.broadcast_not_equal(mx.sym.cast(cap, dtype='float32'), self._null*mx.sym.ones([self.batch_size])), dtype='float32') for cap in self.captions_out]

        # batch normalize feature vectors
        self.features = self._batch_norm(features, mode='train', name='conv_features')
    
        self.init_state = self._get_initial_lstm(features=features)
        self.x = self._word_embedding(inputs=self.captions_in, w=self.embedding_w) #(self.T, batch_size, self.M)
        self.features_proj = self._project_features(features=features)

        for name in self.arguments.keys():
            desc = mx.init.InitDesc(name)
            w = self.arguments[name].as_in_context(self.ctx)
            if name in self.initializers:
                self.initializers[name](desc,w)
            else:
                print name,"has no initializer..."
            self.arguments[name] = w
    
    def getexe(self, sym, bp=True):
        print("Building model...") 
        #sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, step)
        _param_names = [x for x in sym.list_arguments() if x not in self.input_names]
        _aux_names = sym.list_auxiliary_states()

        arg_shapes, _, aux_shapes = sym.infer_shape(**self.input_shapes)
        arg_types, _, aux_types = sym.infer_type(**self.input_types)
        initer = self.weight_initializer

        arg_params = {}
        aux_params = {}

        arg_name2idx = {}
        for i,x in enumerate(sym.list_arguments()):
            arg_name2idx[x] = i
        aux_name2idx = {}
        for i,x in enumerate(sym.list_auxiliary_states()):
            aux_name2idx[x] = i
        for name in _param_names:
            desc = mx.init.InitDesc(name)
            if name in self.arguments.keys():
                w = self.arguments[name]
                #self.initializers[name](desc,w)
            else:
                print name
                w = mx.nd.zeros(arg_shapes[arg_name2idx[name]], dtype=arg_types[arg_name2idx[name]]).as_in_context(self.ctx)
                initer(desc,w)
                self.arguments[name] = w
            arg_params[name] = w

        for name in _aux_names:
            if name in self.auxs.keys():
                w = self.auxs[name]
            else:
                print "aux:",name
                #desc = mx.init.InitDesc(name)
                w = mx.nd.zeros(aux_shapes[aux_name2idx[name]], dtype=aux_types[aux_name2idx[name]]).as_in_context(self.ctx)
                #initer(desc,w)
                self.auxs[name] = w
            aux_params[name] = w

        exec_params = arg_params
        exec_params["data"] = self.image_arr
        #exec_params["features"] = self.features_arr
        #exec_params['attrgt'] = mx.nd.ones([params.batch_size, 1000])
        exec_params["captions"] = self.captions_arr

        exec_grads = {}
        for k,v in exec_params.items():
            #exec_params[k] = v.copyto(self.ctx)
            if k in self.input_names:
                continue
            if bp:
                exec_grads[k] = v.copyto(self.ctx)
        exe = sym.bind(ctx = self.ctx, args = exec_params, args_grad = exec_grads, aux_states = aux_params)
        return exe

    def build_model(self):        
        loss = 0.0
        alpha_list = []
        c,h = self.init_state
        batch_size = self.batch_size
        for t in range(self.T):
            context, alpha = self._attention_layer(self.features, self.features_proj, h, self.att_w, self.att_b, self.att_w_att)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, self.sele_w, self.sele_b)

            cc = mx.sym.concat(self.x[t], context, dim=1)
            _, (h, c) = self.lstm_cell(inputs=cc, states=[h, c])

            logits = self._decode_lstm(self.x[t], h, context, self.decode_w_h, self.decode_b_h, self.decode_w_out, self.decode_b_out, self.decode_w_ctx2out, dropout=self.dropout)
            logp = -mx.sym.log(mx.sym.softmax(logits))
            loss += mx.sym.sum( logp * mx.sym.one_hot(self.captions_out[t], depth=self.V) * mx.sym.broadcast_to(mx.sym.expand_dims(self.mask[t], axis=1), shape=(self.batch_size, self.V)))

        if self.alpha_c > 0:
            #alphas = mx.sym.pack(alpha_list)  # (T, N, L)
            alphas_all = reduce(lambda x,y:x+y, alpha_list)  # (N, L)
            alpha_reg = self.alpha_c * mx.sym.sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg
        loss = loss / batch_size
        loss = mx.sym.MakeLoss(loss)
        self.loss = loss
        self.trainexe = self.getexe(loss)

        return loss, self.trainexe, self.input_names

    def save_params(self, fname):
        arg_params = self.trainexe.arg_dict
        aux_params = self.trainexe.aux_dict
        #self._param_names
        #self._aux_names
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        for n in self.input_names:
            save_dict.pop('arg:%s' % n)
        mx.ndarray.save(fname, save_dict)

    def save(self, prefix, iternum):
        assert self.trainexe is not None
        print("Saving model to %s" %prefix)
        self.loss.save('%s-symbol.json'%prefix)
        self.captiongenerator.save('%s-generator.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, iternum)
        self.save_params(param_name)
        print("Saved checkpoint to %s" %param_name)
    
    def load(self, model_prefix, step):
        print("Loading model...") 
        assert self.arguments is not None
        assert self.auxs is not None
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, step)

        for name in arg_params.keys():
            if name in self.arguments.keys():
                print name,"on"
            else:
                print name,"off"
            self.arguments[name] = arg_params[name].copyto(self.ctx)
        
        for name in self.arguments.keys():
            if name in arg_params.keys():
                continue
                #print name,"have"
            else:
                print name,"not have"

        for name in aux_params.keys():
            if name in self.auxs.keys():
                print name,"on"
            else:
                print name,"off"
            self.auxs[name] = aux_params[name].copyto(self.ctx)        
    
    def build_sampler(self, max_len=20):
        c,h = self.init_state
        batch_size = self.batch_size
        sampled_word_list = []
        alpha_list = []
        beta_list = []

        for t in range(max_len):
            if t == 0:
                #x = self._word_embedding(inputs=mx.sym.ones(shape=[batch_size])*self._start, w=self.embedding_w)
                x = mx.sym.Embedding(data=mx.sym.ones(shape=[batch_size])*self._start, weight=self.embedding_w, input_dim=self.V, output_dim=self.M)
            else:
                #x = self._word_embedding(inputs=sampled_word, w=self.embedding_w)
                x = mx.sym.Embedding(data=sampled_word, weight=self.embedding_w, input_dim=self.V, output_dim=self.M)

            context, alpha = self._attention_layer(self.features, self.features_proj, h, self.att_w, self.att_b, self.att_w_att)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, self.sele_w, self.sele_b)
                beta_list.append(beta)
            
            cc = mx.sym.concat(x, context, dim=1)
            _, (h, c) = self.lstm_cell(inputs=cc, states=[h, c])

            #_, (h, c) = self.lstm_cell(inputs=mx.sym.concat(x, context, dim=1), states=[h, c])

            logits = self._decode_lstm(x, h, context, self.decode_w_h, self.decode_b_h, self.decode_w_out, self.decode_b_out, self.decode_w_ctx2out)
            sampled_word = mx.sym.argmax(logits, axis=1)
            sampled_word_list.append(sampled_word)

        #alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        #betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        wordslist = [mx.sym.expand_dims(w, axis=0) for w in sampled_word_list]
        xx = reduce(lambda x,y: mx.sym.concat(x,y, dim=0), wordslist)
        #print type(xx)
        sampled_captions = mx.sym.transpose(xx)  # (N, max_len)
        #exe = sampled_captions.bind(ctx = self.ctx, args = exec_params, args_grad = exec_grads, aux_states = self.aux_params)
        #sampled_captions.save("gen.json")
        self.captiongenerator = sampled_captions
        exe = self.getexe(sampled_captions, bp=False)
        return alpha_list, beta_list, sampled_captions, exe