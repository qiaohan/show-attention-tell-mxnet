import mxnet as mx

class vggnet19(object):
	"""docstring for vggnet19"""
	def __init__(self, ctx, batchsize):
		super(vggnet19, self).__init__()
		self.ctx = ctx
		self.sym = None
		self.exe = None
		self.image_arr = mx.nd.ones([batchsize,3,224,224]).as_in_context(ctx)
	def save_params(self, fname):
		arg_params = self.exe.arg_dict
		aux_params = self.exe.aux_dict
		#self._param_names
		#self._aux_names
		save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
		save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
		for n in self.input_names:
			save_dict.pop('arg:%s' % n)
		mx.ndarray.save(fname, save_dict)

	def save(self, prefix, iternum):
		assert self.exe is not None
		print("Saving model to %s" %prefix)
		self.sym.save('%s-symbol.json'%prefix)
		param_name = '%s-%04d.params' % (prefix, iternum)
		self.save_params(param_name)
		print("Saved checkpoint to %s" %param_name)
	def load(self, model_prefix, step):
		print("Loading model...") 
		sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, step)
		conv5_3 = mx.sym.transpose(sym.get_internals()['conv5_3_output'], axes=(0,2,3,1))
		self.sym = mx.sym.reshape(data=conv5_3, shape=[-1, 196, 512])
		exec_params={'data':self.image_arr}
		exec_grads={}
		aux_args={}
		for name in arg_params.keys():
			w = arg_params[name]
			exec_params[name] = w.copyto(self.ctx)
			exec_grads[name] = w.copyto(self.ctx)
		for name in aux_params:
			w = aux_params[name]
			aux_args[name] = w.copyto(self.ctx)
		#self.arg_params = exec_params
		#self.aux_params = aux_args
		self.arg_params = arg_params
		self.aux_params = aux_params
		self.exe = self.sym.bind(ctx = self.ctx, args = exec_params, args_grad = exec_grads, aux_states = aux_params)
		return self.exe
	def getfeatures(self, imgbatch):
		mx.nd.array(imgbatch).copyto(self.image_arr)
		return self.exe.forward()[0]
	def getnet(self):
		assert self.sym is not None
		return self.sym, self.arg_params, self.aux_params