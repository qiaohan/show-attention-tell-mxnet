import numpy as np
from PIL import Image
from random import shuffle
import os,json
import multiprocessing as mp

class DataSet(object):
	def __init__(self, data_path, json_path, batchsize, word_to_idx, max_total):
		infos = json.load(open(json_path))
		self.imgpaths = []
		self.captions = [] 
		shuffle(infos)
		for l in infos:
			self.imgpaths.append(l["image_path"])
			self.captions.append(l["caption"])
		self.pathbase = data_path
		self.itptr = 0
		self.totalnum = len(self.imgpaths)
		self.num_batches = len(self.imgpaths)/batchsize
		self.batchsize = batchsize
		self.word_to_idx = word_to_idx
		self.max_len = max_total
		self.processes = []
	def reset(self):
		self.itptr = 0
	def getimg(self, f):
		image = Image.open(f).convert('RGB')
		width, height = image.size
		if width > height:
			left = (width - height) / 2
			right = width - left
			top = 0
			bottom = height
		else:
			top = (height - width) / 2
			bottom = height - top
			left = 0
			right = width
		image = image.crop((left, top, right, bottom))
		image = image.resize([224, 224], Image.ANTIALIAS)
		#print np.asarray(image).shape
		#print f
		return (np.asarray(image).transpose((2,0,1))-120)/255.0
	def sentence2array(self, sentence):
		a = [self.word_to_idx['<NULL>']] * self.max_len
		for i,s in enumerate(sentence.split(" ")):
			if s in self.word_to_idx.keys():
				a[i] = self.word_to_idx[s]
			else:
				a[i] = self.word_to_idx['<NULL>']
		return a
	def next_batch_for_all(self):
		#idxs = [ k+self.itptr for k in range(self.batchsize)] 
		imgpaths = self.imgpaths[self.itptr:self.itptr+self.batchsize]
		captions = self.captions[self.itptr:self.itptr+self.batchsize]
		img_array = [self.getimg(self.pathbase+p) for p in imgpaths]
		caption_array = [self.sentence2array(c) for c in captions]
		self.itptr += self.batchsize
		return np.asarray(caption_array),np.asarray(img_array),imgpaths

def inputQ(queue, dataset, start, end):
	batchsize = dataset.batchsize
	for itptr in xrange(start,end):
		imgpaths = dataset.imgpaths[itptr*batchsize:(itptr+1)*batchsize]
		captions = dataset.captions[itptr*batchsize:(itptr+1)*batchsize]
		img_array = [dataset.getimg(dataset.pathbase+p) for p in imgpaths]
		caption_array = [dataset.sentence2array(c) for c in captions]
		queue.put([np.asarray(caption_array),np.asarray(img_array),imgpaths])
class MpDataSet(object):
	"""docstring for MpDataSet"""
	def __init__(self, pnum, dataset):
		super(MpDataSet, self).__init__()
		#self.dataset = dataset
		self.queue = mp.Queue(10)
		buckets = int(int(dataset.totalnum/dataset.batchsize)/pnum)
		self.totalnum = buckets * pnum * dataset.batchsize
		self.processes = []
		for i in range(pnum):
			sb = buckets*i
			eb = buckets*(i+1)
			self.processes.append(mp.Process(target=inputQ, args=(self.queue,  dataset, sb, eb)))
	def reset(self):
		for p in self.processes:
			p.terminate()
		for p in self.processes:
			p.join()
		while not self.queue.empty():
			self.queue.get()
		for p in self.processes:
			p.start()
	def next_batch_for_all(self):
		#idxs = [ k+self.itptr for k in range(self.batchsize)] 
		return self.queue.get()
		