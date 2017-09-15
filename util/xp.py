import numpy
from chainer import Variable, cuda

class Xp:
	def __init__(self,args):
		if args.use_gpu:
			self.lib = cuda.cupy
			self.is_gpu = True
			cuda.get_device(args.gpu_device).use()
		else:
			self.lib = numpy

	def uniform(self,dim):
		return Variable(self.lib.array(self.lib.uniform.uniform(0,1,dim), dtype=self.lib.float32))
	def normal(self,var,dim):
		return Variable(self.lib.array(self.lib.random.normal(0,var,dim), dtype=self.lib.float32))

	def uniformv(self,dim):
		return Variable(self.lib.array(self.lib.uniform.uniform(0,1,dim), dtype=self.lib.float32),volatile='on')
	def normalv(self,var,dim):
		return Variable(self.lib.array(self.lib.random.normal(0,var,dim), dtype=self.lib.float32),volatile='on')


	def array_int(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.int32))
	def array_intv(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.int32),volatile='on')

	def array_float(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.float32))
	def array_floatv(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.float32),volatile='on')
	
	def get_max(self,x):
		x = x.data.argmax(1)
		if self.is_gpu:	x = cuda.to_cpu(x)
		return list(x)
