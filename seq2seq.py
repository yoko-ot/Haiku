#! -*- coding:utf-8 -*-

import sys,os,shutil,random
import pickle

import numpy as np

from chainer import functions as F
from chainer import links as L
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.cuda

from util.backend import Backend as Xp
from util.vocabulary import Vocabulary
from util.optimizer_setter import get_opt
import util.general_tool as tool

from models.Model0001 import AutoEncoder
#from models.Model0001 import Encoder


import codecs
def generate_minibatch(data_file,batch_size):
	
	batch=[]
	for text in codecs.open(data_file,'r',"utf-8"):
		batch.append(text.strip())
		if len(batch)==batch_size:
			yield batch
			batch=[]
	if len(batch)!=0: yield batch
				


def get_season_word_list(season):#季語についてはまだいれていないので、ここは関係ない。
	tool.trace("季節:\t%s"%season)
	kigo=set()
	for line in  open('dataset/kigo/'+season):
		kigo.add(line.strip())
	return kigo
				
def train(args):
	vocab_haiku 	= Vocabulary(args.haiku_file_train, args.vocab)
	vocab_content  	= Vocabulary(args.content_file_train, args.vocab)
	#バグ

	xp = Xp(args)
	m = AutoEncoder(args)
	if args.use_gpu: m.to_gpu() 
	opt = get_opt(args)
	opt.setup(m)

	season_word_list = get_season_word_list(args.season)#放置
	

	

	for epoch in range(args.epoch_size):
		Loss,Accuracy = [],[]
		print("epoch=",epoch)

		haiku_gen  = generate_minibatch(args.haiku_file_train,	args.batch_size)
		content_word_gen  = generate_minibatch(args.content_file_train, args.batch_size)
		for (haikus,content_words) in zip(haiku_gen,content_word_gen):
			haikus 	= [[vocab_haiku.s2i(x) 	for x in text] 	for text in haikus]
			content_words 	= [[vocab_content.s2i(x) for x in word] for word in content_words]
					
			loss,accuracy = m(haikus,content_words,xp,args)
			loss.backward()
			opt.update()
			Loss.append(loss.data.get())
			Accuracy.append(accuracy.data.get())
			del loss,accuracy

		tool.trace('epoch:',epoch,'Loss:',sum(Loss)/len(Loss),'Accracy:',sum(Accuracy)/len(Accuracy))

                content_word_test_gen=generate_minibatch(args.content_file_test,args.batch_size)
                                      
                for i,content_words_test in enumerate(content_word_test_gen):
#		for i,content_words in enumerate(gen_data(args.content_file_test,args.batch_size)):
#			content_words = [[vocab_content.s2i(x) for x in word] for word in content_words]
                        content_words_test = [[vocab_content.s2i(x) for x in word] for word in content_words_test]
			output = m.generate(args,content_words_test,xp)
                        
			#output='  ' + ''.join([vocab_content.i2s(x) for x in m.generate(args,batch_content_test,xp)])

                        print(output)
                        
	tool.trace('finished.')

############################################################################
############################################################################
############################################################################

from argparse import ArgumentParser
def parse_args():
	p = ArgumentParser()

	p.add_argument('--haiku_file_train', 	'-sFtr', 	default='dataset/haiku/haiku_kana')
	p.add_argument('--content_file_train',  '-cFtr', 	default='dataset/content_word/content_kana')
	p.add_argument('--content_file_test',   '-cFte', 	default='dataset/content/content_kana_test')
		
	p.add_argument('--use_gpu',		'-g', 	action='store_true', default=False)
	p.add_argument('--gpu_device', 	'-gd',	default=0, type=int)

	p.add_argument('--vocab',	'-V', 		default=10000,	type=int)
	p.add_argument('--embed',	'-De', 		default=11, 	type=int)
#	p.add_argument('--hidden',	'-Dh', 		default=13, 	type=int)
        p.add_argument('--hidden',      '-Dh',          default=22,     type=int)


	p.add_argument('--batch_size',	'-bS',	default=20, 	type=int)
	p.add_argument('--epoch_size',	'-eS',	default=1000, 	type=int)
	p.add_argument('--limit_size',	'-lS',	default=41, 	type=int)


	p.add_argument('--opt_model','-opt',default='Adam')
	p.add_argument('--alpha0','-a0', default=0, type=float)
	p.add_argument('--alpha1','-a1', default=0, type=float)
	p.add_argument('--alpha2','-a2', default=0, type=float)
	p.add_argument('--alpha3','-a3', default=0, type=float)

	p.add_argument('--season','-season',default='spring')
		
	return p.parse_args()

if __name__ == '__main__':
	args = parse_args()
	print(args)
        print(chainer.__version__)
	train(args)
