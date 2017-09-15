#! -*- coding:utf-8 -*-

from chainer import functions as F
from chainer import links as L
from chainer import Chain, Variable


import random
import numpy as np
import chainer.cuda


class Encoder(Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(Encoder, self).__init__(
			i2c = L.EmbedID(1, hidden_size),
			i2h = L.EmbedID(1, hidden_size),
			x2e = L.EmbedID(vocab_size, embed_size),
			e2h = L.Linear(embed_size,  4 * hidden_size),
			h2h = L.Linear(hidden_size, 4 * hidden_size),
		)
	def __call__(self, x, c, h):
		e = F.tanh(self.x2e(x))
                self.e = F.tanh(self.x2e(x))
		return F.lstm(c, self.e2h(e) + self.h2h(h))
        def get_embed_x(self, x):
#                self.e = F.tanh(self.x2e(x))
#                return self.e
                return self.x2e(x)
        

	def init(self,x):
		return self.i2c(x),self.i2h(x)

#####################
class Decoder(Chain):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(Decoder, self).__init__(
			y2e = L.EmbedID(vocab_size, embed_size),
			e2h = L.Linear(embed_size,  4 * hidden_size),
			h2h = L.Linear(hidden_size, 4 * hidden_size),
			h2f = L.Linear(hidden_size, embed_size),
			f2y = L.Linear(embed_size, vocab_size),
		)
	def __call__(self, y, c, h):
		e = F.tanh(self.y2e(y))
		c, h = F.lstm(c, self.e2h(e) + self.h2h(h))
		return c, h
	def predict(self,c,h):
		return self.f2y(F.tanh(self.h2f(h)))


class AutoEncoder(Chain):
	def __init__(self, args):
		super(AutoEncoder, self).__init__(
			enc = Encoder(args.vocab, args.embed, args.hidden),
			dec = Decoder(args.vocab, args.embed, args.hidden),
			wc1=L.Linear(args.embed,args.embed),
			wc2=L.Linear(args.embed,args.embed),
		)

	def get_predicted_word(self,encoded_vectors,decoded_vector,args):
                
                attention=self.get_attention(args,encoded_vectors,decoded_vector)
                softmaxed_attention=F.softmax(attention)
                return softmaxed_attention
		#return sum(encoded_vectors)

	def generate(self,args,content_words,xp):
		EOS = xp.array_int([0],is_train=False)

                c,h = self.enc.init(EOS)

		max_content_length = max(len(x) for x in content_words)
		for i in range(len(content_words)):
			content_words[i] += [0]*(max_content_length-len(content_words[i]))
#		content_words = xp.array_int(content_words,is_train=True)
#                content_words = xp.array_int(content_words,is_train=False)
                content_words = xp.array_float(content_words,is_train=False)#floatにしないとエラー出た
		generated = list()
		while len(generated)<= args.limit_size:

                        print(content_words.data.shape)
                        print(h.data.shape)

			y = self.get_predicted_word(content_words, h,args)
			generated.append(xp.get_max(y)[0])
			c,h = self.dec(sliced_haiku[i],c,h)
		return generated

	def get_attention(self,args,encoded_vectors,decoded_vector):
                
#                print("encoded_vectors.data.shape",encoded_vectors.data.shape)
#                print(encoded_vectors.data)
                print("decoded_vector.data.shape",decoded_vector.data.shape)
                print(type(encoded_vectors))
                pre_alpha=F.batch_matmul(encoded_vectors,decoded_vector,transa=True,transb=False)
		alpha=F.softmax(pre_alpha)
                print(alpha.data.shape)
                
#		c_t=F.matmul(encoded_vectors,alpha,transa=False,transb=False)
                c_t=F.batch_matmul(encoded_vectors,alpha,transa=False,transb=False)
                return c_t
        

	def __call__(self,haikus,content_words,xp,args):
		self.zerograds() #勾配をゼロに初期化

		max_content_length = max(len(x) for x in content_words)
		for i in range(len(content_words)):
			content_words[i] += [0]*(max_content_length-len(content_words[i]))#一番長い内容語と同じ長さにする
                content_words = xp.array_int(content_words,is_train=True)
                #content_words = xp.array_float(content_words,is_train=True)

		max_haiku_length = max(len(x) for x in haikus)
                print("max_haiku_length",max_haiku_length)
		for i in range(len(haikus)):
			haikus[i] += [0]*(max_haiku_length-len(haikus[i]))
                        
		sliced_haiku = list()
		for i in range(max_haiku_length):
#			sliced_haiku.append(xp.array_int([haiku[i] for haiku in haikus],is_train=True))
                        sliced_haiku.append(xp.array_float([haiku[i] for haiku in haikus],is_train=True))
#                H=xp.array_int(self.enc.init(content_words))
                H=self.enc.init(content_words)#内容語のベクトルを出しているつもり
                print(self.enc.init(content_words))
                EOS=xp.array_int([0 for x in range(len(haikus))])

		c,h_y = self.enc.init(EOS)#EOSのh
                print("h_y.data.shape",h_y.data.shape)
                
                
		loss, accuracy = [],[]
		for i in range(max_haiku_length):
#			y = self.get_predicted_word(content_words, h,args)#content_words:encoded_vectors, h:decoded_vector
                        
                        y = self.get_predicted_word(H, h_y,args)#内容語のベクトルと、1つ前のyのhから、yを予測しようとしている
                        
##                       # y = self.get_predicted_word(embed_x, h,args)
##                        print("↓予測した俳句のdata.shape")
##                        print(y.data.shape)
##                        print("sliced_haikuのdata.shape")
##                        print(sliced_haiku[i].data.shape)
##                        #y=F.squeeze(y,axis=2)
##                        Y=F.squeeze(y,axis=2)
                        
			loss.append(F.softmax_cross_entropy(y,sliced_haiku[i]))
			accuracy.append(F.accuracy(y,sliced_haiku[i]))
			c,h_y = self.dec(sliced_haiku[i],c,h_y)

		y = self.get_predicted_word(content_words, h,args)
#                y = self.get_predicted_word(embed_x, h,args)
		loss.append(F.softmax_cross_entropy(y, EOS))
		accuracy.append(F.accuracy(y,EOS))
		return sum(loss), sum(accuracy)/len(accuracy)
