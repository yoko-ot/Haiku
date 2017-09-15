# -*- coding: utf-8 -*-

from collections import defaultdict,Counter
import os,codecs

class Vocabulary:
	def __init__(self,data_file,vocab):
		word_freq = Counter()
		for text in codecs.open(data_file,'r','utf-8'):
			word_freq.update(list(text.strip()))
		print('full vocabulary size:',len(word_freq))
		_s2i,_i2s = {'<EOS>':0,'<unk>':1},{0:'<EOS>',1:'<unk>'}
		for sc in word_freq.most_common(vocab-len(_s2i)):
			i,s = len(_s2i), sc[0]
			_s2i[s],_i2s[i] = i,s
		self._s2i,self._i2s = _s2i,_i2s

	def s2i(self, s):
		if s in self._s2i:	return self._s2i[s]
		return 1

	def i2s(self, i):
		if i in self._i2s:	return self._i2s[i].encode('utf-8')
		return '<OOV>'

	def dump(self):
		result = []
		for i,s in sorted(self._i2s.items(),key=lambda x:x[0]):
			result.append(str(i)+':'+s)
		return '\n'.join(result)


