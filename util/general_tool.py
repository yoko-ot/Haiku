# -*- coding: utf-8 -*-

import datetime
def trace(*args):
	message = datetime.datetime.now().strftime('%H:%M:%S')+', '
	message += ', '.join(map(str,args))
	print(message)
