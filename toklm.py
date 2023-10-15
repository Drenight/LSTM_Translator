###############################################################################
# rnn (recurrent neural net) character language model
###############################################################################
import time, os, sys, random, datetime
from random import shuffle
import argparse
import numpy as np
#import matplotlib.pyplot as plt

from toklm_RNN import *
from tqdm import tqdm

import csv

###############################################################################
import socket
hostname = socket.gethostname()
log_file = open(f'log/tokLM_{hostname}.log', 'a')
model_path = './models/'


# My switch !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rev_opti = False #中文没有空格，反过来比较麻烦
small_test = False
small_limit = 100000

# Translation parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
seed = 'I am good.'	# your sentence to translate
target_language_token = '<bos_chn>'	# <bos_sp> <bos_en>
model_fn = 'tokLM-LIB-D-MJ0JR667.pth' # can specify a previously trained model to reload, and skip training
# model_fn = None



if model_fn is None:
	train = True

	#sp_fn = 'sample.model'
	# sp_fn = 'you need to train a sentencepiece model.model'
	sp_fn = 'spm.model'
else:
	train = False
	#train = True

if not train:
	log_file = open(f'log/tokLM_{hostname}_eval.log', 'a')

###############################################################################
def log_message(outf, message):
	print(message)
	if not outf is None:
		outf.write(message)
		outf.write("\n")
		outf.flush()

###############################################################################
# # torch.cuda.set_device(6)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.version.__version__, device)
# exit(0)

###############################################################################
class DataItem:
	def __init__(self, text=None, ndxs=None):
		self.text		= text		# original text
		self.ndxs		= ndxs		# indexes of characters or tokens

###############################################################################
# tunable hyper-parameters
###############################################################################
# these values are used to create the model
###############################################################################
share = False
share = True	# share embedding table with output layer

# embed_size = 20
# embed_size = 50
# embed_size = 100
embed_size = 150
# embed_size = 300
# embed_size = 500

if share:
	proj_size = embed_size
else:
	proj_size = 150
	# proj_size = 200
	#proj_size = 250
	#proj_size = 350

# hidden_size = 3000
# hidden_size = 150
# hidden_size = 200
# hidden_size = 300
hidden_size = 600
# hidden_size = 800
# hidden_size = 1000

rnn_nLayers = 1
# rnn_nLayers = 3
# rnn_nLayers = 4

dropout = 0.001
# dropout = 0.1

specs = [embed_size, hidden_size, proj_size, rnn_nLayers, share, dropout]

###############################################################################
# these values are used within the training code
###############################################################################
# learning_rate = 3
# learning_rate = 0.000001
learning_rate = 0.001
# learning_rate = 0.001
# learning_rate = 1

# initial batchsize
# batch_size = 2
# batch_size = 2
# batch_size = 2
# batch_size = 4
batch_size = 32

# increase the batchsize every epoch by this factor
batch_size_multiplier = 1
# batch_size_multiplier = 1.4
# batch_size_multiplier = 1.6
#batch_size_multiplier = 2

# nEpochs = 1
nEpochs = 2
# nEpochs = 20
# nEpochs = 200
# nEpochs = 1000

L2_lambda = 0.001
# L2_lambda = 0.03

###############################################################################
def train_batch(model, optimizer, criterion, data_source, data_target, data_ndxs, file_source, update=True):
	end_token = '<bos>'
	model.zero_grad() 
	total_loss, total_tokens, total_chars = 0, 0, 0

	for ndx in data_ndxs:
		data_item = data_source[ndx]	
		data_item_tar = data_target[ndx]
		if data_item.ndxs is None:
			data_item.ndxs = model.lookup_ndxs(data_item.text)
		if data_item_tar.ndxs is None:
			data_item_tar.ndxs = model.lookup_ndxs(data_item_tar.text)
		# print(data_item_tar.ndxs)
		# print(start_token)
		if file_source[ndx] == 'Span2Eng.tsv' or file_source[ndx] == 'Manda2Eng.tsv':
			end_token = '<bos_en>'
		elif file_source[ndx] == 'Eng2Span.tsv':
			end_token = '<bos_sp>'
		elif file_source[ndx] == 'Eng2Manda.tsv':
			end_token = '<bos_chn>'
	
		# exit()
		# print(sp.piece_to_id('<bos_en>'))
		# print(sp.piece_to_id('<bos_chn>'))
		# print(sp.piece_to_id('<bos_sp>'))

		# print(data_item_tar.text)
		# print(end_token)
		# print(data_item_tar.ndxs)
		# data_item_tar.ndxs.append(sp.piece_to_id(end_token))
		# print(data_item_tar.ndxs)
		# exit()

		data_item.ndxs.append(sp.piece_to_id(end_token))
		ndxs = torch.tensor(data_item.ndxs, dtype=torch.int64).to(device)
		ndxs_tar = torch.tensor(data_item_tar.ndxs, dtype=torch.int64).to(device)

		# print(ndxs) #tensor([   0,    3, 6398,  702,   80,  645,  167,  368,    4,    0], device='cuda:0')
		# exit()

		out, hidden = model([ndxs])
		out, hidden = model([ndxs_tar[:-1]],hidden)

		# #这个留作生成代码，训练还是用之前类似的代码
		# pre_id = torch.tensor([model.bos_id], dtype=torch.int64).to(device)
		# tensor_list = []
		# cnt = 0
		# while cnt < len(data_item_tar.ndxs):
		# 	cnt += 1
		# 	out, hidden = model([pre_id], hidden)
		# 	tensor_list.append(out[0][0])
		# 	max_index = torch.argmax(out[0][0])
		# 	pre_id = torch.tensor([max_index.item()], dtype=torch.int64).to(device)
		# stacked_tensor = torch.stack(tensor_list) 

		#data_target
		loss = criterion(out[0], ndxs_tar[1:])

		# exit(0)

		total_loss += loss.data.item()
		total_tokens += len(out[0])
		total_chars += len(data_item.text)+1

		if update:
			loss.backward()

	if update:
		optimizer.step()

	return total_loss, total_tokens, total_chars

###############################################################################
def train_model(model, train_data_source, train_data_target, file_source, dev_data_source=None, dev_data_target=None):
	data_list  = [i for i in range(len(train_data_source))]    # list of indexes of data
	if dev_data_source == None:
		shuffle(data_list)
		nDevItems = min(int(0.05*len(train_data_source)), 2000) # use 5% or 2,000 whichever is smaller
		dev_list = data_list[:nDevItems]
		train_list	= data_list[nDevItems:]
		dev_data_source = train_data_source
		dev_data_target = train_data_target
	else:
		train_list	= data_list
		dev_list	= [i for i in range(len(dev_data_source))]    # list of indexes of dev_data

	# define the loss functions
	criterion = nn.CrossEntropyLoss(reduction='sum')

	# choose an optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
	# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
	# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

	###############################################################################
	# train model 
	###############################################################################
	TotalTrainLoss0, TotalTrainChars = 0, 0

	start = time.time()
	bs = batch_size
	for epoch in range(nEpochs):
		if epoch > 0: bs *= batch_size_multiplier
		ibs = int(bs+0.5)
		shuffle(train_list)     # make it a habit to shuffle data, even if random
		model.train()
		TrainLoss0, TrainTokens, TrainChars = 0, 0, 0
		for i in tqdm(range(0, len(train_list), ibs)):
		# for i in range(0, len(train_list), ibs):
			loss0, nTokens, nChars = train_batch(model, optimizer, criterion, train_data_source, train_data_target, train_list[i:i+ibs], file_source, update=True)

			TrainLoss0 += loss0
			TrainTokens += nTokens
			TrainChars += nChars

			if nChars > 0:
				# print(f'{epoch:4} {i:6} {nTokens:5} {nChars:6} {loss0/nTokens:7.3f} {loss0/nChars:7.3f} -- {TrainLoss0/TrainTokens:8.4f} {TrainLoss0/TrainChars:8.4f}')
				pass

		model.eval()
		DevLoss0, DevTokens, DevChars = train_batch(model, optimizer, criterion, dev_data_source, dev_data_target, dev_list, file_source, update=False)
		if epoch == 0: log_message(log_file, f'train={len(train_list):,} {TrainTokens:,} {TrainChars:,} {TrainChars/TrainTokens:0.1f} dev={len(dev_list):,} {DevTokens:,} {DevChars:,} {DevChars/DevTokens:0.1f} bs={batch_size} lr={learning_rate} {model.specs}')

		msg_trn = f'TrainLoss0/TrainTokens:{TrainLoss0/TrainTokens:8.4f} TrainLoss0/TrainChars:{TrainLoss0/TrainChars:8.4f}'
		msg_dev = f'DevLoss0/DevTokens:{DevLoss0/DevTokens:8.4f} DevLoss0/DevChars{DevLoss0/DevChars:8.4f}'
		log_message(log_file, f'epoch:{epoch} msg_trn{msg_trn} -- msg_dev{msg_dev} -- ibs:{ibs:4} {time.time()-start:6.1f}')

		torch.save(model, f'./models/tokLM-{hostname}.pth')
		# torch.save(model, f'./models/tokLM-{hostname}-dec.pth')

	return model

###############################################################################
###############################################################################
def read_datafile(ffn, data_source, data_target):
	with open(ffn, 'r', newline='', encoding='utf-8') as tsvfile:
		# Create a CSV reader with tab ('\t') delimiter
		tsvreader = csv.reader(tsvfile, delimiter='\t')

		# Iterate through the lines in the TSV file
		row_cnt = 0
		for row in tsvreader:
			row_cnt += 1
			# 'row' is a list containing the fields on the current line
			# You can access individual fields by their index
			# print("row:",row)
			field1 = row[0]  # First field
			field2 = row[1]  # Second field
			field4 = row[3]

			if rev_opti:
				words = field2.split()
				words_reversed = list(reversed(words))
				reversed_string = " ".join(words_reversed)
				field2 = reversed_string

		# for sent in sents:
		# 	sent = sent.strip()
		# 	if len(sent) == 0: continue # skip empty lines
			#print(sent)
			data_item = DataItem(field2)
			data_source.append(data_item)
			data_item = DataItem(field4)
			data_target.append(data_item)
			# if len(data_source) > 20: break # uses shortened data list -- useful for development & debugging
	return row_cnt

###############################################################################
def generate(model, seed="The ", n=500, temp=0):
	ori_seed = seed
	if rev_opti:
		words = seed.split()
		words_reversed = list(reversed(words))
		reversed_string = " ".join(words_reversed)
		seed = reversed_string

	model.eval()
	# ndxs = model.lookup_ndxs(seed)[:-1]		# adds <s> ... </s> -- remove </s> with [:-1]
	ndxs = model.lookup_ndxs(seed)
	
	# print(ndxs)
	# exit()

	# train ndxs
	# ndxs = torch.tensor(data_item.ndxs, dtype=torch.int64).to(device)
	# print(ndxs) tensor([   0,    3, 6398,  702,   80,  645,  167,  368,    4,    0], device='cuda:0')
	if target_language_token == '<bos_en>':
		ndxs.append(model.bos_en)
	elif target_language_token == '<bos_sp>':
		ndxs.append(model.bos_sp)
	elif target_language_token == '<bos_chn>':
		ndxs.append(model.bos_chn)
	ndxs_list = ndxs
	# print(ndxs_list)
	# print(model.decode(ndxs_list))

	# exit()
	ndxs = torch.tensor(ndxs, dtype=torch.int64).to(device)

	# print(ndxs)
	# exit()

	c, h = model([ndxs])
	# print(c)
	# exit()
	text = list(ori_seed)
	for i in range(n):
		scores = c[0,-1]
		if temp <= 0:
			_, best = scores.max(0)
			best = best.data.item()
		else:
			#print(scores)
			output_dist = nn.functional.softmax(scores.view(-1).div(temp))#.exp()
			#print(output_dist)
			best = torch.multinomial(output_dist, 1)[0]
			best = best.data.item()

		# print(best)
		ndxs_list.append(best)
		if best == model.eos_id:
			break

		c_in = torch.tensor([best], dtype=torch.int64).to(device)
		c, h = model([c_in], h)

	text = model.decode(ndxs_list[1:-1]) # removes <s> & </s>
	return text

###############################################################################
def count_parameters(model):
	total = 0
	for name, p in model.named_parameters():
		if p.dim() > 1:
			print(f'{p.numel():,}\t{name}')
			total += p.numel()

	print(f'total = {total:,}')

###############################################################################
# start main
###############################################################################
if __name__ == "__main__":
	log_message(log_file, f'\nstart TokLM -- {datetime.datetime.now()} - pytorch={torch.version.__version__}, device={device}')

	if model_fn is None:
		sp = spm.SentencePieceProcessor()
		sp.Load(model_path+sp_fn)
		print(f'loaded {model_path+sp_fn} sentencepiece model')

		model = RNN(specs, sp)
		model.tokenizer = sp_fn
		model = model.to(device)
	else:
		ffn = model_path + model_fn
		print(ffn)
		model = torch.load(ffn, map_location=device)
		log_message(log_file, f'load model {ffn} = {model.specs}')

	print(model)
	count_parameters(model)

	if train:
		path_data = './data/'
		# fns = ['ROCStories_spring2016.txt', 'ROCStories_winter2017.txt']
		fns = ['Span2Eng.tsv', 'Manda2Eng.tsv', 'Eng2Span.tsv', 'Eng2Manda.tsv']
		data_source = []
		data_target = []
		file_source = []
		for fn in fns:
			ffn = path_data + fn
			row_cnt = read_datafile(ffn, data_source, data_target)
			file_source.extend([fn] * row_cnt)  # 记录文件来源
		# print(len(data_source_sp), len(data_target_en), fns[0])
		# exit(0)
		combined_data = list(zip(data_source, data_target, file_source))
		random.shuffle(combined_data)
		data_source, data_target, file_source = zip(*combined_data)
		# print(data_source[0].text, data_target[0].text, file_source)

		if small_test:
			data_source = data_source[:small_limit]
			data_target = data_target[:small_limit]
			file_source = file_source[:small_limit]

		# print(data_source[0].text, data_target[0].text, file_source[0])
		# exit()

		model = train_model(model, data_source, data_target, file_source)
		# model = train_model(model, data_target_en1, data_source_sp, start_token='<bos_sp>')
		# model = train_model(model, data_source_chn, data_target_en2, start_token='<bos_en>')
		# model = train_model(model, data_target_en2, data_source_chn, start_token='<bos_chn>')

	# exit(0)
	mlen = 200
	#temp = 0.1	# this will select a random word based on the probability distribution scaled by p/temp
	temp = 0	# this will select the best word at each step

	# seed = 'estoy tan feliz' #I am so happy
	# seed = '¡Intentemos algo!'
	# global seed

	while True:
		sample = generate(model, seed=seed, n=mlen, temp=temp)
		print(sample)
		break
		print('enter another seed or <control-c> to exit')
		seed = input()

