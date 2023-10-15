###############################################################################
#
###############################################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn

import sentencepiece as spm

###############################################################################
class RNN(nn.Module):
###############################################################################
	def __init__(self, specs, sp):
		super(RNN, self).__init__()
		# ensure that character dictionary doesn't change
		self.sp = sp # SentencePiece model
		self.unk_id = sp.piece_to_id('<unk>')
		self.bos_id = sp.piece_to_id('<bos>')
		self.eos_id = sp.piece_to_id('<eos>')
		self.bos_en = sp.piece_to_id('<bos_en>')
		self.bos_chn = sp.piece_to_id('<bos_chn>')
		self.bos_sp = sp.piece_to_id('<bos_sp>')
		# print("bos_id, eos_id",self.bos_id,self.eos_id) 0 0
		# exit()
		nTokens = sp.get_piece_size()
		self.ori_specs = specs
		self.specs = specs + [nTokens]

		embed_size, hidden_size, proj_size, rnn_nLayers, self.share, dropout = specs
		self.embed = nn.Embedding(nTokens, embed_size)

		#if rnn_nLayers == 1: dropout = 0.0 # dropout is only applied between layers
		self.rnn = nn.LSTM(embed_size, hidden_size, rnn_nLayers, dropout=dropout, batch_first=True, proj_size=proj_size)
		self.dec_rnn = nn.LSTM(embed_size, hidden_size, rnn_nLayers, dropout=dropout, batch_first=True, proj_size=proj_size)

		if not self.share:
			self.out = nn.Linear(proj_size, nTokens, bias=False) # character - CrossEntropy

		self.dropout = nn.Dropout(dropout)

		for p in self.parameters(): # optionally apply different randomization
			if p.dim() > 1:
				#nn.init.xavier_uniform_(p)
				#nn.init.xavier_normal_(p)
				nn.init.kaiming_uniform_(p)	# kaiming works better than xavier or pytorch default
				# nn.init.kaiming_normal_(p)		# no apparent difference between uniform & normal
				pass

	###########################################################################
	def forward(self, seqs, hidden=None):
		if hidden == None:
			nBatch = len(seqs)
			nTokens = len(seqs[0])

			seqs = torch.cat(seqs).view(nBatch, nTokens)

			# if not self.dec_flag:
			embed = self.embed(seqs)
			embed = self.dropout(embed)
			prev, hidden = self.rnn(embed, hidden)
			prev = self.dropout(prev)

			if not self.share:
				out = self.out(prev) # chars
			else:
				out = torch.matmul(prev, torch.t(self.embed.weight)) # uses the embedding table as the output layer
			return out, hidden
		else:
			nBatch = len(seqs)
			nTokens = len(seqs[0])

			seqs = torch.cat(seqs).view(nBatch, nTokens)

			# if not self.dec_flag:
			embed = self.embed(seqs)
			embed = self.dropout(embed)
			prev, hidden = self.dec_rnn(embed, hidden)
			prev = self.dropout(prev)

			if not self.share:
				out = self.out(prev) # chars
			else:
				out = torch.matmul(prev, torch.t(self.embed.weight)) # uses the embedding table as the output layer
			return out, hidden
		# else:
		# 	embed = self.embed(seqs)
		# 	embed = self.dropout(embed)
		# 	prev, hidden = self.dec_rnn(embed, hidden)
		# 	prev = self.dropout(prev)

		# 	if not self.share:
		# 		out = self.out(prev) # chars
		# 	else:
		# 		out = torch.matmul(prev, torch.t(self.embed.weight)) # uses the embedding table as the output layer
		# 	# print("dec out is, len is", len(out[0]))
		# 	# print("out0 is", out[0])
		# 	# exit(0)
		# 	return out, hidden
		
		# exit(0)


	###########################################################################
	def lookup_ndxs(self, text, add_control=True):
		ndxs = self.sp.EncodeAsIds(text)
		if add_control:
			ndxs = [self.bos_id] + ndxs + [self.eos_id]
		return ndxs

	###########################################################################
	def decode(self, ndxs):
		tokens = self.sp.decode(ndxs)
		return tokens
