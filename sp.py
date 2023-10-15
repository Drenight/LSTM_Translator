###############################################################################
# SentencePiece example -- Using SentencePiece is easy!
#“pip install sentencepiece” installs it for Python
###############################################################################
import sentencepiece as spm

#fn = './data/ROCStories_spring2016.txt'
# fn = './data/ROCStories_spring2016.txt,./data/ROCStories_winter2017.txt' # can use ',' separated list
# fn = './data/Span2Eng.tsv, ./data/Manda2Eng.tsv' # can use ',' separated list
fn = './data/Span2Eng.tsv,./data/Manda2Eng.tsv'

vocab_size = 50000 # or whatever size you want – 4k ... 50k are typical sizes

# for a complete list of options see: https://github.com/google/sentencepiece/blob/master/doc/options.md
# be sure to include ' ' at between parameters
params = ''
params = params + ' --input=' + fn					# only required parameters
params = params + ' --model_prefix=./models/spm'	# specify an output name

# optional parameters
params = params + ' --vocab_size=' + str(vocab_size)	# default: 8000
#params = params + ' --input_sentence_size=5000000'
#params = params + ' --seed_sentencepiece_size=2000000'
#params = params + ' --shuffle_input_sentence=true'		# not important if sentence_size > number of sentences in files
#params = params + ' --character_coverage=1.0'			# 1.0 will include all characters in vocab, default: 0.9995
#params = params + ' --character_coverage=0.99997'		# 1.0 will include all characters in vocab, default: 0.9995

# Normalization rule name. Choose from nfkc or identity, etc.  default: "nmt_nfkc"
params = params + ' --normalization_rule_name=nfkc_cf'	# Normalization Form Compatibility Composition with case folding
# params = params + ' --normalization_rule_name=nfkc'		# no case folding

# Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.
params = params + ' --model_type=unigram'		# default
# params = params + ' --model_type=bpe'

# params = params + ' --pad_id=3'					# include <pad> control symbol
params = params + ' --control_symbols=<bos_en>,<bos_sp>,<bos_chn>,<bos>,<eos>'
# params = params + ' --control_symbols=<ara>,<cmn>,<eng>,<rus>,<spa>'

# shrinking_factor pieces with respect to the loss -- default: 0.75 -- closer to 1.0 should produce a better model, but take longer
#params = params + ' --shrinking_factor=0.5'
#params = params + ' --shrinking_factor=0.75'	# default
#params = params + ' --shrinking_factor=0.9'
params = params + ' --shrinking_factor=0.95'	# limit

# trains a vocabulary and write 2 files: ./models/spm.model and ./models/spm.vocab
train = True
#train = False
if train:
	spm.SentencePieceTrainer.Train(params)

# load and test of model
fn_model = './models/spm.model'
    
sp = spm.SentencePieceProcessor()
sp.Load(fn_model)

print(sp.__dict__)
print(sp.this)

print(sp.EncodeAsPieces('Hello world.'))
print(sp.EncodeAsIds('Hello world.'))

print()
print(sp.EncodeAsPieces('liking garage duality.'))
print(sp.EncodeAsIds('liking garage duality.'))

print()
print("tokens at 10, 30, 60, 100, 200")
print(sp.DecodeIds([0, 1, 2, 3, 4, 10, 30, 60, 61, 70, 100, 200])) # just some random tokens

print(sp.piece_to_id('<bos_en>'))
print(sp.piece_to_id('<bos_chn>'))
print(sp.piece_to_id('<bos_sp>'))
