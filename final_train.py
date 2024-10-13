# %%
import torch
import json
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import gensim #For word2vec
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import time
import nltk
import random
from numpy.random import choice as randomchoice
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import pickle

# %%
train_file = sys.argv[1];
val_file = sys.argv[2];
# train_file = 'data/A2_train.jsonl'
# val_file = 'data/A2_val.jsonl'
train_begin_time = time.time();

# %%
with open(train_file) as f:
    train_data = [json.loads(line) for line in f]

# %%
with open(val_file) as f:
    val_data = [json.loads(line) for line in f]

# %%
train_data.extend(val_data) #So we have more data to train on.

# %%
tokenize_func = nltk.tokenize.WordPunctTokenizer().tokenize
punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError: #Classic way to get is_numeric
        return False
def tokenize(sentence, with_num = True):
    s = tokenize_func(sentence.lower());
    # for i in range(len(s)):
    #     s[i] = s[i].strip(punctuations); #removing all punctuations right here.
    #     if(s[i] == ''):
    #         s.pop(i); #removing punctuations. If the word is empty, remove it
    if(with_num):
        return s; #If with_num is true, return the sentence as it is, without converting the numbers to <NUM>
    for i in range(len(s)):
        if(is_numeric(s[i])):
            s[i] = '<NUM>'; #replaces numbers with <NUM>
    return s;

def tokenize_with_num(sentence): #just tokenizes normally. No replacement of numbers
    s = tokenize_func(sentence.lower());
    return s;

def get_embedding_index(sentences, model):
    return ([tokenize_and_get_embedding_index(sentence, model) for sentence in sentences]);

def tokenize_and_get_embedding_index(sentence, vocab, with_num = False):
    s = tokenize(sentence, with_num = with_num);
    # FOr now testing with No UNK, Later will have to add UNK
    tens = torch.tensor([vocab.get(word, vocab['<UNK>']) for word in s if (word not in punctuations and word in vocab)]); #if the word is not in the punctuation, only then we add it.
    if(len(tens) == 0):
        return torch.tensor([vocab.get(word, vocab['<UNK>']) for word in s]) #using UNK in this case.
    else:
        return tens;

# %%
questions = [[word for word in tokenize(d['question']) if word not in punctuations] for d in train_data]
columns = []
for i in range(len(train_data)):
    for k in range(len(train_data[i]['table']['rows'][0])):
        cur_sentence = [];
        for j in range(len(train_data[i]['table']['rows'])):
            cur_sentence.extend([word for word in tokenize(train_data[i]['table']['rows'][j][k])]) # if word not in punctuations])
        columns.append(cur_sentence)
    for k in range(len(train_data[i]['table']['cols'])):
        columns.append([word for word in tokenize(train_data[i]['table']['cols'][k])]) # if word not in punctuations])

# %%
## SET EPOCHS TO 10 IN WORD2VEC #CHANGE

# %%
#Pad :
from collections import Counter
pad = '<PAD>'
unk = '<UNK>'
max_len = max([len(q) for q in questions])
padded_questions = [q + [pad] * min(max_len - len(q), 5) for q in questions]
all_sentences = questions + columns# + padded_questions[:5000]
wordvec = Word2Vec(vector_size=200, window=5, min_count=1, workers=5, epochs=8)

# %%
all_words = [word for sentence in all_sentences for word in sentence]
word_freq = Counter(all_words);
freq_threshold = 3; #replacing all words that appear less than this times with <UNK>
all_sentences_with_unk = [[word if word_freq[word] >= freq_threshold else unk for word in sentence] 
                     for sentence in all_sentences if any(word_freq[word] < freq_threshold for word in sentence)]

all_sentences = all_sentences + all_sentences_with_unk + padded_questions[:800]; #adding questions that have a PAD token next to them.

# %%


# %%
wordvec.build_vocab(all_sentences)
wordvec.train(all_sentences, total_examples=wordvec.corpus_count, epochs=wordvec.epochs)
wordvec.save("word2vec_for_column.model")
wordvec.save("word2vec_for_row.model");
padding_idx = wordvec.wv.key_to_index[pad]
print("word2vec trained on a corpus of ", len(all_sentences), " sentences, with a vocab of ", len(wordvec.wv.key_to_index), " words")

# %%
del all_sentences, padded_questions, questions, columns #deleting these for space conservation

# %%
negative_probability = 0.8;
class custom_dataset(Dataset):
    def __init__(self, dataset = None ,json_path = None):    
        if(dataset == None and json_path == None):
            raise Exception("custom dataset got no data path or dataset");
        if(dataset != None):
            self.data = dataset; #a list.
        else:
            with open(json_path) as f:
                self.data = [json.loads(line) for line in f] #loads it from the file in the other case.
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #we will return a dictionary containing the question the table and the answer
        Q = tokenize_and_get_embedding_index(self.data[index]['question'], wordvec.wv.key_to_index);
        label = 1.0 if random.random() > negative_probability else 0.0;
        if(label == 0.0):
            A =  randomchoice(self.data[index]['table']['cols']); #chooses one of the columns as a negative example.
            while A == self.data[index]['label_col'][0]:
                A = randomchoice(self.data[index]['table']['cols']); #Setting it to a different column.
        else:
            A = self.data[index]['label_col'][0]
        A = tokenize_and_get_embedding_index(A, wordvec.wv.key_to_index)
        return {'question': Q, 'answer': A, 'label': label}

def collate_fn(batch):
    questions = [item['question'] for item in batch] #becomes a list of list of integers
    answers = [item['answer'] for item in batch] #becomes a list of list of integers
    labels = [item['label'] for item in batch] #becomes a list of list of integers
    qlen = [len(q) for q in questions]
    alen = [len(a) for a in answers]
    if 0 in qlen or 0 in alen:
        print('Zero length detected')
        for i in range(len(qlen)):
            if qlen[i] == 0:
                print("q 0 len: ",batch[i]['question'])
            for i in range( len(alen)):
                if alen[i] == 0:
                    print("col answer 0 len: ", batch[i]['answer'])
        raise Exception("bruh")
    questions = pad_sequence(questions, batch_first=True, padding_value=wordvec.wv.key_to_index['<PAD>']).to(torch.int32)# , dtype=torch.int32
    answers = pad_sequence(answers, batch_first=True, padding_value=wordvec.wv.key_to_index['<PAD>']).to(torch.int32)#, dtype=torch.int32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return {'question': questions, 'answer': answers, 'label': labels, 'qlen': qlen, 'alen': alen}

# %%
import torch.nn.utils.rnn as rnn_utils

class LSTM_on_words(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab, dropout=0.3):
        super(LSTM_on_words, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(vocab), padding_idx=padding_idx,freeze=True).to(device);
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=False).to(device);

    def forward(self, x, x_lengths):
        # Embedding
        out = self.embedding(x)
        # Pack padded sequence
        lengths = x_lengths.detach().cpu().numpy();
        out = rnn_utils.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False).to(device);
        out, (hidden, cell) = self.lstm(out)
        return out, (hidden, cell);

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.ReLU = nn.ReLU(inplace=False)
        self.layers = [self.fc1,self.ReLU, self.fc2,self.ReLU, self.fc3]
        self.all_layers = nn.Sequential(*self.layers)
    def forward(self, x):
        out = self.all_layers(x)
        return out

class Column_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab, dropout=0.4):
        super(Column_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.question_lstm = LSTM_on_words(input_size, hidden_size, num_layers, vocab=vocab, dropout=dropout);
        self.column_lstm = LSTM_on_words(hidden_size, hidden_size, num_layers, vocab=vocab, dropout=dropout);
        self.final = FeedForward(2*input_size, hidden_size, 1).to(device);
    def forward(self, question, qlen, column_name, alen):
        packed_outputs, (question, question_cell_state) = self.question_lstm(question, qlen);
        packed_col_outputs, (column_name, column_cell_staes) = self.column_lstm(column_name, alen);
        # print(question.shape, column_name.shape)
        out = torch.cat((question[-1], column_name[-1]), 1); #Concatenate along the first dimension
        #Now out is 32*200 size.
        out = self.final(out);
        return out;

# %%
try:
    print("GPU: ", torch.cuda.get_device_name(device))
except:
    print("no GPU found brah");
# torch.autograd.set_detect_anomaly(True)

# %%
model = Column_Decoder(200, 200, 2, wordvec.wv.vectors, dropout=0.5).to(device) #Using 2 layers in my LSTM now

# %%
batch_size = 256;
learning_rate = 0.001;
num_epochs = 350;
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([5.0]).to(device));
DL = DataLoader(custom_dataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)

# %%
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename);
    model.load_state_dict(checkpoint['model_state_dict']);
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
    epoch = checkpoint['epoch'];
    loss = checkpoint['loss'];
    return model, optimizer, epoch, loss;

def store_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            }, filename);

# %%
epoch = 0

# %%
model.train()
running_loss = 0;
try:
    print(epoch);
except:
    epoch = 0; #setting epoch to 0 if it doesn't exist.
for epoch in range(epoch, num_epochs):
    mean_loss = 0; steps = 0;
    for i,d in enumerate(DL):
        optimizer.zero_grad();
        question = d['question'].to(device);
        answer = d['answer'].to(device);
        label = d['label'].to(device);
        qlen = torch.tensor(d['qlen'])
        alen = torch.tensor(d['alen'])
        try:
            output = model(question, qlen, answer, alen);
        except Exception as e:
            # print("input question:",question);
            # print("columns: ", answer);
            raise Exception(e);
        loss = criterion(output, label.view(-1, 1));
        # print(output)
        loss.backward();
        optimizer.step();
        #calculate norm of the gradients = 
        total_norm = 0
        for p in model.parameters():
            if(p.grad == None):
                continue;
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        mean_loss += loss.item();
        steps += 1;
        # break;
        #if(i%20 == 0):
        #    print("batch ", i, "running mean loss: ", mean_loss/steps, end = '    \r')
    running_loss = mean_loss/steps if running_loss == 0 else (mean_loss*0.1/steps + 0.9*running_loss);
#     if(epoch%1000 == 0):
#         store_checkpoint(model, optimizer, epoch, running_loss, "checkpoint" + str(epoch) + ".pth")
    if(epoch%200 == 0):
        store_checkpoint(model, optimizer, epoch, running_loss, "base_checkpoint_final.pth");
    print("Epoch ", epoch, "loss: ", mean_loss/steps, "running_loss:", running_loss);
    # break;

# %%
torch.save(model.state_dict(), 'column_model_state_dict_with_num.pth'); #stores the model 
# will also save at the end as a pickle file
# %%
# %%
Val_loader = DataLoader(custom_dataset(val_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %%
def calculate_column_accuracy(model, dataloader):
    model.eval();
    total_samples = 0; total_correct = 0;
    for samples in range(5):
        for d in DL:
            question = d['question'].to(device);
            answer = d['answer'].to(device);
            label = d['label'].to(device);
            qlen = torch.tensor(d['qlen'])
            alen = torch.tensor(d['alen'])
            output = model(question, qlen, answer, alen);
            outs = torch.tensor([1 if i > 0 else 0 for i in output]).to(device)
            acc = torch.sum(outs == label); 
            total_correct += acc.item();
            total_samples += label.shape[0];
    print("Val acc: ",total_correct/total_samples, ",", total_correct, " out of ", total_samples);
    model.train();

# %%
calculate_column_accuracy(model, Val_loader) #we calculate the accuracy only for fun

# %%
def predict_column(model, datadict):
    question = tokenize_and_get_embedding_index(datadict['question'], wordvec.wv.key_to_index);
    if(len(question) <= 0):
        print(datadict['question'])
    cols = [];
    question_batch = [];
    for j in range(len(datadict['table']['cols'])):
        cols.append(tokenize_and_get_embedding_index(datadict['table']['cols'][j], wordvec.wv.key_to_index));
        question_batch.append(question);
    #Now we use our model on this 
    col_lengths = torch.tensor([len(i) for i in cols])
    question_lengths = torch.tensor([len(question) for q in question_batch])
    questions = torch.tensor(pad_sequence(question_batch, batch_first=True, padding_value=wordvec.wv.key_to_index['<PAD>'])).to(torch.int32).to(device)# , dtype=torch.int32
    cols = torch.tensor(pad_sequence(cols, batch_first=True, padding_value=wordvec.wv.key_to_index['<PAD>'])).to(torch.int32).to(device)
    #now we run the model on this and get the predictions
    outs = model(questions, question_lengths, cols, col_lengths); 
    return torch.argmax(outs);
def calculate_column_accuracy_over_all(model, dataset):
    model.eval();
    correct = 0; total = 0;
    for i in range(len(dataset)):
        total += 1;
        outs = predict_column(model, dataset[i]);
        val = outs.item();
        if(dataset[i]['label_col'][0] == dataset[i]['table']['cols'][val]):
            correct += 1;
        #print('got :', val, " ", dataset[i]['table']['cols'][val], " and needed ", dataset[i]['label_col'])
        #break;
#         print("running acc:", correct/total, " done: ",total,  end = "                  \r")
    print("\ntotal acc: ", correct/total)
    model.train();
calculate_column_accuracy_over_all(model, val_data)
# %%
wordvec_row = wordvec # CHANGE this to get new wordvec embeddings. but during training these will remain the same.
# %%

# %%
def tokenize_inside_list_and_get_embedding(sentences, vocab, with_num = True):
    s = [];
    for i in range(len(sentences)):
        s.append(tokenize_and_get_embedding_index(sentences[i], vocab, with_num = with_num));
    return s;

# %%
class Row_dataset(Dataset):
    def __init__(self, json_path,length = 4000, offset = 0):    
        with open(json_path) as f:
            self.data = [json.loads(line) for line in f]
        self.length = length;
        random.shuffle(self.data)
        self.offset = offset;
    def __len__(self):
        return min(len(self.data), self.length); #for now only using the first 100 samples for checking if the model is working. It should overfit if it is working.
        return len(self.data)
    def __getitem__(self, index):
        index = (index + self.offset)%len(self.data); #offset the index by a certain value, usually the previous full value.
        Q = tokenize_and_get_embedding_index(self.data[index]['question'], wordvec_row.wv.key_to_index, with_num = True);
        #Q is sent as it is, then we send it all the columns as well, and only some of the rows.
        columns = [tokenize_and_get_embedding_index(self.data[index]['table']['cols'][i], wordvec_row.wv.key_to_index, with_num = True) for i in range(len(self.data[index]['table']['cols']))];
        #keep only 5 rows here, but randomly 5
        label_rows = self.data[index]['label_row'];
        row_ind_neg_list = [];
        for i in range(len(self.data[index]['table']['rows'])):
            if i not in label_rows:
                row_ind_neg_list.append(i);
        num_neg = min(10, len(row_ind_neg_list)); #with each correct row sample we will send 4 negative examples
        neg_list = random.sample(row_ind_neg_list, num_neg);
        #now we need to get the rows out of these indices
        pos_rows = [tokenize_inside_list_and_get_embedding(self.data[index]['table']['rows'][i], wordvec_row.wv.key_to_index, with_num = True) for i in label_rows];
        neg_rows = [tokenize_inside_list_and_get_embedding(self.data[index]['table']['rows'][i], wordvec_row.wv.key_to_index, with_num = True) for i in neg_list];
        labels = [1.0]*len(pos_rows) + [0.0]*len(neg_rows);
        total_rows = pos_rows + neg_rows;
        return Q, columns, total_rows, labels
    

def collate_row_fn(batch):
    Q, columns, total_rows, labels = zip(*batch);
    qlen = [len(q) for q in Q]
    clen = [[len(c) for c in col] for col in columns]
    row_len = [[[len(r) for r in cell] for cell in row] for row in total_rows]
    Q = pad_sequence(Q, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32)# , dtype=torch.int32
    columns = [pad_sequence(col, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32) for col in columns]
    total_rows = [[pad_sequence(row, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']) for row in rows] for rows in total_rows]
    singular_row = []; 
    ends = [];
    new_row_lens = [];
    for i in total_rows:
        for j in i:
            singular_row.extend(j); 
            ends.append(len(singular_row));
    for i in row_len:
        for j in i:
            new_row_lens.extend(j);
    # print(singular_row)
    singular_row = pad_sequence(singular_row, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32)
    #Now we convert all the rows into just one row.
    return {'question': Q, 'columns': columns, 'rows': singular_row, 'qlen': qlen, 'clen': clen, 'row_len': new_row_lens, 'labels': labels, 'ends': ends}

# %%
row_dataloader = DataLoader(Row_dataset(train_file), batch_size=1, shuffle=True, collate_fn=collate_row_fn)

# %%
class FeedForward_row(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward_row, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.ReLU = nn.ReLU(inplace=False)
        self.layers = [self.fc1,self.ReLU, self.fc2,self.ReLU, self.fc3]
        self.all_layers = nn.Sequential(*self.layers)
    def forward(self, x):
        out = self.all_layers(x)
        return out

# %%
class row_detecter_model(nn.Module):
    def __init__(self, hidden_size=200, input_size=200, num_layers=2):
        super(row_detecter_model, self).__init__();
        #self.column_attention_decider = Column_Decoder(200, 200, 2, wordvec_row.wv.vectors).to(device) #decides the attention for each LSTM ran on the rows.
        #Broke the column attention decider into 3 parts as well, for faster training and inference.
        self.question_col_lstm = LSTM_on_words(input_size, hidden_size, num_layers, vocab=wordvec_row.wv.vectors, dropout=0);
        self.column_lstm = LSTM_on_words(hidden_size, hidden_size, num_layers, vocab=wordvec_row.wv.vectors, dropout=0);
        self.FFCol = FeedForward(400, 200, 1).to(device);
        
        self.Question_lstm = LSTM_on_words(200, 200,2,vocab=wordvec_row.wv.vectors).to(device);
        self.row_lstm = LSTM_on_words(200, 200, 2, vocab=wordvec_row.wv.vectors);
        self.FFrow = FeedForward_row(400, 200, 1).to(device);

# %%
row_model = row_detecter_model().to(device);

# %%
loss_criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([10.0]).to(device));
optimizer_row = optim.Adam(row_model.parameters(), lr=0.0002, eps=1e-7);
#optimizer_row = optim.SGD(row_model.parameters(), lr = 0.001, momentum=0.9)

# %%
row_dataloader.dataset.offset = 0
#first we shuffle the input data that we have.
epoch = 0;

# %%

# %%
import time
running_loss = 0; alpha = 0.8
start = time.time()
row_dataloader.dataset.length = len(train_data);
row_model.train()
# row_dataloader.dataset.offset = 0;
epoch = 0;
try:
    print(epoch);
except:
    epoch = 0; #setting epoch to 0 if it doesn't exist.
for epoch in range(epoch, 50): #Only 250 epochs because we do not know the speed of their servers.
    mean_batch_loss = 0;
    total = 0;
    # if(epoch % 5 == 0):
    # row_dataloader.dataset.offset += 20;
    #in every epoch we move our dataset forward by 10 samples.
    #so to cover all 25000 samples we need 2500 epochs which is still too much.
    #so new plan is to move 
    elapsed = time.time() - train_begin_time; #total time elapsed between training.
    #if this is above 6*3600 - 600 seconds then we break out of the loop.
    epoch_start = time.time()
    for (batch_num,D) in enumerate(row_dataloader):
        cols = ([i.to(device) for i in D['columns']])[0]
        Q = D['question'].to(device)
        qlen = torch.tensor([Q.shape[1]])
        #now lets stack Q's together to get the same first dimension as cols.
        _, (question_embed, _) = row_model.Question_lstm(Q, qlen); #this gets us the result we need to pass in the column attention decider.
        question_embed = question_embed[-1].to(device); #using only the last one.
        _, (col_question_embed, _) = row_model.question_col_lstm(Q, qlen);
        col_question_embed = col_question_embed[-1].to(device); #using only the last output.
        Q = Q[0]
        total_rows = D['rows']#this is a list of rows. Each row is a tensor of words.
        ends = (D['ends']) #this is a list of row end points in our total_rows tensor.
        row_lens = torch.tensor(D['row_len']).to(device) #this is a list of row lengths.
        clen = torch.tensor(D['clen'][0]) #list for columns.

## Attention calculation over column names.
        _, (outs, _) = row_model.column_lstm(cols.to(device), clen.to(device)); #here the attention is calculated
        # outs = outs[-1];
        outs = outs[-1]
        col_Qstacked = col_question_embed.repeat(outs.shape[0], 1);
        x = torch.cat((col_Qstacked, outs), 1);
        outs = row_model.FFCol(x);
        attention = (F.softmax(outs, dim=0)).squeeze(1) #the attention is calculated here.
## Attention calculation End.
        
        _, (row_embed_lstm, _) = row_model.row_lstm(total_rows.to(device), row_lens.to(device)) #gets the scores for all the rows in a batch
        #row_embeddings = torch.tensor((len(ends), 200))        
        row_embeddings = torch.zeros((len(ends), 200)).to(device); #removed requires_grad = True.
        # row_embeddings = (attention @ row_embed_lstm[-1].view(len(ends), -1, 200));
        row_size = row_embed_lstm[-1].shape[0]//len(ends);
        # print(row_size, row_embeddings.shape, row_embed_lstm[-1].shape, attention.shape)
        for i in range(len(ends)):
            row_embeddings[i] = attention @ row_embed_lstm[-1][i*row_size:(i+1)*row_size];
        question_embed = question_embed.repeat(len(ends), 1);
        # print(question_embed.shape, row_embeddings.shape, len(ends))
        inp = torch.cat((question_embed,row_embeddings), 1);  
        outs = row_model.FFrow(inp)
        # print(outs) #prints a score for each row in our corpus. and now we must calculate the loss.
        loss = loss_criterion(outs, torch.tensor(D['labels']).to(device).view(-1, 1));
        optimizer_row.zero_grad();
        loss.backward();
        optimizer_row.step();
        mean_batch_loss += loss.item(); total += 1;
        if(batch_num % 5000 == 0):
            print("epoch: ", epoch, ", batch:", batch_num ,"running mean loss: ", mean_batch_loss/total, end = "                             \r");
        # if(batch_num %20 == 0):
        # if(batch_num  == 10):
        # print("average batch time: ", (time.time() - epoch_start)/total, end = "                             \r");
    running_loss = mean_batch_loss/total if running_loss == 0 else ((alpha*running_loss + (1-alpha)*mean_batch_loss/total))
    if(epoch % 50 == 0):
        store_checkpoint(row_model, optimizer_row, epoch, mean_batch_loss/total, "row_model_checkpoint.pth");
    #row_dataloader.dataset.offset += 200; #each epoch we move our dataset forward by 50 samples.
    # if((epoch+1) % 1 == 0):
    # row_dataloader.dataset.offset += 4000; #Offsetting 2000 samples to the dataset every 3 epochs.
    print("SUB EPOCH", epoch,"loss", mean_batch_loss/total, "running_loss = ",running_loss,"time taken: ", time.time() - epoch_start, "       ");""

# %%
torch.save(row_model.state_dict(), 'row_model_checkpoint.pth'); #stores the model

# %%
class Row_final_prediction_dataset(Dataset):
    def __init__(self, datadictlist, length = 8000):    
        self.data = datadictlist; 
        self.length = length;
    def __len__(self):
        return min(len(self.data), self.length); #for now only using the first 100 samples for checking if the model is working. It should overfit if it is working.
        #return len(self.data)
    def __getitem__(self, index):
        Q = tokenize_and_get_embedding_index(self.data[index]['question'], wordvec_row.wv.key_to_index, with_num = True);
        #Q is sent as it is, then we send it all the columns as well, and only some of the rows.
        columns = [tokenize_and_get_embedding_index(self.data[index]['table']['cols'][i], wordvec_row.wv.key_to_index, with_num = True) for i in range(len(self.data[index]['table']['cols']))];
        #keep only 5 rows here, but randomly 5
        label_rows = self.data[index]['label_row'];
        all_rows = [];
        for i in range(len(self.data[index]['table']['rows'])):
            all_rows.append(i); #adding their indices.
        total_rows = [tokenize_inside_list_and_get_embedding(self.data[index]['table']['rows'][i], wordvec_row.wv.key_to_index, with_num = True) for i in all_rows];
        return Q, columns, total_rows, index #not sending any labels when we are predicting, but we are sending the index of the table for easy lookup.

def collate_row_final_fn(batch):
    Q, columns, total_rows, indices = zip(*batch);
    qlen = [len(q) for q in Q]
    clen = [[len(c) for c in col] for col in columns]
    row_len = [[[len(r) for r in row] for row in rows] for rows in total_rows]
    Q = pad_sequence(Q, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32)# , dtype=torch.int32
    columns = [pad_sequence(col, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32) for col in columns]
    total_rows = [[pad_sequence(row, batch_first=True, padding_value=wordvec_row.wv.key_to_index['<PAD>']).to(torch.int32) for row in rows] for rows in total_rows]
    return {'question': Q, 'columns': columns, 'rows': total_rows, 'qlen': qlen, 'clen': clen, 'row_len': row_len, 'indices': indices}

# %%
row_predict_dataloader = DataLoader(Row_final_prediction_dataset(val_data), batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_row_final_fn)

# %%
row_model.eval();
correct = 0; total = 0; total_loss = 0;

for D in row_predict_dataloader:
    table_index = D['indices'][0];
    cols = ([i.to(device) for i in D['columns']])[0]
    Q = D['question'].to(device)
    qlen = torch.tensor([Q.shape[1]])
    #now lets stack Q's together 
    _, (col_question_embed, _) = row_model.question_col_lstm(Q, qlen);
    col_question_embed = col_question_embed[-1].to(device);# to get the same first dimension as cols.
    _, (question_embed, _) = row_model.Question_lstm(Q, qlen); #this gets us the result we need to pass in the column attention decider.
    question_embed = question_embed[-1].to(device); #using only the last one.
    Q = Q[0]
    total_rows = D['rows'][0]#this is a list of rows. Each row is a tensor of words.
    Qstacked = Q.repeat(cols.shape[0], 1);
    row_lens = torch.tensor(D['row_len'][0]).to(device); #this is a list of row lengths.
    qlen = torch.tensor([Qstacked.shape[1]]*Qstacked.shape[0])
    clen = torch.tensor(D['clen'][0]) #list for columns.
    ### Attention calculation over column names.
    _, (outs, _) = row_model.column_lstm(cols.to(device), clen.to(device)); #here the attention is calculated
    # outs = outs[-1];
    outs = outs[-1]
    col_Qstacked = col_question_embed.repeat(outs.shape[0], 1);
    x = torch.cat((col_Qstacked, outs), 1);
    outs = row_model.FFCol(x);
## Attention calculation End.
    attention = (F.softmax(outs, dim=0)) #the attention is calculated here.
    row_embeddings = torch.zeros((len(total_rows), 200)).to(device);
    for i in range(len(total_rows)):
        _, (cur_embed, _) = row_model.row_lstm(total_rows[i].to(device), row_lens[i].to(device)) #gets the scores for the current row.
        actual_embed = cur_embed[-1][0]*attention[0];
        for j in range(1, len(attention)):
            actual_embed += cur_embed[-1][j]*attention[j];
        row_embeddings[i] = actual_embed; 
        #After this we get the actual embedding for our model.
        #Now we need to run our feed forward model on this.
    question_embed = question_embed.repeat(len(total_rows), 1);
    inp = torch.cat((question_embed,row_embeddings), 1);  
    outs = row_model.FFrow(inp)
    correct_labels = train_data[table_index]['label_row'];
    labels = [0.0]*len(total_rows);
    for i in range(len(labels)):
        if i in correct_labels:
            labels[i] = 1.0;
    # print(outs) #prints a score for each row in our corpus. and now we must calculate the loss.
    labels = torch.tensor(labels).to(device);
    outs = outs.squeeze(1);
    loss = loss_criterion(outs, labels);
    total_loss += loss.item();
    # print(outs);
    #assuming only one output is required.
    ans_ind = torch.argmax(outs).item();
    total += 1;
    if(ans_ind in correct_labels):
        correct += 1;
    print("running acc: ", correct/total, ",", correct, " out of ", total, end = "                   \r")
    #print(loss.item());
    # optimizer_row.zero_grad();
    # loss.backward();
    # optimizer_row.step();
    # mean_batch_loss += loss.item(); total += 1;
    # print("loss: ", loss.item(), end = "                             \r");
    if(total >= 1000):
        break; #only calculating accuracies on 1000 samples for now.
row_model.train();
print(correct, " out of ", total, " acc: ", correct/total, "loss: ", total_loss/total);

# %%
upload_file = {};
upload_file['column_model'] = model;
upload_file['row_model'] = row_model;
upload_file['wordvec_for_column'] = wordvec;
upload_file['wordvec_for_row'] = wordvec_row;
with open("model_files.pkl", "wb") as f:
    pickle.dump(upload_file, f); #Pickling all the files together for upload to google



