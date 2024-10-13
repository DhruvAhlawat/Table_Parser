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
test_file = sys.argv[1];
pred_file = sys.argv[2];
# train_file = 'data/A2_train.jsonl'
# val_file = 'data/A2_val.jsonl'
test_begin_time = time.time();

# %%
with open(test_file) as f:
    test_data = [json.loads(line) for line in f]

# %%
with open("model_files.pkl","rb") as f:
    model_files = pickle.load(f)

wordvec = model_files['wordvec_for_column'];
model = model_files['column_model'];
row_model = model_files['row_model'];
wordvec_row = model_files['wordvec_for_row'];
padding_idx = wordvec.wv.key_to_index['<PAD>']
#Now we have all the files required.
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
# %%
## SET EPOCHS TO 10 IN WORD2VEC #CHANGE
# %%
#Pad :
# %%

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

# %%

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

# %%

# %%
# will also save at the end as a pickle file
# %%
# %%
# %%


# %%

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

# %%
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

# %%

#optimizer_row = optim.SGD(row_model.parameters(), lr = 0.001, momentum=0.9)

# %%
#first we shuffle the input data that we have.

# %%

# %%

# row_dataloader.dataset.offset = 0
# %%

# %%
class Row_final_prediction_dataset(Dataset):
    def __init__(self, datadictlist, length = 8000):    
        self.data = datadictlist; 
        self.length = len(datadictlist);
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
row_predict_dataloader = DataLoader(Row_final_prediction_dataset(test_data), batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_row_final_fn)


model = model.to(device);
model.eval();
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

# %%
# I will loop over all the data available and get predictions for each of them and create a json file.
# I will then use the json file to create a csv file.
output = [];
for i in range(len(test_data)):
    col = [test_data[i]['table']['cols'][predict_column(model, test_data[i]).item()]]; #this gets us the column.
    row = [0]; #for now we will just use the first row.
    cell = [[row[0], col[0]]];
    qid = test_data[i]['qid'];
    output.append({'label_col': col, 'label_row': row, 'label_cell': cell, 'qid': qid});
with open(pred_file, 'w') as f:
    for item in output:
        f.write(json.dumps(item) + '\n'); #simply write each line like this.
#we write this file, BUT then we add our row predictions to it as well, to hopefully get some better accuracy.

print("------------------- Column predictions completed --------------------");

del model; #we can delete our model and free the memory
row_model = row_model.to(device);
# %%
row_model.eval();
for D in row_predict_dataloader:
    try:
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
        if(len(total_rows) > 200):
            #then we skip this table as it is too big and it will take too much time to fix this now.
            continue;
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
##      ttention calculation End.
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
        outs = outs.squeeze(1);
        ans_ind = torch.argmax(outs).item();
        output[table_index]['label_row'] = [ans_ind];
        output[table_index]['label_cell'] = [[ans_ind, output[table_index]['label_col'][0]]]; #this is the final output
    except Exception as e:
        print("exception was raised\n\n" , e, "\n");
        print("moving onto next row prediction");
        continue;
# %%

with open(pred_file, 'w') as f:
    for item in output:
        f.write(json.dumps(item) + '\n'); #simply write each line like this.

print("------------------- Row predictions completed --------------------");