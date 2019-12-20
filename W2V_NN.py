#!/usr/bin/env python3

from W2V_utilities import *
from ML_sklearn import *

# load data
RESULT_POS_PATH = './Data/produced/pp_pos_otpl_nd.txt'
RESULT_NEG_PATH = './Data/produced/pp_neg_otpl_nd.txt'
RES_PATH = './Data/produced/pp_test_otpl.txt'

# load the data files = list with each line being a tweet
result_pos = open(RESULT_POS_PATH, "r").read().splitlines()
result_neg = open(RESULT_NEG_PATH, "r").read().splitlines()
test_set = open(RES_PATH, "r").read().splitlines()

# variable definition
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

# create test and train dataframes
train_df = build_train_df(result_pos,result_neg)
test_df = build_test_df(test_set)

# split in test and train
data_train, data_test = train_test_split(train_df, test_size=0.10, random_state=42)
TRAINING_VOCAB = get_vocab(data_train)
TEST_VOCAB = get_vocab(data_test)

# word embedding
word2vec_path = './Data/google_pretrained/GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)

# tokenization and padding
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Tweet"].tolist())

training_sequences,train_word_index,train_nn_data = get_sequences_idx_padding(tokenizer,data_train,MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

X_train = train_nn_data

y_train = data_train.Sentiment.values
y_test = data_test.Sentiment.values

test_sequences,test_word_idx,test_data = get_sequences_idx_padding(tokenizer,data_test,MAX_SEQUENCE_LENGTH)

X_test = test_data

# pca
X_train, X_test = scale_pca(X_train,X_test)
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

# NN
clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(2, X_train.shape[1]), random_state=4, verbose=False, learning_rate='constant')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
compute_accuracy(y_test, y_pred)
print("is the predicted accuracy")

r_TEST_VOCAB = get_vocab(test_df) 
Test_sequences,Test_word_idx,Test_nn = get_sequences_idx_padding(tokenizer,test_df,MAX_SEQUENCE_LENGTH)
r_y_pred = clf.predict(Test_nn)

#create submission
test_id = test_df['Tweet_submission_id'].to_numpy()
create_csv_submission(test_id,r_y_pred, "./Submissions/NN_W2V_SUB.csv")