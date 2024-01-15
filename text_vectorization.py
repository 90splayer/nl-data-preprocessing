#Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
import pandas as  pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print(train.head(2))
print(test.head(2))
print(sample.head(2))

#Set the seed value
SEED = 4243

#Define the label
TARGET = "target"

def plot_null_values(df):
    print("Total number of samples in the data =",df.shape[0])
    sns.heatmap(df.isnull().sum().to_frame(),
                annot=True,
                fmt="d",
                cmap="crest"
                )
    plt.title("Heatmap of NULL values")

#Use the plot_null_values function to plot Null values in the train dataset
print("\t\t\t\tNULL values in Train Dataset")
plot_null_values(train)

#Replace the Null values with str(0)
train.fillna("0",inplace=True)
test.fillna("0",inplace=True)

#Target analysis
sns.histplot(x=train[TARGET])
plt.title("Distribution of target values")

#Group the train DataFrame by "keyword" column and
# count the "target" Series group values
keyword = train.groupby("keyword")["target"].count()

#Convert the above Groupby Object to DataFrame and sort the values
keyword_df = pd.DataFrame(data={"keyword":keyword.index, "count":keyword.values}).sort_values(by=["count"],ascending=False)
keyword_df

#Let's plot the "key" DataFrame
plt.figure(figsize=(12,5))
#Limit the data to top 30 keywords by .head() method
sns.barplot(data=keyword_df.head(30), x = 'keyword',y = 'count')
plt.xticks(rotation = 50)
plt.ylabel('count')
plt.title('Top 30 keywords on Tweets');

#Group the train DataFrame by "keyword" column and
# count the "target" Series group values
location = train.groupby("location")["target"].count()

#Convert the above Groupby Object to DataFrame and sort the values
location_df = pd.DataFrame(data={"location":location.index, "count":location.values}).sort_values(by=["count"],ascending=False)
location_df

#Let's plot the "location" DataFrame
plt.figure(figsize=(12,5))

#Limit the data to top 30 locations by .head() method
sns.barplot(data=location_df.head(30), x = 'location',y = 'count')
plt.xticks(rotation = 50)
plt.ylabel('count')
plt.title('Top 30 locations on Tweets');

#Let's plot the "location" DataFrame
plt.figure(figsize=(12,5))

#Limit the data to top 30 locations by .head() method
sns.barplot(data=location_df.head(30), x = 'location',y = 'count')
plt.xticks(rotation = 50)
plt.ylabel('count')
plt.title('Top 30 locations on Tweets')

#Define a function to get the maximum length of words in each column
def check_max_length_sentence(df):
    max_length = 0
    for text in df["text"]:
        if len(text) > max_length:
            max_length = len(text)
    print("Maximum length of Text column is:",max_length)
    
    max_length = 0
    for text in df["location"]:
        if len(text) > max_length:
            max_length = len(text)
    print("Maximum length of Location column is:",max_length)
    
    max_length = 0
    for text in df["keyword"]:
        if len(text) > max_length:
            max_length = len(text)
    print("Maximum length of Keyword column is:",max_length)

#Maximum lengths of columns in train data
check_max_length_sentence(train)

#Maximum lengths of columns in test data
check_max_length_sentence(test)

def get_sentence_lengths(df):
    df2 = pd.DataFrame(columns=["length"])
    i = 0
    for text in df["text"]:
        df2.loc[i,"length"] = len(text)
        i+=1
    return df2

def plot_sentence_lengths(df):
    lengths_df = get_sentence_lengths(df)
    bins = range(0,161,20)
    plt.hist(lengths_df["length"], bins=bins,alpha=0.3)
    #sns.countplot(data=lengths_df, x="length")
    plt.title("Distribution of sentence lengths")

#Use the plot_sentence_lengths function on train dataset
plot_sentence_lengths(train)

#Use the plot_sentence_lengths function on test dataset
plot_sentence_lengths(test)

#Remove the "id" column form the train DataFrame
train.pop("id")

train.head(3)
test.head(2)

train.drop(['keyword','location'],axis=1,inplace=True)
test.drop(['location','keyword'],axis=1,inplace=True)

#Shuffle the DataFrame
train = shuffle(train, random_state=SEED)
train = shuffle(train, random_state=int(SEED/2))
train.head()

#Convert train data to tf.data.Dataset object
BATCH = 32
#Concatenate the text data of the columns "keyword", "location", "text"
train_tf = tf.data.Dataset.from_tensor_slices(
    (train['text'], train[TARGET]))

#Convert the data into batch
train_tf = train_tf.shuffle(int((SEED*13)/8)).batch(BATCH)

#Convert test data to tf.data.Dataset object
test_tf = tf.data.Dataset.from_tensor_slices(test['text'])

#Convert the data into batch
test_tf = test_tf.batch(BATCH)

max_length = 165
max_tokens = 20_000

#Instantiate the TextVectorization layer
text_vectorization = layers.TextVectorization(max_tokens=max_tokens,
                                              output_mode='int',
                                              output_sequence_length=max_length
                                             )

#Learn the vocabulary
text_vectorization.adapt(train_tf.map(lambda twt, target: twt))

#Get the vocabulary
vocab = text_vectorization.get_vocabulary()
print("Vocabulary size =",len(vocab))

#Convert the list object to NumPy array for decoding the vectorized data
vocab = np.array(vocab)

#Vectorize the train dataset
train_tf = train_tf.map(lambda twt, target: (text_vectorization(twt), target),
                   num_parallel_calls=tf.data.AUTOTUNE)

#Vectorize the test dataset
test_tf = test_tf.map(lambda twt: text_vectorization(twt),
                      num_parallel_calls=tf.data.AUTOTUNE)

#Define a function to print the tokenized data
def print_sample(data_obj):
    for sample, traget in data_obj:
        #Print the first item
        print("1st sample:",sample[0].numpy())
        print("\n")
        #Print the second item
        print("2nd sample:",sample[1].numpy())
        print("\n")
        #Print the third item
        print("3rd sample:",sample[2].numpy())
        print("\n")
        break

#Print the tokenized data
print_sample(train_tf)

#Print the vectorized tweet and the decoded tweet
for tx in train_tf:
    print("\t\t\t\tVectorized Tweet:\n",tx[0][0])
    print("\n\n\t\t\t\tDecoded Tweet:\n", " ".join(vocab[tx[0][0].numpy()]))
    break

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_prof = keras.Sequential(
            [layers.Dense(dense_dim, activation='relu'),
             layers.Dense(embed_dim),]
                                )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.max_pool1 = layers.GlobalMaxPooling1D()
        
    # Define all methods where forward pass is implement
    
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
            
        # Apply the attention layers
        attention_output = self.attention(inputs, inputs, attention_mask = mask)
        #Normalizaion the data
        proj_input = self.layernorm_1(inputs + attention_output)
        #Apply Dense Layer
        proj_output= self.dense_prof(proj_input)
        #Normalization the data and return it
        return self.layernorm_2(proj_input + proj_output)
    
    #Define configuration methods
    def get_config(self):
        config = super().get_config()
        config_update({
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
            'dense_dim': self.dense_dim
        })
        
        return config
    
# Implement Positional embedding as a subclass layers
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_legth = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_embeddings = layers.Embedding(
            input_dim = input_dim, output_dim = output_dim)
        self.positional_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0, limit=length,delta  =1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.positional_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask = None):
        return tf.math.not_equal(inputs , 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim' : self.output_dim,
            'sequence_lenght': self.sequence_legth,
            'input_dim': self.input_dim
        })
        return config
    
# Constract the model
# Define the input
inputs = keras.Input(shape=(None,), dtype='int64')

# apply positional embeddings
pos_embed = PositionalEmbedding(sequence_length=165,
                               input_dim=20_000,
                                output_dim=256)(inputs)

# apply the encoder
encoder = TransformerEncoder(embed_dim=256,
                            dense_dim=32,
                            num_heads =8)(pos_embed)
x = layers.GlobalMaxPooling1D()(encoder)
x = layers.Dropout(0.5)(x)
output = layers.Dense(units=1, activation='sigmoid')(x)
model = keras.Model(inputs= inputs, outputs = output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.98,epsilon=1e-9),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy']
             )

model.summary()

# plot model
keras.utils.plot_model(model=model,
                      to_file='model.png',
                      show_shapes = True,
                      show_layer_names=True,
                      expand_nested=True,
                      show_layer_activations=True,
                      show_trainable=True)

# Define validation Data
val_size = int(0.25 * len(train_tf))

# Split the dta into training and validation
validation_data = train_tf.take(val_size)
train_data = train_tf.skip(val_size)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_gru = EarlyStopping(monitor='val_loss', patience=3)


history = model.fit(train_data,
                    epochs=150,
                    validation_data=validation_data,
                    callbacks=early_stopping_gru
                    )

model.evaluate(validation_data)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

Epochs = len(acc)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(Epochs),acc, label='Training Accuracy')
plt.plot(range(Epochs),val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(Epochs),loss, label='Training loss')
plt.plot(range(Epochs),val_loss, label='Validation loss')
plt.legend(loc='lower right')
plt.title('Training and Validation loss')

# Plotting the training and validation loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting the training and validation accuracy
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

prdictions = model.predict(test_tf)

thresholds = 0.5 
# convert the float values to binary
final_predictions = [1 if i > thresholds else 0 for i in prdictions]
final_predictions[:10]

sample

submisssion = pd.DataFrame(columns=['id','target'])
submisssion['target'] = final_predictions
submisssion['id'] = test.id
submisssion

submisssion.to_csv('submission.csv',index=False)
