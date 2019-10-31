

# import rp layer and sp metrics
from rpnet import sp, RingerRp

# import tensorflow/keras wrapper
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten

# importkeras learning rate multipler. This will be used to apply different learning rates 
# for each layer.
from keras_lr_multiplier import LRMultiplier

# import numpy
import numpy as np

# import sklearn things
from sklearn.utils.class_weight import compute_class_weight

# rpnet utilites
from rpnet import get_output_from


def norm1( data ):
	norms = np.abs( data.sum(axis=1) )
	norms[norms==0] = 1
	return data/norms[:,None]




# create the cv and split in train/validation samples just for sp validation
file = '../data/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_V97_et2_eta0.slim.npz'
raw_data = dict(np.load(file))
data = raw_data['data'][:,1:101]
data = norm1(data)
target = raw_data['target']
del raw_data


print(data.shape)

# Create all necessary splits to separate the data in train and validation sets
# Here, we will use only the fist "sort" just for testing
from sklearn.model_selection import StratifiedKFold, KFold
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(data,target)]
x = data [ splits[0][0] ]
y = target [ splits[0][0] ]
x_val = data [ splits[0][1] ]
y_val = target [ splits[0][1] ]




kernel_size=3
model = Sequential()
model.add(Dense(5, input_shape=(100,) ,activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('tanh'))
  
  


optimizer='adam'

# compile the model
model.compile( optimizer,
               loss = 'mse',
               metrics = ['acc'],
              )


sp_obj = sp(patience=25, verbose=True, save_the_best=True)
sp_obj.set_validation_data( (x_val, y_val) )


# train the model
history = model.fit(x, y,
          epochs          = 1000,
          batch_size      = 1024,
          verbose         = True,
          validation_data = (x_val,y_val),
          callbacks       = [sp_obj],
          class_weight    = compute_class_weight('balanced',np.unique(y),y),
          shuffle         = True)







# The network output
#output = get_output_from( model, 'Activation', x )
#output = get_output_from( model, 'RingerRp', x )








