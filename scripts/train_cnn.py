

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")


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


def norm1_and_cnn_reshape( data ):
	norms = np.abs( data.sum(axis=1) )
	norms[norms==0] = 1
	ret = data/norms[:,None]
	ret = np.array([data])
	return np.transpose(ret, [1,2,0])




# create the cv and split in train/validation samples just for sp validation
file = '../data/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_V97_et0_eta0.slim.npz'
raw_data = dict(np.load(file))
data = raw_data['data'][:,1:101]
data = norm1_and_cnn_reshape(data)
target = raw_data['target']
del raw_data



# Create all necessary splits to separate the data in train and validation sets
# Here, we will use only the fist "sort" just for testing
from sklearn.model_selection import StratifiedKFold, KFold
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(data,target)]
x = data [ splits[0][0] ]
y = target [ splits[0][0] ]
x_val = data [ splits[0][1] ]
y_val = target [ splits[0][1] ]




kernel_size=5
model = Sequential()
model.add(Conv1D(16, kernel_size=kernel_size, activation='relu', input_shape=(100,1) ))
model.add(Conv1D(32, kernel_size=kernel_size, activation='relu' ))
#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(64,  activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
#model.add(Dropout(0.25))
model.add(Dense(5,  activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
#model.add(Dropout(0.25))
model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('sigmoid'))




optimizer='adam'

# compile the model
model.compile( optimizer,
               loss = 'binary_crossentropy',
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








