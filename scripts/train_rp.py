


# import rp layer and sp metrics
from rpnet import sp, monit, RingerRp

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

# matplot lib for plots
import matplotlib.pyplot as plt



# create the cv and split in train/validation samples just for sp validation
file = '../data/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_V97_et0_eta0.slim.npz'
raw_data = dict(np.load(file))
data = raw_data['data'][:,1:101]
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




# create the model
model = Sequential()
model.add(RingerRp(  input_shape=(100,), name='RingerRp') )
#model.add(Dense(5, input_shape=(100,), activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform', name='Dense'))
model.add(Dense(10, activation='tanh'  , kernel_initializer='random_uniform', bias_initializer='random_uniform', name='Hidden' ))
#model.add(Dropout(0.25))
model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform', name='Output'))
model.add(Activation('sigmoid',name='Activation'))





# create the optimizer
#optimizer = LRMultiplier('adam', {'Hidden': 1, 'Output': 1, 'RingerRp':2}, name='Adam' ),
optimizer='adam'

# compile the model
model.compile( optimizer,
               loss = 'binary_crossentropy',
               metrics = ['acc'],
              )


sp_obj = sp(patience=25, verbose=True, save_the_best=True)
sp_obj.set_validation_data( (x_val, y_val) )
monit = monit()

# train the model
history = model.fit(x, y,
          epochs          = 1000,
          batch_size      = 1024,
          verbose         = True,
          validation_data = (x_val,y_val),
          callbacks       = [monit, sp_obj],
          class_weight    = compute_class_weight('balanced',np.unique(y),y),
          shuffle         = True)



weights = monit.getWeights()

from pprint import pprint
alphas = []; betas = []
for w in weights:
  alphas.append( w[0][0] ); betas.append(w[1][0])



# plot all alphas and betas along the training
plt.scatter(alphas, betas )
plt.xlabel('alphas')
plt.ylabel('betas')
plt.savefig('alphas_and_betas.pdf')


# The network output
#output = get_output_from( model, 'Activation', x )
#output = get_output_from( model, 'RingerRp', x )








