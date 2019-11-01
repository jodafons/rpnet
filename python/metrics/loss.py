

import tensorflow.keras.backend as K
import tensorflow as tf
import sklearn

def sp(y_true, y_pred, num_thresholds=1000):

  # Calculate roc curve
  fa, pd, thresholds = sklearn.metrics.roc_curve(K.eval(y_true), K.eval(y_pred))
  sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
  knee = np.argmax(sp)
  return K.variable(1.-sp[knee])



def auc(y_true, y_pred):
  auc = tf.keras.metrics.AUC(y_true, y_pred)[1]
  K.get_session().run(tf.local_variables_initializer())
  return auc
