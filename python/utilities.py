
__all__ = ['get_output_from']

from tensorflow.keras.models import Model

def get_output_from( model, layer_name, data ):
  return Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict( data )
