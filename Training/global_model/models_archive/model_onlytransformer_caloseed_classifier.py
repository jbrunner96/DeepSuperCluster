## Graph Highway network
import tensorflow as tf
import numpy as np

#########################3
# Masking utils
@tf.function
def create_padding_masks(rechits):
    mask_rechits = tf.cast(tf.reduce_sum(rechits,-1) != 0, tf.float32)
    mask_cls = tf.cast(tf.reduce_sum(rechits,[-1,-2]) != 0, tf.float32)
    return mask_rechits, mask_cls

###########################

def get_dense(spec, act, last_act, dropout=0., L2=False, L1=False, name="dense"):
    layers = [] 
    for i, d in enumerate(spec[:-1]):
        if not L1 and not L2:
            layers.append(tf.keras.layers.Dense(d, activation=act, name=name+"_{}".format(i)))
        if not L1 and L2:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer=tf.keras.regularizers.L2(0.001), name=name+"_{}".format(i)))
        if not L2 and L1:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer=tf.keras.regularizers.L1(0.001),name=name+"_{}".format(i)))
        if L1 and L2:
            layers.append(tf.keras.layers.Dense(d, activation=act, kernel_regularizer=tf.keras.regularizers.L1L2(0.001,0.001),name=name+"_{}".format(i)))
        if dropout > 0.:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(spec[-1], activation=last_act, name=name+"_{}".format(i+1)))
    return tf.keras.Sequential(layers, name=name)

def get_conv1d(spec, act, last_act, dropout=0., L2=False, L1=False, name="dense"):
    layers = [] 
    for i, d in enumerate(spec[:-1]):
        if not L1 and not L2:
            layers.append(tf.keras.layers.Conv1D(filters=d,kernel_size=1, activation=act, name=name+"_{}".format(i)))
        if not L1 and L2:
            layers.append(tf.keras.layers.Conv1D(filters=d,kernel_size=1, activation=act, kernel_regularizer=tf.keras.regularizers.L2(0.001), name=name+"_{}".format(i)))
        if not L2 and L1:
            layers.append(tf.keras.layers.Conv1D(filters=d,kernel_size=1, activation=act, kernel_regularizer=tf.keras.regularizers.L1(0.001),name=name+"_{}".format(i)))
        if L1 and L2:
            layers.append(tf.keras.layers.Conv1D(filters=d,kernel_size=1, activation=act, kernel_regularizer=tf.keras.regularizers.L1L2(0.001,0.001),name=name+"_{}".format(i)))
        if dropout > 0.:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Conv1D(filters=spec[-1], kernel_size=1, activation=last_act, name=name+"_{}".format(i+1)))
    return tf.keras.Sequential(layers, name=name)



############################
# From https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
@tf.function
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        # the mask signals the elements to keep so we have to invert it
        scaled_attention_logits += ( (1 - mask) * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

########################################
#from tensorflow.keras.layers import MultiHeadAttention


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, add_axis=False, **kwargs):
    self.num_heads = num_heads
    self.d_model = d_model
    self.add_axis = add_axis
    name = kwargs.pop("name", None)
    super(MultiHeadAttention, self).__init__(name=name)

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.Wq = tf.keras.layers.Conv1D(filters=self.d_model,kernel_size=1, use_bias=False)
    self.Wk = tf.keras.layers.Conv1D(filters=self.d_model,kernel_size=1, use_bias=False)
    self.Wv = tf.keras.layers.Conv1D(filters=self.d_model,kernel_size=1, use_bias=False)
    self.dense = tf.keras.layers.Conv1D(filters=self.d_model,kernel_size=1, use_bias=False)

  def get_config():
        return {
            "num_heads" : self.num_heads,
            "d_model": self.d_model,
            "name": self.name,
            "add_axis": self.add_axis
        }

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    In case it is used for rechits allow for an additional axis
    """
    if not self.add_axis :
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    else:
        x = tf.reshape(x, (batch_size, tf.shape(x)[1], -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    obj_size = tf.shape(q)[1]
   
    if not self.add_axis:
        mask_att = mask[:,tf.newaxis,tf.newaxis,:]
    else:
        mask_att = mask[:,:,tf.newaxis, tf.newaxis,:]
    
    q = self.Wq(q)  # (batch_size, seq_len, (n_rechits,) d_model)
    k = self.Wk(k)  # (batch_size, seq_len, (n_rechits,) d_model)
    v = self.Wv(v)  # (batch_size, seq_len, (n_rechits,) d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask_att)

    if not self.add_axis:
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    else:
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])  # (batch_size, cl_size, seq_len_q (rechits), num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, obj_size, -1, self.d_model))  # (batch_size, cl_size,seq_len_q, d_model)

    # the output is not masked
    output = self.dense(concat_attention) # (batch_size, (clsize), seq_len_q, d_model)

    # To be checked if dense here is needed
    return output, attention_weights


#######################################################################################

class MultiSelfAttentionBlock(tf.keras.layers.Layer):
  def __init__(self, output_dim, num_heads, ff_dim, reduce=None, add_axis=False, **kwargs):
    name = kwargs.pop("name", None)
    super(MultiSelfAttentionBlock, self).__init__(name=name)
    self.output_dim = output_dim
    self.ff_dim = ff_dim
    self.num_heads = num_heads
    self.reduce = reduce # it can be None, sum, mean, max
    self.add_axis = add_axis
    self.activation = kwargs.pop("activation", "relu")
    self.dropout = kwargs.get("dropout", 0.)
    self.l2_reg = kwargs.get("l2_reg", False)

    self.inputW = tf.keras.layers.Conv1D(filters=self.output_dim, kernel_size=1, use_bias=False)

    # Change this 
    self.mha = MultiHeadAttention(self.output_dim, self.num_heads,name=self.name+"_mha",
                                  add_axis=self.add_axis)
    self.ffn = get_conv1d([self.ff_dim, self.output_dim], self.activation, last_act="linear",
                                    L2=self.l2_reg, dropout=self.dropout, name=self.name+"_ff")
      

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(self.dropout)
    self.dropout2 = tf.keras.layers.Dropout(self.dropout)

  def get_config():
        return {
            "output_dim" : self.output_dim,
            "ff_dim" : self.ff_dim,
            "num_heads" : self.num_heads,
            "dropout" : self.dropout,
            "name": self.name,
            "reduce": self.reduce,
            "add_axis": self.add_axis
        }
    
  def call(self, x, mask, training):
    # needed to understand if the layer is used on the clusters or on the rechits
    if not self.add_axis:
        mask_out = mask[:,:,tf.newaxis]
    else:
        mask_out = mask[:,:,:,tf.newaxis]
        
    #projecting the input on the MSA dim
    msa_input = self.inputW(x)
    attn_output, attn_weights = self.mha(msa_input,msa_input,msa_input, mask)    # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(msa_input + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    output_block = self.layernorm2(out1 + ffn_output) * mask_out  # (batch_size, input_seq_len, d_model)

    # Now the aggregation 
    if self.reduce == "sum":
        return  tf.reduce_sum(output_block, -2), attn_weights
    if self.reduce == "mean":
        N_nodes = tf.reduce_sum(mask,-1)[:,tf.newaxis]
        return tf.math.divide_no_nan( tf.reduce_sum(output_block, -2), N_nodes), attn_weights
    if self.reduce == "max":
        return tf.reduce_max(output_block, axis=-2), attn_weights
    else:
        #just return all the nodes
        return output_block, attn_weights

    return out2, attn_weights

     

############################

class RechitsTransformer(tf.keras.layers.Layer):
    '''Transformer for rechits feature extraction
    A single features vector of dimension output_dim is built from arbitrary list of rechits. '''
    def __init__(self, input_dim, output_dim, 
                 num_transf_layers, num_transf_heads,
                 transf_ff_dim, *args, **kwargs):
        self.activation = kwargs.pop("activation", tf.keras.activations.relu)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dropout = kwargs.pop("dropout", 0.01)
        self.l2_reg = kwargs.pop("l2_reg", False)
        self.num_transf_heads = num_transf_heads
        self.num_transf_layers = num_transf_layers
        self.transf_ff_dim = transf_ff_dim
    
        super(RechitsTransformer, self).__init__(*args, **kwargs)

        self.transformer = [
            MultiSelfAttentionBlock(output_dim=self.output_dim,
                                    num_heads=self.num_transf_heads,
                                    ff_dim=self.transf_ff_dim,
                                    reduce=None,
                                    add_axis=True,
                                    name=f"rechits_transf_{i}")
            for i in range(self.num_transf_layers)
        ]

        # Feed-forward output (1 hidden layer) for accumulation
        self.dense_out = get_conv1d([self.output_dim, self.output_dim],
                                    self.activation, last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout,
                                    name="dense_rechits")

    def get_config():
        return {
            "input_dim" : self.input_dim,
            "output_dim": self.output_dim,
            "num_transf_heads": self.num_transf_heads,
            "num_transf_layers": self.num_transf_layers,
            "transf_ff_dim": self.transf_ff_dim,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation
        }
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nrechits, 4]
        # coord = x[:,:,:,0:2] #ieta and iphi as coordinated
        out = x
        for layer in self.transformer:
            out, att_weights = layer(out, mask, training)
        # The last layer is already 
        # Mask for the attention output
        mask_for_output = mask[:,:,:,tf.newaxis]
        # Apply dense layer on each rechit output before the final sum
        dense_output = self.dense_out(out, training=training) 
        # Sum the rechits vectors
        N_rechits = tf.reduce_sum(mask,-1)[:,:,tf.newaxis]
        output = tf.math.divide_no_nan( tf.reduce_sum(dense_output, -2), N_rechits)
        return output

################################
class FeaturesBuilding(tf.keras.layers.Layer):
    '''Features building part of the model
    Includes a transformer for the rechits and a DNN for clusters features
    '''
    
    def __init__(self,  **kwargs):
        self.activation = kwargs.get("activation", tf.nn.selu)
        self.layers_input = kwargs.pop("features_builder_layers_input",[64,64])
        self.output_dim_rechits = kwargs.pop("output_dim_rechits",16)
        self.output_dim_features = kwargs.pop("output_dim_features",32)
        self.rechit_num_transf_layers = kwargs.pop("rechit_num_transf_layers", 2)
        self.rechit_num_transf_heads = kwargs.pop("rechit_num_transf_heads", 4)
        self.rechit_transf_ff_dim = kwargs.pop("rechit_transf_ff_dim", 64)

        self.dropout = kwargs.get("dropout", 0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        name = kwargs.get("name", None)
            
        self.rechits_transf = RechitsTransformer(name="rechit_transformer",
                                     output_dim=self.output_dim_rechits,
                                     input_dim=4,
                                     num_transf_layers=self.rechit_num_transf_layers,
                                     num_transf_heads=self.rechit_num_transf_heads,
                                     transf_ff_dim=self.rechit_transf_ff_dim,
                                     activation=self.activation,
                                     dropout=self.dropout)
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        #append last layer dimension that is the output dimension of the node features
        self.dense_feats = get_conv1d(self.layers_input+[self.output_dim_features],
                                      self.activation,last_act=self.activation,
                                      L2=self.l2_reg, dropout=self.dropout,
                                      name="dense_nodes_feats")
        
        #Layer normalizations
        self.feat_layer_normalization = tf.keras.layers.LayerNormalization(name="norm_nodes_feats",epsilon=1e-3)

        super(FeaturesBuilding, self).__init__(name=name)

    def get_config(self):
        return {
            "features_builder_layers_input": self.layers_input,
            "output_dim_rechits": self.output_dim_rechits,
            "output_dim_features": self.output_dim_features,
            "rechit_num_transf_layers": self.rechit_num_transf_layers,
            "rechit_num_transf_heads": self.rechit_num_transf_heads,
            "rechit_transf_ff_dim": self.rechit_transf_ff_dim,
            "output_dim": self.output_dim,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "name": self.name
        }

    def call(self, cl_features, rechits_features, mask_rechits, mask_cls, training):
        # Cal the rechitGCN and get out 1 vector for each cluster 
        output_rechits = self.rechits_transf(rechits_features, mask_rechits, training=training)
        
        # Layer normalization on the two pieces
        # output_rechits_norm = self.rechit_layer_normalization(output_rechits)
        # cl_features_norm = self.input_layer_normalization(cl_features)

        # Concat the per-cluster feature with the rechits output
        cl_and_rechits = self.concat([cl_features, output_rechits])
        # apply dense layers for feature building 
        cl_and_rechits = self.dense_feats(cl_and_rechits, training=training) 
        # apply normalization and also applying the mask (to be sure)
        cl_and_rechits = self.feat_layer_normalization(cl_and_rechits) * mask_cls[:,:,tf.newaxis]

        return  cl_and_rechits, output_rechits



#############################################
# Putting all the pieces together

class DeepClusterGN(tf.keras.Model):
    '''
    Model parameters:
    - activation
    - output_dim_rechits:  latent space dimension for the rechits per-cluster feature vector
    - output_dim_features: output of the features building step (features per cluster)
    - output_dim_sa_clclass: output of the self-attention layer for cluster classification (default==output_dim_gconv)
    - output_dim_sa_windclass: output of the self-attention layer for windows classification (default==output_dim_gconv)
    - coord_dim:  coordinated space dimension
    - rechit_num_transf_layers: number of transformer layers for rechits feature extraction
    - rechit_num_transf_heads: number of transformer heads in the rechits transformer
    - rechit_transf_ff_dim: hidden dimention of the transformer for the rechits

    - global_transf_layers:  number of transformer layers for the global transformer
    - global_transf_heads:  number of transformer heads in the global transformer
    - global_transf_ff_dim:  hidden dimention of the transformer for the global transformer
    
    - layers_input:  list representing the DNN applied on the [rechit+cluster] concatened features to build the clusters latent space
    - layers_clclass:  list representing the DNN for cluster classification eg [64,64]
    - layers_windclass:  list representing the DNN for window classification eg [64,64]
    - n_windclasses: number of classes for window classification
    - dropout: dropout function to apply on classification DNN
    - l2_reg: activate l2 regularization in all the Dense layers
    - loss_weights:  dictionary "loss_clusters, loss_window, loss_etw, loss_en_resol, loss_en_softF1"
    '''
    def __init__(self, **kwargs):
        self.activation = kwargs.get("activation", tf.nn.selu)
        self.output_dim_features = kwargs.get("output_dim_features", 32)
        self.rechit_num_transf_layers = kwargs.get("rechit_num_transf_layers", 2)
        self.rechit_num_transf_heads = kwargs.get("rechit_num_transf_heads", 4)
        self.rechit_transf_ff_dim = kwargs.get("rechit_transf_ff_dim", 64)

        #global transformer config
        self.global_transf_layers = kwargs.pop("global_transf_layers", 4)
        self.global_transf_heads = kwargs.pop("global_transf_heads", 8)
        self.global_transf_ff_dim = kwargs.pop("global_transf_ff_dim", 512)
        
        self.layers_clclass = kwargs.pop("layers_clclass",[64,64])
        self.accumulator_windclass = kwargs.pop("accumulator_windclass",[64,64])
        self.layers_windclass = kwargs.pop("layers_windclass",[64,64])
        self.accumulator_enregr = kwargs.pop("accumulator_enregr",[64,64])
        self.accumulator_is_calo_seed = kwargs.pop("accumulator_is_calo_seed",[64,64])
        self.layers_enregr = kwargs.pop("layers_enregr",[64,64])
        self.layers_is_calo_seed = kwargs.pop("layers_is_calo_seed",[64,64])
        self.n_windclasses = kwargs.pop("n_windclasses", 1)
        self.dropout = kwargs.get("dropout",0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        self.loss_weights = kwargs.get("loss_weights", {"clusters":1., "window":1., "softF1":1., "et_miss":1., "et_spur":1., "en_regr":1., "softF1_beta":1, "is_calo_seed":1})
        
        super(DeepClusterGN, self).__init__()
        
        self.features_building = FeaturesBuilding(name="features_builder", **kwargs)

        self.transf_global = [ MultiSelfAttentionBlock(output_dim=self.output_dim_features,
                                                       num_heads=self.global_transf_heads,
                                                       ff_dim=self.global_transf_ff_dim, name=f"global_transformer_{i}", **kwargs)
                               for i in range(self.global_transf_layers)]
        
        self.dense_clclass = get_conv1d(name="dense_clclass",
                                        spec=self.layers_clclass+[1],
                                        act=self.activation,
                                        last_act=tf.keras.activations.linear,
                                        dropout=self.dropout, L2=self.l2_reg)

        self.accum_windclass = get_dense(name="accumulator_windclass",
                                               spec=self.accumulator_windclass + [self.output_dim_features],
                                               act=self.activation,
                                               last_act=tf.keras.activations.linear,
                                               dropout=self.dropout, L2=self.l2_reg)
        self.dense_windclass = get_dense(name="dense_windclass",
                                         spec=self.layers_windclass+[self.n_windclasses],
                                         act=self.activation,
                                         last_act=tf.keras.activations.linear,
                                         dropout=self.dropout, L2=self.l2_reg)

        self.accum_enregr = get_dense(name="accumulator_enregr",
                                               spec=self.accumulator_windclass + [self.output_dim_features],
                                               act=self.activation,
                                               last_act=tf.keras.activations.linear,
                                               dropout=self.dropout, L2=self.l2_reg)
        self.dense_enregr = get_dense(name="dense_enregr", spec=self.layers_enregr+[1], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)

        self.accum_is_calo_seed = get_dense(name="accumulator_is_calo_seed",
                                               spec=self.accumulator_windclass + [self.output_dim_features],
                                               act=self.activation,
                                               last_act=tf.keras.activations.linear,
                                               dropout=self.dropout, L2=self.l2_reg)
        self.dense_is_calo_seed = get_dense(name="dense_is_calo_seed", spec=self.layers_is_calo_seed+[1], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)

        #dropout layers
        #self.global_dropout = tf.keras.layers.Dropout(self.dropout)
        #self.global_layernorm = tf.keras.layers.LayerNormalization(name="global_layernorm", epsilon=1e-3)
        #self.windclass_output_dropout = tf.keras.layers.Dropout(self.dropout)
        #self.enregr_dropout = tf.keras.layers.Dropout(self.dropout)
        # Layer normalizations
        self.windclass_layernorm = tf.keras.layers.LayerNormalization(name="windclass_layernorm", epsilon=1e-3)
        self.enregr_layernorm = tf.keras.layers.LayerNormalization(name="enregr_layernorm", epsilon=1e-3)
        self.is_calo_seed_layernorm = tf.keras.layers.LayerNormalization(name="is_calo_seed_layernorm", epsilon=1e-3)
        # Concatenation layers
        self.concat_inputs = tf.keras.layers.Concatenate(axis=-1)
        self.concat_inputs_enregr = tf.keras.layers.Concatenate(axis=-1)
        self.concat_inputs_is_calo_seed = tf.keras.layers.Concatenate(axis=-1)
        self.concat_wind_feats = tf.keras.layers.Concatenate(axis=-1)

    def get_config(self):
        return {
            "features_builder_layers_input": self.feature_building.features_builder_layers_input,
            "rechits_num_transf_layers": self.feature_building.rechit_num_transf_layers,
            "rechits_num_transf_heads": self.feature_building.rechit_num_transf_heads,
            "rechits_transf_ff_dim": self.feature_building.rechit_transf_ff_dim,
            "global_transf_layers": self.global_transf_layers,
            "global_transf_heads": self.global_transf_heads,
            "global_transf_ff_dim": self.global_transf_ff_dim,

            "layers_clclass": self.layers_clclass, 
            "layers_windclass": self.layers_windclass,
            "layers_enregr": self.layers_enregr,
            "layers_is_calo_seed": self.layers_is_calo_seed,

            "accumulator_windclass": self.accumulator_windclass,
            "accumulator_enregr": self.accumulator_enregr,
            "accumulator_is_calo_seed": self.accumulator_is_calo_seed,

            "output_dim_rechits": self.feature_building.output_dim_rechits,
            "output_dim_features": self.output_dim_features,
            
            "n_windclasses": self.n_windclasses,

            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "name": self.name,
            "loss_weights": self.loss_weights
        }

    @tf.function
    def call(self, inputs, training):
        cl_X_initial, wind_X, cl_hits, is_seed, mask_cls, mask_rechits = inputs 
        # Concatenate the seed label on clusters features
        cl_X_initial = tf.concat([tf.cast(is_seed[:,:,tf.newaxis], tf.float32), cl_X_initial], axis=-1)
        #cl_X now is the latent cluster+rechits representation
        cl_X, output_rechits = self.features_building(cl_X_initial, cl_hits, mask_rechits, mask_cls, training)
        mask_cls_to_apply = mask_cls[:,:,tf.newaxis]

        # The output of the features building step has shape [Nbatch, Nclusters, output_dim_features]
        # Call the global transformer
        out_ = cl_X
        for layer in self.transf_global:
            out_, _ = layer(out_, mask_cls, training)
            
        # Dropout + normalization
        #out_ = self.global_dropout(cl_X, training=training)
        #out_ = self.global_layernorm(out_) * mask_cls_to_apply

        # Clusters classification block
        in_clclass = self.concat_inputs([cl_X,out_])
        
        clclass_out = self.dense_clclass(in_clclass, training=training) * mask_cls_to_apply


        # Windows classification block
        # Concatenate with window level features
        in_windcl = self.concat_wind_feats([in_clclass,  tf.repeat(wind_X[:, tf.newaxis, :], tf.shape(in_clclass)[1], axis=1), clclass_out])
        # Norm before dense for wind classification because the sum is performed in the SA layer
        in_windcl = self.windclass_layernorm(in_windcl)
        # Now apply the accumulator dense before accumulating
        in_windcl = self.accum_windclass(in_windcl, training=training)*mask_cls_to_apply
        # accumulate
        in_windcl = tf.reduce_sum(in_windcl, 1) / tf.reduce_sum(mask_cls, 1)[:,tf.newaxis]
        # Apply dense for classification
        windclass_out = self.dense_windclass(in_windcl, training=training)


        # Energy regression block
        # concact clclass inputs, cl decision, rechits transformer output
        in_en_regr = self.concat_inputs_enregr([in_clclass, clclass_out, output_rechits])
        input_en_regr = self.enregr_layernorm(in_en_regr)
        # now apply the accumulator dense before accumulating
        input_en_regr = self.accum_enregr(input_en_regr, training=training)*mask_cls_to_apply
        # accumulate
        input_en_regr = tf.reduce_sum(input_en_regr, 1) / tf.reduce_sum(mask_cls, 1)[:,tf.newaxis]
        # apply dense
        enregr_out = self.dense_enregr(input_en_regr, training=training)

        # Calo Seed classifier
        # concact clclass inputs, cl decision, rechits transformer output
        in_is_calo_seed = self.concat_inputs_is_calo_seed([in_clclass, clclass_out, output_rechits])
        input_is_calo_seed = self.is_calo_seed_layernorm(in_is_calo_seed)
        # now apply the accumulator dense before accumulating
        input_is_calo_seed = self.accum_is_calo_seed(input_is_calo_seed, training=training)*mask_cls_to_apply
        # accumulate
        input_is_calo_seed = tf.reduce_sum(input_is_calo_seed, 1) / tf.reduce_sum(mask_cls, 1)[:,tf.newaxis]
        # apply dense
        is_calo_seed_out = self.dense_is_calo_seed(input_is_calo_seed, training=training)
               
        return (clclass_out, windclass_out, enregr_out, is_calo_seed_out), mask_cls

    ########################
    # Training related methods
    def set_metrics(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss1_tracker = tf.keras.metrics.Mean(name="loss_clusters")
        self.loss2_tracker = tf.keras.metrics.Mean(name="loss_windows")
        self.loss3_tracker = tf.keras.metrics.Mean(name="loss_softF1")
        self.loss4_tracker = tf.keras.metrics.Mean(name="loss_en_resol")
        self.loss5_tracker = tf.keras.metrics.Mean(name="loss_en_softF1")
        self.loss6_tracker = tf.keras.metrics.Mean(name="loss_en_regr")
        self.loss7_tracker = tf.keras.metrics.Mean(name='loss_is_calo_seed')
        self.loss_reg      = tf.keras.metrics.Mean(name="loss_regularization")

    # Customized training loop
    # Based on https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
    @tf.function
    def train_step(self, data):
        x, y, w = data 
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss_clusters = clusters_classification_loss(y, y_pred, w[0])
            loss_softF1 =  soft_f1_score(y,y_pred, w[0], self.loss_weights["softF1_beta"])
            loss_windows = window_classification_loss(y, y_pred, w[0])
            loss_en_resol, loss_en_softF1 = energy_loss(y, y_pred, w[0], self.loss_weights["softF1_beta"])
            loss_en_regr = energy_regression_loss(y, y_pred, w[0])
            loss_is_calo_seed = is_calo_seed_loss(y, y_pred, w[0])
            loss_reg = tf.reduce_sum(self.losses)
            # tf.print(loss_clusters, loss_softF1, loss_windows, loss_en_resol, loss_en_regr, loss_en_softF1, sum(self.losses))
            # Total loss function
            loss =  self.loss_weights["clusters"] * loss_clusters +\
                     self.loss_weights["window"] * loss_windows + \
                     self.loss_weights["softF1"] * loss_softF1 + \
                     self.loss_weights["is_calo_seed"] * loss_is_calo_seed + \
                     loss_reg
                     #self.loss_weights["en_softF1"] *  loss_en_softF1 + \
                     #self.loss_weights["en_regr"] * loss_en_regr + \
                     #self.loss_weights["en_resol"] * loss_en_resol+ \
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss_clusters)
        self.loss2_tracker.update_state(loss_windows)
        self.loss3_tracker.update_state(loss_softF1)
        self.loss4_tracker.update_state(loss_en_resol)
        self.loss5_tracker.update_state(loss_en_softF1)
        self.loss6_tracker.update_state(loss_en_regr)
        self.loss7_tracker.update_state(loss_is_calo_seed)
        self.loss_reg.update_state(loss_reg)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_softF1": self.loss3_tracker.result(),
                "loss_en_resol": self.loss4_tracker.result(),
                "loss_en_softF1": self.loss5_tracker.result(),
                "loss_en_regr": self.loss6_tracker.result(),
                "loss_is_calo_seed": self.loss7_tracker.result(),
                "loss_regularization": self.loss_reg.result()}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y, w  = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss_clusters = clusters_classification_loss(y, y_pred, w[0] )
        loss_softF1 =  soft_f1_score(y,y_pred, w[0], self.loss_weights["softF1_beta"])
        loss_windows = window_classification_loss(y, y_pred, w[0])
        loss_en_resol, loss_en_softF1 = energy_loss(y, y_pred, w[0], self.loss_weights["softF1_beta"])
        loss_en_regr = energy_regression_loss(y, y_pred, w[0])
        loss_is_calo_seed = is_calo_seed_loss(y, y_pred, w[0])
        loss_reg = tf.reduce_sum(self.losses)
        # Total loss function
        loss =  self.loss_weights["clusters"] * loss_clusters +\
                self.loss_weights["window"] * loss_windows + \
                self.loss_weights["softF1"] * loss_softF1 + \
                self.loss_weights["is_calo_seed"] * loss_is_calo_seed + \
                loss_reg
                # self.loss_weights["en_resol"] * loss_en_resol + \
                # self.loss_weights["en_softF1"] *  loss_en_softF1 + \
                # self.loss_weights["en_regr"] * loss_en_regr + \
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss_clusters)
        self.loss2_tracker.update_state(loss_windows)
        self.loss3_tracker.update_state(loss_softF1)
        self.loss4_tracker.update_state(loss_en_resol)
        self.loss5_tracker.update_state(loss_en_softF1)
        self.loss6_tracker.update_state(loss_en_regr)
        self.loss7_tracker.update_state(loss_is_calo_seed)
        self.loss_reg.update_state(loss_reg)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_softF1": self.loss3_tracker.result(),
                "loss_en_resol": self.loss4_tracker.result(),
                "loss_en_softF1": self.loss5_tracker.result(),
                "loss_en_regr": self.loss6_tracker.result(),
                "loss_is_calo_seed": self.loss7_tracker.result(),
                "loss_regularization": self.loss_reg.result()
                }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.loss1_tracker, self.loss2_tracker, self.loss3_tracker,
                self.loss4_tracker, self.loss5_tracker, self.loss6_tracker, self.loss7_tracker, self.loss_reg]



############################################
############################################


## Loss functions
@tf.function
def clusters_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed),  mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true        
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass[:,:,tf.newaxis], dense_clclass, from_logits=True) * mask_cls
    # This should be replaced by the mean over the not masked elements
    # reduced_loss = tf.reduce_sum(tf.reduce_mean(class_loss, axis=-1) * weight) / tf.reduce_sum(weight)
    ncls = tf.reduce_sum(mask_cls, axis=-1)
    # Do not reduce_mean the loss, but sum and divide by number of clusters
    reduced_loss = tf.reduce_sum((tf.reduce_sum(class_loss, axis=-1)/ncls)*weight) / tf.reduce_sum(weight)
    return reduced_loss

## Loss functions
@tf.function
def energy_weighted_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    cl_ets = cl_X[:,:,1]
    # matched_window = tf.cast(y_metadata[:,-1]!=0, tf.float32)
    # compute the weighting mean of the loss based on the energy of each seed in the window
    cl_ets_weights = cl_ets / tf.reduce_sum(cl_ets, axis=-1)[:,tf.newaxis]
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass[:,:,tf.newaxis], dense_clclass, from_logits=True) * mask_cls
    weighted_loss = class_loss * cl_ets_weights
    # mean over the batch
    reduced_loss = tf.reduce_sum(tf.reduce_sum(weighted_loss, axis=-1) * weight) / tf.reduce_sum(weight) 
    return reduced_loss

@tf.function
def window_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    w_flavour = tf.one_hot( tf.cast(y_windclass / 11, tf.int32) , depth=3)

    # Only window multi-class classification
    windclass_loss = tf.keras.losses.categorical_crossentropy(w_flavour, dense_windclass, from_logits=True)
    reduced_loss =   tf.reduce_sum(windclass_loss * weight) / tf.reduce_sum(weight)
    return reduced_loss

@tf.function
def energy_loss(y_true, y_pred, weight, beta=1):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    y_target = tf.cast(y_clclass, tf.float32)[:,:,tf.newaxis]
    cl_en = Et = cl_X[:,:,0:1]
    En_sim_good = y_metadata[:,-2] # en_true_sim --> HARD CODED be careful in the config
    pred_prob = tf.nn.sigmoid(dense_clclass)

    sel_en = tf.squeeze(tf.reduce_sum(cl_en * pred_prob , axis=1))
    en_resolution_loss =  tf.reduce_sum(tf.square( (sel_en/En_sim_good) - 1) * weight ) / tf.reduce_sum(weight) 
    #soft f1 style loss
    tp = tf.reduce_sum(cl_en* pred_prob * y_target, axis=1)
    fn = tf.reduce_sum(cl_en* (1 - pred_prob) * y_target, axis=1)
    fp = tf.reduce_sum(cl_en* pred_prob * (1 - y_target), axis=1)
    soft_f1_loss = 1 - ((1 + beta**2) * tp)/ ( (1+beta**2)*tp + beta* fn + fp + 1e-16)
    reduced_f1 = tf.reduce_sum(tf.squeeze(soft_f1_loss) * weight)  / tf.reduce_sum(weight) 

    return en_resolution_loss , reduced_f1

@tf.function
def soft_f1_score(y_true, y_pred, weight, beta=1):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    y_target = tf.cast(y_clclass, tf.float32)[:,:,tf.newaxis]
    # matched_window = tf.cast(y_metadata[:,-1]!=0, tf.float32)

    pred_prob = tf.nn.sigmoid(dense_clclass)*mask_cls[:,:,None]
    tp = tf.reduce_sum(pred_prob * y_target, axis=1)
    fn = tf.reduce_sum((1 - pred_prob) * y_target, axis=1)
    fp = tf.reduce_sum(pred_prob * (1 - y_target), axis=1)

    soft_f1_loss = 1 - ((1 + beta**2) * tp)/ ( (1+beta**2)*tp + beta* fn + fp + 1e-16) 
    reduced_f1 = tf.reduce_sum(tf.squeeze(soft_f1_loss) * weight) / tf.reduce_sum(weight) 
    return reduced_f1

@tf.function
def huber_loss(y_true, y_pred, delta, weight):
    z = tf.math.abs(y_true - y_pred)
    mask = tf.cast(z < delta,tf.float32)
    return  tf.reduce_sum( (0.5*mask*tf.square(z) + (1.-mask)*(delta*z - 0.5*delta**2))*weight)/tf.reduce_sum(weight)

quantiles = tf.constant([ 0.25, 0.75])[:,tf.newaxis]
@tf.function
def quantile_loss(y_true, y_pred, weight):
    e = y_true - y_pred
    l =  tf.reduce_sum( ( quantiles*e + tf.clip_by_value(-e, tf.keras.backend.epsilon(), np.inf) ) * weight) / tf.reduce_sum(weight)
    return l

@tf.function
def energy_regression_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    cl_ens = cl_X[:,:,0]
    pred_en =  tf.reduce_sum(cl_ens * tf.squeeze(tf.cast(tf.nn.sigmoid(dense_clclass) > 0.5 , tf.float32)), axis=-1)
    calib_pred_en =  pred_en * tf.squeeze(en_regr_factor)
    true_en_gen = y_metadata[:,-3]  # en_true_gen --> HARD CODED be careful in the config

    loss = huber_loss(true_en_gen, calib_pred_en, 5, weight) + quantile_loss(true_en_gen, calib_pred_en,weight )
    return loss

@tf.function
def is_calo_seed_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor, is_seed_calo_seed), mask_cls  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true

    caloclass_loss = tf.keras.losses.binary_crossentropy(y_is_seed_calo_seed, is_seed_calo_seed[:,0], from_logits=True)
    reduced_loss =   tf.reduce_sum(caloclass_loss * weight) / tf.reduce_sum(weight)
    return reduced_loss
