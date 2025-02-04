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

def point_wise_feed_forward_network(d_model, dff, name="fff"):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ],name=name)

# def point_wise_cnn1d(out, hidden, name, act="relu", last_act="linear", L2=False):
#     if not L2:
#         return tf.keras.Sequential([
#                 tf.keras.layers.Conv1D(filters=hidden,kernel_size=1, activation=act, name=name+"_0",kernel_regularizer=tf.keras.regularizers.L2(0.001)),
#                 tf.keras.layers.Conv1D(filters=out,   kernel_size=1, activation=last_act, name=name+"_1",kernel_regularizer=tf.keras.regularizers.L2(0.001)),
#             , name=name)
#     else:
#         return tf.keras.Sequential([
#                 tf.keras.layers.Conv1D(filters=hidden,kernel_size=1, activation=act, name=name+"_0"),
#                 tf.keras.layers.Conv1D(filters=out,   kernel_size=1, activation=last_act, name=name+"_1"),
#             , name=name)

#
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


###########################
#Distance

@tf.function
def dist(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [tf.shape(na)[0], -1, 1])
    nb = tf.reshape(nb, [tf.shape(na)[0], 1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, transpose_a=False, transpose_b=True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D

@tf.function
def dist_batch2(A,B):
    na = tf.reduce_sum(tf.square(A), -1)
    nb = tf.reduce_sum(tf.square(B), -1)
 
    na = tf.reshape(na, [tf.shape(na)[0],tf.shape(na)[1], -1, 1])
    nb = tf.reshape(nb, [tf.shape(nb)[0],tf.shape(nb)[1], 1, -1])
    Dsq = tf.clip_by_value(na - 2*tf.linalg.matmul(A, B, transpose_a=False, transpose_b=True) + nb, 1e-12, 1e12)
    D = tf.sqrt(Dsq)
    return D


#Given a list of [Nbatch, Nelem, Nfeat] input nodes, computes the dense [Nbatch, Nelem, Nelem] adjacency matrices
class Distance(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        self.batch_dim = kwargs.pop('batch_dim',1)
        super(Distance, self).__init__(*args, **kwargs)
        
    def call(self, inputs1, inputs2):
        #compute the pairwise distance matrix between the vectors defined by the first two components of the input array
        #inputs1, inputs2: [Nbatch, Nelem, distance_dim] embedded coordinates used for element-to-element distance calculation
        if self.batch_dim == 1:
            D = dist(inputs1, inputs2)
        if self.batch_dim == 2:
            D = dist_batch2(inputs1, inputs2)
      
        #adjacency between two elements should be high if the distance is small.
        #this is equivalent to radial basis functions. 
        #self-loops adj_{i,i}=1 are included, as D_{i,i}=0 by construction
        adj = tf.math.exp(-1.0*D)

        #optionally set the adjacency matrix to 0 for low values in order to make the matrix sparse.
        #need to test if this improves the result.
        #adj = tf.keras.activations.relu(adj, threshold=0.01)
        return adj

###############################################
# https://arxiv.org/pdf/2004.04635.pdf
#https://github.com/gcucurull/jax-ghnet/blob/master/models.py 
class GHConvI(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, n_iter, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = n_iter
        name = kwargs.get("name", "ghc")

        super(GHConvI, self).__init__(*args, **kwargs)

        self.W_t = self.add_weight(shape=(self.input_dim, self.hidden_dim), name="w_t_"+name, initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t_"+name, initializer="zeros")
        self.theta = self.add_weight(shape=(self.input_dim, self.hidden_dim), name="theta_"+name, initializer="random_normal")
    
    def call(self, x, adj):
        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)
        adj_k = tf.pow(adj, self.k)

        f_het = tf.linalg.matmul(x, self.theta)  #inner infusion
        # Added activation to homogenous component
        f_hom = self.activation(tf.linalg.matmul(adj_k, f_het*norm_k)*norm_k)

        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)
        #tf.print(tf.reduce_mean(f_hom), tf.reduce_mean(f_het), tf.reduce_mean(gate))

        out = gate*f_hom + (1-gate)*f_het
        return out

class GHConvO(tf.keras.layers.Layer):
    def __init__(self, k, *args, **kwargs):
        self.activation = kwargs.pop("activation")
        self.hidden_dim = args[0]
        self.k = k

        super(GHConvO, self).__init__(*args, **kwargs)

        self.W_t = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_t", initializer="random_normal")
        self.b_t = self.add_weight(shape=(self.hidden_dim, ), name="b_t", initializer="zeros")
        self.W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="w_h", initializer="random_normal")
        self.theta = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="theta", initializer="random_normal")
    
    def call(self, x, adj):
        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)
        adj_k = tf.pow(adj, self.k)

        f_hom = tf.linalg.matmul(x, self.theta)
        # Added activation to homogenous component
        f_hom = self.activation(tf.linalg.matmul(adj_k, f_hom*norm_k)*norm_k)

        f_het = tf.linalg.matmul(x, self.W_h)  #outer infusion
        gate = tf.nn.sigmoid(tf.linalg.matmul(x, self.W_t) + self.b_t)
        #tf.print(tf.reduce_mean(f_hom), tf.reduce_mean(f_het), tf.reduce_mean(gate))

        out = gate*f_hom + (1-gate)*f_het
        return out

#######################
## Simple Graph Conv layer
class SGConv(tf.keras.layers.Dense):
    def __init__(self, k, *args, **kwargs):
        super(SGConv, self).__init__(*args, **kwargs)
        self.k = k
    
    def call(self, inputs, adj):
        W = self.weights[0]
        b = self.weights[1]

        #compute the normalization of the adjacency matrix
        in_degrees = tf.reduce_sum(adj, axis=-1)
        #add epsilon to prevent numerical issues from 1/sqrt(x)
        norm = tf.expand_dims(tf.pow(in_degrees + 1e-6, -0.5), -1)
        norm_k = tf.pow(norm, self.k)

        support = (tf.linalg.matmul(inputs, W))
     
        #k-th power of the normalized adjacency matrix is nearly equivalent to k consecutive GCN layers
        adj_k = tf.pow(adj, self.k)
        out = tf.linalg.matmul(adj_k, support*norm_k)*norm_k

        return self.activation(out + b)


############################
# From https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
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

class SelfAttention(tf.keras.layers.Layer):
    '''
    Self attention layer only computing the SA computation
    Input : [Nbatch, Nclusters, input_dim]
    Output:  [Nbatch, Nclusters, output_dim]
    '''    
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        self.activation = kwargs.pop("activation", "relu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        name = kwargs.pop("name", None)

        super(SelfAttention, self).__init__(name=name)

        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.input_dim, self.output_dim), name="Q_sa_"+name, initializer="random_normal")
        self.K = self.add_weight(shape=(self.input_dim, self.output_dim), name="K_sa_"+name, initializer="random_normal")
        self.V = self.add_weight(shape=(self.input_dim, self.output_dim), name="V_sa_"+name, initializer="random_normal")


    def get_config():
        return {
            "input_dim" : self.input_dim,
            "output_dim": self.output_dim,
            "name": self.name
        }
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nfeatures]
        q = tf.matmul(x,self.Q)
        k = tf.matmul(x,self.K)
        v = tf.matmul(x,self.V)
        # mask the padded clusters in the attention distance
        mask_for_attention = mask[:,tf.newaxis,:]
        # Mask for output
        mask_for_nodes = mask[:,:,tf.newaxis]
        # Get self-attention output and attention weights
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Dropout
        sa_output = sa_output * mask_for_nodes
        
        return sa_output, attention_weights

####################################################

class SelfAttentionBlock(tf.keras.layers.Layer):
    '''
    Generic self attention layer that can reduce or not the output feature vectors.
    Input : [Nbatch, Nclusters, input_dim]
    Output:  
            - reduce=none  [Nbatch, Nclusters, output_dim]  
            - reduce=sum or mean   [Nbatch, output_dim]  
    '''    
    def __init__(self, input_dim, output_dim, reduce=None, *args, **kwargs):
        self.activation = kwargs.pop("activation", "relu")
        self.dropout = kwargs.get("dropout", 0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduce = reduce # it can be None, sum, mean, max
        name = kwargs.pop("name", None)

        super(SelfAttentionBlock, self).__init__(name=name)
        
        # Self-attention matrices
        self.Q = self.add_weight(shape=(self.input_dim, self.output_dim), name="Q_sa_"+name, initializer="random_normal")
        self.K = self.add_weight(shape=(self.input_dim, self.output_dim), name="K_sa_"+name, initializer="random_normal")
        self.V = self.add_weight(shape=(self.input_dim, self.output_dim), name="V_sa_"+name, initializer="random_normal")
        self.inputW = self.add_weight(shape=(self.input_dim, self.output_dim), name="input_sa_"+name, initializer="random_normal")

        # Feed-forward output (1 hidden layer)
        self.dense_out = get_conv1d([self.output_dim, self.output_dim], self.activation, last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout, name="output_sa_"+name)
        # Layer normalizations
        self.norm1 = tf.keras.layers.LayerNormalization(name="SA_lnorm1_"+name, epsilon=1e-3, axis=-1)
        self.norm2 = tf.keras.layers.LayerNormalization(name="SA_lnorm2_"+name, epsilon=1e-3, axis=-1)
        # Dropouts
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)

    def get_config():
        return {
            "input_dim" : self.input_dim,
            "output_dim": self.output_dim,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "reduce": self.reduce,
            "name": self.name
        }
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nfeatures]
        q = tf.matmul(x,self.Q)
        k = tf.matmul(x,self.K)
        v = tf.matmul(x,self.V)
        # Apply the dense_input to x to get transformed dimension
        transf_input = tf.matmul(x,self.inputW)
        # mask the padded clusters in the attention distance
        mask_for_attention = mask[:,tf.newaxis,:]
        # Mask for output
        mask_for_nodes = mask[:,:,tf.newaxis]
        # Get self-attention output and attention weights
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Dropout
        sa_output = self.drop1(sa_output, training=training) 
        # Add + layer norm  + mask
        output_sa = self.norm1(transf_input + sa_output) * mask_for_nodes
        # Apply dense
        output_dense = self.dense_out(output_sa, training=training) 
        # Dropout
        output_dense = self.drop2(output_dense, training=training)
        # Add and layer norm
        output_block = self.norm2(output_dense + output_sa) * mask_for_nodes

        # Now the aggregation 
        if self.reduce == "sum":
            return  tf.reduce_sum(output_block, -2), attention_weights
        if self.reduce == "mean":
            N_nodes = tf.reduce_sum(mask,-1)[:,tf.newaxis]
            return tf.math.divide_no_nan( tf.reduce_sum(output_block, -2), N_nodes), attention_weights
        if self.reduce == "max":
            return tf.reduce_max(output_block, axis=-2), attention_weights
        else:
            #just return all the nodes
            return output_block, attention_weights

######################################################################
######################################################################

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, **kwargs):
    self.num_heads = num_heads
    self.d_model = d_model
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
            "name": self.name
        }

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    mask_att = mask[:,tf.newaxis,tf.newaxis,:]
    
    q = self.Wq(q)  # (batch_size, seq_len, d_model)
    k = self.Wk(k)  # (batch_size, seq_len, d_model)
    v = self.Wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask_att)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    # the output is not masked
    output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

    return output, attention_weights


#######################################################################################

class MultiSelfAttentionBlock(tf.keras.layers.Layer):
  def __init__(self, output_dim, num_heads, ff_dim,reduce=None, **kwargs):
    name = kwargs.pop("name", None)
    super(MultiSelfAttentionBlock, self).__init__(name=name)
    self.output_dim = output_dim
    self.ff_dim = ff_dim
    self.num_heads = num_heads
    self.reduce = reduce # it can be None, sum, mean, max
    self.activation = kwargs.pop("activation", "relu")
    self.dropout = kwargs.get("dropout", 0.)
    self.l2_reg = kwargs.get("l2_reg", False)

    self.inputW = tf.keras.layers.Conv1D(filters=self.output_dim, kernel_size=1, use_bias=False)
    self.mha = MultiHeadAttention(self.output_dim, self.num_heads,name=self.name+"_msa", )
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
            "reduce": self.reduce
        }
    
  def call(self, x, mask, training):
    mask_out = mask[:,:,tf.newaxis]
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


######################################################################
######################################################################
############################
## GCN + Self-attention block for rechits feature extraction
# A single features vector of dimension output_dim is built from arbitrary list of rechits. 
# A GHconv block is applied and then a self-attention block 
class RechitsGCN(tf.keras.layers.Layer):

    def __init__(self, nconv, input_dim, output_dim,  *args, **kwargs):
        self.activation = kwargs.pop("activation", tf.keras.activations.relu)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dropout = kwargs.pop("dropout", 0.01)
        self.l2_reg = kwargs.pop("l2_reg", False)
        self.nconv = nconv
    
        super(RechitsGCN, self).__init__(*args, **kwargs)
        
        self.dist = Distance(batch_dim=2)
        self.GCN = GHConvI(name="GHC_rechits", n_iter = self.nconv, input_dim=input_dim, 
                                hidden_dim=output_dim, activation=self.activation)
        
        # Self-attention matrices
        self.Q = tf.keras.layers.Conv1D(filters=self.output_dim,kernel_size=1, use_bias=False, name="Q_sa_rechits")
        self.K = tf.keras.layers.Conv1D(filters=self.output_dim,kernel_size=1, use_bias=False, name="K_sa_rechits")
        self.V = tf.keras.layers.Conv1D(filters=self.output_dim,kernel_size=1, use_bias=False, name="V_sa_rechits")
        #  Dense 
        # Feed-forward output (1 hidden layer)
        self.dense_out = get_conv1d([self.output_dim, self.output_dim], self.activation, last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout, name="dense_rechits")
        # Dropouts
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        #Layer normalizations
        self.sa_normalization = tf.keras.layers.LayerNormalization(name="sa_norm_rechits", epsilon=1e-3, axis=-1)
        self.out_normalization = tf.keras.layers.LayerNormalization(name="out_norm_rechits", epsilon=1e-3, axis=-1)

    def get_config():
        return {
            "input_dim" : self.input_dim,
            "output_dim": self.output_dim,
            "nconv": self.nconv,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation
        }
        
    def call(self, x, mask, training):
        # x has structure  [Nbatch, Nclusters, Nrechits, 4]
        coord = x[:,:,:,0:2] #ieta and iphi as coordinated
        # create mask for adjacency matrix
        adj_mask = m =  mask[:,:,:,tf.newaxis] @ mask[:,:,tf.newaxis, :]
        # compute adjacency and mask it
        adj = self.dist(coord,coord) * adj_mask
        # apply GCN in fully batched style
        out_gcn = self.GCN(x,adj)
        # And now SA layer for aggregation
        q = self.Q(out_gcn)
        k = self.K(out_gcn)
        v = self.V(out_gcn)
        mask_for_attention = mask[:,:,tf.newaxis,:]
        sa_output, attention_weights = scaled_dot_product_attention(q, k, v, mask_for_attention)
        # Mask to compute the output masking the correct rechits
        mask_for_output = mask[:,:,:,tf.newaxis]
        # Layer normalizationa and dropout on SA output
        sa_output = self.drop1(sa_output, training=training)
        sa_output = self.sa_normalization(sa_output) * mask_for_output
        # Apply dense layer on each rechit output before the final sum
        dense_output = self.dense_out(sa_output, training=training) 
        dense_output = self.drop2(dense_output, training=training)
        # Add + Norm
        dense_output = self.out_normalization(dense_output + sa_output) * mask_for_output
        # Sum the rechits vectors
        # output = tf.reduce_sum(convout * mask_for_output, -2)
        # Or doing the mean 
        N_rechits = tf.reduce_sum(mask,-1)[:,:,tf.newaxis]
        output = tf.math.divide_no_nan( tf.reduce_sum(dense_output, -2), N_rechits)
        return output, (sa_output,dense_output, attention_weights, adj)

################################
# Graph building part of the model
class GraphBuilding(tf.keras.layers.Layer):
    
    def __init__(self,  **kwargs):
        self.activation = kwargs.get("activation", tf.nn.selu)
        self.layers_input = kwargs.pop("layers_input",[64,64])
        self.output_dim_rechits = kwargs.pop("output_dim_rechits",16)
        self.output_dim_nodes = kwargs.pop("output_dim_nodes",32)
        #self.coord_sa_dim = kwargs.pop("coord_sa_dim",10)
        #self.coord_dim = kwargs.pop("coord_dim",3)
        self.nconv_rechits = kwargs.pop("nconv_rechits",3)
        self.dropout = kwargs.get("dropout", 0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        name = kwargs.get("name", None)
            
        self.rechitsGCN = RechitsGCN(name="rechit_gcn", output_dim=self.output_dim_rechits, input_dim=4, 
                                nconv=self.nconv_rechits, activation=self.activation, dropout=self.dropout)
        
        #Self-attention for coordinations
        #self.SA_coord = SelfAttention(name="coord_SA", input_dim=self.output_dim_nodes, output_dim=self.coord_sa_dim)
        #self.dense_coord = get_conv1d([self.coord_sa_dim, self.coord_dim], self.activation, last_act=tf.keras.activations.linear,
        #                            L2=self.l2_reg, name="dense_coord")
        #self.dist = Distance(batch_dim=1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        #append last layer dimension that is the output dimension of the node features
        self.dense_feats = get_conv1d(self.layers_input+[self.output_dim_nodes], self.activation,last_act=self.activation,
                                    L2=self.l2_reg, dropout=self.dropout, name="dense_nodes_feats")
        
        #Layer normalizations
        self.feat_layer_normalization = tf.keras.layers.LayerNormalization(name="norm_nodes_feats",epsilon=1e-3)

        super(GraphBuilding, self).__init__(name=name)

    def get_config(self):
        return {
            "layers_input": self.layers_input,
            "output_dim_rechits": self.output_dim_rechits,
            "output_dim_nodes": self.output_dim_nodes,
            #"coord_sa_dim": self.coord_sa_dim,
            #"coord_dim" : self.coord_dim,
            "nconv_rechits": self.nconv_rechits,
            "output_dim": self.output_dim,
            "nconv": self.nconv,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "name": self.name
        }

    def call(self, cl_features, rechits, mask_rechits, mask_cls, training):
        # Conversion from RaggedTensor to dense tensor
        #rechits = rechits.to_tensor()
        # Cal the rechitGCN and get out 1 vector for each cluster 
        output_rechits, (debug) = self.rechitsGCN(rechits, mask_rechits, training=training)
        
        # Layer normalization on the two pieces
        # output_rechits_norm = self.rechit_layer_normalization(output_rechits)
        # cl_features_norm = self.input_layer_normalization(cl_features)

        # Concat the per-cluster feature with the rechits output
        cl_and_rechits = self.concat([cl_features, output_rechits])
        # apply dense layers for feature building 
        cl_and_rechits = self.dense_feats(cl_and_rechits, training=training) 
        # apply normalization and also applying the mask (to be sure)
        cl_and_rechits = self.feat_layer_normalization(cl_and_rechits) * mask_cls[:,:,tf.newaxis]

        #cl_and_rechits are now the baseline node features
        
        # Now apply the coordinate network with a layer normalization before
        # coord_output, coord_att_ws = self.SA_coord(cl_and_rechits, mask_cls, training)
        # the input of the dense is masked, no need to mask the output, it is masked later
        # coord_output = self.dense_coord(coord_output)
        # Build the adjacency matrix
        # adj = self.dist(coord_output,coord_output)
        # mask the padded clusters      
        # adj_mask = m =  mask_cls[:,:,tf.newaxis] @ mask_cls[:,tf.newaxis, :]
        # adj = adj* adj_mask
        
        #return the nodes features, the coordinates , the adjacency matrix, the clusters mask
        return  cl_and_rechits, output_rechits


#############################################
# Putting all the pieces together

class DeepClusterGN(tf.keras.Model):
    '''
    Model parameters:
    - activation
    - output_dim_nodes: latent space dimension for clusters node built from rechits and cluster features
    - output_dim_rechits:  latent space dimension for the rechits per-cluster feature vector
    - output_dim_gconv: output of the graph convolution (default==output_dim_nodes)\
    - output_dim_msa_encoder: output of the self-attention layer for cluster classification (default==output_dim_gconv)
    - output_dim_sa_windclass: output of the self-attention layer for windows classification (default==output_dim_gconv)
    - coord_dim:  coordinated space dimension
    - nconv_rechits: number of convolutions for the rechits GCN
    - nconv: number of convolutions for the global model
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
        self.output_dim_nodes = kwargs.get("output_dim_nodes",32)
        self.output_dim_gconv = kwargs.pop("output_dim_gconv",self.output_dim_nodes)
        self.output_dim_msa_encoder = kwargs.pop("output_dim_msa_encoder",self.output_dim_gconv)
        self.output_dim_sa_windclass = kwargs.pop("output_dim_sa_windclass",self.output_dim_gconv)
        self.output_dim_sa_enregr = kwargs.pop("output_dim_sa_enregr",self.output_dim_gconv)
        self.nconv = kwargs.pop("nconv",3)
        self.nrepeat_msa_encoder = kwargs.pop("nrepeat_msa_encoder",1)
        self.num_heads_msa_encoder = kwargs.pop("num_heads_msa_encoder", 8)
        self.ff_dim_msa_encoder = kwargs.pop("ff_dim_msa_encoder", 256)
        self.num_heads_msa_windclass = kwargs.pop("num_heads_msa_windclass", 8)
        self.ff_dim_msa_windclass = kwargs.pop("ff_dim_msa_windclass", 256)
        self.num_heads_msa_enregr = kwargs.pop("num_heads_msa_enregr", 8)
        self.ff_dim_msa_enregr = kwargs.pop("ff_dim_msa_enregr", 256)
        self.layers_clclass = kwargs.pop("layers_clclass",[64,64])
        self.layers_windclass = kwargs.pop("layers_windclass",[64,64])
        self.layers_enregr = kwargs.pop("layers_enregr",[64,64])
        self.n_windclasses = kwargs.pop("n_windclasses", 1)
        self.dropout = kwargs.get("dropout",0.)
        self.l2_reg = kwargs.get("l2_reg", False)
        self.loss_weights = kwargs.get("loss_weights", {"clusters":1., "window":1., "softF1":1., "et_miss":1., "et_spur":1., "en_regr":1., "softF1_beta":1})
        
        super(DeepClusterGN, self).__init__()
        
        self.graphbuild = GraphBuilding(name="graph_builder", **kwargs)
        # self.GCN = GHConvI(name="GHN_global", n_iter =self.nconv, input_dim=self.output_dim_nodes , 
                                        # hidden_dim=self.output_dim_gconv , activation=self.activation)

        # Clusters classification head
        self.MSA_encoders =[ ] 
        for i in range(self.nrepeat_msa_encoder):
            self.MSA_encoders.append(MultiSelfAttentionBlock(name="MSA_encoder_{}".format(i), output_dim=self.output_dim_msa_encoder, 
                                    num_heads=self.num_heads_msa_encoder, ff_dim=self.ff_dim_msa_encoder, **kwargs))

        self.dense_clclass = get_conv1d(name="dense_clclass", spec=self.layers_clclass+[1], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)
        # Window classification head
        # self.concat_gcn_SAcl = tf.keras.layers.Concatenate(axis=-1)
        # self-attention for windows classification with "mean" reduction
        self.SA_windclass = MultiSelfAttentionBlock(name="MSA_windclass",  output_dim=self.output_dim_sa_windclass, 
                            num_heads=self.num_heads_msa_windclass, ff_dim=self.ff_dim_msa_windclass, reduce="sum", **kwargs)
        self.dense_windclass = get_dense(name="dense_windclass", spec=self.layers_windclass+[self.n_windclasses], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)

        # Energy regression head
        self.SA_enregr = MultiSelfAttentionBlock(name="MSA_enregr",  output_dim=self.output_dim_sa_enregr,
                             num_heads=self.num_heads_msa_enregr, ff_dim=self.ff_dim_msa_enregr, reduce="sum", **kwargs)

        self.dense_enregr = get_dense(name="dense_enregr", spec=self.layers_enregr+[1], act=self.activation,
                                     last_act=tf.keras.activations.linear, dropout=self.dropout, L2=self.l2_reg)

        #dropout layers
        self.gcn_dropout = tf.keras.layers.Dropout(self.dropout)
        # self.SA_windclass_input_dropout = tf.keras.layers.Dropout(self.dropout)
        self.SA_windclass_output_dropout = tf.keras.layers.Dropout(self.dropout)
        self.enregr_dropout = tf.keras.layers.Dropout(self.dropout)
        # Layer normalizations
        self.gcn_output_layernorm = tf.keras.layers.LayerNormalization(name="gcn_output_layernorm", epsilon=1e-3)
        # self.input_SA_windclass_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.SA_windclass_layernorm = tf.keras.layers.LayerNormalization(name="SA_windclass_layernorm", epsilon=1e-3)
        self.SA_enregr_layernorm = tf.keras.layers.LayerNormalization(name="SA_enregr_layernorm", epsilon=1e-3)
        # Concatenation layers
        self.concat_wind_feats = tf.keras.layers.Concatenate(axis=-1)
        self.concat_inputs = tf.keras.layers.Concatenate(axis=-1)
        self.concat_inputs_clclass = tf.keras.layers.Concatenate(axis=-1)

    def get_config(self):
        return {
            "layers_input": self.graphbuild.layers_input,
            "output_dim_rechits": self.graphbuild.output_dim_rechits,
            #"coord_dim" : self.graphbuild.coord_dim,
            #"coord_di_sa" : self.graphbuild.coord_dim_sa,
            "nconv_rechits": self.graphbuild.nconv_rechits,

            "num_heads_msa_encoder": self.num_heads_msa_encoder,
            "ff_dim_msa_encoder": self.ff_dim_msa_encoder,
            "nrepeat_msa_encoder": self.nrepeat_msa_encoder,

            "num_heads_msa_windclass": self.num_heads_msa_windclass,
            "ff_dim_msa_windclass": self.ff_dim_msa_windclass,

            "num_heads_msa_enregr": self.num_heads_msa_enregr,
            "ff_dim_msa_enregr": self.ff_dim_msa_enregr,

            "layers_clclass": self.layers_clclass, 
            "layers_windclass": self.layers_windclass,
            "layers_enregr": self.layers_enregr,
           
            "output_dim_nodes": self.output_dim_nodes,
            "output_dim_gconv": self.output_dim_gconv,
            "output_dim_msa_encoder": self.output_dim_msa_encoder,
            "output_dim_sa_windclass": self.output_dim_sa_windclass,
            "output_dim_sa_enregr": self.output_dim_sa_enregr,
            
            "n_windclasses": self.n_windclasses,

            "nconv": self.nconv,
            "l2_reg": self.l2_reg,
            "dropout": self.dropout,
            "activation": self.activation,
            "name": self.name,
            "loss_weights": self.loss_weights
        }

    def call(self, inputs, training):
        cl_X_initial, wind_X, cl_hits, is_seed, mask_cls, mask_rechits = inputs 
       
        # Concatenate the seed label on clusters features
        cl_X_initial = tf.concat([tf.cast(is_seed[:,:,tf.newaxis], tf.float32), cl_X_initial], axis=-1)
        # Call the graphbuilding step: compute the rechit summary,clusters features and adjacency matrix
        cl_X,  output_rechits = self.graphbuild(cl_X_initial, cl_hits, mask_rechits, mask_cls, training)
        #cl_X now is the latent cluster+rechits representation
        mask_cls_to_apply = mask_cls[:,:,tf.newaxis]
        # Apply first the graph convolution
        #out_gcn = self.GCN(cl_X, adj) 
        # Dropout + normalization
        #out_gcn = self.gcn_dropout(out_gcn, training=training)
        #out_gcn = self.gcn_output_layernorm(out_gcn) * mask_cls_to_apply

        # Clusters classification block
        # Apply self attention: output already masked internally
        output_MSA_encoder = cl_X 
        att_weights_encoders = []
        for encoder in self.MSA_encoders:
            output_MSA_encoder, attweights = encoder(output_MSA_encoder, mask_cls, training)
            att_weights_encoders.append(attweights)
        # Concat with the cluster inputs features
        out_encoder_and_inputs = self.concat_inputs([cl_X, output_rechits, output_MSA_encoder] )
        
        # No need to normalize since the self-attention layer
        # interanally has skip connections and add+norms
        clclass_out = self.dense_clclass(out_encoder_and_inputs, training=training) * mask_cls_to_apply

        out_encoder_and_inputs_and_clclass = self.concat_inputs_clclass([out_encoder_and_inputs,clclass_out])

        # Windows classification block
        # Apply Self-attention for window classification
        out_SA_windcl, att_weights_windclass = self.SA_windclass(out_encoder_and_inputs_and_clclass, mask_cls, training)
        # Concatenate with window level features
        out_SA_windcl = self.concat_wind_feats([wind_X, out_SA_windcl])
        # Norm before dense for wind classification because the sum is performed in the SA layer
        out_SA_windcl = self.SA_windclass_output_dropout(out_SA_windcl, training=training)
        out_SA_windcl = self.SA_windclass_layernorm(out_SA_windcl)
        windclass_out = self.dense_windclass(out_SA_windcl, training=training)

        # Energy regression block
        # Weight the input to en regre buy cluster selection probability 
        # input_en_regr = input_en_regr * clclass_out # mask already applied
        out_SA_enregr, att_weights_enregr = self.SA_enregr(out_encoder_and_inputs_and_clclass, mask_cls, training)
        out_SA_enregr = self.enregr_dropout(out_SA_enregr, training=training)
        out_SA_enregr = self.SA_enregr_layernorm(out_SA_enregr)
        # apply dense
        out_SA_enregr = self.dense_enregr(out_SA_enregr, training=training)
        
        return (clclass_out, windclass_out, out_SA_enregr), mask_cls, \
               ( cl_X, output_rechits, \
                  output_MSA_encoder, out_SA_windcl,\
                 att_weights_encoders,att_weights_windclass, att_weights_enregr)

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

    # Customized training loop
    # Based on https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
    @tf.function
    def train_step(self, data):
        x, y, w = data 
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss_clusters = clusters_classification_loss(y, y_pred, w)
            loss_softF1 =  soft_f1_score(y,y_pred, w, self.loss_weights["softF1_beta"])
            loss_windows = window_classification_loss(y, y_pred, w)
            loss_en_resol, loss_en_softF1 = energy_loss(y, y_pred, w, self.loss_weights["softF1_beta"])
            loss_en_regr = energy_regression_loss(y, y_pred, w)
            # Total loss function
            loss =  self.loss_weights["clusters"] * loss_clusters +\
                    self.loss_weights["window"] * loss_windows + \
                    self.loss_weights["softF1"] * loss_softF1 + \
                    self.loss_weights["en_resol"] * loss_en_resol+ \
                    self.loss_weights["en_softF1"] *  loss_en_softF1 + \
                    self.loss_weights["en_regr"] * loss_en_regr + \
                    sum(self.losses)

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
        # mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_softF1": self.loss3_tracker.result(),
                "loss_en_resol": self.loss4_tracker.result(),
                "loss_en_softF1": self.loss5_tracker.result(),
                "loss_en_regr": self.loss6_tracker.result()}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y, w  = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss_clusters = clusters_classification_loss(y, y_pred, w )
        loss_softF1 =  soft_f1_score(y,y_pred, w, self.loss_weights["softF1_beta"])
        loss_windows = window_classification_loss(y, y_pred, w)
        loss_en_resol, loss_en_softF1 = energy_loss(y, y_pred, w, self.loss_weights["softF1_beta"])
        loss_en_regr = energy_regression_loss(y, y_pred, w)
        # Total loss function
        loss =  self.loss_weights["clusters"] * loss_clusters +\
                self.loss_weights["window"] * loss_windows + \
                self.loss_weights["softF1"] * loss_softF1 + \
                self.loss_weights["en_resol"] * loss_en_resol + \
                self.loss_weights["en_softF1"] *  loss_en_softF1 + \
                self.loss_weights["en_regr"] * loss_en_regr + \
                sum(self.losses)
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss_clusters)
        self.loss2_tracker.update_state(loss_windows)
        self.loss3_tracker.update_state(loss_softF1)
        self.loss4_tracker.update_state(loss_en_resol)
        self.loss5_tracker.update_state(loss_en_softF1)
        self.loss6_tracker.update_state(loss_en_regr)
        # mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(),
                "loss_clusters": self.loss1_tracker.result(),
                "loss_windows": self.loss2_tracker.result(),
                "loss_softF1": self.loss3_tracker.result(),
                "loss_en_resol": self.loss4_tracker.result(),
                "loss_en_softF1": self.loss5_tracker.result(),
                "loss_en_regr": self.loss6_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.loss1_tracker, self.loss2_tracker, self.loss3_tracker,
                self.loss4_tracker, self.loss5_tracker, self.loss6_tracker]




############################################
############################################
## Loss functions
@tf.function
def clusters_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor),  mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata = y_true
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass[:,:,tf.newaxis], dense_clclass, from_logits=True) * mask_cls
    reduced_loss = tf.reduce_sum(tf.reduce_mean(class_loss, axis=-1) * weight) / tf.reduce_sum(weight)
    # This should be fixed by doing an average over the not masked clusters
    return reduced_loss 

@tf.function
def energy_weighted_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    cl_ets = cl_X[:,:,1]
    # matched_window = tf.cast(y_metadata[:,-1]!=0, tf.float32)
    # compute the weighting mean of the loss based on the energy of each seed in the window
    cl_ets_weights = cl_ets / tf.reduce_sum(cl_ets, axis=-1)[:,tf.newaxis]
    class_loss = tf.keras.losses.binary_crossentropy(y_clclass, dense_clclass, from_logits=True) * mask_cls
    weighted_loss = class_loss * cl_ets_weights
    # mean over the batch
    reduced_loss = tf.reduce_sum(tf.reduce_sum(weighted_loss, axis=-1) * weight) / tf.reduce_sum(weight) 
    return reduced_loss

@tf.function
def window_classification_loss(y_true, y_pred, weight):
    (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    # Only window multi-class classification
    windclass_loss = tf.keras.losses.categorical_crossentropy(y_windclass, dense_windclass, from_logits=True)
    reduced_loss =   tf.reduce_sum(windclass_loss * weight) / tf.reduce_sum(weight)
    return reduced_loss


# def energy_loss(y_true, y_pred):
#     (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
#     y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
#     y_target = tf.cast(y_clclass, tf.float32) 
#     # matched_window = tf.cast(y_metadata[:,-1]!=0, tf.float32)

#     pred_prob = tf.nn.sigmoid(dense_clclass)
#     diff = tf.math.abs(y_target - pred_prob)
#     Et = cl_X[:,:,1:2]
#     missing_en = Et * diff * y_target
#     spurious_en =  Et * diff * (1 - y_target)
#     reduced_loss_missing = tf.reduce_mean(tf.squeeze(tf.reduce_sum(missing_en, axis=1))) 
#     reduced_loss_spurious =  tf.reduce_mean(tf.squeeze(tf.reduce_sum(spurious_en, axis=1))) 
#     return reduced_loss_missing,reduced_loss_spurious
@tf.function
def energy_loss(y_true, y_pred, weight, beta=1):
    (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    y_target = tf.cast(y_clclass, tf.float32) 
    cl_en = Et = cl_X[:,:,0:1]
    En_sim_good = y_metadata[:,4]
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
    (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    y_target = tf.cast(y_clclass, tf.float32) 
    # matched_window = tf.cast(y_metadata[:,-1]!=0, tf.float32)

    pred_prob = tf.nn.sigmoid(dense_clclass)
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
    (dense_clclass, dense_windclass, en_regr_factor), mask_cls, _  = y_pred
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    cl_ens = cl_X[:,:,0]
    pred_en =  tf.reduce_sum(cl_ens * tf.squeeze(tf.cast(tf.nn.sigmoid(dense_clclass) > 0.5 , tf.float32)), axis=-1)
    calib_pred_en =  pred_en * tf.squeeze(en_regr_factor)
    true_en_gen = y_metadata[:,2]  # en_true_gen

    loss = huber_loss(true_en_gen, calib_pred_en, 5, weight) + quantile_loss(true_en_gen, calib_pred_en,weight )
    return loss

