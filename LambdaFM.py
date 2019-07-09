'''
LambdaFM: Learning Optimal Ranking with Factorization Machines Using Lambda Surrogates

@author:
Xin xin, Fajie Yuan
@references:
'''
import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--neg', type=int, default=5,
                        help='number of negative samples in which to chose the largest score')

    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        #self.loss_type = loss_type
        self.user_field_M=user_field_M
        self.item_field_M=item_field_M
        #self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        #self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features= tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features= tf.placeholder(tf.int32, shape=[None, None])
            #self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            #self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)
            #self.scores=tf.placeholder(tf.float32,shape=[1,None])
            #self.true_item=tf.placeholder(tf.int32, shape=[1])
            #self.topk=tf.placeholder(tf.int32)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part for positive (u,i)_____________
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.positive_features)
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings, 1)
            self.summed_positive_emb=tf.add(self.summed_user_emb,self.summed_item_positive_emb)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)

            self.squared_user_emb=tf.square(self.user_feature_embeddings)
            self.squared_item_positiv_emb=tf.square(self.positive_feature_embeddings)
            self.squared_user_emb_sum=tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positiv_emb, 1)
            self.squared_positive_emb_sum=tf.add(self.squared_user_emb_sum,self.squared_item_positive_emb_sum)

            # ________ FM part for positive (u,i)__________
            self.FM_positive = 0.5 * tf.subtract(self.summed_positive_emb_square, self.squared_positive_emb_sum)  # None * K
            #if self.batch_norm:
                #FM = self.batch_norm_layer(FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_positive = tf.nn.dropout(self.FM_positive, self.dropout_keep)  # dropout at the FM layer

            # _________positive_________
            self.Bilinear_positive = tf.reduce_sum(self.FM_positive, 1, keepdims=True)  # None * 1
            self.user_feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features),
                                              1)  # None * 1
            self.item_feature_bias_positive = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features),
                1)  # None * 1
            #Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.positive = tf.add_n([self.Bilinear_positive, self.user_feature_bias, self.item_feature_bias_positive])  # None * 1

            # _________ sum_square part for negative (u,j)_____________
            self.negative_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                 self.negative_features)
            self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings, 1)
            self.summed_negative_emb = tf.add(self.summed_user_emb, self.summed_item_negative_emb)
            self.summed_negative_emb_square = tf.square(self.summed_negative_emb)

            self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)
            self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb, 1)
            self.squared_negative_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_negative_emb_sum)

            # ________ FM part for negative (u,j)__________
            self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square, self.squared_negative_emb_sum)  # None * K
            #if self.batch_norm:
                #FM = self.batch_norm_layer(FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep)  # dropout at the FM layer

            # _________negative_________
            self.Bilinear_negative = tf.reduce_sum(self.FM_negative,1,keepdims=True)  # None * 1
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features),
                1)  # None * 1
            self.negative = tf.add_n([self.Bilinear_negative, self.user_feature_bias, self.item_feature_bias_negative])  # None * 1

            # Compute the loss.
            self.loss=-tf.log(tf.sigmoid(self.positive-self.negative))
            self.loss=tf.reduce_sum(self.loss)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            #whether in TopK
            #self.hit=tf.nn.in_top_k(self.scores,self.true_item, self.topk)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print "#params: %d" % total_parameters

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            #bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub,ib  = sess.run([user_feature_embeddings,item_feature_embeddings,user_feature_bias,item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
            #all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    '''def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(train_data['Y']) - batch_size)
        X_user, X_positive, X_negative, Y = [], [], [], []
        # forward get sample
        i = start_index
        all_items = data.binded_items.values()
        while len(X_user) < batch_size and i < len(train_data['X_user']):
            if len(train_data['X_user'][i]) == len(train_data['X_user'][start_index]):
                Y.append([train_data['Y'][i]])
                #get user feature map
                user_feature= train_data['X_user'][i].strip().split('-')
                X_user.append([data.user_fields[item] for item in user_feature[0:]])
                #get positive item feature map
                item_feature= train_data['X_item'][i].strip().split('-')
                X_positive.append([data.item_fields[item] for item in item_feature[0:]])
                #uniform sampler
                user_id=data.binded_users[train_data['X_user'][i]] # get userID
                pos=data.user_positive_list[user_id]   #get positive list for the userID
                candidates = list(set(all_items) -set(pos))  #get negative set
                neg = np.random.choice(candidates)  #uniform sample a negative itemID from negative set
                negative_feature=data.item_map[neg].strip().split('-') #get negative item feature
                X_negative.append([data.item_fields[item] for item in negative_feature[0:]])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_user) < batch_size and i >= 0:
            if len(train_data['X_user'][i]) == len(train_data['X_user'][start_index]):
                Y.append([train_data['Y'][i]])
                #get user feature map
                user_feature= train_data['X_user'][i].strip().split('-')
                X_user.append([data.user_fields[item] for item in user_feature[0:]])
                #get positive item feature map
                item_feature= train_data['X_item'][i].strip().split('-')
                X_positive.append([data.item_fields[item] for item in item_feature[0:]])
                #uniform sampler
                user_id=data.binded_users[train_data['X_user'][i]] # get userID
                pos=data.user_positive_list[user_id]   #get positive list for the userID
                candidates = list(set(all_items) -set(pos))  #get negative set
                neg = np.random.choice(candidates)  #uniform sample a negative itemID from negative set
                negative_feature=data.item_map[neg].strip().split('-') #get negative item feature
                X_negative.append([data.item_fields[item] for item in negative_feature[0:]])
                i = i + 1
            else:
                break
        return {'X_user': X_user,'X_positive': X_positive,'X_negative': X_negative, 'Y': Y}'''
    #negative sampling
    def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        #get sample
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            #uniform sampler
            user_features="-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id=data.binded_users[user_features] # get userID
            pos=data.user_positive_list[user_id]   #get positive list for the userID
            candidates = list(set(all_items) -set(pos))  #get negative set
            #lambda sampler
            neg = np.random.choice(candidates,args.neg)
            user_sample=[]
            item_sample=[]
            for neg_id in neg:
                user_sample.append(train_data['X_user'][index])
                negative_feature = data.item_map[neg_id].strip().split('-')  # get negative item feature
                item_sample.append([int(item) for item in negative_feature[0:]])
            feed_dict = {self.user_features: user_sample, self.positive_features: item_sample, self.train_phase: True,
                         self.dropout_keep:self.keep}
            neg_score=self.sess.run((self.positive),feed_dict=feed_dict).reshape(args.neg)
            chosen_item= item_sample[np.argmax(neg_score)]
            X_negative.append(chosen_item)
        return {'X_user': X_user,'X_positive': X_positive,'X_negative': X_negative}

    '''def shuffle_in_unison_scary(self, a, b):  # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)'''

    def train(self, Train_data):  # fit a dataset
        for epoch in xrange(self.epoch):
            total_loss=0
            t1 = time()
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in xrange(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                loss=self.partial_fit(batch_xs)
                total_loss= total_loss+loss
            t2 = time()
            print("the total loss in %d th iteration is: %f" %(epoch, total_loss))
            #model.evaluate()
        if self.pretrain_flag < 0:
            print "Save model to file as pretrain."
            self.saver.save(self.sess, self.save_file)
        #if self.pretrain_flag < 0:
            #print "Save model to file as pretrain."
            #self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = 0
        mrr=0
        ndcg=0
        rank= []
        for index in xrange(len(data.Test_data['X_user'])):
            user_features = data.Test_data['X_user'][index]
            item_features = data.Test_data['X_item'][index]
            scores=model.get_scores_per_user(user_features)
            #get true item score
            true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
            true_item_score = scores[true_item_id]
            #delete visited scores
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]  # get userID
            visited = data.user_positive_list[user_id]  # get positive list for the userID
            scores = np.delete(scores, visited)
            #whether hit
            sorted_scores = sorted(scores, reverse=True)
            label = sorted_scores[args.topk - 1]
            if true_item_score>=label:
                count=count+1
                rank.append(sorted_scores.index(true_item_score) + 1)
            #print index
        hit_rate=float(count)/len(data.Test_data['X_user'])
        for item in rank:
            mrr=mrr+float(1.0)/item
            ndcg=ndcg+float(1.0)/np.log2(item+1)
        mrr=mrr/len(data.Test_data['X_user'])
        ndcg=ndcg/len(data.Test_data['X_user'])
        print("the Hit Rate is: %f" %hit_rate)
        print("the MRR is: %f" % mrr)
        print("the NDCG is: %f" % ndcg)

    #def test(self):
    #    X_user, X_item = [], []
    #    for index in xrange(len(data.Test_data['X_user'])):
    #        user_features = data.Test_data['X_user'][index]
    #        X_user.append(user_features)
    #        item_features = data.Test_data['X_item'][index]
    #        X_item.append(item_features)
    #    feed_dict = {self.user_features: X_user, self.positive_features: X_item,self.train_phase: False, self.dropout_keep:1.0}
    #    scores = self.sess.run((self.positive), feed_dict=feed_dict)
    #    print scores

    def get_scores_per_user(self, user_feature):  # evaluate the results for an user context, return scorelist
        #num_example = len(Testdata['Y'])
        # get score list for a userID, store in scorelist, indexed by itemID
        #scorelist=[]
        X_user, X_item= [],[]
        #X_item = []
        #Y=[[1]]
        all_items = data.binded_items.values()
        #true_item_id=data.binded_items[item]
        #user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],X_user)
        for itemID in xrange(len(all_items)):
            X_user.append(user_feature)
            item_feature=[int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)
        feed_dict = {self.user_features: X_user, self.positive_features: X_item,self.train_phase: False, self.dropout_keep: 1.0}
        scores=self.sess.run((self.positive),feed_dict=feed_dict)
        scores=scores.reshape(len(all_items))
        return scores

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print( "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s, batch_norm=%d, keep=%.2f"
                % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.keep_prob))

    save_file = '../pretrain/%s_%d' % (args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = FM(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
               args.batch_size, args.lr, args.lamda,  args.keep_prob, args.optimizer, args.batch_norm, args.verbose)
    #model.test()
    model.train(data.Train_data)
    #model.test()
    model.evaluate()
