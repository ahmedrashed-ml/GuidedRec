import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
import time
from collections import deque
import warnings
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import random
import pickle
import math
import tensorflow_addons as tfa
from sklearn import preprocessing
import tensorflow_ranking as tfr
import datetime


tf.reset_default_graph()



BATCH_SIZE = 250
NEGSAMPLES=1
USER_NUM = 943
ITEM_NUM = 1682
DIM = 50
EPOCH_MAX = 500
DEVICE = "/gpu:0"
PERC=0.9
SURROGATE=int(sys.argv[1])
SEED=int(sys.argv[2])
LOSSFUN=sys.argv[3]

print('GraphTFRank',"-",SURROGATE,"-",SEED,"-",LOSSFUN)

np.random.seed(SEED)

import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import preprocessing
tf.reset_default_graph()
def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)





def inferenceDense(phase,ufsize,ifsize,user_batch, item_batch,time_batch,idx_user,idx_item,ureg,ireg, user_num, item_num, dim=5,UReg=0.0,IReg=0.0, device="/cpu:0"):
    with tf.device(device):
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")  
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")  
          
          
        ul1mf=tf.layers.dense(inputs=user_batch, units=20,name='ul1mf',activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))  
        il1mf=tf.layers.dense(inputs=item_batch, units=20,name='il1mf',activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))  
        InferInputMF=tf.multiply(ul1mf, il1mf)  
 
        infer=tf.reduce_sum(InferInputMF, 1, name="inference")  

        regularizer = tf.reduce_sum(tf.add(UReg*tf.nn.l2_loss(ul1mf),IReg*tf.nn.l2_loss(il1mf), name="regularizer"))
    return infer,infer,infer,infer,regularizer
  

def optNDCG(NDCGscore,varlist, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        costNDCG=-NDCGscore
        train_NDCG=tf.train.AdamOptimizer(0.00001).minimize(costNDCG)
    return train_NDCG

def optLoss(NDCGscore,NDCGTrue, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        costLOSS=tf.nn.l2_loss(tf.subtract(NDCGscore, NDCGTrue))
        train_LOSS=tf.train.AdamOptimizer(0.000015).minimize(costLOSS)
    return train_LOSS,costLOSS


def optimization(infer, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):

    with tf.device(device):
        cost =tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,logits=infer)# tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return cost, train_op


def optimizationRank(infer, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):

    with tf.device(device):
        losfun =tfr.losses.make_loss_fn(LOSSFUN)
        rate_batch=tf.dtypes.cast(rate_batch, tf.float32)
        infer=tf.dtypes.cast(infer, tf.float32)
        cost=losfun(tf.reshape(rate_batch, [25, 10]) ,tf.reshape(infer, [25, 10]),None)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return cost, train_op  


def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])




def dcg_score(y_true, y_score, k=10, gains="exponential"):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def get_UserData100k():
    col_names = ["user", "age", "gender", "occupation","PostCode"]
    df = pd.read_csv('../Data/u.user', sep='|', header=None, names=col_names, engine='python')
    del df["PostCode"]
    df["user"]-=1
    df=pd.get_dummies(df,columns=[ "age", "gender", "occupation"])
    del df["user"]
    return df.values

def get_ItemData100k():
    col_names = ["movieid", "movietitle", "releasedate", "videoreleasedate","IMDbURL"
                ,"unknown","Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary"
                ,"Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller"
                ,"War","Western"]
    df = pd.read_csv('../Data/u.item', sep='|', header=None, names=col_names, engine='python')
    df['releasedate'] = pd.to_datetime(df['releasedate'])
    df['year'],df['month']=zip(*df['releasedate'].map(lambda x: [x.year,x.month]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['year']=df['year'].fillna(0.0)

    del df["month"]
    del df["movietitle"]
    del df["releasedate"]
    del df["videoreleasedate"]
    del df["IMDbURL"]
  
    df["movieid"]-=1
    del  df["movieid"]
    return df.values 

def getHR(ranklist, gtItem): 
    if(gtItem in ranklist):
      return 1 
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getNDCG_SIM(ranklist, gtItem,Sim):
    #print(ranklist)
    NewTestItem=-1
    ItemFeatRow=ITEMDATA[np.asarray([gtItem],dtype=np.int32),:]
    RankedFeatRows=ITEMDATA[np.asarray(ranklist,dtype=np.int32),:]
    
    simvals=cosine_similarity(RankedFeatRows,ItemFeatRow)
    simvals=(simvals+1)/2
    #print(simvals)
    simvals[simvals < Sim] = 0.0

    #print(simvals.shape)
    for i in range(len(ranklist)):
      if (simvals[i] > 0.0 and ranklist[i] != gtItem):
        NewTestItem=ranklist[i]
        break
      if (simvals[i] > 0.0 and ranklist[i] == gtItem):
        NewTestItem=gtItem
        break
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == NewTestItem:
            return math.log(2) / math.log(i+2)
    return 0

def my_elementwise_func(x):
    return tf.matmul(x,tf.transpose(x)) 

def Main(train,ItemData=False,UserData=False,Graph=True,lr=0.00003,ureg=0.0,ireg=0.0):

    AdjacencyUsers = [[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)]
    DegreeUsers = [[0 for x in range(1)] for y in range(USER_NUM)]
    DegreeUsersVec= [0 for y in range(USER_NUM)]
    VarUsersVec= [list() for y in range(USER_NUM)]
    VarUsersVals= [0 for y in range(USER_NUM)]
    
    AdjacencyItems = [[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)]
    DegreeItems = [[0 for x in range(1)] for y in range(ITEM_NUM)]
    DegreeItemsVec= [0 for y in range(ITEM_NUM)]
    VarItemsVec= [list() for y in range(ITEM_NUM)]
    VarItemsVals= [0 for y in range(ITEM_NUM)]
    if(Graph):   
      for index, row in train.iterrows():
        userid=int(row['user'])
        itemid=int(row['item'])
        AdjacencyUsers[userid][itemid]=row['rate']/5.0
        AdjacencyItems[itemid][userid]=row['rate']/5.0
        DegreeUsersVec[userid]+=1
        DegreeItemsVec[itemid]+=1
        DegreeUsers[userid][0]+=1
        DegreeItems[itemid][0]+=1
        VarUsersVec[userid].append(row['rate'])
        VarItemsVec[itemid].append(row['rate'])
      
      
      for i in range(USER_NUM):
        VarUsersVals[i]=np.var(VarUsersVec[i]) 
      for i in range(ITEM_NUM):
        VarItemsVals[i]=np.var(VarItemsVec[i])   
      print(np.max(DegreeUsersVec) )
      print(np.min(DegreeUsersVec) )
      print(np.max(DegreeItemsVec) )
      print(np.min(DegreeItemsVec) )   
      
      DUserMax=np.amax(DegreeUsers) 
      DItemMax=np.amax(DegreeItems)
      DegreeUsers=np.true_divide(DegreeUsers, DUserMax)
      DegreeItems=np.true_divide(DegreeItems, DItemMax)
      
      
      AdjacencyUsers=np.asarray(AdjacencyUsers)
      AdjacencyItems=np.asarray(AdjacencyItems)
    
    if(Graph):
      UserFeatures= np.concatenate((np.identity(USER_NUM), AdjacencyUsers,DegreeUsers), axis=1) 
      ItemFeatures= np.concatenate((np.identity(ITEM_NUM), AdjacencyItems,DegreeItems), axis=1)
    else:
      UserFeatures= np.identity(USER_NUM) 
      ItemFeatures= np.identity(ITEM_NUM)      


   
    print(UserFeatures.shape)
    print(ItemFeatures.shape)
    if(UserData):
      UsrDat=get_UserData()
      UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 

    if(ItemData):
      ItmDat=get_ItemData()
      ItemFeatures=np.concatenate((ItemFeatures,ItmDat), axis=1) 
      
    print(UserFeatures.shape)
    print(ItemFeatures.shape)
    

    print("Finish")
    samples_per_batch = len(train) // int(BATCH_SIZE/(NEGSAMPLES+1))
    

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    y_batch = tf.placeholder(tf.float64, shape=[None,1], name="y")
    m_batch = tf.placeholder(tf.float64, shape=[None,1], name="m")
    d_batch = tf.placeholder(tf.float64, shape=[None,1], name="d")
    dw_batch = tf.placeholder(tf.float64, shape=[None,1], name="dw")
    dy_batch = tf.placeholder(tf.float64, shape=[None,1], name="dy")
    w_batch = tf.placeholder(tf.float64, shape=[None,1], name="w")
    ndcgTargets = tf.placeholder(tf.float64, shape=[25,1], name="ndcgTargets")
    
    
    time_batch=tf.concat([y_batch, m_batch,d_batch,dw_batch,dy_batch,w_batch], 1)
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')
    modelphase= tf.placeholder(tf.bool, name='modelphase')
    
    w_user = tf.constant(UserFeatures,name="userids", shape=[USER_NUM,UserFeatures.shape[1]],dtype=tf.float64)
    w_item = tf.constant(ItemFeatures,name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]],dtype=tf.float64)


    infer,infern,inferSig,varlist, regularizer = inferenceDense(phase,UserFeatures.shape[1],ItemFeatures.shape[1],user_batch, item_batch,time_batch,w_user,w_item,ureg,ireg, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    
    infernStopped=tf.stop_gradient(infern)

    Infere2d=tf.reshape(infer, [25, 10])
    rate2d=tf.reshape(rate_batch, [25, 10])
    NDCG2d=tf.concat([Infere2d, rate2d], 1)
    
    Infere3d=tf.reshape(infer, [25, 10,1])
    rate3d=tf.reshape(rate_batch, [25, 10,1])   
    NDCG3d=tf.concat([Infere3d, rate3d], 2) 


    Infere2dn=tf.reshape(infernStopped, [25, 10])
    NDCG2dn=tf.concat([Infere2dn, rate2d], 1)
    
    Infere3dn=tf.reshape(infernStopped, [25, 10,1]) 
    NDCG3dn=tf.concat([Infere3dn, rate3d], 2) 


    ####################################################
    with tf.variable_scope("Surrogate"):
      l11=tf.layers.dense(inputs=NDCG2dn, units=20,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l11")
      l11=tf.layers.dropout(l11,rate=0.0,training=phase)
      l12=tf.layers.dense(inputs=l11, units=20,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l12")
      #l12=tf.layers.dropout(l12,rate=0.1,training=phase)
      ##
      b21=tf.reshape(NDCG3dn, [-1, 2])
      l21=tf.layers.dense(inputs=b21, units=20,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l21")
      #l21=tf.layers.dropout(l21,rate=0.1,training=phase)
      b22=tf.reshape(l21, [-1,10, 20])
      b22=tf.map_fn(my_elementwise_func,  b22)
      b23=tf.reduce_mean(b22,axis=1)
      l22=tf.layers.dense(inputs=b23, units=20,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l22")
      l31=tf.multiply(l22, l12)
      ##
      l4=tf.layers.dense(inputs=l31, units=8,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l4")
      l4=tf.layers.dropout(l4,rate=0.0,training=phase)
      NDCGScore=tf.layers.dense(inputs=l4, units=1,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="NDCGScore")
      #################
      l11n=tf.layers.dense(inputs=NDCG2d, units=20,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l11")
      l12n=tf.layers.dense(inputs=l11n, units=20,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l12")
      ##
      b21n=tf.reshape(NDCG3d, [-1, 2])
      l21n=tf.layers.dense(inputs=b21n, units=20,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l21")
      b22n=tf.reshape(l21n, [-1,10, 20])
      b22n=tf.map_fn(my_elementwise_func,  b22n)
      b23n=tf.reduce_mean(b22n,axis=1)
      l22n=tf.layers.dense(inputs=b23n, units=20,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l22")
      l31n=tf.multiply(l22n, l12n)
      ##
      l4n=tf.layers.dense(inputs=l31n, units=8,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="l4")
      NDCGScoren=tf.layers.dense(inputs=l4n, units=1,reuse=True,trainable=False,activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="NDCGScore")


    ####################################################
    #NDCGScore=NDCGModel(NDCG2d,NDCG3d,modelphase)
    print(NDCGScore)
    #global_step = tf.contrib.framework.get_or_create_global_step()
    cost, train_op = optimizationRank(inferSig, regularizer, rate_batch, learning_rate=lr, reg=0.09, device=DEVICE)
    train_NDCG=optNDCG(NDCGScoren,varlist, device=DEVICE)
    train_Loss,loss_cost=optLoss(NDCGScore,ndcgTargets, device=DEVICE)    

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    finalerror=-1
    degreelist=list()
    predlist=list()
    degreelist0=list()
    predlist0=list()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("epoch, train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20")
        now = datetime.datetime.now()
        textTrain_file = open("./Output/"+now.strftime('%Y%m%d%H%M%S')+'_GraphTFRank'+"_"+str(SURROGATE)+"_"+str(SEED)+"_"+LOSSFUN +".txt", "w",newline='')
        errors = deque(maxlen=samples_per_batch)
        losscost = deque(maxlen=samples_per_batch)
        truecost = deque(maxlen=samples_per_batch)
        start = time.time()
        print(samples_per_batch)
        for i in range(EPOCH_MAX * samples_per_batch):
            #users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = GetTrainSample(dictUsers,BATCH_SIZE,10)
            ################################
            
            if(SURROGATE):
              if((i // samples_per_batch)<0):
                _, pred_batch,cst = sess.run([train_op, infer,cost], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      phase:True})
                inf2d,rt2d = sess.run([Infere2d, rate2d], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      phase:True,
                                                                      modelphase:True})

                totndcg=np.asarray( [ndcg_score(x,y,k=10) for x, y in zip(rt2d.tolist(), inf2d.tolist())])
                _,lscost = sess.run([train_Loss, loss_cost], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      ndcgTargets: totndcg.reshape((-1, 1)),                                                                   
                                                                      phase:True,
                                                                      modelphase:True})
                          
                losscost.append(lscost)
                truecost.append(totndcg)              
              else:
                _, pred_batch,cst = sess.run([train_op, infer,cost], feed_dict={user_batch: users,
                                                        item_batch: items,
                                                        rate_batch: rates,
                                                        phase:True})
                inf2d,rt2d = sess.run([Infere2d, rate2d], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      phase:True,
                                                                      modelphase:True})

                totndcg=np.asarray( [ndcg_score(x,y,k=10) for x, y in zip(rt2d.tolist(), inf2d.tolist())])
                _,lscost = sess.run([train_Loss, loss_cost], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      ndcgTargets: totndcg.reshape((-1, 1)),                                                                   
                                                                      phase:True,
                                                                      modelphase:True})
                          
                losscost.append(lscost)
                truecost.append(totndcg)
                #############################train_NDCG,
                _,pred_batch,cst = sess.run([train_NDCG, infer,NDCGScore], feed_dict={user_batch: users,
                                                                      item_batch: items,
                                                                      rate_batch: rates,
                                                                      phase:True,
                                                                      modelphase:False})
                errors.append(cst)
              #print(pred_batch)
            else:
            ################################
              _, pred_batch,cst = sess.run([train_op, infer,cost], feed_dict={user_batch: users,
                                                                    item_batch: items,
                                                                    rate_batch: rates,
                                                                    phase:True})
              losscost.append(0)
              errors.append(0)
              truecost.append(0)
              pred_batch = clip(pred_batch)
            ################################

            #print(i,' ',samples_per_batch)
            if i % samples_per_batch == 0:
                #print('-----')

                train_err = np.mean(errors)
                train_err1 = np.mean(truecost)
                train_err2 = np.mean(losscost)
                totalhits=0
                ############
                correcthits10=0
                correcthits20=0
                correcthits5=0
                ndcg10=0
                ndcg20=0
                ndcg5=0
                ndcg10_50=0
                ndcg10_90=0
                ndcg10_80=0
                ndcg10_95=0
                ###########
                for userid in dictUsers: 
                  items=dictUsers[userid][3]
                  TestItem=dictUsers[userid][2][0]
                  TestUser=userid
                  users=np.repeat(userid, 100)
                  items=dictUsers[userid][3]
                  pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                          item_batch: items,                                                                                             
                                                          phase:False}) 
                  
                  sorteditems=[x for _, x in sorted(zip(pred_batch,items), key=lambda pair: pair[0],reverse=True)]
                  #######
                  topitems10=sorteditems[:5]
                  correcthits10=correcthits10+getHR(sorteditems[:10],TestItem)
                  correcthits20=correcthits20+getHR(sorteditems[:20],TestItem)
                  correcthits5=correcthits5+getHR(sorteditems[:5],TestItem)

                  ndcg10=ndcg10+getNDCG(sorteditems[:10],TestItem)
                  ndcg20=ndcg20+getNDCG(sorteditems[:20],TestItem)
                  ndcg5=ndcg5+getNDCG(sorteditems[:5],TestItem)

                  ndcg10_50=ndcg10_50#+getNDCG_SIM(topitems10,TestItem,0.5)
                  ndcg10_80=ndcg10_80#+getNDCG_SIM(topitems10,TestItem,0.8)
                  ndcg10_90=ndcg10_90#+getNDCG_SIM(topitems10,TestItem,0.9)                  
                  ndcg10_95=ndcg10_95#+getNDCG_SIM(topitems10,TestItem,0.95) 

                  totalhits=totalhits+1
                  ############
                HitRatio10 = correcthits10/ totalhits
                HitRatio20 = correcthits20/ totalhits
                HitRatio5 = correcthits5/ totalhits
                NDCG10 = ndcg10/ totalhits
                NDCG20 = ndcg20/ totalhits
                NDCG5 = ndcg5/ totalhits

                NDCG_50 = ndcg10_50/ totalhits
                NDCG_80 = ndcg10_80/ totalhits
                NDCG_90 = ndcg10_90/ totalhits
                NDCG_95 = ndcg10_95/ totalhits
                print("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20))                
                textTrain_file.write("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20) +'\n')            
                textTrain_file.flush()

        textTrain_file.close()  
    degreelist,predlist=zip(*sorted(zip(degreelist, predlist)))  
    return 


def GenNDCGSamples():

  #print(ndcg_score([1, 0, 1,0], [0.9, 0.6, 0.7,0.65],k=5))

  yhats=np.random.rand(100,10)
  trues=np.zeros((100,10))

  for row in trues:
    row[np.random.choice(range(10), 5, replace=False)]=1

  ndcgs=0
  totndcg=np.asarray( [ndcg_score(x,y) for x, y in zip(trues.tolist(), yhats.tolist())])
  comp=np.asarray([[[l,m] for l,m in zip(x, y) ] for x, y in zip(trues.tolist(), yhats.tolist())])

  return trues,yhats,comp,totndcg.reshape((-1, 1))

def GetTrainSampleold(DictUsers,BatchSize=1000,topn=10):
  trainusers=list()
  trainitems=list()
  traintargets=list()
  numusers=int(BatchSize/(topn))
  #print(numusers)
  for i in range(numusers):
    batchusers=random.randint(0,USER_NUM-1)
    while len(DictUsers[batchusers][0])==0:
      batchusers=random.choice(list(DictUsers.keys()))
    trainusers.extend(np.repeat(batchusers, topn))
    ##Pos
    
    trainitems.extend(np.random.choice(DictUsers[batchusers][0], int(topn/2),replace=True))
    traintargets.extend(np.ones(int(topn/2)))
    ##Neg
    trainitems.extend(np.random.choice(DictUsers[batchusers][1], int(topn/2),replace=True))
    traintargets.extend(np.zeros(int(topn/2))) 
    
  trainusers=np.asarray(trainusers)
  trainitems=np.asarray(trainitems)    
  traintargets=np.asarray(traintargets)
  return trainusers,trainitems,traintargets 

def GetTrainSample(DictUsers,BatchSize=1000,topn=10):
  trainusers=list()
  trainitems=list()
  traintargets=list()
  numusers=int(BatchSize/(topn))
  #print(numusers)
  for i in range(numusers):
    batchusers=random.randint(0,USER_NUM-1)
    while len(DictUsers[batchusers][0])==0:
      batchusers=random.choice(list(DictUsers.keys()))
    trainusers.extend(np.repeat(batchusers, topn))
    ##Pos
    
    trainitems.extend(np.random.choice(DictUsers[batchusers][0], int(5),replace=True))
    traintargets.extend(np.ones(int(5)))
    ##Neg
    trainitems.extend(np.random.choice(DictUsers[batchusers][1], int(5),replace=True))
    traintargets.extend(np.zeros(int(5))) 
    
  trainusers=np.asarray(trainusers)
  trainitems=np.asarray(trainitems)    
  traintargets=np.asarray(traintargets)
  return trainusers,trainitems,traintargets    

ITEMDATA=get_ItemData100k()

dictUsers=load_data("../Data/UserDict.dat")
df_train=load_data("../Data/RankData.dat")
print(len(df_train))
print(df_train.shape)
warnings.filterwarnings('ignore')
Main(df_train,ItemData=False,UserData=False,Graph=False,lr=0.001,ureg=0.0,ireg=0.0)
tf.reset_default_graph() 
