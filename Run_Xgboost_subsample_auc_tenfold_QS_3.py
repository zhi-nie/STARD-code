# Run STARD using Xgboost
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

def PerfStat_logistic(Y, Y_pred):
    #pos_class = 1
    #neg_class = 0
    y = Y.ravel()
    y_pred = Y_pred.ravel() 
    s =  np.logical_and(y == y_pred, y_pred == 1)
    tp = np.sum( s == True  )
    fp = np.sum( np.logical_and( y != y_pred , y_pred == 1) )
    tn = np.sum( np.logical_and(y == y_pred, y_pred == -1 ))
    fn = np.sum( np.logical_and(y != y_pred , y_pred == -1))
    total = tp + fp + tn + fn
    acc = float(tp + tn) / float(total)
    recall =  float(tp) / float(tp + fn)
    sst =  recall
    spc= float(tn) /float(tn + fp);
    return(acc,spc,sst)
# load data from matlab files


def Unbalanced_classify(X, Y, CV_ID, tr_id, param, subsample, Ycp):
    perf = np.empty((10,16))
    Y1 = Y
    Y1 = (Y1 + 1 ) / 2  
    for i in range(10):
        tr_ind = np.where(CV_ID!= (i + 1) )
        te_ind = np.where(CV_ID == (i + 1))

        
        X_tr = X[tr_ind[0],:]
        Y_tr = Y[tr_ind[0]]
        X_te = X[te_ind[0],:]
        Y_te = Y[te_ind[0]]    
        Y1_tr = Y1[tr_ind[0]]
        Y1_te = Y1[te_ind[0]] 
        
    # build a model
    
        dtest = xgb.DMatrix(X_te, label = Y1_te)
    
        num_round = 5
        preds = np.zeros((len(Y_te),subsample))
        preds_rf = np.zeros((len(Y_te),1)).ravel()
        preds_lr = np.zeros((len(Y_te),1)).ravel()
        preds_gbdt = np.zeros((len(Y_te),1)).ravel()
        
        
        preds_rf_prob = np.zeros((len(Y_te),2))         
        preds_lr_prob = np.zeros((len(Y_te),2))
        preds_gbdt_prob = np.zeros((len(Y_te),2))
        
        for j in range(subsample): 
            tr_idx = tr_id[0][i][0][j].ravel() -1
            X_tr_sub = X_tr[tr_idx,: ]
            Y_tr_sub = Y_tr[tr_idx]
            #print sum(np.where(Y_tr_sub == 1))
            Y1_tr_sub = Y1_tr[tr_idx]
            #t1 =np.sum(Y_tr_sub == 1)
            #t2 = np.sum(Y_tr_sub == -1)
            #Xgboost
            dtrain = xgb.DMatrix( X_tr_sub, label = Y1_tr_sub)      
            bst = xgb.train(param, dtrain, num_round)
            preds[:,j] = bst.predict(dtest)
            #random forest
            clf = RandomForestClassifier(n_estimators=500, n_jobs = 5)
            clf.fit(X_tr_sub, Y_tr_sub.ravel())
            proba = clf.predict_proba(X_te)
            pred_prob = np.multiply(np.amax(proba,axis = 1), 
                                                clf.classes_.take(np.argmax(proba,axis = 1),axis = 0))
            pred_label = clf.classes_.take(np.argmax(proba,axis = 1),axis = 0)  
            #print pred_label.shape
            preds_rf += pred_label
            preds_rf_prob += proba            
            
            #logistic regression
            lr = linear_model.LogisticRegression()
            lr.fit(X_tr_sub, Y_tr_sub.ravel())
            #preds_lr += lr.predict(X_te)
            proba= lr.predict_proba(X_te)              
            pred_prob = np.multiply(np.amax(proba,axis = 1), 
                                              lr.classes_.take(np.argmax(proba,axis = 1),axis = 0))
            pred_label = lr.classes_.take(np.argmax(proba,axis = 1),axis = 0)
            preds_lr += pred_label
            preds_lr_prob += proba
            
            #gradient boosting decision tree 
            clf =  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr_sub,  Y_tr_sub.ravel()) 
            proba = clf.predict_proba(X_te)                            
            pred_prob = np.multiply(np.amax(proba,axis = 1), clf.classes_.take(np.argmax(proba,axis = 1),axis = 0))
           
            pred_label = clf.classes_.take(np.argmax(proba,axis = 1),axis = 0) 
            preds_gbdt += pred_label
            preds_gbdt_prob += proba
            
        #print preds.shape
        #print preds.tolist()
        preds = np.mean(preds, axis = 1 )
        Y_pre = -np.ones(np.shape(preds))
        idx = np.where(preds >= 0.5)
        Y_pre[idx] = 1
        (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre)
        # calucate AUC
        Y_Xgboost_prob = preds;
        idx = np.where(Y_Xgboost_prob < 0.5)
        Y_Xgboost_prob[idx] = Y_Xgboost_prob[idx] - 1
        fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_Xgboost_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        p1 = [acc,spc,sst, auc] 
        Ycp[te_ind[0],0] = Y_Xgboost_prob.ravel()
        #perf[i,:] = np.array(p1)
        # random forest

        # AUC of random forest
        #preds_rf_prob = preds_rf_prob / subsample
        Y_rf_prob = np.multiply(np.amax(preds_rf_prob,axis = 1), 
                                              clf.classes_.take(np.argmax(preds_rf_prob,axis = 1),axis = 0))
        Y_rf_prob = Y_rf_prob / subsample
        fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_rf_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)     
        
        Y_pre_rf = np.sign(Y_rf_prob)
        (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_rf)           
             
        p2 = [acc,spc,sst, auc] 
        Ycp[te_ind[0],1] = Y_rf_prob.ravel()
        # logistic regression
      
        # AUC of logistic regression
        Y_lr_prob = np.multiply(np.amax(preds_lr_prob,axis =1),
                                lr.classes_.take( np.argmax(preds_lr_prob,axis =1), axis = 0))   #np.mean(preds_lr_prob, axis = 1)
        Y_lr_prob = Y_lr_prob/subsample
        fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_lr_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)  
        
        Y_pre_lr = np.sign(Y_lr_prob)
        (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_lr)        
        p3 = [acc,spc,sst, auc] 
        Ycp[te_ind[0],2] = Y_lr_prob.ravel()
        # Gradient boosting decision tree
      
        # AUC of gradient boosting decsion ttree
        Y_gbdt_porb =  np.multiply(np.amax(preds_gbdt_prob,axis =1),
                                clf.classes_.take( np.argmax(preds_gbdt_prob,axis =1), axis = 0))
        #Y_gbdt_porb = np.mean(preds_gbdt_prob,axis = 1)
        Y_gbdt_porb = Y_gbdt_porb/subsample
        fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_gbdt_porb, pos_label=1)    
        auc = metrics.auc(fpr, tpr)  
        
        Y_pre_gbdt = np.sign(Y_gbdt_porb)
        (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_gbdt)        
        p4 = [acc,spc,sst,auc]
        Ycp[te_ind[0],3] = Y_gbdt_porb.ravel()
        perf[i,:] = p1+p2+p3+p4
    a = np.mean(perf, axis=0)
    return a


data_mat = scipy.io.loadmat('./FS_QS_RM3.mat')
datapath = '../DataL12/'

A = data_mat['Xrm2_train']
Y = data_mat['Yrm_tr']
Ypre1 = np.zeros((Y.shape[0],4))
CV_ID = data_mat['CV_IDrm_train']
tr_id = data_mat['tr_idrm_train']

X = np.array(stats.zscore(A, axis=0, ddof=1))
#print tr_id[0][1].shape 
final_perf = np.zeros((4, 16 ))
count = 0
print A.shape
param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}   
a = Unbalanced_classify(X, Y, CV_ID, tr_id, param, 30, Ypre1)

final_perf[count,:] = a
count += 1
np.savetxt('JS_QS_RM3_ag_10fold.txt', Ypre1)

A = data_mat['Xrd2_train']
# features to be deleted due to too small variance that could cause failure of zscore
del_idx = np.array([80,81,82,83,96,97]) -1
A = np.delete(A,del_idx, 1)
Y = data_mat['Yrd_tr']
Ypre2 = np.zeros((Y.shape[0],4))
CV_ID = data_mat['CV_IDrd_train']
tr_id = data_mat['tr_idrd_train']
X = np.array(stats.zscore(A, axis=0, ddof=1))
print A.shape
a = Unbalanced_classify(X, Y, CV_ID, tr_id, param, 30, Ypre2)
final_perf[count,:] = a
count += 1
np.savetxt('JS_QS_RD3_ag_10fold.txt', Ypre2)
np.savetxt('JS_QS3_ag_perf.txt', final_perf)


