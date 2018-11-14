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


def Unbalanced_classify(X, Y, CV_ID, tr_id, param, subsample):
    perf = np.empty((1,12))
    Y1 = Y
    Y1 = (Y1 + 1 ) / 2  
    #for i in range(10):
    tr_ind = np.where(CV_ID!= 2 )
    te_ind = np.where(CV_ID == 2)

        
    X_tr = X[tr_ind[0],:]
    Y_tr = Y[tr_ind[0]]
    X_te = X[te_ind[0],:]
    Y_te = Y[te_ind[0]]    
    Y1_tr = Y1[tr_ind[0]]
    Y1_te = Y1[te_ind[0]] 
    Ycp = np.zeros((len(te_ind[0]),4))
            
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
        tr_idx = tr_id[0][j].ravel() -1
        X_tr_sub = X_tr[tr_idx,: ]
        Y_tr_sub = Y_tr[tr_idx]
        #print sum(np.where(Y_tr_sub == 1))
        Y1_tr_sub = Y1_tr[tr_idx]
        t1 =np.sum(Y_tr_sub == 1)
        t2 = np.sum(Y_tr_sub == -1)
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
        preds_rf += pred_label
        preds_rf_prob += proba                 
            
            #logistic regression
        lr = linear_model.LogisticRegression()
        lr.fit(X_tr_sub, Y_tr_sub.ravel())
        #preds_lr[:,j] = lr.predict(X_te)
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
        #pred_label = clf.classes_.take(np.argmax(proba,axis = 1),axis = 0) 
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
    Ycp[:,0] = Y_Xgboost_prob.ravel()

    Y_rf_prob = np.multiply(np.amax(preds_rf_prob,axis = 1), 
                                                      clf.classes_.take(np.argmax(preds_rf_prob,axis = 1),axis = 0))
    Y_rf_prob = Y_rf_prob / subsample    
    fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_rf_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)   
    Y_pre_rf = np.sign(Y_rf_prob)
    (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_rf)        
    
    p2 = [acc,spc,sst, auc] 
    Ycp[:,1] = Y_rf_prob.ravel()
        # logistic regression
        # AUC of logistic regression
    Y_lr_prob = np.multiply(np.amax(preds_lr_prob,axis =1),
                                       lr.classes_.take( np.argmax(preds_lr_prob,axis =1), axis = 0))  
    
    Y_lr_prob = Y_lr_prob/subsample       
    fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_lr_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)   
    Y_pre_lr = np.sign(Y_lr_prob)
    (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_lr)      
    p3 = [acc,spc,sst, auc] 
    Ycp[:,2] = Y_lr_prob.ravel()    
    
    # Gradient boosting decision tree
    # AUC of gradient boosting decsion ttree
    Y_gbdt_porb =  np.multiply(np.amax(preds_gbdt_prob,axis =1),
                                    clf.classes_.take( np.argmax(preds_gbdt_prob,axis =1), axis = 0))
          
    Y_gbdt_porb = Y_gbdt_porb/subsample    
    fpr, tpr, thresholds = metrics.roc_curve(Y_te, Y_gbdt_porb, pos_label=1)    
    auc = metrics.auc(fpr, tpr)      
    Y_pre_gbdt = np.sign(Y_gbdt_porb)
    (acc,spc,sst) = PerfStat_logistic(Y_te, Y_pre_gbdt)    
    p4 = [acc,spc,sst,auc]
    Ycp[:,3] = Y_gbdt_porb.ravel()
    perf = p1+p2+p3+p4
        #print p1+p2+p3
    #a = np.mean(perf, axis=0)
    return (perf, Ycp)


data_mat = scipy.io.loadmat('./JS_QC_data3_RC_ag.mat')
data_mat2 = scipy.io.loadmat('QC_permLabel.mat')
datapath = '../DataL12/'

A = data_mat['Xrm2']
Y = data_mat['Yrm']
CV_ID = data_mat['CV_IDrm_rc']
tr_idperm = data_mat2['perm_tr_idrm']
Yperm = data_mat2['YrmPerm']
print Y.shape
final_perf = np.zeros((1000, 16 ))
count = 0

param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}   

#delete all the features with small variance which could cause issues with zscore normalization
del_idx = np.array([80,81,82,83,96,97]) -1
A = np.delete(A,del_idx, 1)

X = np.array(stats.zscore(A, axis=0, ddof=1))


for i in xrange(0,1000):
    Y1 = Yperm[:,i]
    Y = Y1.reshape((Y1.shape[0],1))
    #print Y.shape
    tr_id = tr_idperm[i][0]
    a, Ycp = Unbalanced_classify(X, Y, CV_ID, tr_id, param, 30)
    final_perf[count,:]= a
    count += 1
    print count

np.savetxt('JS_QC_data3_independent_RM_perm_p1.txt', final_perf)



