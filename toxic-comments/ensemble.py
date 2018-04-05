from toxic_comments import *
def round_dfs(d):
    for key in d:
        df = d[key]
        for label in df:
            df[label] = df[label].round()
    return d
##lr_fp = 'val/models/lr_preds.csv'
##bnb_fp = 'val/models/bnb_preds.csv'
##mnb_fp = 'val/models/mnb_preds.csv'
##rfc_fp = 'val/models/rf_preds.csv'
##
##lr_preds = pd.read_csv(lr_fp)
##lr_probs = pd.read_csv('val/models/lr_probs.csv')
##
##bnb_preds = pd.read_csv(bnb_fp)
##bnb_probs = pd.read_csv('val/models/bnb_probs.csv')
##
##mnb_preds = pd.read_csv(mnb_fp)
##mnb_probs = pd.read_csv('val/models/mnb_probs.csv')
##
##rfc_preds = pd.read_csv(rfc_fp)
##rfc_probs = pd.read_csv('val/models/rf_probs.csv')
##
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
##
##ensemble_preds = lr_preds.copy()
##ensemble_probs = lr_probs.copy()
##
##ensemble_preds[LABELS] = ((lr_preds[LABELS]*0.7) + (bnb_preds[LABELS]*0.1) + (mnb_preds[LABELS]*0.1) + (rfc_preds[LABELS]*0.1)) 
##ensemble_probs[LABELS] = ((lr_probs[LABELS]*0.7) + (bnb_probs[LABELS]*0.1) + (mnb_probs[LABELS]*0.1) + (rfc_probs[LABELS]*0.1))
##
##ensemble_preds.to_csv('val/ensembles/weighted_avg_preds.csv',index=False)
##ensemble_probs.to_csv('val/ensembles/weighted_avg_probs.csv',index=False)


#get AUROC, CM, Bal Acc
e1 = 'val/ensembles/lr_mnb_bnb_rf_probs.csv'
e2 = 'val/ensembles/lr_mnb_bnb_probs.csv'
e3 = 'val/ensembles/lr_bnb_probs.csv'
e4 = 'val/ensembles/weighted_avg_probs.csv'
true = 'val/ensembles/true_labels.csv'

e1_probs = pd.read_csv(e1)
e2_probs = pd.read_csv(e2)
e3_probs = pd.read_csv(e3)
e4_probs = pd.read_csv(e4)
true_labels = pd.read_csv(true)
#print(e1_probs['toxic'])
d = collections.OrderedDict()
d['e1'] = e1_probs
d['e2'] = e2_probs
d['e3'] = e3_probs
d['e4'] = e4_probs

get_auroc_ensemble(d,true_labels)

d = round_dfs(d)
plot_cm_ensemble(d,true_labels)
