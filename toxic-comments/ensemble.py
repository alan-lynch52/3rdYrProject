import numpy as np
import pandas as pd

lr_fp = 'lr_submission.csv'
bnb_fp = 'bnb_submission.csv'
mnb_fp = 'mnb_submission.csv'
rfc_fp = 'rfc_submission.csv'

lr_preds = pd.read_csv(lr_fp)
bnb_preds = pd.read_csv(bnb_fp)
mnb_preds = pd.read_csv(mnb_fp)
rfc_preds = pd.read_csv(rfc_fp)

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
ensemble_preds = lr_preds.copy()
ensemble_preds[LABELS] = (lr_preds[LABELS] + bnb_preds[LABELS] + mnb_preds[LABELS] + rfc_preds[LABELS]) / 4
print(ensemble_preds[LABELS])
ensemble_preds.to_csv('ensemble_submission.csv',index=False)

