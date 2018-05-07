% creating a demsar diagram with gavins code
% File should contain ERRORS, not ACCURACIES. No headers
% addpath('C:\Users\ajb\Dropbox\Code\Matlab');
% Change to file name for errors

DATA_PATH='model_critdiff.csv';
names={'LR', 'BNB', 'MNB'};

data=csvread(DATA_PATH);
critdiff(data,names,0.05)