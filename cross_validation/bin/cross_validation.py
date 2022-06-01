"""
    10 fold cross-validation
"""
#%% importing modules

import os

import numpy as np
import scipy
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn, TestDataset, test_collate_fn


#%% Auxilliary functions

FOLD_SEED = 11

def _check_gpu():
     # set up torch on gpu if possible
    use_gpu = False
    cuda=torch.device('cpu')
    
    if torch.cuda.is_available():
        cuda=torch.device('cuda:0')
        print(torch.version.cuda)
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f/1e6)
        use_gpu = True
    else:
        print("no cuda available")
    return use_gpu

def _load_data():
    found = False

    #prime:
    tempMat='/home/au709601/NOBACKUP/data/'
    if  not found and os.path.exists(tempMat):
        matDir=tempMat
        found=True
        print(matDir)
        
    #home:
    tempMat='/media/kafkan/storage1/speciale/data/pickle/'
    if  not found and os.path.exists(tempMat):
        matDir=tempMat
        found=True    
        print(matDir)
    assert(found)
    data = ExtSleepContainer.fromDirectory(matDir,'eeg_lr',"calibration_dict.pkl")
    print('Data loaded')
    return data

def _data_samplers(valIdxs, trainIdxs, loadedData, batch_size, L=20, nw=1):
        # get training and validation data points
        trainX, trainCal, trainCalMap, trainY = loadedData.returnBySubject(trainIdxs)
        valX, valCal, valCalMap, valY = loadedData.returnBySubject(valIdxs)
        
        # reshape labels
        trainY = torch.tensor((trainY-1).flatten()).type(torch.long)
        valY = torch.tensor((valY-1).flatten()).type(torch.long)
        
        # Instantiate databases and samplers
        trainDataset=torch.utils.data.TensorDataset(torch.tensor(trainX),trainY)
        valDataset=torch.utils.data.TensorDataset(torch.tensor(valX), valY)
        trainSampler=torch.utils.data.DataLoader(ExtendedDataset(trainDataset,L, trainCal, trainCalMap),batch_size=batch_size,
                                         shuffle=True,drop_last=True,collate_fn=custom_collate_fn, num_workers=nw)
        valSampler=torch.utils.data.DataLoader(ExtendedDataset(valDataset,L, valCal, valCalMap),batch_size=batch_size,
                                          shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=1)
        return trainSampler, valSampler

def _evaluate_test(model, trainer, testIdxs, loadedData, logger, L):
    testX, testCal, testCalMap, testLabels = loadedData.returnBySubject(testIdxs)
    testCal = [c.type(torch.float32) for c in testCal]
    
    ensemble_pred, probs = ensemble_prediction(model,testX, testCal, testCalMap, trainer )
    ensemble_pred += 1
    probs = torch.tensor(probs)
    kappa=cohen_kappa_score(ensemble_pred ,testLabels.T)
    conf = confusion_matrix(testLabels.T, ensemble_pred)

    rolledKappas=np.zeros(L)
    for iRoll in range(L):
        _, rolled_labels =torch.max(torch.tensor(probs[iRoll,:,:]),dim=1)
        rolled_labels = torch.unsqueeze(rolled_labels+1,1).cpu().detach().numpy()
        rolledKappas[iRoll]=cohen_kappa_score(rolled_labels,testLabels.T)

    print('Consensus:',testIdxs,kappa)
    logger.log_metrics({'ensembleKappa':kappa})
    logger.log_metrics({'meanRolledKappa': np.mean(rolledKappas)})
    return kappa, conf

#%% Ensemble test

def ensemble_prediction(model, X, cal, calMap, trainer, L=20):
    pl.seed_everything(97)

    cal = [c.type(torch.float32) for c in cal]
    num_points = X.shape[0]

    #pad end so that length is a multiple of L
    missing = int(np.ceil(num_points/L)*L-num_points)

    paddedX = np.concatenate((X,X[0:missing,:,:,:]),axis=0)
    paddedX =torch.tensor(paddedX)
    padded_length = paddedX.shape[0]

    # pad calibration data
    padCalMap  = np.concatenate((calMap, calMap[0:missing]),axis=0)
    padCalMap = torch.tensor(padCalMap)
    probs = np.zeros((L,num_points,5))

    with torch.no_grad():
        model.eval()

        for j in (range(0,L)):    
            rolledX = torch.roll(paddedX, shifts=j,dims=0)
            rolledCalMap = torch.roll(padCalMap,shifts=j,dims=0)
            rolledTest = TestDataset(rolledX, L, cal, rolledCalMap)
            testLoader=torch.utils.data.DataLoader(rolledTest,batch_size=1,
                                             shuffle=False,drop_last=False,
                                             collate_fn=test_collate_fn)
            logits=np.zeros((padded_length,5))
            trainer.test(model, testLoader,verbose=False)
            test_results = model.test_results
            logits[test_results['idxs'].astype(int),:]=test_results['y_pred']
      
            # get probs w. softmax, roll logits backwards to put probs in the right place
            probs[j,:,:]=scipy.special.softmax(np.roll(logits[0:num_points,:],-j,0),axis=1)
        
        #discard padded end
        probs=probs[:,0:num_points,:] 
        # aggregate
        ensemble_probs=np.sum(probs,axis=0)
        # ensemble_probs=np.prod(probs,axis=0)

    pred = np.argmax(ensemble_probs, axis=1)
    return pred, probs
    

#%% Interface

def cross_validate(init_function,  model_class, epochs, exp_name, savedir, seed=97, batch_size=32, folds=10, L=20, nw=1, steps_up=10):
    pl.seed_everything(seed)
    rng = np.random.RandomState(FOLD_SEED)
    use_gpu = _check_gpu()
    loaded_data = _load_data()
    
    n_subjects = 20
    assert float.is_integer(n_subjects / folds) 
    subj_idxs = np.arange(1, n_subjects+1)
    rng.shuffle(subj_idxs)
    folds = subj_idxs.reshape(10,2)
    
    kappas = np.zeros(len(folds))
    confs = np.zeros((len(folds),5,5))
    
    
    for i, testIdxs in enumerate(folds):
        
        # randomly select validation and training data from remaining folds
        remaining = np.delete(folds, i, axis=0)
        rng.shuffle(remaining)
        valIdxs = remaining[0:1,:]
        trainIdxs = remaining[1:,:]
        
        trainSampler, valSampler =  _data_samplers(valIdxs, trainIdxs, loaded_data, batch_size, L, nw)
        
        model = init_function(trainSampler)

        
        logger = TensorBoardLogger(
            savedir+"/tb_logs",
            name=exp_name,     
            version=i,
        )
        
        checkpoint = ModelCheckpoint(
                        monitor="valKappa", mode="max",
                        dirpath=savedir+"/"+exp_name, 
                        filename=exp_name+f"-fold-{i}-best" + "{valKappa}",
                        save_top_k = 1)

        gpus = 1 if use_gpu else None
        trainer = pl.Trainer(max_epochs=epochs,
                     callbacks=[checkpoint],
                     logger=logger,
                     gpus=gpus,
                     )
        trainer.fit(model, trainSampler, valSampler)
        
        #restore best
        model = model_class.load_from_checkpoint(checkpoint.best_model_path)
        
        # test with ensembling
        kappa, conf = _evaluate_test(model, trainer, testIdxs, loaded_data, logger, L)
        
        kappas[i] = kappa
        confs[i,:,:] = conf
        logger.log_hyperparams(params={"trainIdxs":trainIdxs, "valIdxs":valIdxs, "testIdxs":testIdxs, "best_model":checkpoint.best_model_path })
        logger.finalize("success")
        
    np.save(f"{savedir}/{exp_name}/kappas.npy", kappas)
    np.save(f"{savedir}/{exp_name}/confusion_matrices.npy", confs)
