
def get_session_name(config):
    model_name = "Modelv1"
    if config.MODEL.NAME == 'cnn-lstm':
        model_name = 'CNN-LSTM'
    elif config.MODEL.NAME == 'model_v2':
        model_name = "ModelV2"
    elif config.MODEL.NAME == 'cnn-lstm-se':
        model_name = 'CNN-LSTM-SE'
    elif config.MODEL.NAME == 'conv-lstm':
        model_name = 'Conv-LSTM'
    elif config.MODEL.NAME == 'strans':
        model_name = 'STrans'
    elif config.MODEL.NAME == 'strans-v2': 
        model_name = 'Strans-V2'
    elif config.MODEL.NAME == 'strans-v3': 
        model_name = 'Strans-V3'
    elif config.MODEL.NAME == 'strans-v4': 
        model_name = 'Strans-V4'
    elif config.MODEL.NAME == 'strans-v5': 
        model_name = 'Strans-V5'
    elif config.MODEL.NAME == 'strans-v6': 
        model_name = 'Strans-V6'
    elif config.MODEL.NAME == 'pred':
        model_name = 'Pred'
    elif config.MODEL.NAME == 'vit-2head':
        model_name = 'VIT_2Head'
    session_name = f"BATCH_{config.TRAIN.BATCH_SIZE}_Add_type_{config.MODEL.TEMPORAL.ADDING_TYPE}"
    if config.LOSS.NAME == "mse" or config.LOSS.NAME == "magnitudeweight" or config.LOSS.NAME == "logmagnitudeweight" or config.LOSS.NAME == "quantile" or config.LOSS.NAME == "mae" or config.LOSS.NAME == "huberloss" or config.LOSS.NAME == "combineweightloss" or config.LOSS.NAME == "msle":
        session_name += f"{config.WANDB.GROUP_NAME}_{model_name}_PS-{config.MODEL.PATCH_SIZE}_Lr-{config.OPTIMIZER.LR}_LF-{config.LOSS.NAME}_DR-{config.MODEL.DROPOUT}_LN-{config.MODEL.USE_LAYER_NORM}-ST_{config.MODEL.SPATIAL.TYPE}_{config.MODEL.SPATIAL.NUM_LAYERS}-ON_{config.TRAIN.OUTPUT_NORM}_Seed-{config.MODEL.SEED}_LRS-{config.LRS.USE_LRS}"
    elif  config.LOSS.NAME == "expweightedloss":
        pass
    elif config.LOSS.NAME == "weightedmse":
        session_name += f"{config.WANDB.GROUP_NAME}_{model_name}_PS-{config.MODEL.PATCH_SIZE}_Lr-{config.OPTIMIZER.LR}_LF-{config.LOSS.NAME}-wfn_{config.LOSS.WEIGHT_FUNC}_DR-{config.MODEL.DROPOUT}_LN-{config.MODEL.USE_LAYER_NORM}-ST_{config.MODEL.SPATIAL.TYPE}_{config.MODEL.SPATIAL.NUM_LAYERS}-ON_{config.TRAIN.OUTPUT_NORM}_Seed-{config.MODEL.SEED}_LRS-{config.LRS.USE_LRS}"
    
    elif config.LOSS.NAME == "weightedthresholdmse":
        session_name += f"{config.WANDB.GROUP_NAME}_{model_name}_PS-{config.MODEL.PATCH_SIZE}_Lr-{config.OPTIMIZER.LR}_LF-{config.LOSS.NAME}-{config.LOSS.HIGH_WEIGHT}/{config.LOSS.LOW_WEIGHT}-{config.LOSS.GROUNDTRUTH_THRESHOLD}_DR-{config.MODEL.DROPOUT}_LN-{config.MODEL.USE_LAYER_NORM}-ST_{config.MODEL.SPATIAL.TYPE}_{config.MODEL.SPATIAL.NUM_LAYERS}-ON_{config.TRAIN.OUTPUT_NORM}_Seed-{config.MODEL.SEED}_LRS-{config.LRS.USE_LRS}"
    else:
        raise("Error: Not correct loss function name!")
    session_name += f"GSMAP_time_step{config.MODEL.TIME_STEP}"
    session_name += f"ECMWF_time_step{config.MODEL.ECMWF_TIME_STEP}"
    if config.LRS.USE_LRS == True:
        if config.LRS.NAME == "CosineAnnealingLR":
            session_name += f"_CosineAnnealingLR-{config.LRS.COSINE_T_MAX}-{config.LRS.COSINE_ETA_MIN}"
        elif config.LRS.NAME == "ReduceLROnPlateau":
            session_name += f"_ReduceLROnPlateau-{config.LRS.PLATEAU_MODE}-{config.LRS.PLATEAU_FACTOR}-{config.LRS.PLATEAU_PATIENCE}"
    session_name += f"VIT_Blocks_{config.TRAIN.NUM_VITBLOCKS}"
    return session_name