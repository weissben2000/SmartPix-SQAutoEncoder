training_dict = {
    "MC2D_T240e_2bit": {
        "weights": "./SQ_DAE_regression/model-d11e02f3-SQDAE_regression_Categorical2bit_240eN_Target_0_30_856-checkpoints",
        "tfr": "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords/TFR_val/3srCI_16x16_240eN_raw",
        "name": "MaxConv2D T240e 2bit FE",
        "noise": 240,
        "model_type": "Conv2D",
        "model_output": "max"
    },
    "SQDAE_T240e_2x2bitEncoder": {
        "weights": "../smart-pixels-ml/trained_models//model-74441133-SQMaxConv2D_T240e_v2-checkpoints",
        "tfr": "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords/TFR_val/3srCI_16x16_240eN_raw",
        "name": "SQAutoEncoder T240e 2bitFE 2bitEncoder",
        "noise": 240,
        "model_type": "SQDAE",
        "model_output": "max"
    },
    "MC2D_T80e_2bit": {
        "weights": "/uscms/home/jennetd/nobackup/smart-pixels/noise-paper/trained_models//model-58301193-MaxConv2D_SQ_80e_2bit-checkpoints",
        "tfr": "/uscms/home/jennetd/nobackup/smart-pixels/tfrecords/TFR_val/3sr_16x16_80eNoise_test",
        "name": "MaxConv2D T80e 2bit FE",
        "noise": 80,
        "model_type": "Conv2D",
        "model_output": "max"
    },
    "SQDAE_T80e_2x2bitEncoder": {
        "weights": "./SQ_DAE_regression//model-01193edd-SQDAE_regression_Categorical2bit_80eN_Target_0_30_856-checkpoints",
        "tfr": "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords/TFR_val/3sr_16x16_80eN_raw",
        "name": "SQAutoEncoder T80e 2bitFE 2bitEncoder",
        "noise": 80,
        "model_type": "SQDAE",
        "model_output": "max"
    },
    "SQDAE_T80e_2x3bitEncoder": {
        "weights": "./SQ_DAE_regression//model-9ee54816-SQDAE_regression_Categorical2bit_80eN_Target_0_7_23_134_404_867_1748-checkpoints",
        "tfr": "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords/TFR_val/3sr_16x16_80eN_raw",
        "name": "SQAutoEncoder T80e 2bitFE 3bitEncoder",
        "noise": 80,
        "model_type": "SQDAE",
        "model_output": "max"
    },
    "MC2D_T0e_2bit": {
        "weights": "/uscms/home/jennetd/nobackup/smart-pixels/noise-paper/trained_models//model-60350e7e-MaxConv2D_SQ_0e_2bit-checkpoints",
        "tfr": "/uscms/home/jennetd/nobackup/smart-pixels/tfrecords/TFR_val/3sr_16x16_0eNoise_test",
        "name": "MaxConv2D T0e 2bit FE",
        "noise": 0,
        "model_type": "Conv2D",
        "model_output": "max"
    },
    "SQDAE_T0e_2x2bitEncoder": {
        "weights": "./SQ_DAE_regression//model-00d39d23-SQDAE_Maxregression_Categorical2bit_0eN_Target_0_30_856-checkpoints",
        "tfr": "/uscms/home/bweiss/nobackup/smart-pixels/tfrecords/TFR_val/3srCI_16x16_0eN_raw",
        "name": "SQAutoEncoder T0e 2bitFE 2bitEncoder",
        "noise": 0,
        "model_type": "SQDAE",
        "model_output": "max"
    }
}

