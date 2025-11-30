import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, correlate
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

def alignment_term(dFM, dGT):
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    align_Matrix = 2 * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + np.finfo(float).eps)
    return align_Matrix

def enhanced_alignment_term(align_Matrix):
    enhanced = ((align_Matrix + 1) ** 2) / 4
    return enhanced

def centroid(GT):
    rows, cols = GT.shape
    if np.sum(GT) == 0:
        X = round(cols / 2)
        Y = round(rows / 2)

    else:
        total = np.sum(GT)
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1)
        X = round(np.sum(np.sum(GT, axis=0) * i) / total)
        Y = round(np.sum(np.sum(GT, axis=1) * j) / total)

    return X, Y

def divideGT(GT, X, Y):
    hei, wid = GT.shape
    area = wid * hei

    LT = GT[:Y, :X]
    RT = GT[:Y, X:]
    LB = GT[Y:, :X]
    RB = GT[Y:, X:]

    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei - Y)) / area
    w4 = 1.0 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def Divideprediction(prediction, X, Y):
    hei, wid = prediction.shape

    LT = prediction[:Y, :X]
    RT = prediction[:Y, X:]
    LB = prediction[Y:, :X]
    RB = prediction[Y:, X:]

    return LT, RT, LB, RB

def ssim(prediction, GT):
    dGT = GT.astype(float)
    N = prediction.size

    x = np.mean(prediction)
    y = np.mean(dGT)

    sigma_x2 = np.sum((prediction - x) ** 2) / (N - 1 + np.finfo(float).eps)
    sigma_y2 = np.sum((dGT - y) ** 2) / (N - 1 + np.finfo(float).eps)
    sigma_xy = np.sum((prediction - x) * (dGT - y)) / (N - 1 + np.finfo(float).eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q

def Object(prediction, GT):
    if prediction.size == 0:
        return 0

    if not prediction.dtype == float:
        prediction = prediction.astype(float)

    if (prediction.max() > 1) or (prediction.min() < 0):
        raise ValueError('prediction should be in the range of [0 1]')

    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
        raise ValueError('GT should be of type: logical')

    x = np.mean(prediction[GT])
    sigma_x = np.std(prediction[GT])

    score = 2.0 * x / (x ** 2 + 1.0 + sigma_x + np.finfo(float).eps)
    return score

def S_object(prediction, GT):
    prediction_fg = prediction.copy()
    prediction_fg[~GT] = 0
    O_FG = Object(prediction_fg, GT)

    prediction_bg = 1.0 - prediction
    prediction_bg[GT] = 0
    O_BG = Object(prediction_bg, ~GT)

    u = np.mean(GT)
    Q = u * O_FG + (1 - u) * O_BG
    return Q

def S_region(prediction, GT):
    X, Y = centroid(GT)
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(GT, X, Y)
    prediction_1, prediction_2, prediction_3, prediction_4 = Divideprediction(prediction, X, Y)

    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)

    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def StructureMeasure(prediction, GT):
    if not isinstance(prediction, np.ndarray) or not isinstance(GT, np.ndarray):
        raise ValueError('The prediction and GT should be NumPy arrays.')

    if prediction.dtype != np.float64:
        raise ValueError('The prediction should be of type double.')

    if prediction.max() > 1 or prediction.min() < 0:
        raise ValueError('The prediction should be in the range of [0 1].')
    
    if GT.dtype != np.bool_:
        raise ValueError("GT should be of type: logical")

    y = np.mean(GT)

    if y == 0:
        x = np.mean(prediction)
        Q = 1.0 - x
    elif y == 1:
        x = np.mean(prediction)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_object(prediction, GT) + (1 - alpha) * S_region(prediction, GT)
        if Q < 0:
            Q = 0

    return Q

def original_WFb(FG, GT):
    if not isinstance(FG, np.ndarray) or not isinstance(GT, np.ndarray):
        raise ValueError("Input should be numpy arrays.")
    if FG.dtype != np.float64:
        raise ValueError("FG should be of type: double")
    if (FG.max() > 1) or (FG.min() < 0):
        raise ValueError("FG should be in the range of [0 1]")
    if GT.dtype != np.bool_:
        raise ValueError("GT should be of type: logical")

    dGT = GT.astype(np.float64)

    E = np.abs(FG - dGT)

    Dst, IDXT = distance_transform_edt(1-dGT, return_indices=True)

    K =  np.multiply(cv2.getGaussianKernel(7, 5), (cv2.getGaussianKernel(7, 5)).T)
    Et = E.copy()
    Et[~GT] = Et[IDXT[0][~GT], IDXT[1][~GT]]
    EA = correlate(Et, K, mode='constant')
    MIN_E_EA = E.copy()
    MIN_E_EA[GT & (EA < E)] = EA[GT & (EA < E)]

    B = np.ones_like(GT).astype(np.float64)
    non_ground_dist = Dst[~GT]
    B[~GT] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * non_ground_dist)
    Ew = MIN_E_EA * B

    TPw = np.sum(dGT) - np.sum(Ew[GT])
    FPw = np.sum(Ew[~GT])

    R = 1 - np.mean(Ew[GT])
    P = TPw / (np.finfo(float).eps + TPw + FPw)

    Q = (2) * (R * P) / (np.finfo(float).eps + R + P)

    return Q
def Fmeasure_calu_diceiou(sMap, gtMap, gtsize, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros(gtsize)
    Label3[sMap >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)
    LabelAnd = np.logical_and(Label3, gtMap)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gtMap)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        # PreFtem = 0
        # RecallFtem = 0
        # FmeasureF = 0
        Dice = 0
        # SpecifTem = 0
        IoU = 0
    else:
        IoU = NumAnd / (FN + NumRec)
        # PreFtem = NumAnd / NumRec
        # RecallFtem = NumAnd / num_obj
        # SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        # FmeasureF = (2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem)

    return  Dice,  IoU
def Fmeasure_calu(sMap, gtMap, gtsize, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros(gtsize)
    Label3[sMap >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)
    LabelAnd = np.logical_and(Label3, gtMap)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gtMap)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0
    else:
        IoU = NumAnd / (FN + NumRec)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = (2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem)

    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU

def Enhancedmeasure(FM, GT):
    FM = np.array(FM, dtype=bool)
    GT = np.array(GT, dtype=bool)

    dFM = FM.astype(np.float64)
    dGT = GT.astype(np.float64)

    if np.sum(GT) == 0:
        enhanced_matrix = 1.0 - dFM
    elif np.sum(~GT) == 0:
        enhanced_matrix = dFM
    else:
        align_matrix = alignment_term(dFM, dGT)
        enhanced_matrix = enhanced_alignment_term(align_matrix)

    w, h = GT.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)
    return score

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return 0
def process_image_diceiou(args):
    dataset, name, gtPath, resMapPath, Thresholds = args
    gt = cv2.imread(os.path.join(gtPath, name.split('.')[0]+'.png'), cv2.IMREAD_GRAYSCALE)
    resmap = cv2.imread(os.path.join(resMapPath, name), cv2.IMREAD_GRAYSCALE)
    
    if gt.ndim > 2:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    if not np.issubdtype(gt.dtype, np.bool_):
        gt = gt > 128
    
    if resmap.shape[:2] != gt.shape:
        resmap = cv2.resize(resmap, (gt.shape[1], gt.shape[0]))
        cv2.imwrite(os.path.join(resMapPath, name), resmap)
    
    resmap = resmap.astype(np.float64) / 255.0

    # Smeasure = StructureMeasure(resmap, gt)
    # wFmeasure = original_WFb(resmap, gt)
    # MAE = np.mean(np.abs(gt.astype(np.float64) - resmap))
    
    # threshold_E = np.zeros(len(Thresholds))
    # threshold_F = np.zeros(len(Thresholds))
    # threshold_Pr = np.zeros(len(Thresholds))
    # threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    # threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))
    
    for t in range(len(Thresholds)):
        threshold = Thresholds[t]
        threshold_Dic[t],  threshold_Iou[t] = Fmeasure_calu_diceiou(resmap, gt.astype(np.float64), gt.shape, threshold)
        # Bi_resmap = np.zeros(resmap.shape)
        # Bi_resmap[resmap >= threshold] = 1
        # threshold_E[t] = Enhancedmeasure(Bi_resmap, gt)
    
    return ( threshold_Dic, threshold_Iou)

def process_image(args):
    dataset, name, gtPath, resMapPath, Thresholds = args
    gt = cv2.imread(os.path.join(gtPath, name.split('.')[0]+'.png'), cv2.IMREAD_GRAYSCALE)
    resmap = cv2.imread(os.path.join(resMapPath, name), cv2.IMREAD_GRAYSCALE)
    
    if gt.ndim > 2:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    if not np.issubdtype(gt.dtype, np.bool_):
        gt = gt > 128
    
    if resmap.shape[:2] != gt.shape:
        resmap = cv2.resize(resmap, (gt.shape[1], gt.shape[0]))
        cv2.imwrite(os.path.join(resMapPath, name), resmap)
    
    resmap = resmap.astype(np.float64) / 255.0

    Smeasure = StructureMeasure(resmap, gt)
    wFmeasure = original_WFb(resmap, gt)
    MAE = np.mean(np.abs(gt.astype(np.float64) - resmap))
    
    threshold_E = np.zeros(len(Thresholds))
    threshold_F = np.zeros(len(Thresholds))
    threshold_Pr = np.zeros(len(Thresholds))
    threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))
    
    for t in range(len(Thresholds)):
        threshold = Thresholds[t]
        threshold_Pr[t], threshold_Rec[t], threshold_Spe[t], threshold_Dic[t], threshold_F[t], threshold_Iou[t] = Fmeasure_calu(resmap, gt.astype(np.float64), gt.shape, threshold)
        Bi_resmap = np.zeros(resmap.shape)
        Bi_resmap[resmap >= threshold] = 1
        threshold_E[t] = Enhancedmeasure(Bi_resmap, gt)
    
    return (Smeasure, wFmeasure, MAE, threshold_E, threshold_F, threshold_Pr, threshold_Rec, threshold_Spe, threshold_Dic, threshold_Iou)

if __name__ == "__main__":

    # ResultMapPath = r'/data1/whl/unify/X-Decoder-main/output/colon_lap/run_13_00151740/sun_seg'
    ResultMapPath = r'/data1/whl/unify/X-Decoder-main/output/colon_lap/run_13_00151740/sun_seg'
    #模型分割结果路径

    # DataPath = r'/data1/whl/Datasets/endoscope-labelseg/'
    # DataPath = r'/data1/whl/Datasets/xdecoder_data/polyseg/raw_data/test'
    # DataPath = r'/data1/whl/Datasets/xdecoder_data/'
    DataPath = r'/data1/whl/Datasets/xdecoder_data/SUN-SEG'
    #对照masks绝对路径

    #Datasets = ['cancer', 'polyp', 'adenoma']
    # Datasets = ['CVC-ClinicDB', 'Kvasir', 'ETIS-LaribPolypDB', 'CVC-ColonDB',  'CVC-300']
    # Datasets = ['GIANA','PICCOLO']
    # Datasets = ['polyp', 'adenoma']
    Datasets = ['easy', 'hard']
    #子文件夹名称


    #ResDir = '/data1/ty/wzy/result/public_329_enmarkonly'
    ResDir = r'/data1/whl/unify/X-Decoder-main/output/colon_lap/run_13_00151740/'
    #指标保存绝对路径
    if not os.path.exists(ResDir):
        os.makedirs(ResDir)

    ResName = 'results.txt'
    #指标文件名
    num_threads = 48  # 线程池使用线程数，从经验上看可以设置不超过CPU线程数90%，CPU占用率过高会影响系统进程


    models = os.listdir(ResultMapPath)
    modelNum = len(models)
    datasetNum = len(Datasets)
    Thresholds = np.arange(1, -1/255, -1/255)

    

    for d in range(datasetNum):
        dataset = Datasets[d]
        print(f'处理 {d+1}/{datasetNum}: {dataset} 数据集')
        # filename = dataset + '_' + ResName
        filename = ResName
        fileID = open(os.path.join(ResDir, filename), 'a')
        gtPath = os.path.join(DataPath, dataset,'masks')
        # gtPath = os.path.join(DataPath, dataset,'Mymask')
        # gtPath = DataPath
        gtPath = os.path.join(DataPath, 'test_'+dataset,'annotations')
        # gtPath = os.path.join(DataPath,dataset ,'test','annotations')
        resMapPath = os.path.join(ResultMapPath, dataset)
        imgFiles = [f for f in os.listdir(resMapPath) if f.endswith('.png') or f.endswith('.jpg')]
        imgFiles = np.sort(imgFiles)
        imgNUM = len(imgFiles)

        threshold_Dice = np.zeros((imgNUM, len(Thresholds)))
        threshold_IoU = np.zeros((imgNUM, len(Thresholds)))

        args_list = [(dataset, imgFiles[i], gtPath, resMapPath, Thresholds) for i in range(imgNUM)]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(process_image_diceiou, args_list), total=imgNUM, desc=f'Processing {dataset}'))

        for i, ( thr_Dic, thr_Iou) in enumerate(results):
            
           
            threshold_Dice[i, :] = thr_Dic
            threshold_IoU[i, :] = thr_Iou
        
        
        meanDic = np.mean(threshold_Dice)
        meanIoU = np.mean(threshold_IoU)
        
        fileID.write('\n' + f'(数据集:{dataset}) meanDic:{meanDic:.3f};meanIoU:{meanIoU:.3f}')
        
        fileID.close()