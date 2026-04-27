# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import keyboard
import numpy as np
import pandas as pd
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import fractional_matrix_power

from neuracle_lib.dataServer import DataServerThread
from ui_mi import start_ui, update_text_from_outside

import time
from datetime import datetime

mne.set_log_level('ERROR')

# arithmetic mean only, SPD-safe
def EA_SPDsafe(x, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    n = len(x)
    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar = C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])

    eigvals, eigvecs = np.linalg.eigh(R_bar)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    ref = eigvecs @ D_inv_sqrt @ eigvecs.T

    XEA = ref @ x

    return XEA


def generate_repeating_labels(max_len):
    # Create the repeating pattern [0, 1, ..., 9, 0, 1, ..., 9, ...]
    labels = np.tile(np.arange(2), max_len // 2 + 1)[:max_len]
    return labels


def leftrightflipping_transform(X, left_mat, right_mat):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, num_channels, num_timesamples = X.shape
    transformedX = np.zeros((num_samples, num_channels, num_timesamples))
    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, ch, :] = X[:, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, ch, :] = X[:, left_mat[ind], :]
        else:
            transformedX[:, ch, :] = X[:, ch, :]

    return transformedX

def main(trial_num, train_paths=None):
    ## 配置设备
    neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
    dsi = dict(device_name='DSI-24', hostname='127.0.0.1', port=8844,
               srate=300,
               chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', # 14
                         'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'], n_chan=25)
    neuroscan = dict(device_name='Neuroscan', hostname='127.0.0.1', port=4000,
                     srate=1000,
                     chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'] + ['TRG'], n_chan=65)
    device = [neuracle, dsi, neuroscan]
    ### 设备型号,默认DSI
    target_device = device[1]
    ## 初始化 DataServerThread 线程
    time_buffer = 3  # 缓冲池大小 单位秒


    left_mat = [0, 1, 2, 9, 10, 12, 13, 14, 18]
    right_mat = [6, 5, 4, 21, 11, 23, 22, 15, 19]

    class_display = ['左手', '右手']

    if train_paths is not None:
        train_x_all = []
        train_y_all = []
        for i in range(len(train_paths)):

            train_X = pd.read_csv(train_paths[i]).values
            if len(train_X.shape) == 2:
                train_X = np.split(train_X, int(train_X.shape[1] // 900), axis=1)
                train_X = np.stack(train_X)
            print('train_X.shape', train_X.shape)


            if len(train_X.shape) == 2:
                max_len = int(train_X.shape[1] // 900)
            else:
                max_len = len(train_X)

            train_X = train_X[:, :-1, :]
            train_X = train_X[:, :, 300:900]
            print('train_X.shape', train_X.shape)

            train_y = generate_repeating_labels(max_len)
            print('train_y.shape', train_y.shape)
            csp = mne.decoding.CSP(n_components=6)

            num_trials, num_chn, num_timesamples = train_X.shape

            # Reshape to (num_trials*num_channels, num_timesamples)
            train_X = train_X.reshape(-1, 900)

            # Apply the bandpass filter
            train_X = mne.filter.filter_data(
                data=train_X,
                sfreq=300,
                l_freq=8,
                h_freq=32,
                method='fir', # Use FIR filtering
                # phase='zero', # Zero-phase (non-causal) filtering
                fir_window='hamming' # Hamming window
            )

            # Reshape back to original shape
            train_X = train_X.reshape(num_trials, num_chn, num_timesamples)

            train_X = train_X[:, :, 300:900]
            print('train_X.shape', train_X.shape)

            train_X = EA_SPDsafe(train_X)

            train_y = generate_repeating_labels(max_len)
            print('train_y.shape', train_y.shape)

            train_x_all.append(train_X)
            train_y_all.append(train_y)

    train_X = np.concatenate(train_x_all)
    train_y = np.concatenate(train_y_all)

    aug_train_x = leftrightflipping_transform(train_X, left_mat, right_mat)
    aug_train_y = 1 - train_y

    train_X = np.concatenate((train_X, aug_train_x))
    train_y = np.concatenate((train_y, aug_train_y))

    csp = mne.decoding.CSP(n_components=6)
    train_X_csp = csp.fit_transform(train_X, train_y)
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_X_csp, train_y)


    thread_data_server = DataServerThread(device=target_device['device_name'], n_chan=target_device['n_chan'],
                                          srate=target_device['srate'], t_buffer=time_buffer)

    
    ### 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname=target_device['hostname'], port=target_device['port'])
    if notconnect:
        raise TypeError("Can't connect recorder, Please open the hostport ")
    else:
        # 启动线程
        thread_data_server.Daemon = True
        thread_data_server.start()
        print('Data server connected')
    '''
    在线数据获取演示：每隔一秒钟，获取数据（数据长度 = time_buffer）
    '''

    GLOBAL_CURRENT_TEXT = str('任务提示')
    update_text_from_outside(GLOBAL_CURRENT_TEXT)
    print(GLOBAL_CURRENT_TEXT)

    print('waiting for key press')
    keyboard_is_pressed = False
    while not keyboard_is_pressed:
        if keyboard.is_pressed("space"):
            now = datetime.now()
            print("按下空格的时间：", now.strftime("%Y-%m-%d %H:%M:%S.%f"))
            keyboard_is_pressed = True

            data = thread_data_server.GetBufferData()
            thread_data_server.ResetDataLenCount()

    print('after key press')


    assert keyboard_is_pressed, print('keyboard not pressed')
    assert not notconnect, print('notconnect')

    task_image = r'hint.png'

    prepare_image = r'preparation.png'

    left_image = r'lefthand.png'

    right_image = r'righthand.png'

    allData = []
    N, flagstop = 0, False
    GLOBAL_CURRENT_TEXT = ''

    Xt_aligned = []
    R = 0
    num_samples = 0

    # try:
    while not flagstop:  # get data in one second step
        nUpdate = thread_data_server.GetDataLenCount()
        if nUpdate > (3 * target_device['srate'] - 1):
            N += 1
            data = thread_data_server.GetBufferData()
            allData.append(np.transpose(data, (1, 0)))
            thread_data_server.ResetDataLenCount()
            # print(data.shape)

            data = data[:-1, :]
            data = data[:, 300:900]

            # Apply the bandpass filter
            data = mne.filter.filter_data(
                data=data,
                sfreq=300,
                l_freq=8,
                h_freq=32,
                method='fir', # Use FIR filtering
                # phase='zero', # Zero-phase (non-causal) filtering
                fir_window='hamming' # Hamming window
            )
            # print('after filter', data.shape)
            data = data.reshape(-1, 600)

            # EA
            cov = np.cov(data)
            R = (R * num_samples + cov) / (num_samples + 1)
            num_samples += 1
            sqrtRefEA = fractional_matrix_power(R, -0.5)
            curr_aligned = np.dot(sqrtRefEA, data)
            curr_aligned = csp.transform(curr_aligned.reshape(1,  curr_aligned.shape[0], curr_aligned.shape[1]))

            # PREDICTION happens here
            # could replace this prediction with any algorithm
            class_pred = clf.predict(curr_aligned)[0]

            print(str(class_display[class_pred]))
            GLOBAL_CURRENT_TEXT = str(class_display[class_pred])
            update_text_from_outside(GLOBAL_CURRENT_TEXT)

            time.sleep(0.01)

        if N >= trial_num:
            flagstop = True

    allData = np.concatenate(allData)
    allData = np.transpose(allData, (1, 0))

    print('allData.shape',allData.shape)
    df = pd.DataFrame(allData)

    # 保存为CSV文件，指定文件名
    filename = r'C:\脑电采集\testdata\data_mi_tyl_tests4.csv'
    df.to_csv(filename, index=False)
    thread_data_server.stop()
    input('press to terminate')



if __name__ == '__main__':

    try:
        trial_num = int(input('enter trial number: '))
    except:
        trial_num = 8

    print('starting UI')

    start_ui()

    train_path_1 = r'C:\脑电采集\testdata\data_mi_train.csv'
    train_path_2 = r'C:\脑电采集\testdata\data_mi_trains1.csv'
    train_path_3 = r'C:\脑电采集\testdata\data_mi_tyl_trains2.csv'
    train_path_4 = r'C:\脑电采集\testdata\data_mi_syl_trains2.csv'
    # train_path_5 = r'C:\脑电采集\testdata\data_mi_tyl_trains3.csv'

    train_paths = [train_path_1, train_path_2, train_path_3, train_path_4]

    main(trial_num, train_paths=train_paths)

