from scipy import signal
import numpy as np

def PhaseLaggingIndex(signal1, signal2, index='PLV'):
    """
    Phase locking value of two signals
        @ parameter
            signal1, signal2: n_epochs * n_times
        @ return
            PLV: phase locking value for signal1 and signal 2 (1 * times)
    """
    n_epoch = signal1.shape[0]
    analytic_s1 = signal.hilbert(signal1)  # 希尔伯特变化 # analytic_s1: : n_epochs * n_times
    analytic_s2 = signal.hilbert(signal2)  # analytic_s2: : n_epochs * n_times

    phase1_instant = np.angle(analytic_s1)  # phase1_instant: : n_epochs * n_times
    phase2_instant = np.angle(analytic_s2)  # phase2_instant: : n_epochs * n_times
    delta_phase = phase1_instant - phase2_instant  # 相位差 # delta_phase2: : n_epochs * n_times
    # print(np.sign(np.sin(delta_phase)).shape)
    if index == 'PLI':
        # PLI = np.absolute(np.sum(np.sign(np.sin(delta_phase)), axis=0))/n_epoch
        PLI = np.absolute(np.mean(np.sign(np.sin(delta_phase)), axis=0))
        # PLI = abs(np.mean(np.sign(np.sin(delta_phase)), axis=0))
        pindex = PLI
        # print(pindex.shape)
    elif index == 'PLV':

        PLV = np.absolute(np.sum(np.exp(1j * delta_phase), axis=0)) / n_epoch  # PLV: 1* n_times
        # plv_vector = np.divide(abs(np.sum(np.exp(1j * (delta_phase)), axis=1)),1)
        # print(plv_vector)
        # PLV = np.reshape(plv_vector, (n_epoch, 1, 1), order='F')
        pindex = PLV
    # PLI_mean = np.mean(PLV, axis=0)

    elif index == 'wPLI':
        imz = np.sin(delta_phase)
        wpli_vector = np.divide(abs(np.mean(np.multiply(abs(imz), np.sign(imz)), axis=1)), np.mean(abs(imz), axis=1))
        wpli = np.reshape(wpli_vector, (n_epoch, 1, 1), order='F')
        pindex = wpli
        # print(PLI_mean)
    index_mean = np.mean(pindex)
    # print(index_mean)
    return index_mean


def compute_PLV(data):
    """
    计算每个trial两两电极间的锁相值（PLV）
        @ parameter
            data: trials * timepoints * electrodes
                trials: 实验次数
                timepoints: 时间序列维度
                electrodes: 电极数
        @ return
            plv_data: trials * electrodes * electrodes (每个trial里面电极和电极两两之间的锁相值)
    """
    trials, timepoints, electrodes = data.shape

    # 初始化plv_data为0矩阵，大小为trials * electrodes * electrodes
    plv_data = np.zeros((trials, electrodes, electrodes))

    # 遍历每个trial，每个电极对之间计算PLV值
    for trial in range(trials):
        for i in range(electrodes):
            for j in range(i + 1, electrodes):
                # 计算第i个电极和第j个电极之间的PLV值
                plv_data[trial, i, j] = PhaseLaggingIndex(data[trial, :, i], data[trial, :, j])
                # 计算第j个电极和第i个电极之间的PLV值
                plv_data[trial, j, i] = plv_data[trial, i, j]

    return plv_data