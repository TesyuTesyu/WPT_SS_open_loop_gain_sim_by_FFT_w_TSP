import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def make_window_func(window_name, segment_size):
    #今のところnumpy のみ.scipy.signalを使えばもっとある.
    if window_name=='bartlett':
        window=np.bartlett(segment_size)
    elif window_name=='blackman':
        window=np.blackman(segment_size)
    elif window_name=='hamming':
        window=np.hamming(segment_size)
    elif window_name=='hanning':
        window=np.hanning(segment_size)
    elif window_name=='kaiser':
        window=np.kaiser(segment_size)
    elif window_name=='square':
        window=np.ones(segment_size)

    return window
#一部の時間領域を切り取ってfftするやつ.
def f_fft(y, window_name, fs):
    segment_size = len(y)
    freq_min = 1/(segment_size/fs)#周波数の刻み幅.
    final_freq = fs/2
    fin_num_plt = int(final_freq/freq_min)#plotするときのポイント数.
    freq_segment = np.fft.fftfreq(segment_size, d=1/fs)[0:fin_num_plt]#周波数軸のmatrix.
    window = make_window_func(window_name, segment_size)#窓関数.
    windowed_segment_V = y * window
    FV_segment = np.fft.fft(windowed_segment_V) / (segment_size / 2)
    return freq_segment, FV_segment[0:fin_num_plt]

#LTspiceが出した WPT_SS_v8_freq_control.txtのファイルパス.以下のような形式.
'''
time	V(contac)	V(out)
0.000000000000000e+000	1.256637e-007	2.354353e+001
3.743229611778004e-008	-4.578217e-006	2.354413e+001
1.122968883542075e-007	-1.398605e-005	2.354559e+001
'''
file_path = r"??.txt"

fs=2e6

# データを読み込むプログラム
time = []
v_contac = []
v_out = []
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # ヘッダー行をスキップ
        for line in file:
            columns = line.split()
            time.append(float(columns[0]))
            v_contac.append(float(columns[1]))
            v_out.append(float(columns[2]))
except FileNotFoundError:
    print(f'Error: The file at {file_path} was not found.')
except Exception as e:
    print(f'An error occurred: {e}')

time=np.array(time)
v_contac=np.array(v_contac)
v_out=np.array(v_out)

time_regular = np.arange(0, time[-1], 1/fs)
# 線形補完を実行
f_linear = interpolate.interp1d(time, v_contac, kind='linear', fill_value='extrapolate')
v_contac_regular = f_linear(time_regular)
f_linear = interpolate.interp1d(time, v_out, kind='linear', fill_value='extrapolate')
v_out_regular = f_linear(time_regular)

freq_segment, fft_v_contac_regular=f_fft(v_contac_regular, "square", fs)
freq_segment, fft_v_out_regular=f_fft(v_out_regular, "square", fs)

fft_Gain=fft_v_out_regular/fft_v_contac_regular
phase=np.arctan2(np.imag(fft_Gain),np.real(fft_Gain))

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.title('open loop gain')
ax[0,0].plot(freq_segment[1:] ,20*np.log10(np.abs(fft_Gain[1:])),"k-")
ax[0,0].set_ylabel("Magnitude [dB]")
ax[0,0].set_xscale('log')
ax[1,0].plot(freq_segment[1:] ,phase[1:]*180/np.pi,"k-")
ax[1,0].set_ylabel("Phase [deg]")
ax[1,0].set_xlabel("Frequency [Hz]")
ax[1,0].set_xscale('log')

plt.show()