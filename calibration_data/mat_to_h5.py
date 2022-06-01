import h5py
import sys
import traceback

def to_label_dict(arr):
    num_channels = max(arr.shape)
    s_length = min(arr.shape)
    # invert shape
    strings = dict()
    for i in range(num_channels):
        s = ""
        for j in range(s_length):
            s += chr(arr[j,i])
        # remove trailing whitespace
        s = s.strip()
        strings[s] = i
    return strings

def save_mat_as_h5():
    mat = h5py.File("EEG_raw.mat", mode='r')
    try:
        chanlabels, data, srate = mat["chanlabels"][:], mat['data'][:], mat['srate'][0,0]
        channels = to_label_dict(chanlabels[:])
        print(channels)
        # h5 = h5py.File("EEG.h5", mode="w")
        # try:
        #     h5.create_dataset('channels', data=chanlabels)
        #     h5.create_dataset('data', data=data)
        #     h5.create_dataset('srate', data=srate)
        # except:
        #     print("Error on writing to EEG.h5")
        #     traceback.print_exception(*sys.exc_info())
        # finally:
        #     h5.close()
    except:
        print("Error on reading from EEG_raw.mat")
        traceback.print_exception(*sys.exc_info())
    finally:
        mat.close()
