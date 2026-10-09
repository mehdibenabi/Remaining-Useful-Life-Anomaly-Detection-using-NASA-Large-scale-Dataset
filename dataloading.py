import time
import h5py
import numpy as np
import pandas as pd



def load_data(filename):
    t = time.process_time()

    # Load data
    with h5py.File(filename, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_dev'))             # W
            X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
            X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
            T_dev = np.array(hdf.get('T_dev'))             # T
            Y_dev = np.array(hdf.get('Y_dev'))             # RUL
            A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

            # Test set
            W_test = np.array(hdf.get('W_test'))           # W
            X_s_test = np.array(hdf.get('X_s_test'))       # X_s
            X_v_test = np.array(hdf.get('X_v_test'))       # X_v
            T_test = np.array(hdf.get('T_test'))           # T
            Y_test = np.array(hdf.get('Y_test'))           # RUL
            A_test = np.array(hdf.get('A_test'))           # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))
            X_v_var = np.array(hdf.get('X_v_var'))
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))
            X_v_var = list(np.array(X_v_var, dtype='U20'))
            T_var = list(np.array(T_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))

 

    print('')
    print("Operation time (min): " , (time.process_time()-t)/60)
    print('')
    print ("W_dev shape: " + str(W_dev.shape) + " | W_test shape: " + str(W_test.shape))
    print ("X_s_dev shape: " + str(X_s_dev.shape) + " | X_s_test shape: " + str(X_s_test.shape))
    print ("X_v_dev shape: " + str(X_v_dev.shape) + " | X_v_test shape: " + str(X_v_test.shape))
    print ("A_dev shape: " + str(A_dev.shape) + " | A_test shape: " + str(A_test.shape))

    Y_cols = ['RUL']

    train_df = pd.DataFrame(
        np.hstack([A_dev, W_dev, X_s_dev, X_v_dev, Y_dev]),
        columns=A_var + W_var + X_s_var + X_v_var + Y_cols
    )

    train_df[['unit', 'Fc', 'hs']] = train_df[['unit', 'Fc', 'hs']].astype(int)

    test_df = pd.DataFrame(
        np.hstack([A_test, W_test, X_s_test, X_v_test, Y_test]),
        columns=A_var + W_var + X_s_var + X_v_var + Y_cols
    )

    test_df[['unit', 'Fc', 'hs']] = test_df[['unit', 'Fc', 'hs']].astype(int)

    return W_var, X_s_var, X_v_var, T_var, A_var, train_df, test_df
