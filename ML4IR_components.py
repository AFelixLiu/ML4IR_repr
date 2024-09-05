try:
    import numpy as np
    from collections import defaultdict

    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Lambda

    import keras as keras
    import keras.backend as K
except ModuleNotFoundError:
    raise Exception(
        "Please make sure rdkit, tensorflow and keras are installed!\n" + "In an anaconda environment run the following commands:\n" + "\t conda install -c rdkit rdkit\n" + "\t conda install -c conda-forge tensorflow\n" + "\t conda install keras")


# Generates the inputs of the NN training as a dictionary with keys ("smiles","X","Y" for train, val and test sets and "IDs")
# Molecular fragments which occur less than Nmin times in the training set are discarded 
# Nmin and radius are also stored in the result
def GenerateDATA(smiles, targets, radius=11, splits=None, Nmin=4):
    if splits is None:
        splits = [0.7, 0.15, 0.15]
    allIDs = defaultdict(int)
    morganfps = np.empty(len(smiles), dtype=object)

    indexes = np.random.permutation(len(smiles))

    train_indexes = indexes[:int(len(indexes) * splits[0])]
    val_indexes = indexes[int(len(indexes) * splits[0]):int(len(indexes) * (splits[0] + splits[1]))]
    test_indexes = indexes[int(len(indexes) * (splits[0] + splits[1])):]

    for smiles_ind in range(len(smiles)):
        curr_smiles = smiles[smiles_ind]
        mol = Chem.MolFromSmiles(curr_smiles)
        fp = AllChem.GetMorganFingerprint(mol, radius)
        morganfps[smiles_ind] = fp
        if smiles_ind in train_indexes:
            for ID in [*fp.GetNonzeroElements()]:
                allIDs[ID] += 1

    IDlist = np.empty(len(allIDs.keys()), dtype=object)
    IDlistcounter = 0
    for ID in allIDs.keys():
        if allIDs[ID] >= Nmin:
            IDlist[IDlistcounter] = ID
            IDlistcounter += 1
    IDlist = list(IDlist[:IDlistcounter])

    # fppercent = np.zeros(len(smiles))

    N_fragments = len(IDlist)
    Xs = np.zeros((len(smiles), N_fragments))
    for i in range(len(smiles)):
        for fragmentID in [*morganfps[i].GetNonzeroElements()]:
            if fragmentID in IDlist:
                location = IDlist.index(fragmentID)
                Xs[i][location] = morganfps[i][fragmentID]

    result = {"smiles_train": smiles[train_indexes],
              "smiles_val": smiles[val_indexes],
              "smiles_test": smiles[test_indexes],
              "X_train": Xs[train_indexes],
              "X_val": Xs[val_indexes],
              "X_test": Xs[test_indexes],
              "Y_train": targets[train_indexes],
              "Y_val": targets[val_indexes],
              "Y_test": targets[test_indexes],
              "IDs": IDlist,
              "radius": radius,
              "Nmin": Nmin}

    return result


# Generates the input vector to evaluate a new compound with a given trained model.
# The IDs have to be taken from the file which was used to train the network, the radius has to be set to the same number (or more)
def GenerateFP(smiles, IDs, radius=11):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprint(mol, radius)

    N_fragments = len(IDs)
    X = np.zeros(N_fragments)

    for fragmentID in [*fp.GetNonzeroElements()]:
        if fragmentID in IDs:
            location = IDs.index(fragmentID)
            X[location] = fp[fragmentID]
    return X


# Keras implementation of simplified EMD
def EMDloss(Y1, Y2):
    normed_Y1 = (Y1 / K.sum(Y1, axis=1)[:, None])
    normed_Y2 = (Y2 / K.sum(Y2, axis=1)[:, None])
    diff = normed_Y1 - normed_Y2

    return K.sum(K.abs(K.cumsum(diff, axis=1)))


def TrainModel(data, layers, lossfunc):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=len(data["X_train"][0]), activation='relu'))
    for i in range(len(layers) - 1):
        model.add(Dense(layers[i + 1], activation='relu'))
    model.add(Dense(len(data["Y_train"][0]), activation='linear'))
    model.add(Lambda(lambda x: K.abs(x)))

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=lossfunc, optimizer=opt)

    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=50, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=True)]

    model.fit(data["X_train"], data["Y_train"], epochs=999999, batch_size=25, callbacks=cb,
              validation_data=(data["X_val"], data["Y_val"]))

    return model
