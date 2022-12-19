import tensorflow as tf
from tqdm import tqdm
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

Ztrain = np.load('Z_train.npy')
Zmax = np.max(Ztrain,axis=0)
Zmin = np.min(Ztrain,axis=0)
Ztrain = 2*(Ztrain-Zmin)/(Zmax-Zmin)-1
Tl  = 29999

Z_train = np.zeros((Tl,Ztrain.shape[1],6))
for i in range(6):
    Z_train[:,:,i] = Ztrain[i*Tl:(i+1)*Tl,:]

Ztest = np.load('Z_test.npy')
Ztest = 2*(Ztest-Zmin)/(Zmax-Zmin)-1
Z_test = np.zeros((Tl,Ztest.shape[1],69))
for i in range(69):
    Z_test[:,:,i] = Ztest[i*Tl:(i+1)*Tl,:]

F_train = np.load('Ftrain.npy')
F_test = np.load('Ftest.npy')

look_back = 150

DDN = []
DDNV = []
DDNT = []

for i in range(6):
  DDN.append(np.concatenate((F_train[:,:,i],Z_train[:,:,i]),axis=1))

def create_dataset(dataset, look_back=look_back):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[(i+1):(i+look_back+1),:4]
        b = dataset[i:(i+look_back),4:]
        A = np.concatenate((a,b),axis=1)
        X.append(A)
        Y.append(dataset[i + look_back, 4:])
    return np.array(X), np.array(Y)

XY=[]
for i in range(len(DDN)):
    XY.append(create_dataset(DDN[i]))

xtrain, ytrain = zip(*XY)
xtrain = list(xtrain)
ytrain = list(ytrain)
x_train = np.concatenate(xtrain,axis=0)
y_train = np.concatenate(ytrain,axis=0)


ls=25
Inp = tf.keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2]))
h = tf.keras.layers.LSTM(ls,return_sequences=False)(Inp)
h = tf.keras.layers.Dense(4,activation='linear')(h)
mdl = tf.keras.Model(Inp,h)

wtsName = str(ls)+'LS_wts-2.hdf5'

call_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-9, patience = 40)
call_checkpoint = tf.keras.callbacks.ModelCheckpoint(wtsName, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

Adopt= tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, clipnorm=1)
mdl.compile(loss='mean_squared_error', optimizer=Adopt)

#mdl.fit(x_train,y_train, epochs=2000,batch_size = 8192, validation_data = (x_train,y_train),callbacks = [call_stopping,call_checkpoint])
mdl.load_weights(wtsName)
mdl.evaluate(x_train,y_train)

Inp2 = tf.keras.layers.Input(batch_shape=(1,1,x_train.shape[2]))
h2 = tf.keras.layers.LSTM(25,return_sequences=False,stateful=True)(Inp2)
h2 = tf.keras.layers.Dense(4,activation='linear')(h2)
mdlP = tf.keras.Model(Inp2,h2)
mdlP.set_weights(mdl.get_weights())


def statefulPrediction(dof):
    mdlP.reset_states()
    FT = F_test[:,:,dof]
    YT = Z_test[:,:,dof]
    FT = FT[1:,:]
    YT = YT[0:-1,:]


    datasetT=np.concatenate((FT,YT),axis=1)
    dataT = np.reshape(datasetT,(-1,1,8))
    mdlP.reset_states()

    @tf.function
    def makePred(lstmPred,Ti):
        return lstmPred(Ti)

    zz1=[]
    x0 = dataT[0:1,:,:]
    for i in range(000,29900):
      zzz = makePred(mdlP,x0)
      zz1.append(zzz)
      zzz = np.reshape(zzz,(1,1,4))
      x0 = np.concatenate((dataT[i+1:i+2,:,0:4],zzz),axis=2)

    z = np.squeeze(np.concatenate(zz1,axis=0))

    return z

TestInd = 53

y0 = statefulPrediction(TestInd)
y0 = ((y0+1)/2)*(Zmax-Zmin)+Zmin


Decoder = tf.keras.models.load_model('decoder')
Xhat = Decoder.predict(y0)
Xtrue = np.load('X_test.npy')

Xhat.shape


plt.figure(figsize = (13.5,9))
plt.plot(Xhat[:,-40])
plt.plot(Xtrue[:29990,-40,TestInd],'--')
plt.xlim([0, 29990])

ZZ = []
for i in range(69):
    Z = statefulPrediction(i)
    ZZ.append(((Z+1)/2)*(Zmax-Zmin)+Zmin)

XX = []
for i in range(69):
    XX.append(Decoder.predict(ZZ[i]))

from sklearn.metrics import mean_squared_error
Err = []
for i, x in enumerate(XX):
    Err.append(mean_squared_error(Xtrue[:29900,:,i],x))

plt.figure()
plt.hist(Err)
plt.show()
