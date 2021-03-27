```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm 
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')
from time import sleep
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title
def pretty(X):
    return pd.DataFrame(X)
d = 2
def plotDataPoints(ax,**kwargs):
#     fig,ax = plt.subplots()
    colors  = kwargs.pop('color',["#F44336","#536DFE","#4CAF50","#FF9800"])
    markers = kwargs.pop("marker",["*"]*4)
    data   = [_w0,_w1,_w2,_w3]
    labels = ["$\omega_0$","$\omega_1$","$\omega_2$","$\omega_3$"]
    for i,v in enumerate(data):
        ax.scatter(v[:,0],v[:,1],color=colors[i],label=labels[i],**kwargs,marker=markers[i])
    ax.legend()
#     %matplotlib notebook
```


```python
df     = pd.read_csv("data1.txt",delimiter="\t",header=None)
c      = int(df.shape[1]/d)
w0     = np.hstack([df.iloc[:,:2],np.full((10,1),0)])
_w0    = w0[:,:2]
w1     = np.hstack([df.iloc[:,2:4],np.full((10,1),1)])
_w1    = w1[:,:2]
w2     = np.hstack([df.iloc[:,4:6],np.full((10,1),2)])
_w2    = w2[:,:2]
w3     = np.hstack([df.iloc[:,6:],np.full((10,1),3)])
_w3    = w3[:,:2]
X      = np.concatenate([w0,w1,w2,w3])
_Xdf   = pretty(X).rename(columns = {0:'x1',1:'x2',2:'class'},inplace=False)
_Xdf['class']= _Xdf['class'].astype('int')
_X     = X[:,:2]
target = X[:,2]
wi     = [_w0,_w1,_w2,_w3]
```


```python
_,ax = plt.subplots(figsize=(10,10))
plotDataPoints(ax,marker=["H","*","o","p"],s=300)
ax.set_title("40 muestras")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(bbox_to_anchor=(1.05, 1))
# plt.tight_layout()
# plt.savefig("images/01.png")
```




    <matplotlib.legend.Legend at 0x7efd1b61ac40>




    
![png](README_files/README_2_1.png)
    


# Practica #1:  Riesgo condicional

a) Implemente el calculo del vector de medias, matriz de covarianza y probabilidad a priori definidas en
las Ecuaciones 32–34 de la clase AD-03. No se permite el uso de funciones predefinidas en bibliotecas o
toolboxes.

## Vector de medias


```python
def mean(x):
    return x.sum(axis=0)/x.shape[0]
```

Vector de medias para la clase w1


```python
w0_mean = mean(_w0)
w0_mean
```




    array([2.16, 2.49])



Vector de medias para la clase w2


```python
w1_mean = mean(_w1)
w1_mean
```




    array([ 3.95, -0.84])



Vector de medias para la clase w3


```python
w2_mean = mean(_w2)
w2_mean
```




    array([-1.57,  3.53])



Vector de medias para la clase w4


```python
w3_mean = mean(_w3)
w3_mean
```




    array([-5.99, -4.6 ])




```python
uk = [w0_mean,w1_mean,w2_mean,w3_mean]
```

# Matriz de covarianza


```python
def cov(x,uk):
    ni = x.shape[0]
    d  = x.shape[1]
    matrix = np.zeros((d,d))
    for i in range(d):
        X  = x[:,i]
        for j in range(d):
            Y=x[:,j]
            matrix[i][j]= ((X-uk[i])*(Y-uk[j])).sum()/(ni-1)
    return matrix
```

Matriz de covarianza para la clase w1


```python
w0_cov = cov(_w0,w0_mean)
w0_cov
```




    array([[ 9.32488889, 10.11733333],
           [10.11733333, 11.84988889]])



Matriz de covarianza para la clase w2


```python
w1_cov = cov(_w1,w1_mean)
w1_cov
```




    array([[ 8.36277778,  8.86777778],
           [ 8.86777778, 13.02488889]])



Matriz de covarianza para la clase w3


```python
w2_cov = cov(_w2,w2_mean)
w2_cov
```




    array([[7.63344444, 2.99344444],
           [2.99344444, 9.78233333]])



Matriz de covarianza para la clase w4


```python
w3_cov = cov(_w3,w3_mean)
w3_cov
```




    array([[ 8.62544444, -2.64444444],
           [-2.64444444, 27.64444444]])




```python
covX = [w0_cov,w1_cov,w2_cov,w3_cov]
```

# Probabilidad a priori


```python
data=[_w0,_w1,_w2,_w3]
total = sum(list(map(lambda x:x.shape[0],data)))
```

Probabilidad a priori de la clase w1


```python
Pw0 = _w0.shape[0]/total
Pw0
```




    0.25



Probabilidad a priori de la clase w2


```python
Pw1 = _w1.shape[0]/total
Pw1
```




    0.25



Probabilidad a priori de la clase w3


```python
Pw2 = _w2.shape[0]/total
Pw2
```




    0.25



Probabilidad a priori de la clase w4


```python
Pw3 = _w3.shape[0]/total
Pw3
```




    0.25




```python
Pwi = [Pw0,Pw1,Pw2,Pw3]
Pwi
```




    [0.25, 0.25, 0.25, 0.25]



b) Use las funciones en el inciso (a) para calcular las probabilidades posteriores de cada muestra en la
Tabla 1, implementado la Ecuaci´on 1 de la clase AD-03. La funcion de verosimilitud esta dada por la
funcion de densidad Gaussiana multivariante (Ecuacion 10 de la clase AD-03).

# Funcion de densidad Gaussiana Multivariante


```python
def px(x,ui,sigmai,pwi):
    d  = x.shape[0]
    sigmaDet = np.linalg.det(sigmai)
    denominator = 1/(np.power(2*np.pi,d/2) * np.power(sigmaDet,0.5))
    sigmaIv     = np.linalg.inv(sigmai)
    xMinusMean  = (x-ui)
    xMinusMeanT = -0.5*xMinusMean.T 
    exp         = np.exp(xMinusMeanT@sigmaIv@xMinusMean)
    return denominator*exp
```


```python
# gix(_w0[0],0) + gix(_w0[0],1)+gix(_w0[0],2)+gix(_w0[0],3)+  _rc(0,_w0[0])+_rc(1,_w0[0])+_rc(2,_w0[0])+_rc(3,_w0[0])
```




    0.9999999999999999



# Funcion discriminante Bayesiana


```python
def gix(x,wIndex):
    parameters  = zip(uk,covX,Pwi)
    denominator = np.array(list(map (lambda params: px(x,*params),parameters)))
    return (denominator[wIndex]*Pwi[wIndex]) / denominator.sum()

def bayesClassifier(x):
    data = []
    for i in range(c):
        data.append(gix(x,i))
    return np.argmax(np.array(data))

```


```python
for wii in wi:
    for xi in wii:
        for i in range(c):
            p = gix(xi,i)
            print("P(w{2}|{0}) = {1}".format(xi,p,i))
        print("_"*100)
```

    P(w0|[0.1 1.1]) = 0.18716282129244174
    P(w1|[0.1 1.1]) = 0.0005055445575647302
    P(w2|[0.1 1.1]) = 0.05967286387910771
    P(w3|[0.1 1.1]) = 0.0026587702708857756
    ____________________________________________________________________________________________________
    P(w0|[6.8 7.1]) = 0.23757298882055713
    P(w1|[6.8 7.1]) = 0.009452393627164093
    P(w2|[6.8 7.1]) = 0.0029744910756511287
    P(w3|[6.8 7.1]) = 1.2647662760554928e-07
    ____________________________________________________________________________________________________
    P(w0|[-3.5 -4.1]) = 0.1307783061448698
    P(w1|[-3.5 -4.1]) = 0.0007858781052554419
    P(w2|[-3.5 -4.1]) = 0.014506459567952597
    P(w3|[-3.5 -4.1]) = 0.10392935618192214
    ____________________________________________________________________________________________________
    P(w0|[2.  2.7]) = 0.22097356535132923
    P(w1|[2.  2.7]) = 0.001294023087747058
    P(w2|[2.  2.7]) = 0.027542661572876537
    P(w3|[2.  2.7]) = 0.00018974998804716502
    ____________________________________________________________________________________________________
    P(w0|[4.1 2.8]) = 0.1279833526584118
    P(w1|[4.1 2.8]) = 0.09635363379262601
    P(w2|[4.1 2.8]) = 0.025597661798881295
    P(w3|[4.1 2.8]) = 6.535175008090629e-05
    ____________________________________________________________________________________________________
    P(w0|[3.1 5. ]) = 0.18982354592429143
    P(w1|[3.1 5. ]) = 0.0006643746194834972
    P(w2|[3.1 5. ]) = 0.05944860506470119
    P(w3|[3.1 5. ]) = 6.3474391523887e-05
    ____________________________________________________________________________________________________
    P(w0|[-0.8 -1.3]) = 0.2074081386290973
    P(w1|[-0.8 -1.3]) = 0.0029985839702820227
    P(w2|[-0.8 -1.3]) = 0.029467511311315326
    P(w3|[-0.8 -1.3]) = 0.010125766089305361
    ____________________________________________________________________________________________________
    P(w0|[0.9 1.2]) = 0.21775138941212804
    P(w1|[0.9 1.2]) = 0.0015192802015135559
    P(w2|[0.9 1.2]) = 0.02984278032074696
    P(w3|[0.9 1.2]) = 0.0008865500656114253
    ____________________________________________________________________________________________________
    P(w0|[5.  6.4]) = 0.2372924631662967
    P(w1|[5.  6.4]) = 0.001477100749750657
    P(w2|[5.  6.4]) = 0.011228461711590327
    P(w3|[5.  6.4]) = 1.974372362275675e-06
    ____________________________________________________________________________________________________
    P(w0|[3.9 4. ]) = 0.23188755337261227
    P(w1|[3.9 4. ]) = 0.0056282979229511435
    P(w2|[3.9 4. ]) = 0.012468768559372198
    P(w3|[3.9 4. ]) = 1.5380145064408506e-05
    ____________________________________________________________________________________________________
    P(w0|[7.1 4.2]) = 0.00016886304450698834
    P(w1|[7.1 4.2]) = 0.2476911856811973
    P(w2|[7.1 4.2]) = 0.0021394006191776454
    P(w3|[7.1 4.2]) = 5.506551180519831e-07
    ____________________________________________________________________________________________________
    P(w0|[-1.4 -4.3]) = 0.008261160543151288
    P(w1|[-1.4 -4.3]) = 0.10524492961305654
    P(w2|[-1.4 -4.3]) = 0.020753661340055386
    P(w3|[-1.4 -4.3]) = 0.1157402485037368
    ____________________________________________________________________________________________________
    P(w0|[4.5 0. ]) = 1.8701531748614916e-07
    P(w1|[4.5 0. ]) = 0.247914382669119
    P(w2|[4.5 0. ]) = 0.002035328502080891
    P(w3|[4.5 0. ]) = 5.010181348261948e-05
    ____________________________________________________________________________________________________
    P(w0|[6.3 1.6]) = 1.6585977154410202e-08
    P(w1|[6.3 1.6]) = 0.24905275967560217
    P(w2|[6.3 1.6]) = 0.0009441218483749033
    P(w3|[6.3 1.6]) = 3.1018900457838232e-06
    ____________________________________________________________________________________________________
    P(w0|[4.2 1.9]) = 0.008969975254292701
    P(w1|[4.2 1.9]) = 0.22427001948860872
    P(w2|[4.2 1.9]) = 0.016677892639619083
    P(w3|[4.2 1.9]) = 8.21126174794719e-05
    ____________________________________________________________________________________________________
    P(w0|[ 1.4 -3.2]) = 8.747974219385748e-07
    P(w1|[ 1.4 -3.2]) = 0.24198668999423845
    P(w2|[ 1.4 -3.2]) = 0.0037020428974534896
    P(w3|[ 1.4 -3.2]) = 0.004310392310886172
    ____________________________________________________________________________________________________
    P(w0|[ 2.4 -4. ]) = 3.467022678043516e-12
    P(w1|[ 2.4 -4. ]) = 0.24722967043765967
    P(w2|[ 2.4 -4. ]) = 0.0007833996301756425
    P(w3|[ 2.4 -4. ]) = 0.001986929928697686
    ____________________________________________________________________________________________________
    P(w0|[ 2.5 -6.1]) = 3.788362223713315e-20
    P(w1|[ 2.5 -6.1]) = 0.23974143416927987
    P(w2|[ 2.5 -6.1]) = 0.0002797696783047116
    P(w3|[ 2.5 -6.1]) = 0.009978796152415453
    ____________________________________________________________________________________________________
    P(w0|[8.4 3.7]) = 3.979632221696758e-09
    P(w1|[8.4 3.7]) = 0.2496328037198604
    P(w2|[8.4 3.7]) = 0.00036710295203995724
    P(w3|[8.4 3.7]) = 8.93484674452299e-08
    ____________________________________________________________________________________________________
    P(w0|[ 4.1 -2.2]) = 1.7608691047657403e-12
    P(w1|[ 4.1 -2.2]) = 0.24908734076235783
    P(w2|[ 4.1 -2.2]) = 0.0007192672666118581
    P(w3|[ 4.1 -2.2]) = 0.0001933919692693973
    ____________________________________________________________________________________________________
    P(w0|[-3.  -2.9]) = 0.15619447844086676
    P(w1|[-3.  -2.9]) = 0.00039312714860292294
    P(w2|[-3.  -2.9]) = 0.02778713266374494
    P(w3|[-3.  -2.9]) = 0.0656252617467854
    ____________________________________________________________________________________________________
    P(w0|[0.5 8.7]) = 2.6292275276785584e-16
    P(w1|[0.5 8.7]) = 2.5852911931431442e-11
    P(w2|[0.5 8.7]) = 0.24940796642892474
    P(w3|[0.5 8.7]) = 0.000592033545222094
    ____________________________________________________________________________________________________
    P(w0|[2.9 2.1]) = 0.2015240800460432
    P(w1|[2.9 2.1]) = 0.023551061420234463
    P(w2|[2.9 2.1]) = 0.024752838074095767
    P(w3|[2.9 2.1]) = 0.0001720204596265617
    ____________________________________________________________________________________________________
    P(w0|[-0.1  5.2]) = 1.5533662901823767e-07
    P(w1|[-0.1  5.2]) = 6.68324586264229e-08
    P(w2|[-0.1  5.2]) = 0.24822274286349968
    P(w3|[-0.1  5.2]) = 0.001777034967412633
    ____________________________________________________________________________________________________
    P(w0|[-4.   2.2]) = 7.597865357305958e-12
    P(w1|[-4.   2.2]) = 1.317588567288523e-10
    P(w2|[-4.   2.2]) = 0.20391499668093066
    P(w3|[-4.   2.2]) = 0.04608500317971262
    ____________________________________________________________________________________________________
    P(w0|[-1.3  3.7]) = 2.710234398076642e-07
    P(w1|[-1.3  3.7]) = 5.2091174928240996e-08
    P(w2|[-1.3  3.7]) = 0.24376775869971662
    P(w3|[-1.3  3.7]) = 0.0062319181856686605
    ____________________________________________________________________________________________________
    P(w0|[-3.4  6.2]) = 7.428363006346377e-25
    P(w1|[-3.4  6.2]) = 2.1117754361483553e-15
    P(w2|[-3.4  6.2]) = 0.23330464767109427
    P(w3|[-3.4  6.2]) = 0.01669535232890362
    ____________________________________________________________________________________________________
    P(w0|[-4.1  3.4]) = 2.045383226025233e-16
    P(w1|[-4.1  3.4]) = 1.6883337967046204e-12
    P(w2|[-4.1  3.4]) = 0.21285284727785417
    P(w3|[-4.1  3.4]) = 0.037147152720457294
    ____________________________________________________________________________________________________
    P(w0|[-5.1  1.6]) = 4.5035575873536286e-14
    P(w1|[-5.1  1.6]) = 8.477359878455745e-12
    P(w2|[-5.1  1.6]) = 0.16270562991855944
    P(w3|[-5.1  1.6]) = 0.08729437007291817
    ____________________________________________________________________________________________________
    P(w0|[1.9 5.1]) = 0.01229114960299741
    P(w1|[1.9 5.1]) = 6.767601273161763e-05
    P(w2|[1.9 5.1]) = 0.23715193253302716
    P(w3|[1.9 5.1]) = 0.0004892418512438157
    ____________________________________________________________________________________________________
    P(w0|[-2.  -8.4]) = 6.506136097557759e-11
    P(w1|[-2.  -8.4]) = 0.10816928061951014
    P(w2|[-2.  -8.4]) = 0.0002422931088395525
    P(w3|[-2.  -8.4]) = 0.14158842620658896
    ____________________________________________________________________________________________________
    P(w0|[-8.9  0.2]) = 1.2780085332671314e-26
    P(w1|[-8.9  0.2]) = 8.77550138715224e-18
    P(w2|[-8.9  0.2]) = 0.02663963191526081
    P(w3|[-8.9  0.2]) = 0.2233603680847392
    ____________________________________________________________________________________________________
    P(w0|[-4.2  7.7]) = 1.7216004569835474e-37
    P(w1|[-4.2  7.7]) = 9.053469696491138e-20
    P(w2|[-4.2  7.7]) = 0.21652904432388065
    P(w3|[-4.2  7.7]) = 0.03347095567611936
    ____________________________________________________________________________________________________
    P(w0|[-8.5 -3.2]) = 1.0733290252849963e-11
    P(w1|[-8.5 -3.2]) = 8.113481383766768e-12
    P(w2|[-8.5 -3.2]) = 0.010999372350258637
    P(w3|[-8.5 -3.2]) = 0.2390006276308946
    ____________________________________________________________________________________________________
    P(w0|[-6.7 -4. ]) = 7.231894984404815e-05
    P(w1|[-6.7 -4. ]) = 8.263679729284512e-08
    P(w2|[-6.7 -4. ]) = 0.013914066303238524
    P(w3|[-6.7 -4. ]) = 0.23601353211012013
    ____________________________________________________________________________________________________
    P(w0|[-0.5 -9.2]) = 1.6654497138551935e-19
    P(w1|[-0.5 -9.2]) = 0.1192244145445471
    P(w2|[-0.5 -9.2]) = 6.654194208112023e-05
    P(w3|[-0.5 -9.2]) = 0.13070904351337176
    ____________________________________________________________________________________________________
    P(w0|[-5.3 -6.7]) = 0.0320798464431739
    P(w1|[-5.3 -6.7]) = 0.0004562189949390247
    P(w2|[-5.3 -6.7]) = 0.002055047881875568
    P(w3|[-5.3 -6.7]) = 0.21540888668001149
    ____________________________________________________________________________________________________
    P(w0|[-8.7 -6.4]) = 3.423434751683875e-05
    P(w1|[-8.7 -6.4]) = 1.6755241764153365e-08
    P(w2|[-8.7 -6.4]) = 0.0015122292076396673
    P(w3|[-8.7 -6.4]) = 0.24845351968960175
    ____________________________________________________________________________________________________
    P(w0|[-7.1 -9.7]) = 0.0017872001996606865
    P(w1|[-7.1 -9.7]) = 0.00027944197092399036
    P(w2|[-7.1 -9.7]) = 9.589241589653596e-05
    P(w3|[-7.1 -9.7]) = 0.24783746541351878
    ____________________________________________________________________________________________________
    P(w0|[-8.  -6.3]) = 0.00041786836861744533
    P(w1|[-8.  -6.3]) = 1.4264254166328436e-07
    P(w2|[-8.  -6.3]) = 0.0019447262181772136
    P(w3|[-8.  -6.3]) = 0.2476372627706637
    ____________________________________________________________________________________________________


## Grafica 2D del clasificador bayesiano


```python
plot_classifier(bayes_fx,
                title="Clasificador Bayesiano",
                plot_step=.3,
                filename="bayes",
                edgecolor="#000",
                save_img=True
               )
```




    <AxesSubplot:title={'center':'Clasificador Bayesiano'}, xlabel='$x_1$', ylabel='$x_2$'>




    
![png](README_files/README_46_1.png)
    


c) Calcule el riesgo condicional de las muestras de entrenamiento (Ecuaci´on 19 de la clase AD-02).

#  Riesgo condicional


```python
def lambdaAlphaiWj(i,j):
    return 0 if(i==j) else 1

def _rc(xi):
    def __rc(i):
        k = bayesClassifier(xi)
        res =0 
        for j in range(c):
            res += lambdaAlphaiWj(k,j)*gix(xi,j)
        return res
    return sum(list(map(lambda i:__rc(i),range(c))))
for wi in [_w0,_w1,_w2,_w3]:
    for xi in wi:
        val = _rc(xi)
        print("Riego condicional {}\t=>\t{}".format(xi,val))
    print("_"*100)
# _rc(_w0[0])
```

    Riego condicional [0.1 1.1]	=>	0.25134871483023286
    Riego condicional [6.8 7.1]	=>	0.04970804471777131
    Riego condicional [-3.5 -4.1]	=>	0.4768867754205207
    Riego condicional [2.  2.7]	=>	0.11610573859468304
    Riego condicional [4.1 2.8]	=>	0.48806658936635283
    Riego condicional [3.1 5. ]	=>	0.24070581630283427
    Riego condicional [-0.8 -1.3]	=>	0.17036744548361085
    Riego condicional [0.9 1.2]	=>	0.12899444235148777
    Riego condicional [5.  6.4]	=>	0.050830147334813044
    Riego condicional [3.9 4. ]	=>	0.072449786509551
    ____________________________________________________________________________________________________
    Riego condicional [7.1 4.2]	=>	0.009235257275210743
    Riego condicional [-1.4 -4.3]	=>	0.5370390059850528
    Riego condicional [4.5 0. ]	=>	0.008342469323523987
    Riego condicional [6.3 1.6]	=>	0.0037889612975913665
    Riego condicional [4.2 1.9]	=>	0.10291992204556502
    Riego condicional [ 1.4 -3.2]	=>	0.0320532400230464
    Riego condicional [ 2.4 -4. ]	=>	0.011081318249361406
    Riego condicional [ 2.5 -6.1]	=>	0.04103426332288066
    Riego condicional [8.4 3.7]	=>	0.0014687851205584966
    Riego condicional [ 4.1 -2.2]	=>	0.003650636950568498
    ____________________________________________________________________________________________________
    Riego condicional [-3.  -2.9]	=>	0.3752220862365331
    Riego condicional [0.5 8.7]	=>	0.0023681342843010753
    Riego condicional [2.9 2.1]	=>	0.19390367981582715
    Riego condicional [-0.1  5.2]	=>	0.0071090285460011105
    Riego condicional [-4.   2.2]	=>	0.18434001327627736
    Riego condicional [-1.3  3.7]	=>	0.024928965201133586
    Riego condicional [-3.4  6.2]	=>	0.06678140931562293
    Riego condicional [-4.1  3.4]	=>	0.14858861088858333
    Riego condicional [-5.1  1.6]	=>	0.34917748032576224
    Riego condicional [1.9 5.1]	=>	0.051392269867891374
    ____________________________________________________________________________________________________
    Riego condicional [-2.  -8.4]	=>	0.4336462951736442
    Riego condicional [-8.9  0.2]	=>	0.10655852766104328
    Riego condicional [-4.2  7.7]	=>	0.13388382270447743
    Riego condicional [-8.5 -3.2]	=>	0.04399748947642163
    Riego condicional [-6.7 -4. ]	=>	0.05594587155951946
    Riego condicional [-0.5 -9.2]	=>	0.4771638259465129
    Riego condicional [-5.3 -6.7]	=>	0.13836445327995398
    Riego condicional [-8.7 -6.4]	=>	0.006185921241593081
    Riego condicional [-7.1 -9.7]	=>	0.008650138345924853
    Riego condicional [-8.  -6.3]	=>	0.009450948917345289
    ____________________________________________________________________________________________________


# Práctica #2

a) Use las funciones en el inciso (a) de la Practica #1 para entrenar un clasificador basado en funciones
discriminantes Gaussianas cuadraticas (Ecuacion 26 de la clase AD-03)

## Funcion discriminante cuadratica


```python
def _gix(xi,covs,uk,pw):
    result = []
    for i in range(c):
        w_cov     = covs[i] 
        ui        = uk[i]
        pwi       = pw[i] 
        sigmaIv   = np.linalg.inv(w_cov)
        Wi        = -0.5 * sigmaIv
        wi        = sigmaIv@ui
        sigma_det = np.linalg.det(w_cov)
        w0i       = -0.5*ui.T@sigmaIv@ui-0.5*np.log(sigma_det)+np.log(pwi)
        xiT       = xi.T
        res       = xiT@Wi@xi + wi.T@xi+w0i
        result.append(res)
        
    return np.argmax(result)
```

## Grafica 2D del clasificador basadado en funciones discriminantes guassianas Cuadraticas


```python
plot_classifier(quadratic_fx,title="Clasificador Cuadratico",plot_step=.3,filename="quadratic",save_img=True)
```




    <AxesSubplot:title={'center':'Clasificador Cuadratico'}, xlabel='$x_1$', ylabel='$x_2$'>




    
![png](README_files/README_55_1.png)
    


# Practica #3

a) A partir de los datos en la Tabla 1, use las funciones en el inciso (a) de la Practica #1 para entrenar un clasificador de mınima distancia Euclidiana y un clasificador de mınima distancia Mahalanobis
(Ecuaciones 18 y 23 de la clase AD-03).


```python
def cMinEucli(x,uk):
    result = []
    for i in range(c):
        ui         = uk[i]
        xMinusMean = x-ui
        res =np.sqrt(xMinusMean.T@xMinusMean)
        result.append(res)
    return np.argmin(np.array(result))
#     return result

def cMinMaha(x,uk,sigmas):
    result = []
    combined_sigma             = np.zeros((2,2))
    combined_sigma_denominator =  0
    for i in range(c):
        ni = wi[i].shape[0]
        combined_sigma_denominator+= 1/(ni-1)
        combined_sigma            += (ni-1)*sigmas[i]
    combined_sigma = combined_sigma*combined_sigma_denominator
        
    for i in range(c):
        ui         = uk[i]
        sigmaInv   = np.linalg.inv(combined_sigma)
        xMinusMean = x-ui
        res = np.sqrt(xMinusMean.T@sigmaInv@xMinusMean)
        result.append(res)
    return np.argmin(np.array(result))
```

# Grafica 2D Clasificador de minima distancia euclidiana


```python
plot_classifier(med_fx,title="Clasificador de minima distancia Euclidiana",plot_step=.3,filename="mde",save_img=True)
```




    <AxesSubplot:title={'center':'Clasificador de minima distancia Euclidiana'}, xlabel='$x_1$', ylabel='$x_2$'>




    
![png](README_files/README_60_1.png)
    


# Grafica 2D Clasificador de minima distancia Mahalanobis


```python
plot_classifier(mmd,title="Clasificador de minima distancia Mahalanobis",plot_step=.3,filename="mmd",save_img=True)
```




    <AxesSubplot:title={'center':'Clasificador de minima distancia Mahalanobis'}, xlabel='$x_1$', ylabel='$x_2$'>




    
![png](README_files/README_62_1.png)
    


b) Clasifique los siguientes patrones de prueba: $\mathbf{x_1} = [-1.7,-2.6]^T$, $\mathbf{x_2}=[0.3,-7.6]^T,\mathbf{x_3}=[-7.3,5.0]^T y \quad \mathbf{x_4}=[5.6,2.8]^T$, usando los dos clasificadores implementados en el inciso anterior, asi como el clasificador
cuadratico implementado en la Practica #2.


```python
x1 = np.array([-1.7,-2.6])
x2 = np.array([0.3,-7.6])
x3 = np.array([-7.3,5.0])
x4 = np.array([5.6,2.8])
data = [x1,x2,x3,x4]
def classifyPatterns(data,fx,**kwargs):
    filename = kwargs.get("filename","default")
    title    = kwargs.get("title","default")
    names = ['x1','x2','x3','x4']
    colors        = ["#F44336","#536DFE","#4CAF50","#FF9800"]
    ax= plot_classifier(fx,title=title,plot_step=.3,filename=filename,show_patterns=False)
    for i,xi in enumerate(data):
        val = fx(xi)
        label = "$\mathbf{x}_"+str(i+1)+ "\longrightarrow w_"+str(val)+"$"
        ax.scatter(xi[0],xi[1],color=colors[val],s=500,edgecolors="#000",label=label)
    ax.legend(bbox_to_anchor=(1.05, 1),title="Patrones")
    plt.savefig("images/{}".format(filename),dpi=300,bbox_inches='tight')
```

### Clasificador cuadratico


```python
classifyPatterns(data,quadratic_fx,filename="quadratic_3",title="Clasificador Cuadrático")
```


    
![png](README_files/README_66_0.png)
    


### Clasificador de minima distancia Euclidiana


```python
classifyPatterns(data,med_fx,filename="mde_3",title="Clasificador de mínima distancia Euclidiana")
```


    
![png](README_files/README_68_0.png)
    


### Clasificador de minima distancia Mahalanobis


```python
classifyPatterns(data,mmd,filename="mmd_3",title="Clasificador de mínima distancia Mahalanobis")
```


    
![png](README_files/README_70_0.png)
    


### Clasificador Bayesiano


```python
classifyPatterns(data,bayes_fx,filename="bayes_3",title="Clasificador Bayesiano")
```


    
![png](README_files/README_72_0.png)
    



```python
def get_accuracy(fx):
    target_pred=np.zeros((40))
    for i,x in enumerate(_X):
        target_pred[i]=fx(x)

    acc = target_pred == target
    acc =(target.shape[0] - np.bitwise_not(acc).sum())/target.shape[0]
    return acc
get_accuracy(bayes_fx),get_accuracy(quadratic_fx),get_accuracy(med_fx),get_accuracy(mmd)
```




    (0.9, 0.9, 0.75, 0.825)




```python
def plot_classifier(fx,**kwargs):
    show_patterns = kwargs.get("show_patterns",True)
    save          = kwargs.get("save_img",False)
    colors        = kwargs.get("colors",["#F44336","#536DFE","#4CAF50","#FF9800"])
    labels        = ["$\omega_0$","$\omega_1$","$\omega_2$","$\omega_3$"]
    plot_step     = kwargs.get('plot_step',1)
    filename      = kwargs.get("filename","01")
    edgecolor     = kwargs.get("edgecolor","#000")
    xMin          = -10
    xMax          = 10
    x             = np.arange(xMin, xMax , plot_step)
    y             = np.arange(xMin,xMax,plot_step)
    xx , yy       = np.meshgrid(x,y)
    _,ax          = plt.subplots(figsize=(10,10))
#     data          = []
    title         = kwargs.get('title',"Clasificador")
    _data =dict()
#     with tqdm(total=xx.shape[0],desc="Processing(0)") as pbar:
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x1      = xx[i][j]
            x2      = yy[i][j]
            outcome = fx(np.array([x1,x2]))
#                 data.append((x1,x2,outcome))
            if not outcome  in _data:
                _data[outcome] =[]
            _data[outcome].append([x1,x2])
#         pbar.update(1)
    
#     print(_data)
#     with tqdm(total=len(_data),desc="Rendering(0)") as pbar:
    for key, value in _data.items():
        _value = np.array(value)
        ax.scatter(_value[:,0],_value[:,1],color=colors[key],alpha=.5)
#         pbar.update(1)

    if(show_patterns):
        data =[]
#         with tqdm(total=_X.shape[0],desc="Processing(1)") as pbar:
        for x in  _X:
            outcome = fx(np.array([x[0],x[1]]))
            data.append((x[0],x[1],outcome))
#             pbar.update(1)

#         with tqdm(total=len(data),desc ="Rendering(1)") as pbar:
        for (x1,x2,color) in data:
            ax.scatter(x1,x2,
                       color=colors[color],
                       alpha=1,
                       marker="H",
                       s=200,
                       edgecolor=edgecolor,
                       linewidth=1.5
                      )
#             pbar.update(1)

    ax.set_title(title)
#     ax.set_title("Clasificador Bayesiano")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    markers =list(
       map( lambda x: Line2D([], [], 
               marker='H', 
               color=x[0],
               label=x[1], 
               markersize=10,
               linestyle='',
#                linewidth=2
              ),zip(colors,labels)
    ) )
    ax.legend(handles=markers,bbox_to_anchor=(1.05, 1),title="Clases")
    if(save):
        plt.savefig("images/{}".format(filename),bbox_inches='tight',dpi=300)
    
    return ax
def bayes_fx(xi):
    return bayesClassifier(xi)
def quadratic_fx(xi):
    return _gix(xi,covX,uk,Pwi)
def med_fx(xi):
    return cMinEucli(xi,uk)
def mmd(xi):
    return cMinMaha(xi,uk,covX)
```

# Grafica 3D: Riesgo condicional


```python
%matplotlib inline
plot_step = 0.5
x= np.arange(-10,10,plot_step)
y= np.arange(-10,10,plot_step)
# z= np.sin(10*x*y)
X,Y = np.meshgrid(x,y)
Z   = np.dstack((X,Y))
data = np.zeros(X.shape)
index=2
for i,z in enumerate(Z) :
#     print(z)
    for j,xi in enumerate(z):
        outcome = bayesClassifier(xi)
        cr      = _rc(outcome,xi)
#         data[i][j]= px(xi,uk[index],covX[index],Pwi[index])
        data[i][j]= cr 
        
# Z = np.sin(X*5)*np.cos(Y*5)
fig= plt.figure(figsize=(10,10))
ax= fig.add_subplot(111, projection= '3d')
# print(data.shape)
surf=ax.plot_surface(
    X,Y,
    data+.1,
    cmap='afmhot',
    linewidth=2,antialiased='True',rstride=5,cstride=1,alpha=.5
                    )
ax.contourf(X, Y, data,1000, zdir='z', offset=0,cmap='afmhot')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.set_xlim([-0.5, 1.5])
# ax.set_ylim([-0.5, 1.5])
# ax.set_zlim([-1.5, 1.5])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$R(a_i | x)$")
fig.colorbar(surf,aspect=30,fraction=0.046, pad=0.04)
plt.savefig("images/3d")
# plt.show()
# Z[0].shape
```


    
![png](README_files/README_76_0.png)
    

