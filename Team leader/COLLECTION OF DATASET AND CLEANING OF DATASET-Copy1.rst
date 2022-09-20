
.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import scipy.stats as stats
    import seaborn as sns
    
    %matplotlib inline

.. code:: ipython3

    df = pd.read_csv('CKD dataset.csv')
    data = df
    data.head()
    




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>bp</th>
          <th>sg</th>
          <th>al</th>
          <th>su</th>
          <th>rbc</th>
          <th>pc</th>
          <th>pcc</th>
          <th>ba</th>
          <th>bgr</th>
          <th>...</th>
          <th>wbcc</th>
          <th>rbcc</th>
          <th>htn</th>
          <th>dm</th>
          <th>cad</th>
          <th>appet</th>
          <th>pe</th>
          <th>ane</th>
          <th>class</th>
          <th>Unnamed: 25</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>48.0</td>
          <td>80.0</td>
          <td>1.020</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>normal</td>
          <td>notpresent</td>
          <td>notpresent</td>
          <td>...</td>
          <td>44.0</td>
          <td>7800.0</td>
          <td>5.2</td>
          <td>yes</td>
          <td>yes</td>
          <td>no</td>
          <td>good</td>
          <td>no</td>
          <td>no</td>
          <td>ckd</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>7.0</td>
          <td>50.0</td>
          <td>1.020</td>
          <td>4.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>normal</td>
          <td>notpresent</td>
          <td>notpresent</td>
          <td>...</td>
          <td>38.0</td>
          <td>6000.0</td>
          <td>NaN</td>
          <td>no</td>
          <td>no</td>
          <td>no</td>
          <td>good</td>
          <td>no</td>
          <td>no</td>
          <td>ckd</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>62.0</td>
          <td>80.0</td>
          <td>1.010</td>
          <td>2.0</td>
          <td>3.0</td>
          <td>normal</td>
          <td>normal</td>
          <td>notpresent</td>
          <td>notpresent</td>
          <td>...</td>
          <td>31.0</td>
          <td>7500.0</td>
          <td>NaN</td>
          <td>no</td>
          <td>yes</td>
          <td>no</td>
          <td>poor</td>
          <td>no</td>
          <td>yes</td>
          <td>ckd</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>48.0</td>
          <td>70.0</td>
          <td>1.005</td>
          <td>4.0</td>
          <td>0.0</td>
          <td>normal</td>
          <td>abnormal</td>
          <td>present</td>
          <td>notpresent</td>
          <td>...</td>
          <td>32.0</td>
          <td>6700.0</td>
          <td>3.9</td>
          <td>yes</td>
          <td>no</td>
          <td>no</td>
          <td>poor</td>
          <td>yes</td>
          <td>yes</td>
          <td>ckd</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NaN</td>
          <td>51.0</td>
          <td>80.0</td>
          <td>1.010</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>normal</td>
          <td>normal</td>
          <td>notpresent</td>
          <td>notpresent</td>
          <td>...</td>
          <td>35.0</td>
          <td>7300.0</td>
          <td>4.6</td>
          <td>no</td>
          <td>no</td>
          <td>no</td>
          <td>good</td>
          <td>no</td>
          <td>no</td>
          <td>ckd</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 26 columns</p>
    </div>



.. code:: ipython3

    data.shape
    




.. parsed-literal::

    (400, 26)



.. code:: ipython3

    df.info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 400 entries, 0 to 399
    Data columns (total 26 columns):
    age            0 non-null float64
    bp             391 non-null float64
    sg             388 non-null float64
    al             353 non-null float64
    su             354 non-null float64
    rbc            351 non-null float64
    pc             248 non-null object
    pcc            335 non-null object
    ba             396 non-null object
    bgr            396 non-null object
    bu             356 non-null float64
    sc             381 non-null float64
    sod            383 non-null float64
    pot            313 non-null float64
    hemo           312 non-null float64
    pcv            348 non-null float64
    wbcc           329 non-null float64
    rbcc           294 non-null float64
    htn            269 non-null float64
    dm             398 non-null object
    cad            398 non-null object
    appet          398 non-null object
    pe             399 non-null object
    ane            399 non-null object
    class          399 non-null object
    Unnamed: 25    400 non-null object
    dtypes: float64(15), object(11)
    memory usage: 81.3+ KB
    

.. code:: ipython3

    df.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>bp</th>
          <th>sg</th>
          <th>al</th>
          <th>su</th>
          <th>rbc</th>
          <th>bu</th>
          <th>sc</th>
          <th>sod</th>
          <th>pot</th>
          <th>hemo</th>
          <th>pcv</th>
          <th>wbcc</th>
          <th>rbcc</th>
          <th>htn</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>0.0</td>
          <td>391.000000</td>
          <td>388.000000</td>
          <td>353.000000</td>
          <td>354.000000</td>
          <td>351.000000</td>
          <td>356.000000</td>
          <td>381.000000</td>
          <td>383.000000</td>
          <td>313.000000</td>
          <td>312.000000</td>
          <td>348.000000</td>
          <td>329.000000</td>
          <td>294.000000</td>
          <td>269.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>NaN</td>
          <td>51.483376</td>
          <td>76.469072</td>
          <td>1.017408</td>
          <td>1.016949</td>
          <td>0.450142</td>
          <td>148.036517</td>
          <td>57.425722</td>
          <td>3.072454</td>
          <td>137.528754</td>
          <td>4.627244</td>
          <td>12.526437</td>
          <td>38.884498</td>
          <td>8406.122449</td>
          <td>4.707435</td>
        </tr>
        <tr>
          <th>std</th>
          <td>NaN</td>
          <td>17.169714</td>
          <td>13.683637</td>
          <td>0.005717</td>
          <td>1.352679</td>
          <td>1.099191</td>
          <td>79.281714</td>
          <td>50.503006</td>
          <td>5.741126</td>
          <td>10.408752</td>
          <td>3.193904</td>
          <td>2.912587</td>
          <td>8.990105</td>
          <td>2944.474190</td>
          <td>1.025323</td>
        </tr>
        <tr>
          <th>min</th>
          <td>NaN</td>
          <td>2.000000</td>
          <td>50.000000</td>
          <td>1.005000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>22.000000</td>
          <td>1.500000</td>
          <td>0.400000</td>
          <td>4.500000</td>
          <td>2.500000</td>
          <td>3.100000</td>
          <td>9.000000</td>
          <td>2200.000000</td>
          <td>2.100000</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>NaN</td>
          <td>42.000000</td>
          <td>70.000000</td>
          <td>1.010000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>99.000000</td>
          <td>27.000000</td>
          <td>0.900000</td>
          <td>135.000000</td>
          <td>3.800000</td>
          <td>10.300000</td>
          <td>32.000000</td>
          <td>6500.000000</td>
          <td>3.900000</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>NaN</td>
          <td>55.000000</td>
          <td>80.000000</td>
          <td>1.020000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>121.000000</td>
          <td>42.000000</td>
          <td>1.300000</td>
          <td>138.000000</td>
          <td>4.400000</td>
          <td>12.650000</td>
          <td>40.000000</td>
          <td>8000.000000</td>
          <td>4.800000</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>NaN</td>
          <td>64.500000</td>
          <td>80.000000</td>
          <td>1.020000</td>
          <td>2.000000</td>
          <td>0.000000</td>
          <td>163.000000</td>
          <td>66.000000</td>
          <td>2.800000</td>
          <td>142.000000</td>
          <td>4.900000</td>
          <td>15.000000</td>
          <td>45.000000</td>
          <td>9800.000000</td>
          <td>5.400000</td>
        </tr>
        <tr>
          <th>max</th>
          <td>NaN</td>
          <td>90.000000</td>
          <td>180.000000</td>
          <td>1.025000</td>
          <td>5.000000</td>
          <td>5.000000</td>
          <td>490.000000</td>
          <td>391.000000</td>
          <td>76.000000</td>
          <td>163.000000</td>
          <td>47.000000</td>
          <td>17.800000</td>
          <td>54.000000</td>
          <td>26400.000000</td>
          <td>8.000000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.isna().sum()




.. parsed-literal::

    age            400
    bp               9
    sg              12
    al              47
    su              46
    rbc             49
    pc             152
    pcc             65
    ba               4
    bgr              4
    bu              44
    sc              19
    sod             17
    pot             87
    hemo            88
    pcv             52
    wbcc            71
    rbcc           106
    htn            131
    dm               2
    cad              2
    appet            2
    pe               1
    ane              1
    class            1
    Unnamed: 25      0
    dtype: int64



