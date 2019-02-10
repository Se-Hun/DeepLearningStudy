# Tensorflow 첫걸음

참고문헌 : 
1. colab-tutorial의 텐서플로우 첫걸음[Google의 colab](https://colab.research.google.com)

colab-tutorial을 따라 공부하였으므로 자세한 코드 실행결과나 더 구체적인 사항은 **colab-tutorial의 pandas 간단 소개** 부분을 참고하도록 하자.

## 목표

Tensorflow를 이용하여 LinearRegressor 모델을 통한 입력 특성 하나를 기반으로 지역별 주택 가격 중앙값을 예측해보도록 하자.

## 시작

우선, 다음과 같이 필요한 라이브러리를 로드하자.

    from __future__ import print_function

    import math

    from IPython import display
    from matplotlib import cm
    from matplotlib import gridspec
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    import tensorflow as tf
    from tensorflow.python.data import Dataset

    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

다음으로 다음과 1990년 캘리포니아 인구조사 자료 데이터셋을 로드하자.

    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

마지막으로 경사하강법(비용을 최소화하는 W와 b를 찾는 알고리즘)의 성능에 악영향을 줄 수 있는 의도치 않은 정룔 효과를 방지하고자 데이터를 무작위로 추출한다.

또한, 일반적으로 사용하는 학습률 범위에서 보다 쉽게 학습할 수 있도록 median_house_value를 천 단위로 조정한다.

    california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
    california_housing_dataframe["median_house_value"] /= 1000.0
    california_housing_dataframe

결과는 다음과 같다.

![Tesnsorflow_step1_1](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/Tesnsorflow_step1_1.PNG)

## 데이터 조사

데이터를 본격적으로 다루기 전에 잠시 살펴보는 것이 좋다.

따라서, 각 열(feature)에 대해 개수, 평균, 표준편차, 최대값, 최소값, 다양한 분위 등 몇 가지 유용한 통계를 `DataFrame.describe()`함수를 이용하여 출력해보자.

    california_housing_dataframe.describe()

결과는 다음과 같다.

![Tesnsorflow_step1_2](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/Tesnsorflow_step1_2.PNG)

## 본격적인 모델 만들기

입력 : 입력 특성은 total_rooms를 사용한다.

출력 : 이번 실습에서 예측하고자 하는 것은 지역별 주택의 중앙값이다. 즉, median_house_value 특성이 레이블에 해당하게 된다.

알고리즘 : Tensorflow가 제공하는 LinearRegressor 인터페이스를 사용한다.

### 1단계 : 특성 정의 및 특성 열 구성

텐서플로우에서 특성의 데이터 유형(숫자형 데이터인가? 범주형 데이터인가?)을 지정하려면 **특성 열**이라는 구조체를 사용한다.

특성 열은 특성 데이터에 대한 설명만 저장하여 특성 데이터 자체는 포함하지 않는다.

우선, total_rooms라는 숫자형 데이터 하나만을 사용하겠다.

다음 코드에서는 california_housing_dataframe에서 total_rooms 데이터를 추출하고 numeric_column으로 특성 열을 정의하여 데이터가 숫자임을 지정한다.

    # Define the input feature: total_rooms.
    my_feature = california_housing_dataframe[["total_rooms"]]

    # Configure a numeric feature column for total_rooms.
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]

### 2단계 : label(target) 정의

우리의 목표는 median_house_value 값을 예측 하는 것이므로 label은 median_house_value이다. 다음과 같이 label을 median_house_value로 정의한다.

    # Define the label.
    targets = california_housing_dataframe["median_house_value"]

### 3단계 : LinearRegressor 구성

우리는 LinearRegressor를 사용하여 선형 회귀 모델을 구성할 것이다.

또한, **GradientDescentOptimizer**를 사용하여 경사하강법을 구현하고 이를 통해 모델을 학습시킬 것이다.

이 때, **learning_rate**는 경사 단계의 크기를 조절하는 것으로 얼마만큼 이동해나갈지 이 매개변수를 통하여 조정할 수 있다.

    # Use gradient descent as the optimizer for training the model.
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # clip_gradients_by_norm() 함수를 이용하면 경사 제한을 적용시킬 수 있다.
    # 경사 제한은 학습 중에 경사가 너무 커져서 경사하강법이 실패하는 경우가 나타나지 않도록 제한하는 것을 의미한다.

    # Configure the linear regression model with our feature columns and optimizer.
    # Set a learning rate of 0.0000001 for Gradient Descent.
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

### 4단계 : 입력 함수 정의

이제 모델 구성은 끝났기에 본격적으로 학습을 시키는 단계이다. 즉, 입력 함수를 정의하는 단계라고 할 수 있다.

우선, pandas 특성 데이터를 Numpy 배열의 dict로 변환한다.

다음으로 TensorFlow가 제공하는 Dataset 라이브러리를 활용하여 이 데이터로부터 데이터 세트 개체를 생성하고 **batch_size 크기의 배치로 나누어 지정한 세대 수(num_epochs)만큼 반복**한다.(**batch** : 데이터를 나누는 수, **epoch** : 모든 batch들을 한 번 돌리면 epoch 한 번)

위의 과정이 필요한 이유는 메모리 크기의 한계때문에 빅데이터를 한번에 학습시킬 수 없기 때문이다.

다음으로, shuffle을 True로 설정하면 학습 중에 데이터가 모델에 무작위로 전달되도록 데이터가 뒤섞인다. 이 때, buffer_size 인수는 shuffle에서 무작위로 추출할 데이터셋의 크기를 지정한다.

마지막으로 입력 함수에서 데이터 세트에 대한 반복자(Iterator)를 만들고 다음 데이터 배치를 LinearRegressor에 반환한다.

이 모든 과정의 코드는 다음과 같다.

    def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

### 5단계 : 모델 학습

linear_regressor의 `train()`함수를 호출하여 모델을 학습시킬 수 있다.

이 때, my_feature 및 target을 인수로 전달할 수 있도록 my_input_fn을 lambda에 매핑한다.

코드는 다음과 같다.

    _ = linear_regressor.train(
        input_fn = lambda:my_input_fn(my_feature, targets),
        steps=100
    )

### 6단계 : 모델 평가

이 부분은 훈련 데이터셋과 테스트 데이터셋으로 나누어 진행해야 하므로 다음번에 공부해보도록 하자.