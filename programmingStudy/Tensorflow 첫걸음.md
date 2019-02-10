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

