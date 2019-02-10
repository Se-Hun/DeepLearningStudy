# Pandas 간단 공부

참고문헌 : 
1. colab-tutorial의 Pandas 간단 소개[Google의 colab](https://colab.research.google.com)

colab-tutorial을 따라 공부하였으므로 자세한 코드 실행결과나 더 구체적인 사항은 **colab-tutorial의 pandas 간단 소개** 부분을 참고하도록 하자.

## Pandas란?

열 중심 데이터 분석 라이브러리이다. 입력 데이터를 처리하고 분석하는 데 효과적인 도구이다.

## Pandas의 기본 개념

    import pandas as pd
    pd.__version__

위의 코드를 입력하면 Pandas 라이브러리를 가져와서 라이브러리 버전을 출력할 수 있다.

본격적으로 들어가보자!

Pandas의 기본 데이터 구조는 두 가지 클래스로 구현된다.

1. **DataFrame** : 행 및 이름 지정된 열이 포함된 관계형 데이터 테이블이라고 생각할 수 있다. 즉, 엑셀 같은 표 구조를 만들어준다.
2. **Series** : 하나의 열이다. DataFrame에는 하나 이상의 Series와 각 Series의 이름이 포함된다.

## DataFrame 객체와 Series 객체 만들기

다음과 같이 Series 객체를 만들 수 있다.

    pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

실행결과는 다음과 같다.

![pandas_1](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_1.PNG)

다음과 같은 코드를 입력하면 DataFrame 객체를 생성할 수 있다.

    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    population = pd.Series([852469, 1015785, 485199])
    
    pd.DataFrame({ 'City name': city_names, 'Population': population})
    # City name은 DataFrame의 열 이름이 되고 그 열에 city_names라는 Series를 전달해준다.

실행결과는 다음과 같다.

![pandas_2](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_2.PNG)

이처럼 **DataFrame 객체는 DataFrame에서의 열 이름(key 값이 됨)에 Series(values 값이 됨)를 전달해줌**으로써 만들 수 있다.

이 때, Series의 길이가 일치하지 않는 경우, 누락된 값은 특수 NA/NaN 값으로 채워지게 된다.

## CSV 파일을 DataFrame으로 로드하기

    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
    california_housing_dataframe.describe()

위의 코드로 CSV 파일을 로드할 수 있다.

결과는 다음과 같다.

![pandas_3](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_3.PNG)

`DataFrame.describe()` 함수를 이용하면 DataSet에 대한 통계를 보여준다. 즉, 위의 결과는 데이터셋들이 아닌 통계값들인 것이다.

`DataFrame.head()` 함수는 데이터셋의 앞 일부분을 보여준다.

    california_housing_dataframe.head()

결과는 다음과 같다.

![pandas_4](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_4.PNG)

pandas는 또한 다음과 같이 그래프를 그릴 수 있는데, 한 예로 `DataFrame.hist()` 함수를 이용하면 히스토그램을 그릴 수 있다.

    california_housing_dataframe.hist('housing_median_age')

결과는 다음과 같다.

![pandas_5](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_5.PNG)

## DataFrame에서 특정 데이터에만 접근해보기

우선, 다음과 같은 코드를 입력하여 제일 처음 만들었던 DataFrame에 이름을 부여하자.

    cities = pd.DataFrame({ 'City name': city_names, 'Population': population})
    cities['City name]

다음과 같이 cities라는 DataFrame이 생성된 것을 볼 수 있다.

![pandas_6](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_6.PNG)

다음으로 다음과 같은 코드를 입력하여 City name이라는 Series의 1번 데이터에 접근해보자.

    cities['City name'][1]

결과는 다음과 같다.

![pandas_7](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_7.PNG)

마지막으로 다음과 같은 코드를 입력하여 0번부터 1번 데이터에 접근해보도록 하자.

    cities[0:2]

![pandas_8](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_8.PNG)

## 데이터 조작

다음과 같이 Python의 기본 산술 연산을 Series에 적용할 수 있다.

    population / 1000

결과는 다음과 같다.

![pandas_9](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_9.PNG)

또한, 다음과 같이 Numpy의 라이브러리 함수들을 적용시킬 수 있다.

    import numpy as np
    np.log(population)

결과는 다음과 같다.

![pandas_10](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_10.PNG)

세번째로 다음과 같이 `Series.apply()` 함수를 이용하여 JavaScript의 map함수 같은 기능을 구현할 수 있다. 인수로는 lambda 함수를 사용해야만 한다.

    population.apply(lambda val: val > 10000000)

결과는 다음과 같다.

![pandas_11](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_11.PNG)

마지막으로 DataFrame을 수정할 수도 있다. 다음 코드를 입력하여 두 개의 Series를 추가해보자.

    cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
    cities['Population density'] = cities['Population'] / cities['Area square miles']
    cities

결과는 다음과 같다.

![pandas_12](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/pandas_12.PNG)

