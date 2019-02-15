# Tensorflow 기본 문법

참고문헌 : 
1. colab-tutorial의 Tensorflow, Helloworld // Tensorflow_programming_concepts

[Google의 colab](https://colab.research.google.com)


colab-tutorial을 따라 공부하였으므로 자세한 코드 실행결과나 더 구체적인 사항은 **colab-tutorial의 Tensorflow, Helloworld** 부분을 참고하도록 하자.

## HelloWorld

    from __future__ import print_function
    import tensorflow as tf
    c = tf.constant('Hello, World!')
    with tf.Session() as sess:
        print(sess.run(c))

결과는 다음과 같다.

![기본-1](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-1.PNG)

## Tensor의 개념

Tensorflow는 Tensor를 만들고 없애고 조작하는 프로그래밍 모델이다. 즉, **Tensorflow에서 대부분의 코드 행은 Tensor의 연산이다.**

****

**스칼라 : **0-d배열(0차원 텐서이다.) ex) 5

**벡터 : **1-d배열(1차원 텐서이다.) ex) [2, 3, 5, 7, 11]

**행렬 : **2-d배열(2차원 텐서이다.) ex) [[3.1, 8.2, 4.9]  [4.3, -2.7, 6.5]]

****

## 그래프

Tensorflow는 그래프 데이터 구조인데, 그래프의 **노드는 연산이고 엣지는 텐서이다.**

텐서는 그래프를 따라 흐르고, 각 노드에서 연산에 의해 조작된다.

한 연산의 출력 텐서는 보통 다음 연산의 입력 텐서가 된다.

## 상수와 변수

상수를 정의하려면 `tf.constant` 연산자를 이용하여 다음과 같이 그 값을 전달하면 된다.

    x = tf.constant([5.2])

유사하게 다음과 같이 변수를 만들 수 있다.

    y = tf.Variable([5])
    
    # 변수를 먼저 만든 다음에 값을 할당할 수도 있다.
    y = tf.Variable([0])
    y = y.assign([5])

텐서는 이와같이 그래프에서 상수 또는 변수로 저장될 수 있는데, 상수는 값이 변하지 않는 텐서를 가지고 변수는 값이 변할 수 있는 텐서를 가진다.

이 때, 주의할 점은 **상수와 변수가 그래프에서 또 다른 연산이라는 것이다. 즉, 상수는 항상 같은 텐서 값을 반환하는 연산이고, 변수는 할당된 텐서를 반환하는 연산이다.**

또한, `tf.add`와 같은 연산과 정의되어진 상수 또는 변수를 병합할 수 있다. `tf.add`연산을 평가할 때 `tf.constant` 또는 `tf.Variable` 연산을 호출하여 값을 얻은 다음 그 값의 합으로 새 텐서를 반환한다.

## 세션

그래프는 반드시 텐서플로우 세션 내에서 실행되어야만 한다. 세션은 다음과 같은 그래프의 상태를 가진다.

    with tf.Session() as sess:
      initialization = tf.global_variables_initializer()
      print(y.eval())

`tf.Variable`을 사용할 때 위에서와 같이 세션 시작 시에 `tf.global_variables_initializer`를 호출하여 명시적으로 초기화해주어야 한다.

****

**텐서플로우 프로그래밍은 기본적으로 두 단계 과정을 거친다.**

1. 상수, 변수, 연산을 그래프로 결합한다.
2. 이 상수, 변수, 연산을 세션 내에서 평가한다.

****

## 간단한 텐서플로우 프로그램 만들기

### Import 시키기

    import tensorflow as tf

위의 명령문을 통해 tensorflow를 import 시킨다.

다른 일반적인 import 명령문은 다음과 같은 것들이 있다.

    import matplotlib.pyplot as plt  # 데이터셋 시각화
    import numpy as np               # 저수준 숫자 Python 라이브러리
    import pandas as pd              # 고수준 숫자 Python 라이브러리

다음과 같이 코드를 작성하여 보자.

    from __future__ import print_function
    
    import tensorflow as tf
    
    g = tf.Graph()
    
    with g.as_default():
      x = tf.constant(8, name="x_const")
      y = tf.constant(5, name="y_const")
      sum = tf.add(x, y, name="x_y_sum")
      
      withd tf.Session() as sess:
        print(sum.eval())

결과는 다음과 같다.

![기본-2](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-2.PNG)

## 벡터 덧셈

다음과 같이 일반적인 수학 연산을 할 수 있다.

    with tf.Graph().as_default():
      primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
      
      ones = tf.ones([6], dtype=tf.int32)  # 각 element들이 1인 벡터(모든 원소가 1인 행렬)
      
      just_beyond_primes = tf.add(primes, ones)
      
      with tf.Session() as sess:
        print(just_beyond_primes.eval())

결과는 다음과 같다.

![기본-3](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-3.PNG)

## 텐서의 Shape

Shape는 텐서의 크기와 차원 수를 결정한다.

다음과 같은 코드를 통해 Shape의 개념을 살펴보자.

    with tf.Graph().as_default():
      scalar = tf.zeros([])
      
      vector = tf.zeros([3])
      
      matrix = tf.zeros([2, 3])
      
      with tf.Session() as sess:
        print('scalar has shape', sclar.get_shape(), 'and Value:\n', scalar.eval())
        print('vector has shape', vector.get_shape(), 'and Value:\n', vector.eval())
        print('matrix has shape', matrix.get_shape(), 'and Value:\n', matrix.eval())

결과는 다음과 같다.

![기본-4](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-4.PNG)

## 브로드캐스팅

수학에서는 같은 형태의 텐서만 연산을 할 수 있지만 텐서플로우에서는 다른 형태의 텐서간에도 연산이 가능하게 한다.

브로드캐스팅은 **요소간 연산에서 더 작은 배열을 확장하여 더 큰 배열과 같은 형태를 가지게 하는 기능**이다.

이 때, 텐서가 브로드캐스팅되면 텐서의 항목은 복사된다.

다음과 같은 코드를 통해 알아보자.

    with tf.Graph().as_default():
      primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
      
      ones = tf.constant(1, dtype=tf.int32)
      
      just_beyond_primes = tf.add(primes, ones)
      
      with tf.Session() as sess:
        print(just_beyond_primes.eval())

결과는 다음과 같다.

![기본-5](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-5.PNG)

## 행렬 곱셈

선형대수학처럼 텐서플로우에서도 두 개의 행렬을 곱할 때는 첫 번째 행렬의 column 수와 두 번째 행렬의 row 수가 같아야 한다.

다음과 같은 코드를 통해 알아보자.

    with tf.Graph().as_default():
      x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)
      y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)
      
      matrix_multiply_result = tf.matmul(x, y)
      
      with tf.Session() as sess:
        print(matrix_multiply_result.eval())

결과는 다음과 같다.

![기본-6](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-6.PNG)

## 텐서 Shape 변경하기

`tf.reshape` 메소드를 이용하면 텐서의 Shape를 변경할 수 있다.

예를 들어, 8x2 형태의 텐서를 2x8 텐서나 4x4 형태로 변경할 수 있다.

다음과 같은 코드를 통해 알아보자.

    with tf.Graph().as_default():
      matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)
      reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
      reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])
      
      with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped matrix (2x8):")
        print(reshaped_2x8_matrix.eval())
        print("Reshaped matrix (4x4):")
        print(reshaped_4x4_matrix.eval())

결과는 다음과 같다.

![기본-7](https://github.com/Se-Hun/DeepLearningStudy/tree/master/programmingStudy/png)

또한, 다음과 같은 코드처럼 차원 수도 변경할 수 있다.

    with tf.Graph().as_default():
      matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)
      
      reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])
      one_dimensional_vector = tf.reshape(matrix, [16])
      
      with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped 3-D tensor (2x2x4):")
        print(reshaped_2x2x4_tensor.eval())
        print("1-D vector:")
        print(one_dimensional_vector.eval())

결과는 다음과 같다.

![기본-8](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-8.PNG)

## 변수 초기화, 할당

텐서플로우에서는 변수 초기화가 자동으로 실행되지 않는다.

다음과 같은 코드에서는 오류가 발생한다.

    with g.as_default():
      with tf.Session() as sess:
        try:
          v.eval()
        except tf.errors.FailedPreconditionError as e:
          print("Caught expected error: ", e)

다음과 같은 오류가 발생한것을 알 수 있다.

![기본-9](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-9.PNG)

변수를 초기화하는 가장 쉬운 방법은 `tf.global_variables_initializer`를 호출하는 것이다.

다음과 같은 코드를 실행시켜 보자.

    with g.as_default():
      with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        sess.run(initialization)
        print(v.eval())
        print(w.eval())

다음과 같은 결과를 얻을 수 있다.

![기본-10](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-10.PNG)

초기화된 변수는 같은 세션 내에서는 값을 유지하지만 새 세션을 시작하면 다시 초기화해주어야 한다.

마지막으로, 변수 값을 할당하려면 할당 작업을 사용해야 한다. 할당 작업을 만들기만 하면 되는 것이 아니라, 초기화와 마찬가지로 session에서 `run()`시켜주어야만 한다.

다음과 같은 코드를 실행시켜 보자.

    with g.as_default():
      with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print(v.eval())
      
      assignment = tf.assign(v, [7])
      print(v.eval())
      sess.run(assignment)
      print(v.eval())

다음과 같은 결과를 얻을 수 있다.

![기본-11](https://github.com/Se-Hun/DeepLearningStudy/blob/master/programmingStudy/png/%EA%B8%B0%EB%B3%B8-11.PNG)

