#행렬, 벡터 모듈 'numpy'
from numpy import *

#타입 어노테이션 (자료형 명시) 모듈 'typing'
from typing import Callable, List

'''
여러 변수를 인자로 받는 함수의 편미분
--------------------------------------------
사용법 : deriv(함수, 변수)
--------------------------------------------
'''
def deriv(func : Callable[[ndarray], ndarray],
          input_ : ndarray,
          delta : float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

#합성함수 사용
Array_Function = Callable[[ndarray], ndarray]

Chain = List[Array_Function]

'''
손실함수(평균제곱오차)
--------------------------------------------
사용법 : MSE(예측값, 실제값)
--------------------------------------------
'''
def MSE(p : ndarray, y : ndarray):
    assert p.shape == y.shape, \
           '''Uncorrect shape between p value and y value'''

    return mean(power(y - p, 2))

#행렬의 모양의 일치 여부 확인
def assert_same_shape(a1 : ndarray, a2 : ndarray):
    assert a1.shape == a2.shape, \
           '''두 행렬의 모양이 일치하지 않습니다.'''

def Multiply_Weight(P_Operation)
