#모듈 불러오기
from modules import *

#연산 수행 클래스
class Operation(object):
    def __init__(self):
        pass

    def forward(self, input_ : ndarray):
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad : ndarray) -> ndarray:
        self.input_grad()

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad : ndarray) -> ndarray:
        raise NotImplementedError()

#인자를 받는 연산 수행 클래스
class P_Operation(Operation):
    def __init__(self, parameter : ndarray) -> ndarray:
        super().__init__()
        self.parameter = parameter

    def backward(self, output_grad : ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.parameter_grad = self._p_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.parameter, self.p_grad)

        return self.input_grad

    def _param_grad(self, output_grad : ndarray) -> ndarray:
        raise NotImplementedError()

#가중치 행렬곱 연산 수행 클래스
class W_Multiply(P_Operation):
    def __init__(self, W : ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        return dot(self.input_, self.parameter)

    def _input_grad(self, output_grad : ndarray) -> ndarray:
        return dot(output_grad, transpose(self.parameter, (1, 0)))

    def _p_grad(self, output_grad : ndarray):
        return dot(transpose(self.input_, (1, 0)), output_grad)
#편향 연산 수행 클래스
class B_Add(P_Operation):
    def __init__(self, B : ndarray):
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        return self.input_ + self.parameter

    def _input_grad(self, output_grad : ndarray) -> ndarray:
        return ones_like(self.input_) * output_grad

    def _p_grad(self, output_grad : ndarray) -> ndarray:
        p_grad = ones_like(self.parameter) * output_grad
        return sum(p_grad, axis=0).reshape(1, p_grad.shape[1])

#활성화 함수 Sigmoid 클래스
class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return 1/(1 + exp(-1 * self.input_))

    def _input_grad(self, output_grad : ndarray) -> ndarray:
        sigmoid_backward = self.output * (1 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
