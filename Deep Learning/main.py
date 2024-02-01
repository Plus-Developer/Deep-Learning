#함수 모음
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
