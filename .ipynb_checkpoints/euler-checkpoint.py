"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn
from scipy.integrate._ivp.ivp import OdeSolver








class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Forward Euler ODE solver compatible with scipy.integrate.solve_ivp.

    规则：
        y_{n+1} = y_n + h * f(t_n, y_n)

    使用：
        solve_ivp(f, (t0, t_bound), y0, method=ForwardEuler, h=0.01, dense_output=True)
    """

    def __init__(
            self,
            fun,
            t0,
            y0,
            t_bound,
            vectorized: bool,
            h:float | None=None,
            **extraneous
    ):
        # 对多余参数发警告
        if extraneous:
            warn("参数多了")
        
        # 调用父类 OdeSolver 的初始化
        super(ForwardEuler, self).__init__(fun=fun,
                         t0=t0,
                         y0=y0,
                         t_bound=t_bound,
                         vectorized=vectorized,
                        )
        
        # 题目说 direction 应该为 +1（要求 t_bound > t0）
        if not (t_bound > t0):
            raise ValueError("ForwardEuler expects t_bound > t0 so that direction = +1.")
        self.direction = 1.0  # 强制正向


         # 步长设置
        if h is None:
            total = float(self.t_bound - self.t)
            self.h = total / 100.0
        else:
            if h <= 0:
                raise ValueError("Step size h must be positive.")
            self.h = float(h)

        # 根据题意，欧拉法不需要雅可比矩阵或 LU 分解
        self.njev = 0
        self.nlu = 0

    def _step_impl(self):
        """
        执行一步向前欧拉法：
            y_{n+1} = y_n + dt* f(t_n, y_n)，
            其中 dt = min(h, t_bound - t_n)，确保不跨越终点。
        """
        t = self.t
        y = self.y

        # 离终点还有多远
        distbound = self.t_bound - t
        if distbound <= 0.0:
            return True, None

        # 取当前步长（不能超过终点）
        dt = min(self.h, distbound) # direction 已被固定为 +1

        # 计算导数并更新
        f = self.fun(t, y)
        # 欧拉一步
        y_new = y + dt * f
        t_new = t + dt

        # 记录这一步的“起点信息”，供稠密输出使用
        self._last_t_old = t
        self._last_y_old = y.copy()
        self._last_f_old = f.copy()

        # 更新状态
        self.t = t_new
        self.y = y_new

        return True, None
    
    def _dense_output_impl(self):
        """
        返回最近一步 [t_old, t] 的稠密输出对象（线性插值）。
        """
        return ForwardEulerOutput(
            t_old=self._last_t_old,
            t=self.t,
            y_old=self._last_y_old,
            f_old=self._last_f_old,
        )

class ForwardEulerOutput(DenseOutput):
    """
    Forward Euler 的稠密输出类。
    实现线性插值：
        y(tau) = y_old + (tau - t_old) * f_old
    """
    def __init__(
            self, 
            t_old: float, 
            t: float, 
            y_old: np.ndarray, 
            f_old: np.ndarray
            ):
        super(ForwardEuler, self).__init__(t_old, t)
        # 存储用于插值的参数
        self._y0 = np.asarray(y_old)
        self._f0 = np.asarray(f_old)
    
    def _call_impl(self, t_eval):
        """
        实现线性插值：
            y(t_eval) = y0 + (t_eval - t_old) * f0
        支持标量或数组输入。
        """
        dt = np.atleast_1d(t_eval) - self.t_old           # 计算 (t_eval - t_old)
        y = self._y0[:, None] + self._f0[:, None] * dt[None, :]  # y0 + f0 * Δt
        return y[:, 0] if np.isscalar(t_eval) else y
