"""四问共用：径向环形有限体积 + 热湿耦合 BDF 求解器。

模型依据：建模思路-2(1).pdf，尤其式(13)(15)(35)(44)-(47)。
所有长度/时间/压力/能量用 m/s/Pa/J。T 为摄氏度，只有 Arrhenius 用 K。
C = kg_water/kg_dry_solid，Y = kg_water/kg_dry_air；严禁直接相减。
几何中心和真实外表面均设节点，配对偶环形控制体；边界节点不是幽灵点。
Numba 仅加速纯数值内核，可选，不影响模型。无跨调用表面缓存，不裁剪已接受解。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return (lambda f: f) if not args or not callable(args[0]) else args[0]

R0_M = 0.02
L0_M = 0.25
C0 = 2.55
T0_C = 28.0
P_PA = 101325.0
HM_M_S = 8e-7
H_W_M2K = 25.0
BETA0 = C0 * (1.0/0.98 - 1.0)
PSAT0_PA = 610.8 * np.exp(17.27*T0_C/(T0_C+237.3))
# 下面这个参考映射是PDF式(13)明确采用的补充假设。
# 它不是 hm 的唯一换算，也不是物理意义上的空气与材料含水率之差。
KP0 = (820.0/(1.0+C0))*HM_M_S*(C0-0.01963)/PSAT0_PA
GX, GW = np.polynomial.legendre.leggauss(16)
GX, GW = (GX+1.0)/2.0, GW/2.0

@dataclass(frozen=True)
class Config:
    case: int = 2                  # 1=附录2；2=附录3（问题2和问题3）；4=附录4
    n: int = 160                   # 径向间隔数；节点数为 n+1
    grid_power: float = 1.6         # 近表面加密的材料坐标网格
    shrink: bool = False
    beta_factor: float = 1.0
    kp_factor: float = 1.0
    h_factor: float = 1.0
    D_factor: float = 1.0
    k_factor: float = 1.0
    face_method: str = 'integral'   # 取值为 'integral' 时使用16点积分；取值为 'harmonic' 时使用调和平均对照
    sensible: bool = True          # 取值为 false 时表示命名对照模型，不是基准模型
    air_tail: str = 'mean'          # 末61条记录的精确均值；取值为 'last' 时使用末值对照
    rtol: float = 2e-8
    atol_C: float = 2e-10
    atol_T: float = 2e-8
    max_step_early_s: float = 15.0
    max_step_late_s: float = 180.0
    horizon_h: float = 480.0        # 安全计算上限，不是声称的烘干时长
    target_C: float = 0.15
    target_margin: float = 1e-7
    stop_at_target: bool = True
    split_input_knots: bool = True

    def validate(self):
        if self.case not in (1, 2, 4) or self.n < 8:
            raise ValueError('case must be 1,2,4 and n must be >=8.')
        if self.shrink and self.case != 4:
            raise ValueError('Shrinkage is only enabled for appendix 4.')
        if not 1 <= self.grid_power <= 2.5:
            raise ValueError('Require 1 <= grid_power <= 2.5.')
        if self.beta_factor <= 0 or min(self.kp_factor, self.h_factor) < 0:
            raise ValueError('beta>0 and transfer factors >=0 are required.')
        if min(self.D_factor, self.k_factor, self.rtol, self.atol_C, self.atol_T,
               self.max_step_early_s, self.max_step_late_s, self.horizon_h) <= 0:
            raise ValueError('Material factors and integration settings must be positive.')
        if not 0 <= self.target_margin < self.target_C < C0:
            raise ValueError('Invalid target or safety margin.')
        if self.face_method not in ('integral', 'harmonic'):
            raise ValueError('Unknown face-method.')
        if self.air_tail not in ('mean', 'last'):
            raise ValueError('air_tail must be mean or last.')

@dataclass
class Inputs:
    air: np.ndarray                 # [s, 摄氏度, kg水/kg干空气]
    radius: np.ndarray              # [s, cm]，与提供的记录一致
    source: str = ''

    @classmethod
    def load(cls, directory: str | Path) -> 'Inputs':
        d = Path(directory)
        try:
            a = np.loadtxt(d/'air.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
            r = np.loadtxt(d/'radius.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
        except OSError as exc:
            raise FileNotFoundError(f'Missing data/air.csv or data/radius.csv in {d}') from exc
        obj = cls(a, r, str(d.resolve()))
        obj.validate()
        return obj

    def validate(self):
        for a, ncol, name in ((self.air, 3, 'air'), (self.radius, 2, 'radius')):
            if a.ndim != 2 or a.shape[1] != ncol or len(a) < 2:
                raise ValueError(f'{name}: wrong column count or insufficient records.')
            if not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0) or a[0, 0] != 0:
                raise ValueError(f'{name}: finite values, strictly increasing seconds from 0 required.')
        if np.any(self.air[:, 2] < 0) or np.any(self.air[:, 1] <= -100):
            raise ValueError('Invalid air temperature or humidity.')
        if np.any(self.radius[:, 1] <= 0) or abs(self.radius[0, 1]-2.0) > 1e-10:
            raise ValueError('radius.csv must be in cm and start at 2 cm, not 0.02.')
        if np.any(np.diff(self.radius[:, 1]) > 1e-10):
            raise ValueError('Expected nonincreasing measured radius.')
        if self.air[-1, 0] != 14400 or self.radius[-1, 0] != 259200:
            raise ValueError('Expected input end times of 14400 s and 259200 s.')

    def tail(self, mode='mean'):
        if mode == 'last':
            return self.air[-1, 1:].copy()
        return self.air[self.air[:, 0] >= self.air[-1, 0]-3600, 1:].mean(axis=0)

    def audit(self):
        ta, ya = self.tail()
        phi = P_PA*ya/(0.62198+ya)/float(psat_pa(ta))
        return dict(air_rows=len(self.air), radius_rows=len(self.radius),
                    air_start_s=float(self.air[0, 0]), air_end_s=float(self.air[-1, 0]),
                    air_every_60s=bool(np.all(np.diff(self.air[:, 0]) == 60)),
                    radius_end_s=float(self.radius[-1, 0]),
                    radius_every_1800s=bool(np.all(np.diff(self.radius[:, 0]) == 1800)),
                    radius_initial_cm=float(self.radius[0, 1]), radius_final_cm=float(self.radius[-1, 1]),
                    last_hour_T_C=float(ta), last_hour_Y_kg_kg=float(ya),
                    beta0_kg_kg=BETA0, kp0_kg_m2_s_Pa=KP0,
                    equilibrium_RH=float(phi), equilibrium_C_kg_kg=float(BETA0*phi/(1-phi)))

@njit(cache=True)
def psat_pa(T):
    return 610.8*np.exp(17.27*T/(T+237.3))

@njit(cache=True)
def latent_heat_J_kg(T):
    return 2.501e6-2361.0*T

@njit(cache=True)
def film_flux(Cs, Ts, Ya, beta, kp):
    # 有符号，单位kg/(m2·s)。允许凝结，热边界使用同一符号。
    ce = max(Cs, 0.0)              # 仅用于保护本构关系试算时的取值
    aw = ce/(ce+beta)
    pva = P_PA*Ya/(0.62198+Ya)
    return kp*(aw*psat_pa(Ts)-pva)

@njit(cache=True)
def properties(C, T, case):
    c = max(C, 1e-12)              # 不覆盖已经接受的 C 状态
    if case == 1:
        return 820., 2600., 0.36, 7e-9*np.exp(-0.89/c)
    if case == 2:
        return (650.+128.*c, (1450.+4186.*c)/(1.+c),
                0.21+0.38*c/(1.+c), 2.4e-3*np.exp(-0.45/c-3850./(T+273.15)))
    return (760.+90.*c, (1850.+4000.*c)/(1.+c),
            0.12+0.20*c/(1.+c), 4.2e-4*np.exp(-0.30/c-3850./(T+273.15)))

def material_constants(case):
    if case == 1: return 820./3.55, 2600., 2600.
    if case == 2: return (650.+128.*C0)/3.55, 1450., 4186.
    if case == 4: return (760.+90.*C0)/3.55, 1850., 4000.
    raise ValueError('Unknown case.')

def parameters(cfg, inp):
    ta, ya = inp.tail(cfg.air_tail)
    rd, cd, cw = material_constants(cfg.case)
    # 此处固定参数顺序，使编译后的数值内核不依赖Python对象。
    return np.array([cfg.case, cfg.shrink, BETA0*cfg.beta_factor, KP0*cfg.kp_factor,
                     H_W_M2K*cfg.h_factor, cfg.D_factor, cfg.k_factor,
                     cfg.sensible, cfg.face_method == 'harmonic', ta, ya, rd, cd, cw], dtype=float)

@njit(cache=True)
def environment(t, p, air, radius, tail_side):
    if tail_side:
        ta, ya = p[9], p[10]
    else:
        ta = np.interp(t, air[:, 0], air[:, 1])
        ya = np.interp(t, air[:, 0], air[:, 2])
    R = R0_M
    if p[1] > 0.5:
        # 72 h测量结束后半径保持末值，这是PDF明确给出的延拓情景。
        R = 0.01*np.interp(t, radius[:, 0], radius[:, 1])
    return ta, ya, R

@njit(cache=True)
def rhs_core(t, y, x, xf, weights, p, air, radius, tail_side):
    """式(45)：共享界面通量。辅助变量为 Mout、Qconv、Qlatent、Qsensible。"""
    m = len(x)
    dy = np.zeros_like(y)
    ta, ya, R = environment(t, p, air, radius, tail_side)
    J = (R/R0_M)**2
    rho_d = p[11]/J
    cw = p[13] if p[7] > 0.5 else 0.0
    V = np.pi*L0_M*R**2*weights
    md = np.pi*L0_M*R0_M**2*p[11]*weights
    H = np.empty(m); k = np.empty(m); D = np.empty(m)
    case = int(p[0])
    for i in range(m):
        rho, cp, ki, Di = properties(y[2*i], y[2*i+1], case)
        H[i] = rho*cp            # 采用附录中的 rho 乘 cp，而不是 rho_d*(cd+cw*C)
        k[i], D[i] = p[6]*ki, p[5]*Di
    fm = np.zeros(m+1); fe = np.zeros(m+1)
    for f in range(1, m):
        a, b = f-1, f
        ca, cb = y[2*a], y[2*b]
        ta_i, tb_i = y[2*a+1], y[2*b+1]
        dr = R*(x[b]-x[a])
        area = 2.0*np.pi*L0_M*R*xf[f]
        if p[8] < 0.5:
            # D_f = integral_0^1 D(Ca+s*(Cb-Ca), (Ta+Tb)/2) ds。
            # 这样可以处理低含水率干层的强非线性，而无需额外设置通量上限。
            df = 0.0
            for q in range(len(GX)):
                cq = ca+(cb-ca)*GX[q]
                df += GW[q]*properties(cq, 0.5*(ta_i+tb_i), case)[3]
            df *= p[5]
        else:
            df = 2.*D[a]*D[b]/max(D[a]+D[b], 1e-300)
        kf = 2.*k[a]*k[b]/(k[a]+k[b])
        fm[f] = -area*rho_d*df*(cb-ca)/dr           # 单位：kg/s
        fe[f] = -area*kf*(tb_i-ta_i)/dr            # 单位：W
        fe[f] += cw*0.5*(ta_i+tb_i)*fm[f]          # 水分显热项；使用同一个 fm
    Cs, Ts = y[2*m-2], y[2*m-1]
    je = film_flux(Cs, Ts, ya, p[2], p[3])         # 单位kg/(m2·s)，不再额外乘以 rho_d
    As = 2.0*np.pi*L0_M*R
    conv = As*p[4]*(ta-Ts)
    lat = As*latent_heat_J_kg(Ts)*je
    sens = As*cw*Ts*je
    fm[m] = As*je
    fe[m] = lat+sens-conv
    for i in range(m):
        dc = (fm[i]-fm[i+1])/md[i]
        dy[2*i] = dc
        dy[2*i+1] = (fe[i]-fe[i+1]-cw*y[2*i+1]*md[i]*dc)/(V[i]*H[i])
    dy[2*m], dy[2*m+1], dy[2*m+2], dy[2*m+3] = fm[m], conv, lat, sens
    return dy


def colored_jacobian(fun, m):
    """6色有限差分局部雅可比；诊断变量对应的列严格为零。"""
    rows, cols = [], []
    for j in range(2*m):
        i = j//2
        rr = list(range(2*max(0, i-1), 2*min(m, i+2)))
        if i == m-1: rr += list(range(2*m, 2*m+4))
        rows.extend(rr); cols.extend([j]*len(rr))
    rows, cols = np.array(rows), np.array(cols)
    def jac(t, y):
        f = fun(t, y)
        h = np.sqrt(np.finfo(float).eps)*np.maximum(np.abs(y[:2*m]), 1.)
        delta = np.empty((6, len(y)))
        for color in range(6):
            yp = y.copy(); js = np.arange(color, 2*m, 6)
            yp[js] += h[js]
            delta[color] = fun(t, yp)-f
        vals = delta[cols % 6, rows]/h[cols]
        return csc_matrix((vals, (rows, cols)), shape=(2*m+4, 2*m+4))
    return jac

@dataclass
class Run:
    config: Config
    inputs: Inputs
    x: np.ndarray
    xf: np.ndarray
    weights: np.ndarray
    p: np.ndarray
    segments: list
    end_s: float
    reached: bool
    threshold_s: float | None
    wall_s: float

    @property
    def m(self): return len(self.x)

    @property
    def dry_masses(self):
        return self.p[11]*np.pi*L0_M*R0_M**2*self.weights

    def sample(self, times):
        t = np.atleast_1d(np.asarray(times, dtype=float))
        if not np.isfinite(t).all() or np.any(t < -1e-9) or np.any(t > self.end_s+1e-7):
            raise ValueError(f'Times must lie within [0, {self.end_s}] seconds.')
        ans = np.empty((2*self.m+4, len(t)))
        ends = np.array([b for a, b, s in self.segments])
        idx = np.searchsorted(ends, np.clip(t, 0., self.end_s), side='left')
        idx = np.minimum(idx, len(ends)-1)
        for j in np.unique(idx):
            mask = idx == j; a, b, sol = self.segments[j]
            ans[:, mask] = sol.sol(np.clip(t[mask], a, b))
        return ans

    def radius_at(self, times):
        t = np.asarray(times, dtype=float)
        if self.config.shrink:
            return 0.01*np.interp(t, self.inputs.radius[:, 0], self.inputs.radius[:, 1])
        return np.full_like(t, R0_M, dtype=float)

    def fields(self, times, positions_cm=None):
        t = np.atleast_1d(np.asarray(times, dtype=float))
        y = self.sample(t)
        C, T = y[:2*self.m:2].T, y[1:2*self.m:2].T
        if positions_cm is None: return C, T
        radii = np.asarray(positions_cm, dtype=float)*0.01
        if np.any(radii < 0): raise ValueError('Distances from the axis cannot be negative.')
        R = self.radius_at(t)
        co = np.full((len(t), len(radii)), np.nan); to = co.copy()
        for i in range(len(t)):
            valid = radii <= R[i]+1e-12
            xi = np.clip(radii[valid]/R[i], 0., 1.)
            # 有界线性重构：其最大值必然出现在已包含的节点上。
            co[i, valid] = np.interp(xi, self.x, C[i])
            to[i, valid] = np.interp(xi, self.x, T[i])
        return co, to

    def diagnostics(self, times):
        t = np.atleast_1d(np.asarray(times, dtype=float))
        y = self.sample(t); m = self.m
        C, T = y[:2*m:2].T, y[1:2*m:2].T
        air = self.inputs.air
        ta = np.interp(t, air[:, 0], air[:, 1]); ya = np.interp(t, air[:, 0], air[:, 2])
        after = t > air[-1, 0]
        ta[after], ya[after] = self.p[9], self.p[10]
        R = self.radius_at(t); md = self.dry_masses
        pva = P_PA*ya/(0.62198+ya)
        z = np.log(np.maximum(pva, 1e-300)/610.8)
        dp = np.where(pva > 0, 237.3*z/(17.27-z), -np.inf)
        aw = C[:, -1]/(C[:, -1]+self.p[2])
        ps = aw*psat_pa(T[:, -1]); je = self.p[3]*(ps-pva)
        water0 = float(md@self.segments[0][2].y[:2*m:2, 0])
        mass = C@md
        return dict(time_s=t, radius_cm=R*100, T_air_C=ta, Y_air_kg_kg=ya,
                    T_dew_C=dp, T_center_C=T[:, 0], T_surface_C=T[:, -1],
                    T_mean_C=T@self.weights, C_center=C[:, 0], C_surface=C[:, -1],
                    C_max=C.max(axis=1), C_mean=C@self.weights,
                    max_location_cm=R*100*self.x[np.argmax(C, axis=1)],
                    j_e_kg_m2_s=je, a_w_surface=aw, p_surface_Pa=ps, p_air_Pa=pva,
                    water_kg=mass, water_out_kg=y[2*m],
                    mass_residual_kg=mass-water0+y[2*m],
                    convection_J=y[2*m+1], latent_J=y[2*m+2], sensible_J=y[2*m+3])

    def metrics(self):
        t = np.unique(np.r_[np.linspace(0., min(14400., self.end_s), 721),
                             np.linspace(0., self.end_s, 1201), self.end_s])
        d = self.diagnostics(t); C, T = self.fields(t)
        eqta, eqya = self.inputs.tail(self.config.air_tail)
        phi = P_PA*eqya/(0.62198+eqya)/float(psat_pa(eqta))
        ceq = self.p[2]*phi/(1-phi) if 0 < phi < 1 else None
        return dict(config=asdict(self.config), threshold_reached=self.reached,
                    drying_s=self.end_s if self.reached else None,
                    drying_h=self.end_s/3600 if self.reached else None,
                    crossing_015_s=self.threshold_s, computed_end_s=self.end_s,
                    end_Cmax=float(C[-1].max()), end_Csurface=float(C[-1, -1]),
                    end_Cmean=float(C[-1]@self.weights), end_Tcenter_C=float(T[-1, 0]),
                    end_Tsurface_C=float(T[-1, -1]), end_radius_cm=float(d['radius_cm'][-1]),
                    equilibrium_C=ceq, beta=self.p[2], Kp=self.p[3],
                    initial_dry_mass_kg=float(self.dry_masses.sum()),
                    initial_water_kg=float(d['water_kg'][0]),
                    final_water_kg=float(d['water_kg'][-1]),
                    water_out_kg=float(d['water_out_kg'][-1]),
                    max_mass_relative_residual=float(np.max(np.abs(d['mass_residual_kg']))/d['water_kg'][0]),
                    min_C_sampled=float(C.min()), min_T_C_sampled=float(T.min()),
                    max_T_C_sampled=float(T.max()),
                    accepted_steps=sum(len(s.t)-1 for a, b, s in self.segments),
                    nfev=sum(s.nfev for a, b, s in self.segments),
                    njev=sum(s.njev for a, b, s in self.segments),
                    wall_s=self.wall_s)

    def independent_energy_audit(self):
        """独立路径积分，而不是 U=integral rho*cp*T；本模型不采用后一种写法。

        在每个已接受的BDF区间上，取6个Gauss节点计算稠密多项式；
        通过Lagrange矩阵对其不超过5次的插值多项式求导，而不是调用RHS。
        随后积分 sum(V H dT/dt + cw T md dC/dt)，并与累计的
        Qconv-Qlatent-Qsensible 比较。该检验针对时间离散实现，不代表真实物理。
        """
        q, w = np.polynomial.legendre.leggauss(6); q, w = (q+1)/2, w/2
        b = np.array([1./np.prod(q[i]-np.delete(q, i)) for i in range(len(q))])
        dm = np.zeros((len(q), len(q)))
        for i in range(len(q)):
            for j in range(len(q)):
                if i != j: dm[i, j] = b[j]/b[i]/(q[i]-q[j])
            dm[i, i] = -dm[i].sum()
        accum = 0.; maxerr = 0.; maxref = 1.
        cw = self.p[13] if self.config.sensible else 0.
        md = self.dry_masses
        _, cd0, cw0 = material_constants(self.config.case)
        for a, end, sol in self.segments:
            for ta, tb in zip(sol.t[:-1], sol.t[1:]):
                dt = tb-ta
                ts = ta+dt*q; y = sol.sol(ts)
                # 先减去常数列，以抑制浮点消减误差。
                yp = (y-y[:, [0]])@dm.T/dt
                C, T = y[:2*self.m:2], y[1:2*self.m:2]
                Cd, Td = yp[:2*self.m:2], yp[1:2*self.m:2]
                if self.config.case == 1: rho = np.full_like(C, 820.)
                elif self.config.case == 2: rho = 650.+128.*C
                else: rho = 760.+90.*C
                H = rho*(cd0+cw0*C)/(1+C)
                V = np.pi*L0_M*self.weights[:, None]*self.radius_at(ts)[None, :]**2
                power = (V*H*Td+cw*T*md[:, None]*Cd).sum(axis=0)
                accum += dt*float(power@w)
                yt = sol.sol(tb)
                expected = yt[2*self.m+1]-yt[2*self.m+2]-yt[2*self.m+3]
                maxerr = max(maxerr, abs(accum-expected))
                maxref = max(maxref, abs(yt[2*self.m+1]), abs(yt[2*self.m+2]+yt[2*self.m+3]))
        return dict(path_storage_J=accum, boundary_net_J=float(expected),
                    final_energy_residual_J=float(accum-expected),
                    max_abs_energy_residual_J=maxerr,
                    max_relative_energy_residual=maxerr/maxref,
                    reference_energy_J=maxref,
                    audit='Six-point Gauss integration of differentiated BDF dense polynomials; engineering path balance, not a state-function energy test.')


def simulate(cfg: Config, inp: Inputs, initial=None, progress=False) -> Run:
    cfg.validate(); inp.validate()
    x = 1-(1-np.linspace(0., 1., cfg.n+1))**cfg.grid_power
    xf = np.r_[0., (x[:-1]+x[1:])/2., 1.]
    weights = np.diff(xf**2)
    p = parameters(cfg, inp); m = len(x)
    y = np.zeros(2*m+4); y[:2*m:2] = C0; y[1:2*m:2] = T0_C
    if initial is not None:
        state = np.asarray(initial, dtype=float)
        if state.shape != (2*m,) or not np.isfinite(state).all() or np.min(state[::2]) < 0:
            raise ValueError('Initial physical states must be finite interleaved C,T of shape 2*(n+1).')
        y[:2*m] = state
    atol = np.r_[np.tile([cfg.atol_C, cfg.atol_T], m), 1e-12, 1e-4, 1e-4, 1e-5]
    def strict_event(t, y): return np.max(y[:2*m:2])-(cfg.target_C-cfg.target_margin)
    def threshold_event(t, y): return np.max(y[:2*m:2])-cfg.target_C
    strict_event.terminal = True; strict_event.direction = -1
    threshold_event.terminal = False; threshold_event.direction = -1
    tend = cfg.horizon_h*3600
    knots = [0., tend, min(tend, inp.air[-1, 0])]
    if cfg.split_input_knots:
        knots += list(inp.air[(inp.air[:, 0] > 0)&(inp.air[:, 0] < tend), 0])
        if cfg.shrink:
            knots += list(inp.radius[(inp.radius[:, 0] > 0)&(inp.radius[:, 0] < tend), 0])
    knots = np.unique(knots)
    segments = []; reached = False; crossing = None
    tic = time.perf_counter()
    for a, b in zip(knots[:-1], knots[1:]):
        tail = a >= inp.air[-1, 0]
        fun = lambda t, state: rhs_core(t, state, x, xf, weights, p, inp.air, inp.radius, tail)
        sol = solve_ivp(fun, (a, b), y, method='BDF', jac=colored_jacobian(fun, m),
                        rtol=cfg.rtol, atol=atol, dense_output=True,
                        max_step=cfg.max_step_late_s if tail else cfg.max_step_early_s,
                        events=[strict_event, threshold_event] if cfg.stop_at_target else None)
        if not sol.success:
            raise RuntimeError(f'BDF failed at t={sol.t[-1]:.9g} s: {sol.message}')
        if np.min(sol.y[:2*m:2]) < -max(100*cfg.atol_C, 1e-9):
            raise RuntimeError('Negative accepted C. Refine grid/tolerances; do not clip the solution.')
        if not np.isfinite(sol.y).all(): raise RuntimeError('Nonfinite accepted state.')
        end = float(sol.t[-1]); segments.append((a, end, sol)); y = sol.y[:, -1]
        if cfg.stop_at_target:
            if crossing is None and len(sol.t_events[1]): crossing = float(sol.t_events[1][0])
            if len(sol.t_events[0]): reached = True; break
        if progress and (a == 0 or b == 14400 or b % 21600 == 0):
            print(f'  case={cfg.case}, n={cfg.n}, t={end/3600:.2f} h, Cmax={y[:2*m:2].max():.6f}', flush=True)
    return Run(cfg, inp, x, xf, weights, p, segments, end, reached, crossing, time.perf_counter()-tic)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, choices=[1, 2, 4], default=2)
    parser.add_argument('--n', type=int, default=160)
    parser.add_argument('--data-dir', default=str(Path(__file__).parent/'data'))
    args = parser.parse_args()
    cfg = Config(case=args.case, n=args.n, shrink=args.case == 4,
                 horizon_h=0.5 if args.case == 1 else 480., stop_at_target=args.case != 1)
    run = simulate(cfg, Inputs.load(args.data_dir), progress=True)
    print(json.dumps(run.metrics(), indent=2, ensure_ascii=False))


