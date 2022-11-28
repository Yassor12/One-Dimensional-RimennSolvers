import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


class Parms:
    """
    This class represent the base class for all solver it contain the common parameters for all solver
    """

    name__ = ""

    def __init__(self, CFL, gamma, tag):
        self.CFL = CFL
        self.gamma = gamma
        self.tag = tag
        self.dirs = tag.split("-")
        try:
            os.makedirs(f"Result/{self.dirs[0]}/")
        except:
            pass

    def set_init(
        self,
        x_left,
        x_right,
        x0,
        x_num,
        timeout,
        d_left,
        d_right,
        u_left,
        u_right,
        p_left,
        p_right,
    ):
        self.x_left = x_left
        self.x_right = x_right
        self.x0 = x0
        self.x_num = x_num
        self.timeout = timeout
        self.d_left = d_left
        self.d_right = d_right
        self.u_left = u_left
        self.u_right = u_right
        self.p_left = p_left
        self.p_right = p_right

    def pre_calcuations(self):
        self.dx = (self.x_right - self.x_left) / self.x_num
        self.xx = np.zeros(self.x_num + 1)

    def results(self, show=None):
        plt.figsize = (5, 5)
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        axs[0, 0].set_title("Density")
        axs[0, 0].plot(self.xx, self.rho)
        axs[1, 0].set_title("Velocity")
        axs[1, 0].plot(self.xx, self.vel)
        axs[0, 1].set_title("Pressue")
        axs[0, 1].plot(self.xx, self.pre)
        axs[1, 1].set_title("Energy")
        axs[1, 1].plot(self.xx, self.En)
        axs[0, 0].grid()
        axs[1, 0].grid()
        axs[0, 1].grid()
        axs[1, 1].grid()
        fig.suptitle(f"{self.dirs[0]} - {self.name__}")
        if self.tag:
            plt.savefig(f"Result/{self.dirs[0]}/{self.dirs[1]}.pdf", dpi=200)

        if show:
            plt.show()


class Exact(Parms):
    name__ = "Exact"

    def __init__(self, CFL, gamma, tag):
        super().__init__(CFL=CFL, gamma=gamma, tag=tag)

    def set_init(self, parm):
        super().set_init(**parm)

    def pre_calcuations(self):
        super().pre_calcuations()
        self.c_left = np.sqrt(self.gamma * self.p_left / self.d_left)
        self.c_right = np.sqrt(self.gamma * self.p_right / self.d_right)
        self.rho = np.zeros(self.x_num + 1)
        self.vel = np.zeros(self.x_num + 1)
        self.pre = np.zeros(self.x_num + 1)
        self.En = np.zeros(self.x_num + 1)

    def __find_p_u(self):
        tol, nriter, q_max = 1.0e-6, 20, 2.0
        du = self.u_right - self.u_left
        ducrit = 2.0 / (self.gamma - 1.0) * (self.c_left + self.c_right) - du
        if ducrit <= 0.0:
            raise ValueError("The data resulting a vaccum !!")
        # p = Guess( dl,ul,pl,cl,dr,ur,pr,cr )
        # compute guess value from pvrs riemann solver
        pvrs = 0.5 * (self.p_left + self.p_right) - 0.125 * (
            self.u_right - self.u_left
        ) * (self.d_left + self.d_right) * (self.c_left + self.c_right)
        p_min = np.minimum(self.p_left, self.p_right)
        p_max = np.maximum(self.p_left, self.p_right)
        q_rat = p_max / p_min
        if (q_rat <= q_max) and (p_min <= pvrs and pvrs <= p_max):
            # pvrs solution
            p = np.maximum(tol, pvrs)
        else:
            if pvrs <= p_min:
                # use two-rarefraction solution
                pnu = (
                    self.c_left
                    + self.c_right
                    - 0.5 * (self.gamma - 1.0) * (self.u_right - self.u_left)
                )
                pde = self.c_left / self.p_left ** (
                    (self.gamma - 1.0) / (2.0 * self.gamma)
                ) + self.c_right / self.p_right ** (
                    (self.gamma - 1.0) / (2.0 * self.gamma)
                )
                p = (pnu / pde) ** (2.0 * self.gamma / (self.gamma - 1.0))
            else:
                # two-shock approximation
                gel = np.sqrt(
                    ((2.0 / (self.gamma + 1.0)) / self.d_left)
                    / (
                        ((self.gamma - 1.0) / (self.gamma + 1.0)) * self.p_left
                        + np.maximum(tol, pvrs)
                    )
                )
                ger = np.sqrt(
                    ((2.0 / (self.gamma + 1.0)) / self.d_right)
                    / (
                        ((self.gamma - 1.0) / (self.gamma + 1.0)) * self.p_right
                        + np.maximum(tol, pvrs)
                    )
                )
                p = (
                    gel * self.p_left
                    + ger * self.p_right
                    - (self.u_right - self.u_left)
                ) / (gel + ger)
                p = np.maximum(tol, p)
        p0 = p
        for k in range(nriter):
            if p <= self.p_left:
                # rarefraction wave
                prat = p / self.p_left
                fl = (
                    (2.0 / (self.gamma - 1.0))
                    * self.c_left
                    * (prat ** ((self.gamma - 1.0) / (2.0 * self.gamma)) - 1.0)
                )
                fld = (1.0 / (self.d_left * self.c_left)) * prat ** (
                    -(self.gamma + 1.0) / (2.0 * self.gamma)
                )
            else:
                # shock wave
                ak = (2.0 / (self.gamma + 1.0)) / self.d_left
                bk = ((self.gamma - 1.0) / (self.gamma + 1.0)) * self.p_left
                qrt = np.sqrt(ak / (bk + p))
                fl = (p - self.p_left) * qrt
                fld = (1.0 - 0.5 * (p - self.p_left) / (bk + p)) * qrt
            if p <= self.p_right:
                # rarefraction wave
                prat = p / self.p_right
                fr = (
                    (2.0 / (self.gamma - 1.0))
                    * self.c_right
                    * (prat ** ((self.gamma - 1.0) / (2.0 * self.gamma)) - 1.0)
                )
                frd = (1.0 / (self.d_right * self.c_right)) * prat ** (
                    -(self.gamma + 1.0) / (2.0 * self.gamma)
                )
            else:
                # shock wave
                ak = (2.0 / (self.gamma + 1.0)) / self.d_right
                bk = ((self.gamma - 1.0) / (self.gamma + 1.0)) * self.p_right
                qrt = np.sqrt(ak / (bk + p))
                fr = (p - self.p_right) * qrt
                frd = (1.0 - 0.5 * (p - self.p_right) / (bk + p)) * qrt
            p = p - (fl + fr + du) / (fld + frd)
            cha = 2.0 * np.abs((p - p0) / (p + p0))
            if cha < tol:
                break
            if p < 0.0:
                p = tol
            p0 = p
        if cha >= tol:
            print("divergence in Newton-Raphson scheme")

        # compute velocity u
        u = 0.5 * (self.u_left + self.u_right + fr - fl)
        return p, u

    def solver(self):
        self.pre_calcuations()
        p_m, u_m = self.__find_p_u()
        for j in range(self.x_num + 1):
            self.xs = self.x_left + j * self.dx
            self.s = (self.xs - self.x0) / self.timeout
            p_sample, u_sample, d_sample = self.get_sample(p_m, u_m, self.s)
            self.xx[j] = self.xs
            self.rho[j] = d_sample
            self.vel[j] = u_sample
            self.pre[j] = p_sample
            self.En[j] = p_sample / (d_sample * (self.gamma - 1.0))

    def get_sample(self, pm, um, s):
        if s <= um:
            # sample point is to the left of the contact
            if pm <= self.p_left:
                # left fan
                shl = self.u_left - self.c_left
                if s <= shl:
                    # left data state
                    d = self.d_left
                    u = self.u_left
                    p = self.p_left
                else:
                    cml = self.c_left * (pm / self.p_left) ** (
                        (self.gamma - 1.0) / (2.0 * self.gamma)
                    )
                    stl = um - cml
                    if s > stl:
                        # middle left state
                        d = self.d_left * (pm / self.p_left) ** (1.0 / self.gamma)
                        u = um
                        p = pm
                    else:
                        # an left state (inside fan)
                        u = (2.0 / (self.gamma + 1.0)) * (
                            self.c_left + (0.5 * (self.gamma - 1.0)) * self.u_left + s
                        )
                        c = (2.0 / (self.gamma + 1.0)) * (
                            self.c_left + (0.5 * (self.gamma - 1.0)) * (self.u_left - s)
                        )
                        d = self.d_left * (c / self.c_left) ** (
                            2.0 / (self.gamma - 1.0)
                        )
                        p = self.p_left * (c / self.c_left) ** (
                            2.0 * self.gamma / (self.gamma - 1.0)
                        )
            else:
                # left shock
                pml = pm / self.p_left
                sl = self.u_left - self.c_left * np.sqrt(
                    ((self.gamma + 1.0) / (2.0 * self.gamma)) * pml
                    + ((self.gamma - 1.0) / (2.0 * self.gamma))
                )
                if s <= sl:
                    # left data state
                    d = self.d_left
                    u = self.u_left
                    p = self.p_left
                else:
                    # middle left state (behind shock)
                    d = (
                        self.d_left
                        * (pml + ((self.gamma - 1.0) / (self.gamma + 1.0)))
                        / (pml * ((self.gamma - 1.0) / (self.gamma + 1.0)) + 1.0)
                    )
                    u = um
                    p = pm
        else:
            # right of contact
            if pm > self.p_right:
                # right shock
                pmr = pm / self.p_right
                sr = self.u_right + self.c_right * np.sqrt(
                    ((self.gamma + 1.0) / (2.0 * self.gamma)) * pmr
                    + ((self.gamma - 1.0) / (2.0 * self.gamma))
                )
                if s >= sr:
                    # right data state
                    d = self.d_right
                    u = self.u_right
                    p = self.p_right
                else:
                    # middle right state (behind shock)
                    d = (
                        self.d_right
                        * (pmr + ((self.gamma - 1.0) / (self.gamma + 1.0)))
                        / (pmr * ((self.gamma - 1.0) / (self.gamma + 1.0)) + 1.0)
                    )
                    u = um
                    p = pm
            else:
                # right fan
                shr = self.u_right + self.c_right
                if s >= shr:
                    # right data state
                    d = self.d_right
                    u = self.u_right
                    p = self.p_right
                else:
                    cmr = self.c_right * (pm / self.p_right) ** (
                        (self.gamma - 1.0) / (2.0 * self.gamma)
                    )
                    Str = um + cmr
                    if s <= Str:
                        # middle right state
                        d = self.d_right * (pm / self.p_right) ** (1.0 / self.gamma)
                        u = um
                        p = pm
                    else:
                        # fan right state (inside fan)
                        u = (2.0 / (self.gamma + 1.0)) * (
                            -self.c_right
                            + (0.5 * (self.gamma - 1.0)) * self.u_right
                            + s
                        )
                        c = (2.0 / (self.gamma + 1.0)) * (
                            self.c_right
                            - (0.5 * (self.gamma - 1.0)) * (self.u_right - s)
                        )
                        d = self.d_right * (c / self.c_right) ** (
                            2.0 / (self.gamma - 1.0)
                        )
                        p = self.p_right * (c / self.c_right) ** (
                            2.0 * self.gamma / (self.gamma - 1.0)
                        )
        return p, u, d


class LFEuler(Parms):
    """
    This class is represent Lax-Friedrichs method to solve the Euler-equation
    """

    name__ = "Lax-Friedrichs"

    def __init__(self, CFL, gamma, tag):
        super().__init__(CFL=CFL, gamma=gamma, tag=tag)
        self._gamma = self.gamma - 1.0
        self.time = 0.0

    def set_init(self, parm, dur):
        super().set_init(**parm)
        self.dur = dur

    def pre_calcuations(self):
        super().pre_calcuations()
        #      conservative initial data
        self.uLeft = np.array(
            [
                self.d_left,
                self.d_left * self.u_left,
                self.p_left / self._gamma
                + 0.5 * self.d_left * self.u_left * self.u_left,
            ]
        )
        self.uRight = np.array(
            [
                self.d_right,
                self.d_right * self.u_right,
                self.p_right / self._gamma
                + 0.5 * self.d_right * self.u_right * self.u_right,
            ]
        )
        self.xx = np.linspace(
            self.x_left + 0.5 * self.dx, self.x_right - 0.5 * self.dx, self.x_num
        )
        self.u = np.zeros((self.x_num + 2, 3))
        Nx = len(self.xx)
        dx = 0.5 / Nx
        u = np.zeros((Nx, 3))
        u[:, 0] = np.where(self.xx + dx <= self.x0, self.uLeft[0], self.uRight[0])
        u[:, 1] = np.where(self.xx + dx <= self.x0, self.uLeft[1], self.uRight[1])
        u[:, 2] = np.where(self.xx + dx <= self.x0, self.uLeft[2], self.uRight[2])
        self.u[1:-1, :] = u

    def dt_step(self, CFL=None):
        if not CFL:
            CFL = self.CFL
        self.rho = self.u[:, 0]
        self.rhou = self.u[:, 1]
        self.rhoE = self.u[:, 2]
        self.p_right = self._gamma * (self.rhoE - 0.5 * self.rhou**2 / self.rho)
        a = np.sqrt(self.gamma * self.p_right / self.rho)
        smax = np.amax(np.abs(self.rhou / self.rho) + a)
        self.dt = CFL * self.dx / smax

    def get_flux(self):
        Nx = self.u.shape[0] - 2
        tmp = 0.5 * self.dx / self.dt
        self.flux = np.zeros((Nx + 2, 3))
        for j in range(Nx + 1):
            fL = self.euler_flux(j)
            fR = self.euler_flux(j + 1)
            self.flux[j] = 0.5 * (fL + fR) + tmp * (self.u[j] - self.u[j + 1])

    def euler_flux(self, i):
        rho, rhou, rhoE = self.u[i]
        u = rhou / rho
        p_right = self._gamma * (rhoE - 0.5 * rhou * u)
        return np.array([rhou, rhou * u + p_right, u * (rhoE + p_right)])

    def solver(self):
        self.pre_calcuations()
        for n in range(self.dur):
            # set transmissible boundary conditions
            self.u[0] = self.u[1]
            self.u[self.x_num + 1] = self.u[self.x_num]

            # impose CFL condition to find dt : time step size
            if n <= 5:
                self.dt_step(CFL=0.2)
            else:
                self.dt_step()
            # check that time timeout has not been exceeded

            if (self.time + self.dt) >= self.timeout:
                self.dt = self.timeout - self.time

            self.tau = self.dt / self.dx
            self.time = self.time + self.dt

            # compute HLL intercell flux Flux(i)
            self.get_flux()

            # advance via conservative formula
            self.u[1 : self.x_num + 1] -= self.tau * (
                self.flux[1 : self.x_num + 1] - self.flux[0 : self.x_num]
            )

            timedif = np.abs(self.time - self.timeout)
            if timedif <= 1.0e-7:
                break

        self.rho = self.u[1:-1, 0]
        self.vel = self.u[1:-1, 1] / self.rho
        self.En = self.u[1:-1, 2] / self.rho - 0.5 * self.vel**2
        self.pre = self._gamma * self.rho * self.En


class HLL(Parms):
    def __init__(self, CFL, gamma, tag):
        super().__init__(CFL=CFL, gamma=gamma, tag=tag)
        self._gamma = self.gamma - 1.0
        self.time = 0.0
        
    def set_init(self, parm, dur):
        super().set_init(**parm)
        self.dur = dur

    def pre_calcuations(self):
        super().pre_calcuations()
        #      conservative initial data
        self.uLeft = np.array(
            [
                self.d_left,
                self.d_left * self.u_left,
                self.p_left / self._gamma
                + 0.5 * self.d_left * self.u_left * self.u_left,
            ]
        )
        self.uRight = np.array(
            [
                self.d_right,
                self.d_right * self.u_right,
                self.p_right / self._gamma
                + 0.5 * self.d_right * self.u_right * self.u_right,
            ]
        )
        self.xx = np.linspace(
            self.x_left + 0.5 * self.dx, self.x_right - 0.5 * self.dx, self.x_num
        )
        self.u = np.zeros((self.x_num + 2, 3))
        Nx = len(self.xx)
        dx = 0.5 / Nx
        u = np.zeros((Nx, 3))
        u[:, 0] = np.where(self.xx + dx <= self.x0, self.uLeft[0], self.uRight[0])
        u[:, 1] = np.where(self.xx + dx <= self.x0, self.uLeft[1], self.uRight[1])
        u[:, 2] = np.where(self.xx + dx <= self.x0, self.uLeft[2], self.uRight[2])
        self.u[1:-1, :] = u

    def dt_step(self, CFL=None):
        if not CFL:
            CFL = self.CFL
        self.rho = self.u[:, 0]
        self.rhou = self.u[:, 1]
        self.rhoE = self.u[:, 2]
        self.p_right = self._gamma * (self.rhoE - 0.5 * self.rhou**2 / self.rho)
        a = np.sqrt(self.gamma * self.p_right / self.rho)
        smax = np.amax(np.abs(self.rhou / self.rho) + a)
        self.dt = CFL * self.dx / smax

    def euler_flux(self, i):
        rho, rhou, rhoE = self.u[i]
        u = rhou / rho
        p_right = self._gamma * (rhoE - 0.5 * rhou * u)
        return np.array([rhou, rhou * u + p_right, u * (rhoE + p_right)])

    def get_flux(self):
        self.flux = np.zeros_like(self.u)
        for j in range(self.x_num + 1):
            # consadred local variable
            d_left, rhou_left, rhoE_left = self.u[j, :]
            u_left = rhou_left / d_left
            p_left = self._gamma * (rhoE_left - 0.5 * rhou_left * u_left)
            c_left = np.sqrt(self.gamma * p_left / d_left)

            d_right, rhou_right, rhoE_right = self.u[j + 1, :]
            u_right = rhou_right / d_right
            p_right = self._gamma * (rhoE_right - 0.5 * rhou_right * u_right)
            c_right = np.sqrt(self.gamma * p_left / d_right)

            # parameters
            tol, qmax = 1.0e-6, 2.0

            # compute guess value from pvrs riemann solver
            pvrs = 0.5 * (p_left + p_right) - 0.125 * (u_right - u_right) * (
                d_left + d_right
            ) * (c_right + c_left)
            pmin = np.minimum(p_left, p_right)
            pmax = np.maximum(p_left, p_right)
            qrat = pmax / pmin
            if (qrat <= qmax) and (pmin <= pvrs and pvrs <= pmax):
                # use pvrs solution as guess
                p_max = np.maximum(tol, pvrs)
            else:
                if pvrs <= pmin:
                    pnu = c_left + c_right - 0.5 * self._gamma * (u_right - u_left)
                    pde = c_left / p_left**self._gamma / (
                        2.0 * self.gamma
                    ) + c_right / p_right**self._gamma / (2.0 * self.gamma)
                    p_max = (pnu / pde) ** 2.0 * self.gamma / self._gamma
                else:
                    rtl = np.sqrt(
                        ((2.0 / (self.gamma + 1.0)) / d_left)
                        / (
                            (self.gamma / (self.gamma + 1.0)) * p_left
                            + np.maximum(tol, pvrs)
                        )
                    )
                    rtr = np.sqrt(
                        ((2.0 / (self.gamma + 1.0)) / d_right)
                        / (
                            (self.gamma / (self.gamma + 1.0)) * p_right
                            + np.maximum(tol, pvrs)
                        )
                    )
                    p = (rtl * p_left + rtr * p_right - (u_right - u_left)) / (
                        rtl + rtr
                    )
                    p_max = np.maximum(tol, p)
            if p_max <= p_left:
                SL = u_left - c_left
            else:
                SL = u_left - c_left * np.sqrt(
                    1.0 + (self._gamma / (2.0 * self.gamma)) * (p_max / p_left - 1.0)
                )
            if p_max <= p_right:
                SR = u_right + c_right
            else:
                SR = u_right + c_right * np.sqrt(
                    1.0 + (self._gamma / (2.0 * self.gamma)) * (p_max / p_right - 1.0)
                )
        
            if SL >= 0.0:
                self.flux[j] = self.euler_flux(j)
            elif SR < 0.0:
                self.flux[j] = self.euler_flux(j + 1)
            else:
                flux_left = self.euler_flux(j)
                flux_right = self.euler_flux(j + 1)
                dS = 1.0 / (SR - SL)
                S2 = SR * SL
                self.flux[j] = (
                        SR * flux_left
                        - SL * flux_right
                        + S2 * (self.u[j + 1] - self.u[j])
                    ) * dS

    def solver(self):
        self.pre_calcuations()
        for n in range(self.dur):
            # set transmissible boundary conditions
            self.u[0] = self.u[1]
            self.u[self.x_num + 1] = self.u[self.x_num]

            # impose CFL condition to find dt : time step size
            if n <= 5:
                self.dt_step(CFL=0.2)
            else:
                self.dt_step()
            # check that time timeout has not been exceeded

            if (self.time + self.dt) >= self.timeout:
                self.dt = self.timeout - self.time

            self.tau = self.dt / self.dx
            self.time = self.time + self.dt

            # compute HLL intercell flux Flux(i)
            self.get_flux()

            # advance via conservative formula
            self.u[1 : self.x_num + 1] -= self.tau * (
                self.flux[1 : self.x_num + 1] - self.flux[0 : self.x_num]
            )

            timedif = np.abs(self.time - self.timeout)
            if timedif <= 1.0e-7:
                break

        self.rho = self.u[1:-1, 0]
        self.vel = self.u[1:-1, 1] / self.rho
        self.En = self.u[1:-1, 2] / self.rho - 0.5 * self.vel**2
        self.pre = self._gamma * self.rho * self.En

class HLLC(Parms):
    def __init__(self, CFL, gamma, tag):
        super().__init__(CFL=CFL, gamma=gamma, tag=tag)
        self._gamma = self.gamma - 1.0
        self.time = 0.0

    def set_init(self, parm, dur):
        super().set_init(**parm)
        self.dur = dur

    def pre_calcuations(self):
        super().pre_calcuations()
        #      conservative initial data
        self.uLeft = np.array(
            [
                self.d_left,
                self.d_left * self.u_left,
                self.p_left / self._gamma
                + 0.5 * self.d_left * self.u_left * self.u_left,
            ]
        )
        self.uRight = np.array(
            [
                self.d_right,
                self.d_right * self.u_right,
                self.p_right / self._gamma
                + 0.5 * self.d_right * self.u_right * self.u_right,
            ]
        )
        self.xx = np.linspace(
            self.x_left + 0.5 * self.dx, self.x_right - 0.5 * self.dx, self.x_num
        )
        self.u = np.zeros((self.x_num + 2, 3))
        Nx = len(self.xx)
        dx = 0.5 / Nx
        u = np.zeros((Nx, 3))
        u[:, 0] = np.where(self.xx + dx <= self.x0, self.uLeft[0], self.uRight[0])
        u[:, 1] = np.where(self.xx + dx <= self.x0, self.uLeft[1], self.uRight[1])
        u[:, 2] = np.where(self.xx + dx <= self.x0, self.uLeft[2], self.uRight[2])
        self.u[1:-1, :] = u

    def dt_step(self, CFL=None):
        if not CFL:
            CFL = self.CFL
        rho = self.u[:, 0]
        rhou = self.u[:, 1]
        rhoE = self.u[:, 2]
        p_right = self._gamma * (rhoE - 0.5 * rhou**2 / rho)
        a = np.sqrt(self.gamma * p_right / rho)
        smax = np.amax(np.abs(rhou / rho) + a)
        self.dt = CFL * self.dx / smax

    def euler_flux(self, i):
        rho, rhou, rhoE = self.u[i]
        u = rhou / rho
        p_right = self._gamma * (rhoE - 0.5 * rhou * u)
        return np.array([rhou, rhou * u + p_right, u * (rhoE + p_right)])

    def get_flux(self):
        self.flux = np.zeros_like(self.u)
        for j in range(self.x_num + 1):
            # consadred local variable
            d_left, rhou_left, rhoE_left = self.u[j, :]
            u_left = rhou_left / d_left
            p_left = self._gamma * (rhoE_left - 0.5 * rhou_left * u_left)
            c_left = np.sqrt(self.gamma * p_left / d_left)

            d_right, rhou_right, rhoE_right = self.u[j + 1, :]
            u_right = rhou_right / d_right
            p_right = self._gamma * (rhoE_right - 0.5 * rhou_right * u_right)
            c_right = np.sqrt(self.gamma * p_left / d_right)

            # parameters
            tol, qmax = 1.0e-6, 2.0

            # compute guess value from pvrs riemann solver
            pvrs = 0.5 * (p_left + p_right) - 0.125 * (u_right - u_right) * (
                d_left + d_right
            ) * (c_right + c_left)
            pmin = np.minimum(p_left, p_right)
            pmax = np.maximum(p_left, p_right)
            qrat = pmax / pmin
            if (qrat <= qmax) and (pmin <= pvrs and pvrs <= pmax):
                # use pvrs solution as guess
                p_max = np.maximum(tol, pvrs)
            else:
                if pvrs <= pmin:
                    pnu = c_left + c_right - 0.5 * self._gamma * (u_right - u_left)
                    pde = c_left / p_left**self._gamma / (
                        2.0 * self.gamma
                    ) + c_right / p_right**self._gamma / (2.0 * self.gamma)
                    p_max = (pnu / pde) ** 2.0 * self.gamma / self._gamma
                else:
                    rtl = np.sqrt(
                        ((2.0 / (self.gamma + 1.0)) / d_left)
                        / (
                            (self.gamma / (self.gamma + 1.0)) * p_left
                            + np.maximum(tol, pvrs)
                        )
                    )
                    rtr = np.sqrt(
                        ((2.0 / (self.gamma + 1.0)) / d_right)
                        / (
                            (self.gamma / (self.gamma + 1.0)) * p_right
                            + np.maximum(tol, pvrs)
                        )
                    )
                    p = (rtl * p_left + rtr * p_right - (u_right - u_left)) / (
                        rtl + rtr
                    )
                    p_max = np.maximum(tol, p)
            if p_max <= p_left:
                SL = u_left - c_left
            else:
                SL = u_left - c_left * np.sqrt(
                    1.0 + (self._gamma / (2.0 * self.gamma)) * (p_max / p_left - 1.0)
                )
            if p_max <= p_right:
                SR = u_right + c_right
            else:
                SR = u_right + c_right * np.sqrt(
                    1.0 + (self._gamma / (2.0 * self.gamma)) * (p_max / p_right - 1.0)
                )
            SM = (p_right - p_left + d_left * u_left * (SL - u_left) - d_right* u_right * (SR - u_right)) / (
        d_left * (SL - u_left) - d_right * (SR - u_right)
    )   
            if SL >= 0.0:
                self.flux[j] = self.euler_flux(j)
            elif SR < 0.0:
                self.flux[j] = self.euler_flux(j + 1)
            elif SM <= 0.0:
                self.flux[j] = self.euler_flux(j + 1)
                tmp = d_right * (SR - u_right) / (SR - SM)
                self.flux[j, 0] += SR * (tmp - d_right)
                self.flux[j, 1] += SR * (tmp * SM - rhou_right)
                self.flux[j, 2] += SR * (
                tmp * (rhoE_right / d_right + (SM - u_right) * (SM + p_right / (d_right * (SR - u_right)))) - rhoE_right
            )
            else:
                self.flux[j] = self.euler_flux(j)
                tmp = d_left * (SL - u_left) / (SL - SM)
                self.flux[j, 0] += SL * (tmp - d_left)
                self.flux[j, 1] += SL * (tmp * SM - rhou_left)
                self.flux[j, 2] += SL * (
                tmp * (rhoE_left / d_left + (SM - u_left) * (SM + p_left / (d_left * (SL - u_left)))) - rhoE_left
            )




    def solver(self):
        self.pre_calcuations()
        for n in range(self.dur):
            # set transmissible boundary conditions
            self.u[0] = self.u[1]
            self.u[self.x_num + 1] = self.u[self.x_num]

            # impose CFL condition to find dt : time step size
            if n <= 5:
                self.dt_step(CFL=0.2)
            else:
                self.dt_step()
            # check that time timeout has not been exceeded

            if (self.time + self.dt) >= self.timeout:
                self.dt = self.timeout - self.time

            self.tau = self.dt / self.dx
            self.time = self.time + self.dt

            # compute HLL intercell flux Flux(i)
            self.get_flux()

            # advance via conservative formula
            self.u[1 : self.x_num + 1] -= self.tau * (
                self.flux[1 : self.x_num + 1] - self.flux[0 : self.x_num]
            )

            timedif = np.abs(self.time - self.timeout)
            if timedif <= 1.0e-7:
                break

        self.rho = self.u[1:-1, 0]
        self.vel = self.u[1:-1, 1] / self.rho
        self.En = self.u[1:-1, 2] / self.rho - 0.5 * self.vel**2
        self.pre = self._gamma * self.rho * self.En
