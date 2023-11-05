import numpy as np
from scipy.stats import norm


def basket_price(vols, correls, w, T, K):
    M = np.sum(w)
    V2 = 0
    for i in range(len(w)):
        for j in range(len(w)):
            V2 += w[i] * w[j] * np.exp(vols[i] * vols[j] * correls[i][j] * T)

    m = 2 * np.log(M) - 0.5 * np.log(V2)
    v2 = np.log(V2) - 2 * np.log(M)

    d1 = (m - np.log(K) + v2) * v2 ** (-0.5)
    d2 = d1 - v2 ** 0.5

    return M * norm.cdf(d1) - K * norm.cdf(d2)


def xi(a,b,c,d,k):
    r1 = np.cos(k*np.pi*(d-a)/(b-a))*np.exp(d) - np.cos(k*np.pi*(c-a)/(b-a))*np.exp(c)
    r2 = k*np.pi/(b-a)*np.sin(k*np.pi*(d-a)/(b-a))*np.exp(d) - k*np.pi/(b-a)*np.sin(k*np.pi*(c-a)/(b-a))*np.exp(c)
    return (r1+r2)/(1+(k*np.pi/(b-a))**2)

def vpsi(a,b,c,d,ks):
    ks = np.maximum(ks,1)
    r = (np.sin(ks*np.pi*(d-a)/(b-a)) - np.sin(ks*np.pi*(c-a)/(b-a)))*(b-a)/(ks*np.pi)
    r[0] = d-c
    return r

def vv_put_orig(a,b,ks):
    r=2/(b-a)*(-xi(a,b,a,0,ks)+vpsi(a,b,a,0,ks))
    r[0] = r[0]/2
    return r

def phi_heston2_pre(omega, kappa, theta, alpha, rho, T):
    a0=kappa-rho*alpha*omega*1j
    gamma=np.power(alpha**2*(omega**2+omega*1j)+a0*a0, 0.5)
    G=(a0-gamma)/(a0+gamma)
    a1=1.0/alpha/alpha*((1-np.exp(-gamma*T))/(1-G*np.exp(-gamma*T)))*(a0-gamma)
    a2=kappa*theta/alpha/alpha*(T*(a0-gamma)-2*np.log((1-G*np.exp(-gamma*T))/(1-G)))
    return a1, a2

def phi_heston2(omega, kappa, theta, alpha, rho, V0, T):
    a1, a2 = phi_heston2_pre(omega, kappa, theta, alpha, rho, T)
    return np.exp(V0*a1+a2)

def phi_heston2_post(a, V0):
    a1, a2 = a
    return np.exp(V0*a1+a2)

def cumulants(kappa, theta, alpha, rho, V0, t):
    c1 = (1-np.exp(-kappa*t))*(theta-V0)/(2*kappa)-theta*t/2
    c21 = V0 / (4*kappa**3) * (4*kappa**2*(1+(rho*alpha*t-1)*np.exp(-kappa*t))
                               +kappa*(4*rho*alpha*(np.exp(-kappa*t)-1)-2*alpha**2*t*np.exp(-kappa*t))
                               +alpha**2*(1-np.exp(-2*kappa*t)))
    c22 = theta / (8*kappa**3) * (8*kappa**3*t - 8*kappa**2*(1+rho*alpha*t+(rho*alpha*t-1)*np.exp(-kappa*t))
                                  +2*kappa*((1+2*np.exp(-kappa*t))*alpha**2*t+8*(1-np.exp(-kappa*t))*rho*alpha)
                                  +alpha**2*(np.exp(-2*kappa*t)+4*np.exp(-kappa*t)-5))
    return c1, c21+c22



def value(F, Ks, V0, kappa, theta, alpha, rho, T):
    Ks=np.array(Ks)
    L=12
    c1, c2 = cumulants(kappa, theta, alpha, rho, theta, T)
    a=c1-L*c2**0.5
    b=c1+L*c2**0.5
    N=50

    xs=np.log(F/Ks)
    ks=np.array(range(N))
    xxs=ks*np.pi/(b-a)
    h = phi_heston2(xxs, kappa, theta, alpha,rho,V0,T)*vv_put_orig(a,b,ks)
    res = Ks*np.sum(h*np.exp(1j * (xs-a)*xxs)).real
    return res


class valuatorAtm:
    def __init__(self, heston_params, T_rem):
        na = heston_params.shape[0]
        N = 50
        L = 12

        self.phi_heston_pre = []
        self.vv_put_orig = []
        self.exp = []
        for i in range(na):
            kappa, theta, alpha, rho = heston_params[i, 1:]
            c1, c2 = cumulants(kappa, theta, alpha, rho, theta, T_rem)
            a = c1 - L * c2 ** 0.5
            b = c1 + L * c2 ** 0.5

            ks=np.array(range(N))
            xxs=ks*np.pi/(b-a)
            put_orig = vv_put_orig(a,b,ks)
            self.vv_put_orig.append(put_orig)
            heston_pre = phi_heston2_pre(xxs, kappa, theta, alpha, rho, T_rem)
            self.phi_heston_pre.append(heston_pre)
            self.exp.append(np.exp(1j * a * xxs))

    def value(self, i_asset, V0, S0):
        aa = self.phi_heston_pre[i_asset]
        h = phi_heston2_post(aa, V0) * self.vv_put_orig[i_asset]
        res = S0 * np.sum(h * self.exp[i_asset]).real
        return res


def test():
    S0=50
    F=50
    Ks=np.array([50])
    V0=0.3**2
    kappa=2
    theta=0.3**2
    alpha=0.3
    rho=-0.7
    L=12
    T=0.25
    c1, c2 = cumulants(kappa, theta, alpha, rho, theta, T)
    a=c1-L*c2**0.5
    b=c1+L*c2**0.5
    N=50

    xs=np.log(F/Ks)
    ks=np.array(range(N))
    xxs=ks*np.pi/(b-a)
    h = phi_heston2(xxs, kappa, theta, alpha,rho,V0,T)*vv_put_orig(a,b,ks)
    res = Ks[0]*np.sum(h*np.exp(1j * (xs[0]-a)*xxs)).real

    va = valuatorAtm(np.array([[V0, kappa, theta, alpha, rho]]), T)
    vav = va.value(0, V0, S0)
    return vav - res



if __name__ == '__main__':
    vols = np.array([0.2, 0.3, 0.4])
    correls = np.array([[1.0, 0.7, 0.5],
                        [0.7, 1.0, 0.4],
                        [0.5, 0.4, 1.0]])
    w = np.array([0.3, 0.3, 0.4])
    print(basket_price(vols, correls, w, 1, 1))
