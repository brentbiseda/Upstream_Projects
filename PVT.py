
# coding: utf-8

# In[15]:


import math


# In[3]:


# PVT Functions -----------------------------------------------------------

# Calculation of Gas Psuedopressure for Miscelaneous Gases
def mp(p, t, sg, n2, co2, h2s):
    mp = 0
    Pold = 0
    Xold = 0
    Pstep = p / 20
    for N in range(20):
        Pnew = Pold + Pstep
        Xnew = 2 * Pnew / Z(Pnew, t, sg, n2, co2, h2s) / Ug(Pnew, t, sg, n2, co2, h2s)
        mp = mp + (Xold + Xnew) / 2 * Pstep
        Pold = Pnew
        Xold = Xnew

    return mp


# In[9]:


# ------------------------------------------------------------------------
# T = temperature of interest (degrees F)
# P = pressure of interest (psia)
# SG = gas specific gravity (air=1)
# N2 = % Nitrogen
# CO2 = % CO2
# H2S = % H2S
# ------------------------------------------------------------------------
# Calculation of Critical Temperature for Miscelaneous Gases
def Tcm(sg, n2, co2, h2s):
    yn2 = n2 / 100
    yco2 = co2 / 100
    yh2s = h2s / 100
    gasg = (sg - 0.9672 * yn2 - 1.5195 * yco2 - 1.1765 * yh2s) / (1 - yn2 - yco2 - yh2s)
    Tcm = 168 + 325 * gasg - 12.5 * gasg ** 2
    Tcm = (1 - yn2 - yco2 - yh2s) * Tcm + 227.3 * yn2 + 547.6 * yco2 + 672.4 * yh2s
    cwa = 120 * ((yco2 + yh2s) ** 0.9 - (yco2 + yh2s) ** 1.6) + 15 * (yh2s ** 0.5 - yh2s ** 4)
    Tcm = Tcm - cwa
    return Tcm


# In[11]:


# ------------------------------------------------------------------------
# Calculation of Critical Pressure for Miscelaneous Gases
def Pcm(sg, n2, co2, h2s):
    yn2 = n2 / 100
    yco2 = co2 / 100
    yh2s = h2s / 100
    gasg = (sg - 0.9672 * yn2 - 1.5195 * yco2 - 1.1765 * yh2s) / (1 - yn2 - yco2 - yh2s)
    Pcm = 677 + 15 * gasg - 37.5 * gasg ** 2
    Pcm = (1 - yn2 - yco2 - yh2s) * Pcm + 493 * yn2 + 1071 * yco2 + 1306 * yh2s
    Tc = 168 + 325 * gasg - 12.5 * gasg ** 2
    Tc = (1 - yn2 - yco2 - yh2s) * Tc + 227.3 * yn2 + 547.6 * yco2 + 672.4 * yh2s
    cwa = 120 * ((yco2 + yh2s) ** 0.9 - (yco2 + yh2s) ** 1.6) + 15 * (yh2s ** 0.5 - yh2s ** 4)
    Pcm = Pcm * (Tc - cwa) / (Tc + yh2s * (1 - yh2s) * cwa)
    return Pcm


# In[20]:


# ------------------------------------------------------------------------
# Calculation of Z Factor for Miscelaneous Gases
def Z(p, t, sg, n2, co2, h2s):
    tr = (t + 460) / Tcm(sg, n2, co2, h2s)
    pr = p / Pcm(sg, n2, co2, h2s)
    A = 0.064225133
    B = 0.53530771 * tr - 0.61232032
    C = 0.31506237 * tr - 1.0467099 - 0.57832729 / tr ** 2
    D = tr
    E = 0.68157001 / tr ** 2
    F1 = 0.68446549
    G = 0.27 * pr
    rho = 0.27 * pr / tr  # Initial guess
    rhoold = rho
    for i in range(100):
        frho = A * rho ** 6 + B * rho ** 3 + C * rho ** 2 + D * rho + E * rho ** 3 * (1 + F1 * rho ** 2) * math.exp(-1*F1 * rho ** 2) - G
        dfrho = 6 * A * rho ** 5 + 3 * B * rho ** 2 + 2 * C * rho + D + E * rho ** 2 * (3 + F1 * rho ** 2 * (3 - 2 * F1 * rho ** 2)) * math.exp(-1*F1 * rho ** 2)
        rho = rho - frho / dfrho
        test = abs((rho - rhoold) / rho)
        if (test < 0.00001):
            break
        rhoold = rho
    Z = 0.27 * pr / rho / tr
    return Z


# In[27]:


# ------------------------------------------------------------------------
def Ug(p, t, sg, n2, co2, h2s):
    zm = Z(p, t, sg, n2, co2, h2s)
    mw = 28.97 * sg
    A = (9.4 + 0.02 * mw) * (t + 460) ** 1.5 / (209 + 19 * mw + (t + 460)) / 10000
    B = 3.5 + 986 / (t + 460) + 0.01 * mw
    C = 2.4 - 0.2 * B
    rho = p * mw / zm / 669.8 / (t + 460)
    Ug = A * math.exp(B * rho ** C)
    return Ug


# In[29]:


# ------------------------------------------------------------------------
# Calculation of Gas Compressibility for Miscelaneous Gases
def Cg(p, t, sg, n2, co2, h2s):
    tr = (t + 460) / Tcm(sg, n2, co2, h2s)
    pr = p / Pcm(sg, n2, co2, h2s)
    A = 0.064225133
    B = 0.53530771 * tr - 0.61232032
    C = 0.31506237 * tr - 1.0467099 - 0.57832729 / tr ** 2
    D = tr
    E = 0.68157001 / tr ** 2
    F1 = 0.68446549
    G = 0.27 * pr
    rho = 0.27 * pr / tr  # Initial guess
    rhoold = rho
    for i in range(1000):
        frho = A * rho ** 6 + B * rho ** 3 + C * rho ** 2 + D * rho + E * rho ** 3 * (1 + F1 * rho ** 2) * math.exp(-1*F1 * rho ** 2) - G
        dfrho = 6 * A * rho ** 5 + 3 * B * rho ** 2 + 2 * C * rho + D + E * rho ** 2 * (3 + F1 * rho ** 2 * (3 - 2 * F1 * rho ** 2)) * math.exp(-1*F1 * rho ** 2)
        rho = rho - frho / dfrho
        test = abs((rho - rhoold) / rho)
        if (test < 0.00001):
            break
        rhoold = rho
    zm = 0.27 * pr / rho / tr
    der = 1 / rho / tr * (5 * A * rho ** 5 + 2 * B * rho ** 2 + C * rho + 2 * E * rho ** 2 * (1 + F1 * rho ** 2 - F1 ** 2 * rho ** 4) * math.exp(-1*F1 * rho ** 2))
    cr = 1 / pr / (1 + rho / zm * der)
    Cg = cr / Pcm(sg, n2, co2, h2s)
    return Cg


# In[31]:


# ------------------------------------------------------------------------
# Calculation of Pressure from P/Z for Miscelaneous Gases
def pressure(poverz, t, sg, n2, co2, h2s):
    p = poverz
    for i in range(100):
        NewZ = Z(p, t, sg, n2, co2, h2s)
        NewP = poverz * NewZ
        check = abs((NewP - p) / NewP)
        if (check < 0.00001):
            break
        p = NewP
    pressure = NewP
    return pressure


# In[33]:


# ------------------------------------------------------------------------
# rsw = gas/water ratio (SCF/STBW)
# T = temperature of interest (degrees F)
# P = pressure (psia)
# salt = salt concentration (weight %)
# ------------------------------------------------------------------------
def Cw(t, p, rsw, salt):
    A = 3.8546 - 0.000134 * p
    B = -0.01052 + 0.000000477 * p
    C = 0.000039267 - 0.00000000088 * p
    Cw = (A + B * t + C * t ** 2) / 1000000
    Cw = Cw * (1 + 0.0089 * rsw)  # Dissolved gas correction
    Cw = Cw * ((-0.052 + 0.00027 * t - 0.00000114 * t ** 2 + 0.000000001121 * t ** 3) * salt ** 0.7 + 1)
    return Cw


# In[34]:


# ------------------------------------------------------------------------
# Gas saturated water
def Bw(t, p, salt):
    A = 0.9911 + 0.0000635 * t + 0.00000085 * t ** 2
    B = -0.000001093 - 0.000000003497 * t + 0.00000000000457 * t ** 2
    C = -0.00000000005 + 6.429E-13 * t - 1.43E-15 * t ** 2
    Bw = A + B * p + C * p ** 2
    Bw = Bw * ((0.000000051 * p + (0.00000547 - 0.000000000195 * p) * (t - 60) + (-0.0000000323 + 0.00000000000085 * p) * (t - 60) ** 2) * salt + 1)
    return Bw


# In[36]:


# ------------------------------------------------------------------------
def Uw(t, p, salt):
    Tc = 5 / 9 * (t - 32)
    tk = Tc + 273.15
    Sum = -7.419242 * (0.65 - 0.01 * Tc) ** (1 - 1)
    Sum = Sum - 0.29721 * (0.65 - 0.01 * Tc) ** (2 - 1)
    Sum = Sum - 0.1155286 * (0.65 - 0.01 * Tc) ** (3 - 1)
    Sum = Sum - 0.008685635 * (0.65 - 0.01 * Tc) ** (4 - 1)
    Sum = Sum + 0.001094098 * (0.65 - 0.01 * Tc) ** (5 - 1)
    Sum = Sum + 0.00439993 * (0.65 - 0.01 * Tc) ** (6 - 1)
    Sum = Sum + 0.002520658 * (0.65 - 0.01 * Tc) ** (7 - 1)
    Sum = Sum + 0.0005218684 * (0.65 - 0.01 * Tc) ** (8 - 1)
    psat = 22088 * math.exp((374.136 - Tc) * Sum / tk)
    Uw = 0.02414 * 10 ** (247.8 / (tk - 140)) * (1 + (p / 14.504 - psat) * 0.0000010467 * (tk - 305))
    Uw = Uw * (1 - 0.00187 * salt ** 0.5 + 0.000218 * salt ** 2.5 + (t ** 0.5 - 0.0135 * t) * (0.00276 * salt - 0.000344 * salt ** 1.5))
    return Uw


# In[37]:


# ------------------------------------------------------------------------
def RSwat(t, p, salt):
    A = 2.12 + 0.00345 * t - 0.0000359 * t ** 2
    B = 0.0107 - 0.0000526 * t + 0.000000148 * t ** 2
    C = -0.000000875 + 0.0000000039 * t - 0.0000000000102 * t ** 2
    RSwat = A + B * p + C * p ** 2
    RSwat = RSwat * (1 - (0.0753 - 0.000173 * t) * salt)
    return RSwat


# In[40]:


# ------------------------------------------------------------------------
# API = crude api gravity
# GOR = gas/oil ratio (SCF/STBO)
# T = temperature of interest (degrees F)
# SG = gas specific gravity (air=1)
# SEPT = separator temperature (degrees F)
# SEPP = separator pressure (psia)
# ------------------------------------------------------------------------
# Calculation of gas bubble point
def BP(api, gor, t, sg, sept, sepp):
    gasgs = sg * (1 + 0.00005912 * api * sept * 0.434294 * math.log(sepp / 114.7, 10))
    if (api > 30):
        c1 = 0.0178
        c2 = 1.187
        c3 = 23.931
    else:
        c1 = 0.0362
        c2 = 1.0937
        c3 = 25.724
    BP = (gor / (c1 * gasgs * exp(c3 * api / (460 + t)))) ** (1 / c2)
    return BP


# In[41]:


# ------------------------------------------------------------------------
# Calculation of solution gas/oil ratio
def RS(p, api, t, sg, sept, sepp):
    gasgs = sg * (1 + 0.00005912 * api * sept * 0.434294 * log(sepp / 114.7, 10))
    if (api > 30):
        c1 = 0.0178
        c2 = 1.187
        c3 = 23.931
    else:
        c1 = 0.0362
        c2 = 1.0937
        c3 = 25.724
    RS = c1 * gasgs * p ^ c2 * exp(c3 * (api / (t + 460)))
    return RS


# In[42]:


# ------------------------------------------------------------------------
# Calculation of oil formation volume factor
def Bo(p, pbp, api, gor, t, sg, sept, sepp):
    gasgs = sg * (1 + 0.00005912 * api * sept * 0.434294 * log(sepp / 114.7, 10))
    if (api > 30):
        c1 = 0.000467
        c2 = 0.000011
        c3 = 0.000000001337
    else:
        c1 = 0.0004677
        c2 = 0.00001751
        c3 = -0.00000001811
    Bo = 1 + c1 * gor + c2 * (t - 60) * (api / gasgs) + c3 * gor * (t - 60) * (api / gasgs)
    if (p > pbp):  # correct if pressure is greater than bubble poin
        co = Coabp(api, gor, p, t, sg, sept, sepp)
        Bo = Bo * math.exp(co * (pbp - p))
    return Bo


# In[43]:


# ------------------------------------------------------------------------
# Calculation of oil viscosity
def Uo(api, p, t, gor, pbp):
    C = 3.0324 - 0.02023 * api
    B = 10 ** C
    A = B * t ** -1.163
    uod = 10 ** A - 1
    A = 10.715 * (gor + 100) ** -0.515
    B = 5.44 * (gor + 150) ** -0.338
    uobo = A * uod ** B
    if (p < pbp):
        Uo = A * uod ** B
    else:
        uobp = A * uod ** B
        A = 2.6 * p ** 1.187 * math.exp(-0.0000898 * p - 11.513)
        Uo = uobp * (p / pbp) ** A
    return Uo


# In[44]:


# ------------------------------------------------------------------------
# Above bubble point only
def Coabp(api, gor, p, t, sg, sept, sepp):
    gasgs = sg * (1 + 0.00005912 * api * sept * 0.434294 * math.log(sepp / 114.7, 10))
    a1 = -1433
    a2 = 5
    a3 = 17.2
    a4 = -1180
    a5 = 12.61
    a6 = 10 ** 5
    Coabp = (a1 + a2 * gor + a3 * t + a4 * gasgs + a5 * api) / (a6 * p)
    return Coabp

