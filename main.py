from ReimannSolvers import Exact, LFEuler, HLL, HLLC


print("Test 1 : Start")
# Test 1
exact_solution1 = Exact(CFL=0.9, gamma=1.4, tag="Test1-Exact")

lf_euler_solution1 = LFEuler(CFL=0.9, gamma=1.4, tag="Test1-Lax")

hll_solver1 = HLL(CFL=0.9, gamma=1.4, tag="Test1-HLL")

e_parm1 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.3,
    "x_num": 100,
    "timeout": 0.2,
    "d_left": 1.0,
    "d_right": 0.125,
    "u_left": 0.75,
    "u_right": 0.0,
    "p_left": 1.0,
    "p_right": 0.1,
}

l_parm1 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.12,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1000.0,
    "p_right": 0.01,
}


hll_parm1 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.3,
    "x_num": 300,
    "timeout": 0.2,
    "d_left": 1.0,
    "d_right": 0.125,
    "u_left": 0.75,
    "u_right": 0.0,
    "p_left": 1.0,
    "p_right": 0.1,
}


exact_solution1.set_init(parm=e_parm1)
exact_solution1.solver()
exact_solution1.results(show=False)


lf_euler_solution1.set_init(parm=l_parm1, dur=600)
lf_euler_solution1.solver()
lf_euler_solution1.results(show=False)

hll_solver1.set_init(parm=hll_parm1, dur=600)
hll_solver1.solver()
hll_solver1.results(show=False)

print("Test 1 : End")
# Test 2


print("Test 2 : Start")
exact_solution2 = Exact(CFL=0.9, gamma=1.4, tag="Test2-Exact")

lf_euler_solution2 = LFEuler(CFL=0.9, gamma=1.4, tag="Test2-Lax")

hll_solver2 = HLL(CFL=0.9, gamma=1.4, tag="Test2-HLL")

e_parm2 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.15,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": -2.0,
    "u_right": 2.0,
    "p_left": 0.4,
    "p_right": 0.4,
}

l_parm2 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.15,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": -2.00,
    "u_right": 2.0,
    "p_left": 0.4,
    "p_right": 0.4,
}


hll_parm2 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.15,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": -2.0,
    "u_right": 2.0,
    "p_left": 0.4,
    "p_right": 0.4,
}


exact_solution2.set_init(parm=e_parm2)
exact_solution2.solver()
exact_solution2.results(show=False)


lf_euler_solution2.set_init(parm=l_parm2, dur=500)
lf_euler_solution2.solver()
lf_euler_solution2.results(show=False)

hll_solver2.set_init(parm=hll_parm2, dur=600)
hll_solver2.solver()
hll_solver2.results(show=False)

print("Test 2 : End")
# Test 3

print("Test 3 : Start")
exact_solution3 = Exact(CFL=0.9, gamma=1.4, tag="Test3-Exact")

lf_euler_solution3 = LFEuler(CFL=0.9, gamma=1.4, tag="Test3-Lax")

hll_solver3 = HLL(CFL=0.9, gamma=1.4, tag="Test3-HLL")

e_parm3 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.012,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1000.0,
    "p_right": 0.01,
}

l_parm3 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.012,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1000.0,
    "p_right": 0.01,
}


hll_parm3 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 300,
    "timeout": 0.012,
    "d_left": 1.0,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1000.0,
    "p_right": 0.01,
}


exact_solution3.set_init(parm=e_parm3)
exact_solution3.solver()
exact_solution3.results(show=False)


lf_euler_solution3.set_init(parm=l_parm3, dur=600)
lf_euler_solution3.solver()
lf_euler_solution3.results(show=False)

hll_solver3.set_init(parm=hll_parm3, dur=600)
hll_solver3.solver()
hll_solver3.results()


print("Test 3 : End")
# Test 4

print("Test 4 : Start")
exact_solution4 = Exact(CFL=0.9, gamma=1.4, tag="Test4-Exact")

lf_euler_solution4 = LFEuler(CFL=0.9, gamma=1.4, tag="Test4-Lax")

hll_solver4 = HLL(CFL=0.9, gamma=1.4, tag="Test4-HLL")

e_parm4 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.4,
    "x_num": 300,
    "timeout": 0.035,
    "d_left": 5.99924,
    "d_right": 5.99242,
    "u_left": 19.5975,
    "u_right": -6.19633,
    "p_left": 460.894,
    "p_right": 46.0950,
}

l_parm4 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.4,
    "x_num": 300,
    "timeout": 0.035,
    "d_left": 5.99924,
    "d_right": 5.99242,
    "u_left": 19.5975,
    "u_right": -6.19633,
    "p_left": 460.894,
    "p_right": 46.0950,
}


hll_parm4 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.4,
    "x_num": 300,
    "timeout": 0.035,
    "d_left": 5.99924,
    "d_right": 5.99242,
    "u_left": 19.5975,
    "u_right": -6.19633,
    "p_left": 460.894,
    "p_right": 46.0950,
}


exact_solution4.set_init(parm=e_parm4)
exact_solution4.solver()
exact_solution4.results(show=False)


lf_euler_solution4.set_init(parm=l_parm4, dur=600)
lf_euler_solution4.solver()
lf_euler_solution4.results(show=False)

hll_solver4.set_init(parm=hll_parm4, dur=600)
hll_solver4.solver()
hll_solver4.results()

print("Test 4 : End")

# Test 5

print("Test 5 : Start")

hll_solver5 = HLL(CFL=0.9, gamma=1.4, tag="Test5-HLL")
hllc_solver5 = HLLC(CFL=0.9, gamma=1.4, tag="Test5-HLLC")

hll_parm5 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 100,
    "timeout": 2.0,
    "d_left": 1.4,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1.0,
    "p_right": 1.0,
}


hllc_parm5 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.5,
    "x_num": 100,
    "timeout": 2.0,
    "d_left": 1.4,
    "d_right": 1.0,
    "u_left": 0.0,
    "u_right": 0.0,
    "p_left": 1.0,
    "p_right": 1.0,
}


hll_solver5.set_init(parm=hll_parm5, dur=600)
hll_solver5.solver()
hll_solver5.results()


hllc_solver5.set_init(parm=hllc_parm5, dur=300)
hllc_solver5.solver()
hllc_solver5.results()

print("Test 5 : End")

# Test 5

print("Test 5 : Start")

hll_solver6 = HLL(CFL=0.9, gamma=1.4, tag="Test6-HLL")
hllc_solver6 = HLLC(CFL=0.9, gamma=1.4, tag="Test6-HLLC")

hll_parm6 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.4,
    "x_num": 100,
    "timeout": 2.0,
    "d_left": 1.4,
    "d_right": 1.0,
    "u_left": 0.1,
    "u_right": 0.1,
    "p_left": 1.0,
    "p_right": 1.0,
}


hllc_parm6 = {
    "x_left": 0.0,
    "x_right": 1.0,
    "x0": 0.4,
    "x_num": 100,
    "timeout": 2.0,
    "d_left": 1.4,
    "d_right": 1.0,
    "u_left": 0.1,
    "u_right": 0.1,
    "p_left": 1.0,
    "p_right": 1.0,
}


hll_solver6.set_init(parm=hll_parm6, dur=600)
hll_solver6.solver()
hll_solver6.results()


hllc_solver6.set_init(parm=hllc_parm6, dur=300)
hllc_solver6.solver()
hllc_solver6.results()

print("Test 6 : End")
