from scipy.optimize import fsolve
import math

def guess():
    # Známé hodnoty ve stupních
    b_deg = -20.6
    B0_deg = -3.0
    P_deg = 2.1
    L0_deg = 139.5
    l_deg = 161.3
    rho_deg = 27.5

    # Konverze na radiány
    b = math.radians(b_deg)
    B0 = math.radians(B0_deg)
    P = math.radians(P_deg)
    L0 = math.radians(L0_deg)
    l = math.radians(l_deg)
    rho=math.radians(rho_deg)

    # Definice funkce pro rovnice
    def equations1(x):
        Q=math.radians(x)
        eq1 = b - math.asin(math.sin(B0) * math.cos(rho) + math.cos(B0) * math.sin(rho) * math.cos(P - Q))
        return eq1
    def equations2(x):
        Q=math.radians(x)
        eq2 = l - (math.asin((math.sin(rho) * math.sin(P - Q)) / math.cos(b)) + L0)
        return eq2

    # Počáteční odhad pro rho a Q
    initial_guesses = [0,60,120,180,240,300,360]

    for initial_guess in initial_guesses:
        # Řešení rovnic
        result = fsolve(equations1, initial_guess, maxfev=2000)
        print(f"Q = {result}")

        result = fsolve(equations2, initial_guess, maxfev=2000)
        print(f"Q = {result} \n")

'''
B0 = math.radians(5.30)
P = math.radians(25.73)
L0 = math.radians(261.02)
rho=math.atan(-9.18/28.2)'''

B0 = math.radians(6.9)
P = math.radians(25.5)
L0 = math.radians(61.4)
rho=math.asin(0.4)
Q=math.radians(330)
b = math.asin(math.sin(B0) * math.cos(rho) + math.cos(B0) * math.sin(rho) * math.cos(P - Q))
print(f"b = {math.degrees(b)}")

b=math.radians(34)
l = (math.asin((math.sin(rho) * math.sin(P - Q)) / math.cos(b)) + L0)
print(f"l = {math.degrees(l)}")
