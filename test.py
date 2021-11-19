epsilon = 0.1
xl = 0
xu = 4
interval = xu-xl
iteration = 0

while interval > epsilon:
    print(f'Iteration = {iteration}')
    x_mid = (xu+xl)/2
    deriv = 6*x_mid**2 + 3 - 6*x_mid - 2*x_mid**3
    if deriv == 0:
        break
    elif deriv > 0:
        xl = x_mid
        iteration += 1
    elif deriv < 0:
        xu = x_mid
        iteration += 1
    interval = abs(xu-xl)
    print(f'f prime = {deriv} at x = {x_mid}')
    print(f'lower bound = {xl}, upper bound = {xu}')
    print(f'interval = {interval}')



