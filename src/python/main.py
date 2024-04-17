import json
import numpy as np
import time

BIG_NUMBER = 1e10 # Revisar si es necesario.

def main():

	# Ejemplo para leer una instancia con json
	instance_name = "aspen_simulation.json"
	filename = "././data/" + instance_name
	with open(filename) as f:
		instance = json.load(f)
	
	K = instance["n"]
	m = 6
	n = 6
	N = 5 # Tome N como numero de Breakpoints
	
	# Ejemplo para definir una grilla de m x n.
	grid_x = np.linspace(min(instance["x"]), max(instance["x"]), num=m, endpoint=True)
	grid_y = np.linspace(min(instance["y"]), max(instance["y"]), num=n, endpoint=True)
	x = instance['x']
	y = instance['y']

	#--------------------- FUNCIONES AUXILIARES (PARA ARMADO FUNCIONES PRINCIPALES)---------------------------------------------

	def f_en_tramo(x0,y0,x1,y1,x): #Calcula el valor de la recta
		return (((y1-y0)/(x1-x0))*(x-x0)) + y0 # Simplemente calcula con la formula provista en el PDF

	def estimar_error_y(sol,x, y): #Calcula el valor de la recta
		i = 0  # Establecemos i = 0
		error = 0 # Establecemos error = 0 para sumar cada error del tramo

		while i < (len(sol) - 1): # Desde i=0 hasta sol-1 (ya que tomamos el valor i e i+1 de la sol)
			sub_x, sub_y = subconjunto(x, y, sol[i][0], sol[i+1][0]) # Generamos subconjunto de X e Y correspondiente al tramo
			sub_x = np.array(sub_x) # Convertimos a Array de numpy
			prediccion = f_en_tramo(sol[i][0], sol[i][1], sol[i+1][0], sol[i+1][1], sub_x) # Calculamos la estimación para cada punto
			error = error + calcular_error(prediccion,sub_y) # Calculamos el error de ese tramo y sumamos al anterior
			i = i + 1 #Pasamos de valor de i
		return error # Retornamos la suma de errores
	
	def calcular_error(vector1, vector2): # Tomo como entrada el vector predicción y vector 'y' reales
		diferencia = np.abs(vector1 - vector2) # Calcula diferencia en valor absoluto de cada predicción 
		error = np.sum(diferencia) # Sumamos todos los errores
		return error # Retornamos suma de errores
	
	def subconjunto(x,y,x0,x1):
		sub_X = [x_i for x_i in x if x0 <= x_i <= x1] # Generamos subconjunto de x entre X_0 y X_1
		indice_inferior = x.index(sub_X[0]) # Conseguimos indice inferior
		indice_superior = indice_inferior + len(sub_X) # De igual manera para el indice Superior
		sub_Y = y[indice_inferior:indice_superior] # Generamos subconjunto de y respecto al subconjunto de x
		return sub_X, sub_Y # Retornamos ambos subconjuntos

	#--------------------------------------------------------------------------------------------------------------------
	
	#-- FUERZA BRUTA ----------------------------------------------------------------------------------------------------
 

	sol = [] #Inicializo Solucion como vacio

	def fuerza_bruta(grid_x, grid_y, x, y, N, sol_parcial):
		if(len(grid_x) < N - (len(sol_parcial)) ): #Caso base de grilla con menos de los necesarios
			return {'error':1e10}
		
		elif(len(sol_parcial) == N): # Si el largo de la solucion es el necesario
			error_actual = estimar_error_y(sol_parcial, x, y)
			return {'error':error_actual,'puntos':sol_parcial.copy()}
		else:
			sol_global = {'error':1e10}
			if(N - (len(sol_parcial)) == 1): # Si queda un solo valor de sol, tiene que ser el ultimo x de la grilla
				grid_x = [grid_x[-1]] # Hacemos que grid_x solo sea ese valor
			for i in grid_y: # Recorremos la grilla en Y
				sol_parcial.append((grid_x[0], i)) # Sumamos valores de Y
				parcial = fuerza_bruta(grid_x[1:], grid_y, x, y, N, sol_parcial) # Evaluamos en ese valor de Y
				if(parcial['error'] < sol_global['error']): # Evaluamos error ultima iteracion es mejor o peor y remplazamos
					sol_global = parcial
				sol_parcial.pop() # Quitamos para seguir probando con el resto
			if(len(sol_parcial) > 0): # Si la sol_parcial no es 0 (es decir, ya se agrego x1) puedo ir evaluando opciones salteando valores de x
				parcial = fuerza_bruta(grid_x[1:], grid_y, x, y, N, sol_parcial)
			if(parcial['error'] < sol_global['error']): # Si tienen mejor error, remplazo
				sol_global = parcial
		return  sol_global
	
	inicio_fuerzabruta = time.time()
	print('Brute Force ',fuerza_bruta(grid_x,grid_y,x,y,5,sol))
	fin_fuerzabruta = time.time()
	tiempo_fuerzabruta = fin_fuerzabruta - inicio_fuerzabruta
	print(tiempo_fuerzabruta)
	
	def backtracking(grid_x, grid_y, x, y, N, sol_parcial):
		if(len(grid_x) < N - (len(sol_parcial)) ): #Caso base de grilla con menos de los necesarios
			return {'error':1e10}
		
		elif(len(sol_parcial) == N): # Si el largo de la solucion es el necesario
			error_actual = estimar_error_y(sol_parcial, x, y)
			return {'error':error_actual,'puntos':sol_parcial.copy()}
		else:
			sol_global = {'error':1e10}
			if(N - (len(sol_parcial)) == 1): # Si queda un solo valor de sol, tiene que ser el ultimo x de la grilla
				grid_x = [grid_x[-1]] # Hacemos que grid_x solo sea ese valor
			for i in grid_y: # Recorremos la grilla en Y
				sol_parcial.append((grid_x[0], i)) # Sumamos valores de Y
				if(estimar_error_y(sol_parcial, x, y) < sol_global['error']):
					parcial = backtracking(grid_x[1:], grid_y, x, y, N, sol_parcial) # Evaluamos en ese valor de Y
					if(parcial['error'] < sol_global['error']): # Evaluamos error ultima iteracion es mejor o peor y remplazamos
						sol_global = parcial
				sol_parcial.pop() # Quitamos para seguir probando con el resto
			if(len(sol_parcial) > 0): # Si la sol_parcial no es 0 (es decir, ya se agrego x1) puedo ir evaluando opciones salteando valores de x
				parcial = backtracking(grid_x[1:], grid_y, x, y, N, sol_parcial)
			if(parcial['error'] < sol_global['error']): # Si tienen mejor error, remplazo
				sol_global = parcial
		return  sol_global

	inicio_backtrack = time.time()
	print('Backtracking: ',backtracking(grid_x,grid_y,x,y,6,sol))
	fin_backtrack = time.time()
	tiempo_backtrack = fin_backtrack - inicio_backtrack
	print(tiempo_backtrack)
	#--------------------------------------------------------------------------------------------------------------------------
	def make_cube(N, M, Z):
		matriz = [[[{'error':1e10, 'puntos':[(None,None),(None, None)]} for _ in range(Z)] for _ in range(M)] for _ in range(N)]
		return matriz
	
	def prog_dinamica(grid_x, grid_y, x, y, N):
		N = N - 1
		Z = len(grid_x)
		M = len(grid_y)
		memoria = make_cube(N,Z,M)
		x0 = grid_x[0]
		for k in range(1,Z):
			for i in range(M):
				for p in range(M):
					if memoria[0][k][p]['error'] > estimar_error_y([(x0, grid_y[i]), (grid_x[k], grid_y[p])], x, y):
						memoria[0][k][p] = {'error': estimar_error_y([(x0, grid_y[i]), (grid_x[k], grid_y[p])], x, y),'puntos': [(x0, grid_y[i]), (grid_x[k], grid_y[p])]}
		for i in range(1,N):
			for k in range(2,Z):
				for p in range(M):
					for l in range(k):
						for t in range(M):
							error = memoria[i - 1][l][t]['error']
							if(memoria[i][k][p]['error'] > error):
								error = memoria[i - 1][l][t]['error'] + estimar_error_y([(grid_x[l], grid_y[t]), (grid_x[k], grid_y[p])], x, y)
								if(memoria[i][k][p]['error'] > error):
									memoria[i][k][p]['error'] = error
									puntos = memoria[i - 1][l][t]['puntos']
									puntos.append((grid_x[k], grid_y[p]))
									memoria[i][k][p]['puntos'] = puntos.copy()
									puntos.pop()
		diccionario_menor_error = min(memoria[N - 1][Z - 1], key=lambda x: x["error"])
		return diccionario_menor_error
		

	inicio_progri = time.time()
	print('Programación Dinámica: ' , prog_dinamica(grid_x,grid_y,x,y,4))
	fin_progri = time.time()
	tiempo_progri = fin_progri - inicio_progri
	print(tiempo_progri)


	#----------------
	''' # Experimentando nuestras primeras soluciones
	def prog_dinamica_(grid_x,grid_y,x,y,N):
		N = N-1
		Z = len(grid_x)
		M = len(grid_y)
		memoria = make_cube(N, Z, M)
		x0 = grid_x[0]
		i = 0
		p = 0
		k = 1
		while k < Z:
			while i < M:
				while p < M:
					if(memoria[0][k][p]['error'] > estimar_error_y([(x0,grid_y[i]),(grid_x[k],grid_y[p])], x, y)):
						memoria[0][k][p] = {'error':estimar_error_y([(x0,grid_y[i]),(grid_x[k],grid_y[p])], x, y),'puntos':[(x0,grid_y[i]),(grid_x[k],grid_y[p])]}
					p = p + 1
				p = 0
				i = i + 1
			k = k + 1
			i = 0
		i = 1
		p = 0
		k = 2
		l = 0
		t = 0
		while i < N:
			while k < Z:
				while p < M:
					while l < k:
						while t < M:
							error = memoria[i-1][l][t]['error'] 
							if(memoria[i][k][p]['error'] > error):
								error = memoria[i-1][l][t]['error'] + estimar_error_y([(grid_x[l],grid_y[t]), (grid_x[k],grid_y[p])], x, y)
								if(memoria[i][k][p]['error'] > error):
									memoria[i][k][p]['error'] = error
									puntos = memoria[i-1][l][t]['puntos']
									puntos.append((grid_x[k],grid_y[p]))
									memoria[i][k][p]['puntos'] = puntos.copy()
									puntos.pop()
							t = t + 1
						l = l + 1
						t = 0
					p = p + 1
					l = 0
				k = k + 1
				p = 0
			i = i + 1
			k = 2

		diccionario_menor_error = min(memoria[N-1][Z-1], key=lambda x: x["error"])
		return diccionario_menor_error
 

	inicio_progri = time.time()
	print('Programacion Dinámica Bottom Up: ',prog_dinamica_(grid_x,grid_y,x,y,5))
	fin_progri = time.time()
	tiempo_progri = fin_progri - inicio_progri
	print(tiempo_progri)
	'''
	#----------------

	best = {}
	best['sol'] = [None]*(N+1)
	best['obj'] = BIG_NUMBER
	
	# Posible ejemplo (para la instancia titanium) de formato de solucion, y como exportarlo a JSON.
	# La solucion es una lista de tuplas (i,j), donde:
	# - i indica el indice del punto de la discretizacion de la abscisa
	# - j indica el indice del punto de la discretizacion de la ordenada.
	best['sol'] = [(0, 0), (1, 0), (2, 0), (3, 2), (4, 0), (5, 0)]
	best['obj'] = 5.927733333333339

	# Represetnamos la solucion con un diccionario que indica:
	# - n: cantidad de breakpoints
	# - x: lista con las coordenadas de la abscisa para cada breakpoint
	# - y: lista con las coordenadas de la ordenada para cada breakpoint
	solution = {}
	solution['n'] = len(best['sol'])
	solution['x'] = [grid_x[x[0]] for x in best['sol']]
	solution['y'] = [grid_y[x[1]] for x in best['sol']]
	solution['obj'] = best['obj']

	# Se guarda el archivo en formato JSON
	with open('solution_' + instance_name, 'w') as f:
		json.dump(solution, f)

	
if __name__ == "__main__":
	main()