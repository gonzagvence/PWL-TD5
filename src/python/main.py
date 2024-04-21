import json
import numpy as np
import time


def main():

	# Ejemplo para leer una instancia con json
	instance_name = "aspen_simulation.json"
	filename = "././data/" + instance_name
	with open(filename) as f:
		instance = json.load(f)
	
	K = instance["n"]
	m = 5
	n = 5
	N = 3 # Tome N como numero de Breakpoints
	
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
		if(sub_X==[]): # Casos vacios
			return [],[]
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
	print('Brute Force ',fuerza_bruta(grid_x,grid_y,x,y,N,sol))
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
	print('Backtracking: ',backtracking(grid_x,grid_y,x,y,N,sol))
	fin_backtrack = time.time()
	tiempo_backtrack = fin_backtrack - inicio_backtrack
	print(tiempo_backtrack)
	#--------------------------------------------------------------------------------------------------------------------------
	def make_cube(N, M, Z):
		matriz = [[[{'error':1e10, 'puntos':[(None,None),(None, None)]} for _ in range(Z)] for _ in range(M)] for _ in range(N)]
		return matriz
	
	def prog_dinamica(grid_x, grid_y, x, y, N):
		N = N - 1 #Seteamos N en N-1 ya que todos los algoritmos tomamos N como número de Breakpoints.
		Z = len(grid_x) 
		M = len(grid_y)
		memoria = make_cube(N,Z,M) # Realizamos un cubo de tamaño NxZxM para almacenar como memoria los resultados parciales
		x0 = grid_x[0] # Guardamos primer valor de grilla de X para el 'caso base' de la P. Dinámica, donde buscaremos la mejor recta desde X0 Yi a Xj Yl
		for k in range(1,Z): # #Iteramos grilla de X desde el 1 (ya que el 0 esta fijado en X0)
			for i in range(M): #Iteramos 2 veces sobre grilla de Y para obtener todos los breakpoints posibles
				for p in range(M):
					if memoria[0][k][p]['error'] > estimar_error_y([(x0, grid_y[i]), (grid_x[k], grid_y[p])], x, y): #Si es menor al anterior guardado, remplazar
						memoria[0][k][p] = {'error': estimar_error_y([(x0, grid_y[i]), (grid_x[k], grid_y[p])], x, y),'puntos': [(x0, grid_y[i]), (grid_x[k], grid_y[p])]}
		for i in range(1,N): # Agarro desde '2 piezas' para adelante, ya que los valores con 1 sola pieza los calcule en el caso base
			for k in range(2,Z): # Tomo a partir de la tercer posicion de la grilla de X ya que la de las segunda ya estan calculados
				for p in range(M): # Itero sobre Grilla de y para probar todos los valores
					for l in range(k): #Itero sobre rango de K, ya que l nunca puede ser mayor a k. Ya que se tiene que cumplir que los X sean crecientes
						for t in range(M): #Itero sobre grilla de y para tomar el segundo valor
							error = memoria[i - 1][l][t]['error'] # Me guardo el error de ir hasta Xl Yt 
							if(memoria[i][k][p]['error'] > error):
								error = memoria[i - 1][l][t]['error'] + estimar_error_y([(grid_x[l], grid_y[t]), (grid_x[k], grid_y[p])], x, y) # Guardamos error de ir Xk Yp como el anterior calculado más la suma de ir de ese a Xk Yp
								if(memoria[i][k][p]['error'] > error): #Remplazo solución si es mejor
									memoria[i][k][p]['error'] = error
									puntos = memoria[i - 1][l][t]['puntos']
									puntos.append((grid_x[k], grid_y[p]))
									memoria[i][k][p]['puntos'] = puntos.copy()
									puntos.pop()
		diccionario_menor_error = min(memoria[N - 1][Z - 1], key=lambda x: x["error"]) #Devuelvo el mínimo error llegado la ultima posición de grilla de X, con los breakpoints que pedimos
		return diccionario_menor_error
		

	inicio_progri = time.time()
	print('Programación Dinámica: ' , prog_dinamica(grid_x,grid_y,x,y,N))
	fin_progri = time.time()
	tiempo_progri = fin_progri - inicio_progri
	print(tiempo_progri)


	#----------------
	print('-------------------------------------')
	#----------------
	#Experimento 1: Como varía la calidad de la predicción a medida que aumentamos el tamaño de grilla o cantidad de Breakpoints.
	# Vamos a probar si a mayor número de breakpoints y/o mayor tamaño de grilla hay mejores predicciones:
	def exp1():
		print('Experimento Número 1: Calidad de Predicción')
		print('----Set de Datos 1: Titanium----')
		instance_name = "titanium.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 6 y 5 Breakpoints:')
		grid_x_6 = np.linspace(min(instance["x"]), max(instance["x"]), num=6, endpoint=True)
		grid_y_6 = np.linspace(min(instance["y"]), max(instance["y"]), num=6, endpoint=True)
		print('Brute Force: ',fuerza_bruta(grid_x_6,grid_y_6,x,y,5,sol))
		print('Backtracking: ',backtracking(grid_x_6,grid_y_6,x,y,5,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_6,grid_y_6,x,y,5))
		grid_x_8 = np.linspace(min(instance["x"]), max(instance["x"]), num=8, endpoint=True)
		grid_y_8 = np.linspace(min(instance["y"]), max(instance["y"]), num=8, endpoint=True)
		print('Ahora, con Grillas de Tamaño 8 y 6 Breakpoints:')
		print('Brute Force: ',fuerza_bruta(grid_x_8,grid_y_8,x,y,6,sol))
		print('Backtracking: ',backtracking(grid_x_8,grid_y_8,x,y,6,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_8,grid_y_8,x,y,6))
		print('----Set de Datos 2: Aspen----')
		instance_name = "aspen_simulation.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 5 y 3 Breakpoints:')
		grid_x_5 = np.linspace(min(instance["x"]), max(instance["x"]), num=5, endpoint=True)
		grid_y_5 = np.linspace(min(instance["y"]), max(instance["y"]), num=5, endpoint=True)
		print('Brute Force: ',fuerza_bruta(grid_x_5,grid_y_5,x,y,3,sol))
		print('Backtracking: ',backtracking(grid_x_5,grid_y_5,x,y,3,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_5,grid_y_5,x,y,3))
		grid_x_6 = np.linspace(min(instance["x"]), max(instance["x"]), num=6, endpoint=True)
		grid_y_6 = np.linspace(min(instance["y"]), max(instance["y"]), num=6, endpoint=True)
		print('Ahora, con Grillas de Tamaño 6 y 4 Breakpoints:')
		print('Brute Force: ',fuerza_bruta(grid_x_6,grid_y_6,x,y,4,sol))
		print('Backtracking: ',backtracking(grid_x_6,grid_y_6,x,y,4,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_6,grid_y_6,x,y,4))
		print('----Set de Datos 3: Ethanol----')
		instance_name = "ethanol_water_vle.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 6 y 5 Breakpoints:')
		grid_x_6 = np.linspace(min(instance["x"]), max(instance["x"]), num=6, endpoint=True)
		grid_y_6 = np.linspace(min(instance["y"]), max(instance["y"]), num=6, endpoint=True)
		print('Brute Force: ',fuerza_bruta(grid_x_6,grid_y_6,x,y,5,sol))
		print('Backtracking: ',backtracking(grid_x_6,grid_y_6,x,y,5,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_6,grid_y_6,x,y,5))
		grid_x_7 = np.linspace(min(instance["x"]), max(instance["x"]), num=7, endpoint=True)
		grid_y_7 = np.linspace(min(instance["y"]), max(instance["y"]), num=7, endpoint=True)
		print('Ahora, con Grillas de Tamaño 7 y 6 Breakpoints:')
		print('Brute Force: ',fuerza_bruta(grid_x_7,grid_y_7,x,y,6,sol))
		print('Backtracking: ',backtracking(grid_x_7,grid_y_7,x,y,6,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_7,grid_y_7,x,y,6))
		print('----Set de Datos 4: Optimistic----')
		instance_name = "optimistic_instance.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 5 y 3 Breakpoints:')
		grid_x_5 = np.linspace(min(instance["x"]), max(instance["x"]), num=5, endpoint=True)
		grid_y_5 = np.linspace(min(instance["y"]), max(instance["y"]), num=5, endpoint=True)
		print('Brute Force: ',fuerza_bruta(grid_x_5,grid_y_5,x,y,3,sol))
		print('Backtracking: ',backtracking(grid_x_5,grid_y_5,x,y,3,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_5,grid_y_5,x,y,3))
		grid_x_6 = np.linspace(min(instance["x"]), max(instance["x"]), num=6, endpoint=True)
		grid_y_6 = np.linspace(min(instance["y"]), max(instance["y"]), num=6, endpoint=True)
		print('Ahora, con Grillas de Tamaño 6 y 3 Breakpoints:')
		print('Brute Force: ',fuerza_bruta(grid_x_6,grid_y_6,x,y,3,sol))
		print('Backtracking: ',backtracking(grid_x_6,grid_y_6,x,y,3,sol))
		print('Programación Dinámica: ' , prog_dinamica(grid_x_6,grid_y_6,x,y,3))
		#----------------------------------------
		print('----Probamos aumentar mucho el tamaño de grilla y número de breakpoints----')
		print('----Set de Datos 1: Titanium----')
		instance_name = "titanium.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Grillas de Tamaño 20 y 10 Breakpoints:')
		grid_x_20 = np.linspace(min(instance["x"]), max(instance["x"]), num=20, endpoint=True)
		grid_y_20 = np.linspace(min(instance["y"]), max(instance["y"]), num=20, endpoint=True)
		print('Programación Dinámica: ' , prog_dinamica(grid_x_20,grid_y_20,x,y,10))
		#---------------------------------

		return
	
	#Experimento 2: Performance.
	#Queremos analizar que variables modifican el rendimiento de nuestros algoritmos. Para esta tarea, vamos a ir variando Tamaño de grilla, cantidad de breakpoints, lenguajes y algoritmos para medir su tiempo de cómputo
	def exp2():
		print('Experimento Número 2: Perfomance')
		print('----Set de Datos 1: Titanium----')
		instance_name = "titanium.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 6 y 5 Breakpoints:')
		grid_x_6 = np.linspace(min(instance["x"]), max(instance["x"]), num=6, endpoint=True)
		grid_y_6 = np.linspace(min(instance["y"]), max(instance["y"]), num=6, endpoint=True)
		inicio = time.time()
		fuerza_bruta(grid_x_6,grid_y_6,x,y,5,sol)
		fin = time.time()
		print('Brute Force: ',fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_6,grid_y_6,x,y,5,sol)
		fin = time.time()
		print('Backtracking: ',fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_6,grid_y_6,x,y,5)
		fin = time.time()
		print('Programación Dinámica: ' ,fin-inicio, ' segundos' )
		#----------------------------------------------------------
		grid_x_8 = np.linspace(min(instance["x"]), max(instance["x"]), num=8, endpoint=True)
		grid_y_8 = np.linspace(min(instance["y"]), max(instance["y"]), num=8, endpoint=True)
		print('Ahora, con Grillas de Tamaño 8 y 6 Breakpoints:')
		inicio = time.time()
		fuerza_bruta(grid_x_8,grid_y_8,x,y,6,sol)
		fin = time.time()
		print('Brute Force: ',fin-inicio,' segundos')
		inicio = time.time()
		backtracking(grid_x_8,grid_y_8,x,y,6,sol)
		fin = time.time()
		print('Backtracking: ',fin-inicio,' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_8,grid_y_8,x,y,6)
		fin = time.time()
		print('Programación Dinámica: ' ,fin-inicio,' segundos' )
		#-----------------------------------------------------------
		print('----Set de Datos 2: Aspen----')
		instance_name = "aspen_simulation.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 5 y 3 Breakpoints:')
		grid_x_5 = np.linspace(min(instance["x"]), max(instance["x"]), num=5, endpoint=True)
		grid_y_5 = np.linspace(min(instance["y"]), max(instance["y"]), num=5, endpoint=True)
		inicio = time.time()
		fuerza_bruta(grid_x_5,grid_y_5,x,y,3,sol)
		fin = time.time()
		print('Brute Force: ',fin-inicio,' segundos')
		inicio = time.time()
		backtracking(grid_x_5,grid_y_5,x,y,3,sol)
		fin = time.time()
		print('Backtracking: ',fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_5,grid_y_5,x,y,3)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos')
		#-----------------------------------------------------------
		grid_x_9 = np.linspace(min(instance["x"]), max(instance["x"]), num=9, endpoint=True)
		grid_y_9 = np.linspace(min(instance["y"]), max(instance["y"]), num=9, endpoint=True)
		print('Ahora, con Grillas de Tamaño 9 y 3 Breakpoints:')
		inicio = time.time()
		fuerza_bruta(grid_x_9,grid_y_9,x,y,3,sol)
		fin = time.time()
		print('Brute Force: ', fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_9,grid_y_9,x,y,3,sol)
		fin = time.time()
		print('Backtracking: ', fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_9,grid_y_9,x,y,3)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos')
		print('----Set de Datos 3: Ethanol----')
		instance_name = "ethanol_water_vle.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 4 y 4 Breakpoints:')
		grid_x_4 = np.linspace(min(instance["x"]), max(instance["x"]), num=4, endpoint=True)
		grid_y_4 = np.linspace(min(instance["y"]), max(instance["y"]), num=4, endpoint=True)
		inicio = time.time()
		fuerza_bruta(grid_x_4,grid_y_4,x,y,4,sol)
		fin = time.time()
		print('Brute Force: ', fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_4,grid_y_4,x,y,4,sol)
		fin = time.time()
		print('Backtracking: ', fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_4,grid_y_4,x,y,4)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos' )
		#-----------------------------------------------------------
		grid_x_7 = np.linspace(min(instance["x"]), max(instance["x"]), num=7, endpoint=True)
		grid_y_7 = np.linspace(min(instance["y"]), max(instance["y"]), num=7, endpoint=True)
		print('Ahora, con Grillas de Tamaño 7 y 7 Breakpoints:')
		inicio = time.time()
		fuerza_bruta(grid_x_7,grid_y_7,x,y,7,sol)
		fin = time.time()
		print('Brute Force: ', fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_7,grid_y_7,x,y,7,sol)
		fin = time.time()
		print('Backtracking: ', fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_7,grid_y_7,x,y,7)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos')
		print('----Set de Datos 4: Optimistic----')
		instance_name = "optimistic_instance.json"
		filename = "././data/" + instance_name
		with open(filename) as f:
			instance = json.load(f)
		x = instance['x']
		y = instance['y']
		print('Primero, con Grillas de Tamaño 3 y 2 Breakpoints:')
		grid_x_3 = np.linspace(min(instance["x"]), max(instance["x"]), num=3, endpoint=True)
		grid_y_3 = np.linspace(min(instance["y"]), max(instance["y"]), num=3, endpoint=True)
		inicio = time.time()
		fuerza_bruta(grid_x_3,grid_y_3,x,y,2,sol)
		fin = time.time()
		print('Brute Force: ', fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_3,grid_y_3,x,y,2,sol)
		fin = time.time()
		print('Backtracking: ', fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_3,grid_y_3,x,y,2)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos' )
		#-----------------------------------------------------------
		grid_x_10 = np.linspace(min(instance["x"]), max(instance["x"]), num=10, endpoint=True)
		grid_y_10 = np.linspace(min(instance["y"]), max(instance["y"]), num=10, endpoint=True)
		print('Ahora, con Grillas de Tamaño 10 y 2 Breakpoints:')
		inicio = time.time()
		fuerza_bruta(grid_x_10,grid_y_10,x,y,2,sol)
		fin = time.time()
		print('Brute Force: ', fin-inicio, ' segundos')
		inicio = time.time()
		backtracking(grid_x_10,grid_y_10,x,y,2,sol)
		fin = time.time()
		print('Backtracking: ', fin-inicio, ' segundos')
		inicio = time.time()
		prog_dinamica(grid_x_10,grid_y_10,x,y,2)
		fin = time.time()
		print('Programación Dinámica: ' , fin-inicio, ' segundos')
		return
	




	
	#-----------------
	#Ejemplo de como guardamos y exportamos la solución: (elegimos todo lo que queremos de la solución, menos el archivo que está en la parte superior)
	grid_x = np.linspace(min(instance["x"]), max(instance["x"]), num=20, endpoint=True)
	grid_y = np.linspace(min(instance["y"]), max(instance["y"]), num=20, endpoint=True)
	best = prog_dinamica(grid_x,grid_y,x,y,10)
	best['n'] = len(best['puntos'])

	# Se guarda el archivo en formato JSON
	with open('solucion_' + instance_name, 'w') as f:
		json.dump(best, f)

	
if __name__ == "__main__":
	main()