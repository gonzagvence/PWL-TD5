#include <string>
#include <iostream>
#include <fstream>
#include "include/json.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <ctime>

using namespace std;

// Para libreria de JSON.
using namespace nlohmann;
using json = nlohmann::json;


struct Punto {
    double x;
    double y;
};


vector<double> f_en_tramo(double x0, double y0, double x1, double y1, const vector<double>& x) { //Calcula el valor de la recta
    vector<double> prediccion;
    
    double pendiente = (y1 - y0) / (x1 - x0); 
    
    for (double xi : x) {
        prediccion.push_back(pendiente * (xi - x0) + y0);
    }

    //calcula la formula provista en el PDF
    
    return prediccion;
}



// Función para estimar el error total de la recta

double calcular_error(vector<double>& vector1, vector<double>& vector2) { //Tomo como entrada el vector predicción y vector 'y' reales
    if (vector1.size() != vector2.size()) { 
        cerr << "Error: los vectores tienen diferentes longitudes." << endl;
        return -1; 
    }

    double error = 0.0;
    for (size_t i = 0; i < vector1.size(); ++i) {
        error += abs(vector1[i] - vector2[i]); //Calcula diferencia en valor absoluto de cada predicción y las vas sumando
    }
    return error; //Retornamos suma de errores
}

pair<vector<double>, vector<double>> subconjunto(vector<double>& x, vector<double>& y, double x0, double x1) {
    vector<double> sub_X;
    vector<double> sub_Y;
    size_t indice_inferior = 0;

    
    // Generamos subconjunto de x entre x0 y x1
    for (size_t i = 0; i < x.size(); ++i) {  
        if (x[i] >= x0 && x[i] <= x1) { 
            sub_X.push_back(x[i]);
            if (sub_X.size() == 1) { // Guardamos el índice inferior
            indice_inferior = i;
        }
        }
    }

    if(sub_X.size() == 0) {  // Casos vacios
        return make_pair(sub_X,sub_X);
    }

    // Calculamos el índice superior
    size_t indice_superior = indice_inferior + sub_X.size();
    
    // Generamos subconjunto de y respecto al subconjunto de x
    for (size_t i = indice_inferior; i < indice_superior; ++i) {
        sub_Y.push_back(y[i]);
    }
    
    return make_pair(sub_X, sub_Y); // Retornamos ambos subconjuntos
}



double estimar_error_y(const vector<pair<double, double>>& sol,vector<double>& x, vector<double>& y) { //Calcula el valor de la recta
    double error = 0.0; //Establecemos error = 0 para sumar cada error del tramo
    
    for (size_t i = 0; i < sol.size() - 1; ++i) { // Desde i=0 hasta sol-1 (ya que tomamos el valor i e i+1 de la sol)
        vector<double> sub_x, sub_y;
        tie(sub_x, sub_y) = subconjunto(x, y, sol[i].first, sol[i + 1].first); //Generamos subconjunto de X e Y correspondiente al tramo
        vector<double> sub_x_np(sub_x.begin(), sub_x.end()); // Convertimos a Array de "numpy"
        vector<double> prediccion = f_en_tramo(sol[i].first, sol[i].second, sol[i + 1].first, sol[i + 1].second, sub_x_np); //Calculamos la estimación para cada punto
        error += calcular_error(prediccion, sub_y); //Calculamos el error de ese tramo y sumamos al anterior
    }
    
    return error; //Retornamos la suma de errores
}

pair<double, vector<pair<double, double>>> fuerza_bruta(vector<double>& grid_x, vector<double>& grid_y, vector<double>& x, vector<double>& y, size_t N, vector<pair<double, double>>& sol_parcial) {
    if (grid_x.size() < N - sol_parcial.size()) {  //Caso base de grilla con menos de los necesarios
        return {numeric_limits<double>::max(), {}};
    } else if (sol_parcial.size() == N) { //Si el largo de la solucion es el necesario
        double error_actual = estimar_error_y(sol_parcial, x, y);
        return {error_actual, sol_parcial};
    } else {
        pair<double, vector<pair<double, double>>> sol_global = {numeric_limits<double>::max(), {}};
        if (N - sol_parcial.size() == 1) { //Si queda un solo valor de sol, tiene que ser el ultimo x de la grilla
            grid_x = {(grid_x.back())};  //Hacemos que grid_x solo sea ese valor
        }
        vector<double> grid_x_sliced = {grid_x.begin() + 1, grid_x.end()};
        for (double i : grid_y) {  //Recorremos la grilla en Y
            sol_parcial.push_back({grid_x[0], i}); //Sumamos valores de Y
            auto parcial = fuerza_bruta(grid_x_sliced, grid_y, x, y, N, sol_parcial); //Evaluamos en ese valor de Y
            if (parcial.first < sol_global.first) { //Evaluamos error ultima iteracion es mejor o peor y remplazamos
                sol_global = parcial;
            }
            sol_parcial.pop_back(); //Quitamos para seguir probando con el resto
        }
        if (sol_parcial.size() > 0) { //Si la sol_parcial no es 0 (es decir, ya se agrego x1) puedo ir evaluando opciones salteando valores de x
            auto parcial = fuerza_bruta(grid_x_sliced, grid_y, x, y, N, sol_parcial);
            if (parcial.first < sol_global.first) { //Si tienen mejor error, remplazo
                sol_global = parcial;
            }
        }
        return sol_global;
    }
}



pair<double, vector<pair<double, double>>> backtracking(vector<double>& grid_x, vector<double>& grid_y, vector<double>& x, vector<double>& y, size_t N, vector<pair<double, double>>& sol_parcial) {
    if (grid_x.size() < N - sol_parcial.size()) { //Caso base de grilla con menos de los necesarios
        return {numeric_limits<double>::max(), {}};
    } else if (sol_parcial.size() == N) {  //Si el largo de la solucion es el necesario
        double error_actual = estimar_error_y(sol_parcial, x, y);
        return {error_actual, sol_parcial};
    } else {
        pair<double, vector<pair<double, double>>> sol_global = {numeric_limits<double>::max(), {}};
        if (N - sol_parcial.size() == 1) { //Si queda un solo valor de sol, tiene que ser el ultimo x de la grilla
            grid_x = {grid_x.back()}; //Hacemos que grid_x solo sea ese valor
        }
        vector<double> grid_x_sliced = {grid_x.begin() + 1, grid_x.end()};
        for (double i : grid_y) {
            sol_parcial.push_back({grid_x[0], i}); //Sumamos valores de Y

            if (estimar_error_y(sol_parcial, x, y) < sol_global.first) {
                auto parcial = backtracking(grid_x_sliced, grid_y, x, y, N, sol_parcial); //Evaluamos en ese valor de Y
                if (parcial.first < sol_global.first) { //Evaluamos error ultima iteracion es mejor o peor y remplazamos
                    sol_global = parcial;
                }
            }
            sol_parcial.pop_back(); //Quitamos para seguir probando con el resto
        }
        if (sol_parcial.size() > 0) { //Si la sol_parcial no es 0 (es decir, ya se agrego x1) puedo ir evaluando opciones salteando valores de x
            auto parcial = backtracking(grid_x_sliced, grid_y, x, y, N, sol_parcial);
            if (parcial.first < sol_global.first) { //Si tienen mejor error, remplazo
                sol_global = parcial;
            }
        }
        return sol_global;
    }
}

//---------------------------------------------------------------------------------------------

vector<vector<vector<pair<double, vector<Punto>>>>> make_cube(size_t N, size_t M, size_t Z) {
    vector<vector<vector<pair<double, vector<Punto>>>>> memoria(N, vector<vector<pair<double, vector<Punto>>>>(Z, vector<pair<double, vector<Punto>>>(M, {numeric_limits<double>::max(), {}})));
    return memoria;
}

vector<Punto> prog_dinamica(vector<double>& grid_x, vector<double>& grid_y, vector<double>& x, vector<double>& y, size_t N) {
    N = N - 1; //Seteamos N en N-1 ya que todos los algoritmos tomamos N como número de Breakpoints.
    size_t Z = grid_x.size();
    size_t M = grid_y.size();
    auto memoria = make_cube(N, Z, M); //Realizamos un cubo de tamaño NxZxM para almacenar como memoria los resultados parciales
    double x0 = grid_x[0]; //Guardamos primer valor de grilla de X para el 'caso base' de la P. Dinámica, donde buscaremos la mejor recta desde X0 Yi a Xj Yl
    size_t i = 0;
    size_t p = 0;
    size_t k = 1;
    while (k < Z) { //Iteramos grilla de X desde el 1 (ya que el 0 esta fijado en X0)
        while (i < M) { //Iteramos 2 veces sobre grilla de Y para obtener todos los breakpoints posibles
            while (p < M) {
                double error = estimar_error_y({{x0, grid_y[i]}, {grid_x[k], grid_y[p]}}, x, y);
                if (memoria[0][k][p].first > error) {  //Si es menor al anterior guardado, remplazar
                    memoria[0][k][p] = {error, {{x0, grid_y[i]}, {grid_x[k], grid_y[p]}}};
                }
                ++p;
            }
            p = 0;
            ++i;
        }
        ++k;
        i = 0;
    }
    i = 1;
    p = 0;
    k = 2;
    size_t l = 0;
    size_t t = 0;
    while (i < N) { //Agarro desde '2 piezas' (i=1) para adelante, ya que los valores con 1 sola pieza los calcule en el caso base
        while (k < Z) { //Tomo a partir de la tercer posicion (k=2) de la grilla de X ya que la de las segunda ya estan calculados
            while (p < M) { //Itero sobre Grilla de y para probar todos los valores
                while (l < k) { //Itero sobre rango de K, ya que l nunca puede ser mayor a k. Ya que se tiene que cumplir que los X sean crecientes
                    while (t < M) { //Itero sobre grilla de y para tomar el segundo valor
                        double error = memoria[i - 1][l][t].first; //Me guardo el error de ir hasta Xl Yt
                        if (memoria[i][k][p].first > error) {
                            error += estimar_error_y({{grid_x[l], grid_y[t]}, {grid_x[k], grid_y[p]}}, x, y); //Guardamos error de ir Xk Yp como el anterior calculado más la suma de ir de ese a Xk Yp
                            if (memoria[i][k][p].first > error) { //Remplazo solución si es mejor
                                memoria[i][k][p] = {error, memoria[i - 1][l][t].second};
                                memoria[i][k][p].second.push_back({grid_x[k], grid_y[p]});
                            }
                        }
                        ++t;
                    }
                    ++l;
                    t = 0;
                }
                ++p;
                l = 0;
            }
            ++k;
            p = 0;
        }
        ++i;
        k = 2;
    }

    auto it = min_element(memoria[N - 1][Z - 1].begin(), memoria[N - 1][Z - 1].end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }); //Devuelvo el mínimo error llegado la ultima posición de grilla de X, con los breakpoints que pedimos
    return it->second;
}



//.........Para.crear.grilla................//

vector<double> gridmake(vector<double> lista, int m){  //funcion equivalente a linspace
     
    vector <double> grilla;
    float min = *min_element(lista.begin(), lista.end());
    float max = *max_element(lista.begin(), lista.end());

    float delta = (max - min) / (m-1);
    float temp = min;

    for (int i=0; i < m; ++i) {

        grilla.push_back(temp);
        temp += delta;

    }
  
    return grilla;
}

void imprimir_res(pair<double, vector<pair<double, double>>> resultado) {
    cout << "Error mínimo estimado: " << resultado.first << endl;
    cout << "Puntos correspondientes: ";
    for (const auto& punto : resultado.second) {
        cout << "(" << punto.first << ", " << punto.second << ") ";
    }
    cout << endl;
}

void imprimir_res_pd(vector<Punto> resultado){
    cout << "Puntos con menor error: ";
    for (const auto& punto : resultado) {
        cout << "(" << punto.x << ", " << punto.y << ") ";
    }
    cout << endl;
}

void crear_grillas(vector<double> x, vector<double> y, size_t n, size_t m, vector<pair<double, double>>& sol) {
    vector<double> grid_x = gridmake(x, n);
    vector<double> grid_y = gridmake(y,n);

    cout << "Brute Force: ";
    auto resultado_f = fuerza_bruta(grid_x, grid_y, x, y, m, sol);
    imprimir_res(resultado_f);
    cout << "Backtracking: ";
    auto resultado_b = backtracking(grid_x, grid_y, x, y, m, sol);
    imprimir_res(resultado_b);
    cout << "Programación Dinámica: ";
    auto resultado_pd = prog_dinamica(grid_x, grid_y, x, y, m);
    imprimir_res_pd(resultado_pd);
}

void calcular_tiempos(vector<double> x, vector<double> y, size_t n, size_t m, vector<pair<double, double>>& sol) {
    vector<double> grid_x = gridmake(x, n);
    vector<double> grid_y = gridmake(y,n);
    
    clock_t inicio = clock();
    fuerza_bruta(grid_x, grid_y, x, y, m, sol);
    clock_t final = clock();
    double tiempo_fuerza_bruta = (final - inicio) / (double)CLOCKS_PER_SEC;
    cout << "Tiempo de ejecución (fuerza bruta): " << tiempo_fuerza_bruta << " segundos" << endl;
    
    inicio = clock();
    backtracking(grid_x, grid_y, x, y, m, sol);
    final = clock();
    double tiempo_backtracking = (final - inicio) / (double)CLOCKS_PER_SEC;
    cout << "Tiempo de ejecución (backtracking): " << tiempo_backtracking << " segundos" << endl;

    inicio = clock();
    prog_dinamica(grid_x, grid_y, x, y, m);
    final = clock();
    double tiempo_pd = (final - inicio) / (double)CLOCKS_PER_SEC;
    cout << "Tiempo de ejecución (programacion dinamica): " << tiempo_pd << " segundos" << endl;
}

void titanium1(vector<pair<double, double>>& sol, size_t exp) {

    cout << "----Set de Datos 1: Titanium----\n";
    string instance_name = "../../data/titanium.json";
    cout << "Reading file " << instance_name << endl;
    ifstream input(instance_name);

    json instance;
    input >> instance;
    input.close();

    vector<double> x = instance["x"];
	vector<double> y = instance["y"];

    if( exp == 1) {
        // Experimento 1 con diferentes configuraciones de grilla y breakpoints
        cout << "Primero, con Grillas de Tamaño 6 y 5 Breakpoints:\n";
        crear_grillas(x,y,6,5, sol);

        cout << "Ahora, con Grillas de Tamaño 8 y 6 Breakpoints:\n";
        crear_grillas(x,y,8,6, sol);

        //----------------------------------------
        cout << "----Probamos aumentar mucho el tamaño de grilla y número de breakpoints----\n";

        cout << "Grillas de Tamaño 20 y 10 Breakpoints:\n";

        vector<double> grid_x = gridmake(x, 20);
        vector<double> grid_y = gridmake(y,20);

        cout << "Programación Dinámica: ";
        auto resultado_pd = prog_dinamica(grid_x, grid_y, x, y, 10);
        imprimir_res_pd(resultado_pd);
    } else if (exp == 2) {
        // Experimento 2: Performance.
        cout << "Primero, con Grillas de Tamaño 6 y 5 Breakpoints:\n";
        calcular_tiempos(x,y,6,5, sol);

        cout << "Ahora, con Grillas de Tamaño 8 y 6 Breakpoints:\n";
        calcular_tiempos(x,y,8,6, sol);
    }

    
    
}

void aspen1(vector<pair<double, double>>& sol, size_t exp) {

    cout << "----Set de Datos 2: Aspen----\n";
    string instance_name = "../../data/aspen_simulation.json";
    cout << "Reading file " << instance_name << endl;
    ifstream input(instance_name);

    json instance;
    input >> instance;
    input.close();

    vector<double> x = instance["x"];
	vector<double> y = instance["y"];

    if (exp == 1) {
        // Experimento 1 con diferentes configuraciones de grilla y breakpoints
        cout << "Primero, con Grillas de Tamaño 5 y 3 Breakpoints:\n";
        crear_grillas(x,y,5,3, sol);

        cout << "Ahora, con Grillas de Tamaño 6 y 4 Breakpoints:\n";
        crear_grillas(x,y,6,4, sol);
    } else if (exp == 2) {
        // Experimento 2: Performance.
        cout << "Primero, con Grillas de Tamaño 5 y 3 Breakpoints:\n";
        calcular_tiempos(x,y,5,3, sol);

        cout << "Ahora, con Grillas de Tamaño 9 y 3 Breakpoints:\n";
        calcular_tiempos(x,y,9,3, sol);
    }
    
    
}

void ethanol1(vector<pair<double, double>>& sol, size_t exp) {

    cout << "----Set de Datos 3: Ethanol----\n";
    string instance_name = "../../data/ethanol_water_vle.json";
    cout << "Reading file " << instance_name << endl;
    ifstream input(instance_name);

    json instance;
    input >> instance;
    input.close();

    vector<double> x = instance["x"];
	vector<double> y = instance["y"];

    if (exp == 1 ){
        // Experimento 1 con diferentes configuraciones de grilla y breakpoints
        cout << "Primero, con Grillas de Tamaño 6 y 5 Breakpoints:\n";
        crear_grillas(x,y,6,5, sol);

        cout << "Ahora, con Grillas de Tamaño 7 y 6 Breakpoints:\n";
        crear_grillas(x,y,7,6, sol);

    } else if (exp == 2) {
        // Experimento 2: Performance.
        cout << "Primero, con Grillas de Tamaño 4 y 4 Breakpoints:\n";
        calcular_tiempos(x,y,4,4, sol);

        cout << "Ahora, con Grillas de Tamaño 7 y 7 Breakpoints:\n";
        calcular_tiempos(x,y,7,7, sol);
    }
    
}

void optimistic1(vector<pair<double, double>>& sol, size_t exp) {

    cout << "----Set de Datos 4: Optimistic----\n";
    string instance_name = "../../data/optimistic_instance.json";
    cout << "Reading file " << instance_name << endl;
    ifstream input(instance_name);

    json instance;
    input >> instance;
    input.close();

    vector<double> x = instance["x"];
	vector<double> y = instance["y"];

    if(exp == 1){
        // Experimento 1 con diferentes configuraciones de grilla y breakpoints
        cout << "Primero, con Grillas de Tamaño 5 y 3 Breakpoints:\n";
        crear_grillas(x,y,5,3, sol);

        cout << "Ahora, con Grillas de Tamaño 6 y 3 Breakpoints:\n";
        crear_grillas(x,y,6,3, sol);

    } else if (exp == 2) {
        // Experimento 2: Performance.
        cout << "Primero, con Grillas de Tamaño 3 y 2 Breakpoints:\n";
        calcular_tiempos(x,y,3,2, sol);

        cout << "Ahora, con Grillas de Tamaño 10 y 2 Breakpoints:\n";
        calcular_tiempos(x,y,10,2, sol);
    }
    
}

// Experimento 1: Como varía la calidad de la predicción a medida que aumentamos el tamaño de grilla o cantidad de Breakpoints.
// Vamos a probar si a mayor número de breakpoints y/o mayor tamaño de grilla hay mejores predicciones:
void exp1() {

    vector<pair<double, double>> sol;

    cout << "-------------------------------------" << endl;
    cout << "Experimento Número 1: Calidad de Predicción\n";

    titanium1(sol,1);
    aspen1(sol,1);
    ethanol1(sol,1);
    optimistic1(sol,1);

    
}

// Experimento 2: Performance.
// Queremos analizar que variables modifican el rendimiento de nuestros algoritmos. Para esta tarea, vamos a ir variando Tamaño de grilla, cantidad de breakpoints, lenguajes y algoritmos para medir su tiempo de cómputo
void exp2() {
    vector<pair<double, double>> sol;

    cout << "-------------------------------------" << endl;
    cout << "Experimento Número 2: Performance" << endl;
    
    titanium1(sol,2);
    aspen1(sol,2);
    ethanol1(sol,2);
    optimistic1(sol,2);


}



int main(int argc, char** argv) {
    string instance_name = "../../data/titanium.json";
    cout << "Reading file " << instance_name << endl;
    ifstream input(instance_name);

    json instance;
    input >> instance;
    input.close();

    int K = instance["n"];
    int m = 6;
    int n = 6;
    int N = 5;

    //--------------------armado-de-instance-------------//
    
    vector<double> x = instance["x"];
	vector<double> y = instance["y"];

    //--------------------armado-de-grilla---------------//

    vector<double> grid_x = gridmake(x,m);
    vector<double> grid_y = gridmake(y,n);


    // Aca empieza la magia.

    // Ejemplo para guardar json.
    // Probamos guardando el mismo JSON de instance, pero en otro archivo.
    ofstream output("test_output.out");

    //////////////////FUERZA BRUTA///////////////////

    vector<pair<double, double>> sol_parcial;

    clock_t start_fuerza_bruta = clock();
    auto resultado = fuerza_bruta(grid_x, grid_y, x, y, N, sol_parcial);
    clock_t end_fuerza_bruta = clock();
    double tiempo_fuerza_bruta = (end_fuerza_bruta - start_fuerza_bruta) / (double)CLOCKS_PER_SEC;
    
    cout << "Error mínimo estimado: " << resultado.first << endl;
    cout << "Puntos correspondientes: ";
    for (const auto& punto : resultado.second) {
        cout << "(" << punto.first << ", " << punto.second << ") ";
    }
    cout << endl;

    cout << "Tiempo de ejecución (fuerza bruta): " << tiempo_fuerza_bruta << " segundos" << endl;


    //////////////////BACKTRACKING//////////////////
    
    vector<pair<double, double>> sol_parcial2;
    
    clock_t start_backtrack = clock();
    auto resultado_b = backtracking(grid_x, grid_y, x, y, N, sol_parcial2);
    clock_t end_backtrack = clock();
    double tiempo_backtrack = (end_backtrack - start_backtrack) / (double)CLOCKS_PER_SEC;
    
    cout << "Error mínimo estimado: " << resultado_b.first << endl;
    cout << "Puntos correspondientes: ";
    for (const auto& punto : resultado_b.second) {
        cout << "(" << punto.first << ", " << punto.second << ") ";
    }
    cout << endl;
    
    cout << "Tiempo de ejecución (backtracking): " << tiempo_backtrack << " segundos" << endl;


    //////////////////PROGRAMACION DINAMICA//////////////////

    clock_t start_prog_din = clock();
    auto resultado_p = prog_dinamica(grid_x, grid_y, x, y, N);
    clock_t end_prog_din = clock();
    double tiempo_prog_din = (end_prog_din - start_prog_din) / (double)CLOCKS_PER_SEC;

    // Imprimir el resultado
    cout << "Puntos con menor error: ";
    for (const auto& punto : resultado_p) {
        cout << "(" << punto.x << ", " << punto.y << ") ";
    }
    cout << endl;

    cout << "Tiempo de ejecución (programacion dinamica): " << tiempo_prog_din << " segundos" << endl;


    output << instance;
    output.close();


    //////////////////EXPERIMENTOS//////////////////

    // Experimento 1: Como varía la calidad de la predicción a medida que aumentamos el tamaño de grilla o cantidad de Breakpoints.
	// Vamos a probar si a mayor número de breakpoints y/o mayor tamaño de grilla hay mejores predicciones:
    // exp1();

    // Experimento 2: Performance.
	// Queremos analizar que variables modifican el rendimiento de nuestros algoritmos. Para esta tarea, vamos a ir variando Tamaño de grilla, cantidad de breakpoints, lenguajes y algoritmos para medir su tiempo de cómputo
    // exp2();

    return 0;
}
