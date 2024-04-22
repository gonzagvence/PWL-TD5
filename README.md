# PWL-TD5

## Motivación
La aproximación de datos mediante funciones es una herramienta fundamental en diversas disciplinas, ya que permite modelar de manera eficiente fenómenos complejos que exhiben diversos comportamientos. Aplicaciones en ingeniería, química, física, economía y negocios, entre otras, son algunos ejemplos de estas áreas. En muchos casos, los datos se originan o correponden a fenómenos complejos, correspondientes a funciones desconocidas, y con variaciones bruscas o cambios abruptos, con comportamientos intrínsecamente no lineales, no cóncavos ni convexos. Contar con buenas aproximaciones de las funciones subyacentes resulta en un input clave para procesos más complejos.
Las funciones lineales continuas a trozos (PWL, continuous piecewise linear) constituyen una herramienta versátil para estos contextos. Intuitivamente, el domino de valores se particiona en una colección de segmentos, y dentro de cada segmento los datos se aproximan mediante una función lineal. Para garantizar continuidad, funciones correspondientes a segmentos contiguos deben coincidir en el punto de unión.

## Introducción:

En el siguiente trabajo, se nos pidió como objetivo central crear algoritmos los cuales fueran capaces de aproximar un conjunto de datos mediante funciones piecewise linear. Para esto, teníamos diferentes conjuntos de datos para trabajar (Aspen, Ethanol, Titanium y Optimistic) y diferentes parámetros como grillas y N cantidad de breakpoints (o de piezas, dependiendo cómo definamos la función).

Las grillas y los breakpoints son muy importantes, ya que las grillas nos darán el espacio de X e Y discretizado, de manera de que podamos tomar puntos para formar cada función lineal. El armado de estas funciones lineales, va a depender de estas grillas y los puntos del conjunto de datos. Nuestro objetivo, explicado brevemente, será encontrar funciones lineales que se 'conecten' entre breakpoints de tal manera de que:

    1. Cumpla con la restricción de máximos breakpoints.
    2. Minimice el error con el conjunto de datos.

## Compilación cpp:
  
  ```make //desde la carpeta src```
  
## Ejecución cpp: 

  ```./pwl_fit```
