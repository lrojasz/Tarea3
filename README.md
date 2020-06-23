# Tarea de Programación 3

La tarea de programación consiste de varias diferentes funciones que resuelven partes independientes de la tarea. 
El resumen de los contenidos de este repositorio se encuentra a continuación. 

## Contenidos del Repositorio GIT

Esta tarea consiste de lo siguiente: 
	1) Un archivo .ipynb utilizado para programar en el entorno Jupyter. 
	2) Un archivo .py que contiene el código en formato de Python. 
	3) Los archivos .csv que contienen los datos dados en el enunciado. 
	4) Las imagenes correspondientes a la solución de la tarea. 

### Punto 1

Este punto se resuelve con el código que se encuentra en la función de curvaDeAjuste. 
Este calcula diferentes posibles curvas para las funciones de densidad marginales. 
El de mejor ajuste se despliega independiente de los datos, y todas las imagenes se guardan en el archivo. 

```python
def curvaDeAjuste():
	[...]
	return
```

### Punto 2

Ya habiendo realizado una prueba de independencia, se debe asumir que los datos son independientes para graficar la función de densidad conjunta. 
Cuando los datos son independientes, la densidad conjunta es igual al producto de las densidades marginales.  
Por lo tanto, el producto de las funciones de densidad marginales anteriormente calculados nos otorga la función de densidad conjunta. 

```python
def fDensiddadConjunta():
	[...]
	return
```


### Punto 3

La correlacaión, covarianza y coeficiente de correlación son datos que se calculan de forma algebraica e indican de alguna forma el comportamiento de la función analizada. 
El cálculo de estos datos se obtiene utilizando el archivo "xyp.csv", para aplicar un solo for-loop y facilitar los cálculos. 
Los datos correspondientes se calculan e imprimen en la función correlacionYcovarianza.

```python
def correlacionYcovarianza():
	[...]
	return
```


### Punto 4

Las gráficas correspondientes a las funciones de densidad marginal y densidad conjunta se producen justo después del cálculo algebraico de las funciones.  
Las gráficas de densidad marginal se encuentran bajo los nombres "x_Voigt.png" y "y_Voigt.png". 
La gráfica de densidad conjunta se encuentra bajo el nombre "densidadConjunta.png".
El código correspondiente se encuentra en las siguientes funciones: 

```python
def curvaDeAjuste():
	[...]
	return
```

```python
def fDensidadConjunta():
	[...]
	return
```



## Construido con

* [Jupyter](https://jupyter.org/) - GUI utilizado

## Autores

* **Laura Rojas** - [lrojasz](https://github.com/lrojasz)

## Reconocimientos

* Se utilizaron bibliotecas de numpy y scipy para manipulación de datos y archivos.
* Se utilizaron bibliotecas de matplotlib y pl_toolkits para realizar las gráficas.
* Se utilizaron las fórmulas correspondientes a la presentación 9 - variables aleatorias múltiples para realizar cálculos numéricos.

