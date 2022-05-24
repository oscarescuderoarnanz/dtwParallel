# Dynamic Time Warping 

This package allows to measure the similarity between two time sequences, i.e. it finds the optimal alignment between two time-dependent sequences. It allows the calculation for univariate time series and multivariate time series. In addition, in the case of univariate time series, it incorporates the possibility of visualising the result in order to obtain a better understanding of the method used. On the other hand, the calculation of the DTW distance for multivariate time series can be parallelised. 

## Usage

The package can be used in three different ways:

1) Downloading the repository:

    a) Entering the parameters by terminal.

    ```
    python3 dtwParallel/__main__.py -x 1 2 3 -y 1 1 1 -d "euclidean" -t "d" False False
    ```
    
    Otros ejemplos que incluyen no necesariamente introducir todos los parámetros son:
    
    ```
    python3 dtwParallel/__main__.py -x 1 2 3 -y 1 1 1
    ```
    
    ```
    python3 dtwParallel/__main__.py -x 1 2 3 -y 1 1 1 -d "euclidean"
    ```
   
    b) By file.
    
	Ejemplo de cálculo para series temporales univariantes. 

        ```
        python3 dtwParallel/__main__.py Libro1.csv
        ```
        
	Ejemplo de cálculo para series temporales multivariantes. 

        ```
        python3 dtwParallel/__main__.py X_train.npy
        ```

2) Making use of the API. 



## Requirements

* Python >= 3.6.1 and <= 3.8


Note that you should have also the following packages installed in your system:
- numpy
- pandas
- matplotlib
- seaborn
- gower
- setuptools
- scipy

## Installation
```
pip install -i https://test.pypi.org/simple/ dtwParallel
```

## Documentation
Para cualquier modificación de los parámetros por defecto se puede llevar a cabo la configuración del fichero configuration.py. 

## Examples


## Running tests


## License

Licensed under GBSD 2-Clause License.
