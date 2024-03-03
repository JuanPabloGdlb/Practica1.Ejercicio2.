import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


# Cargar el conjunto de datos
data = pd.read_csv("spheres1d10.csv")

# Definir el porcentaje de datos para entrenamiento y prueba
train_percentage = 0.8
test_percentage = 0.2

# Definir la cantidad de particiones
num_partitions = 5

for i in range(num_partitions):
    print(f"Partición {i+1}:")
    
    # Dividir los datos en entrenamiento y prueba
    train_data, test_data = train_test_split(data, test_size=test_percentage)
    
    # Separar características (X) de etiquetas (y)
    X_train = train_data.drop(columns=['1'])
    y_train = train_data['1']
    X_test = test_data.drop(columns=['1'])
    y_test = test_data['1']
    
    # Inicializar y entrenar el perceptrón
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    
    # Evaluar el perceptrón en los datos de prueba
    y_pred = perceptron.predict(X_test)
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo en la partición de prueba: {accuracy:.2f}")
    print()

# Definir las técnicas a comparar
tecnicas = ["Particionamiento 80-20", "Validación cruzada k-fold (k=5)"]

# Definir los archivos de datos
archivos = ["spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"]

for archivo in archivos:
    print(f"Dataset: {archivo}")
    data = pd.read_csv(archivo)
    
    for tecnica in tecnicas:
        print(f"Técnica: {tecnica}")
        accuracies = []

        if tecnica == "Particionamiento 80-20":
            # Particionamiento de entrenamiento y prueba (80% - 20%)
            X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)
            
            # Inicializar y entrenar el perceptrón
            perceptron = Perceptron()
            perceptron.fit(X_train, y_train)
            
            # Evaluar el perceptrón en los datos de prueba
            y_pred = perceptron.predict(X_test)
            
            # Calcular la precisión del modelo
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        elif tecnica == "Validación cruzada k-fold (k=5)":
            # Validación cruzada k-fold (k=5)
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(data):
                X_train, X_test = data.iloc[train_index, :-1], data.iloc[test_index, :-1]
                y_train, y_test = data.iloc[train_index, -1], data.iloc[test_index, -1]
                
                # Inicializar y entrenar el perceptrón
                perceptron = Perceptron()
                perceptron.fit(X_train, y_train)
                
                # Evaluar el perceptrón en los datos de prueba
                y_pred = perceptron.predict(X_test)
                
                # Calcular la precisión del modelo y agregarla a la lista
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
        
        # Calcular el promedio de precisión para la técnica actual
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Precision promedio: {avg_accuracy:.2f}")
    
    print()
