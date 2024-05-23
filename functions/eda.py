
def explore_data(data):
    print("Dimensiones del dataset:", data.shape)
    print("\nPrimeras filas del dataset:")
    print(data.head())
    print("\nResumen estadístico del dataset:")
    print(data.describe())
    print("\nNúmero de valores nulos por columna:")
    print(data.isnull().sum())
