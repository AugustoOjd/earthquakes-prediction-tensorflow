import pandas as pd

todas_tablas = pd.read_html('http://www.iris.washington.edu/latin_am/evlist.phtml?region=mundo')
table_csv_done = pd.read_csv('earthquakes.csv', index_col=0)


df = table_csv_done.rename(columns={'FECHA - HORA (UTC) día-mes-año hora:min:seg': 'FECHA (UTC)', 'MAG explicado abajo': 'MAG', 'PROF km': 'KM', 'LOCALIDAD haga clic para ver mapa': 'LOCALIDAD'})

# df.to_csv('earthquakes_edited.csv')
# print(df)

# ---------------------------------------------------------------------------------------------

create_df_train = pd.read_csv('earthquakes_edited.csv', index_col=0)

data_to_train = create_df_train.iloc[0:80]
data_to_test = create_df_train[80:101]

# AHORA CREAMOS LOS DOS ARCHIVOS CSV PARA TRAIN Y TEST EL MODELO

data_to_train.to_csv('model/earthquakes_train_model.csv')
data_to_test.to_csv('model/earthquakes_test_model.csv')

print('finish')