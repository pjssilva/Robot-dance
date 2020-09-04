"""
Simple script to retrieve the official information of the Covid-19 epidemic in the 
state of São Paulo, Brazil, and convert it to a format usable by Robot Dance.

One of the main objectives it to group the data in DRS instead of cities.
"""

import urllib.request as request
import os.path as path
import pandas as pd
import numpy as np

##################### Data
data_url = "https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/dados_covid_sp.csv"
icu_url = "https://raw.githubusercontent.com/seade-R/dados-covid-sp/master/data/plano_sp_leitos_internacoes.csv"

# Subregions of the São Paulo metropolitan area
sub_rmsp = {
    "Sub região norte - RMSP": ["Caieiras", "Cajamar", "Francisco Morato", "Franco Da Rocha", "Mairiporã"],
    "Sub região leste - RMSP": ["Arujá", "Biritiba Mirim", "Ferraz De Vasconcelos", "Guararema", "Guarulhos", 
                                "Itaquaquecetuba", "Mogi Das Cruzes", "Poá", "Salesópolis", "Santa Isabel",
                                "Suzano"],
    "Sub região sudeste - RMSP": ["Diadema", "Mauá", "Ribeirão Pires", "Rio Grande Da Serra", "Santo André",
                                  "São Bernardo Do Campo", "São Caetano Do Sul"],
    "Sub região sudoeste - RMSP": ["Cotia", "Embu Das Artes", "Embu-Guaçu", "Itapecerica Da Serra", "Juquitiba", 
                                   "São Lourenço Da Serra", "Taboão Da Serra", "Vargem Grande Paulista"],
    "Sub região oeste - RMSP": ["Barueri", "Carapicuíba", "Itapevi", "Jandira", "Osasco", 
                                "Pirapora Do Bom Jesus", "Santana De Parnaíba"],
    "Mun. São Paulo": ["São Paulo"]
}

# A date that I know is in the data (so I can select a single date to get information that
# does not change in time, like populuation)
valid_date = pd.to_datetime("2020-03-01")

# name of the mobility matrix file
mobility_name = "move_mat_new.csv"
mobility_cities = "names_new.csv"

##################### Script

# Retrieve number of ICU units per DRS
head, tail = path.split(icu_url)
request.urlretrieve(icu_url, filename=tail)

pre_icu = pd.read_csv(tail, sep=";", parse_dates=[0], decimal=",")
last_date = pre_icu["datahora"].max()
pre_icu = pre_icu[pre_icu["datahora"] == last_date].copy()
pre_icu.set_index("nome_drs", inplace=True)

# Map the names of DRS from one file to the other
map_names = {
    "Grande SP Norte": "Sub região norte - RMSP",
    "Grande SP Leste": "Sub região leste - RMSP",
    "Grande SP Sudeste": "Sub região sudeste - RMSP",
    "Grande SP Sudoeste": "Sub região sudoeste - RMSP",
    "Grande SP Oeste": "Sub região oeste - RMSP",
    "Município de São Paulo": "Mun. São Paulo"
}
for d in pre_icu.index:
    if d.startswith("DRS"):
        map_names[d] = d[7:]

# Create a dictionary with the desired information
icu = {}
for d in map_names.keys():
    if d.startswith("Estado"):
        continue
    icu[map_names[d]] = pre_icu.loc[d, "total_covid_uti_mm7d"] / pre_icu.loc[d, "pop"]

# Download epidemy data
heat, tail = path.split(data_url)
request.urlretrieve(data_url, filename=tail)

# Read data for processing
sp = pd.read_csv(tail, sep=";", parse_dates=[4])
sp["nome_munic"] = [s.title() for s in sp["nome_munic"]]
# Cut out ignored nome_munic
sp = sp[sp["nome_munic"] != "Ignorado"]
sp.set_index("nome_munic", inplace=True)
# Replace the original DRS by the subdivisions used in the São Paulo Covid-19 reponse plan.
for name_sub, cities_sub in sub_rmsp.items():
    for c in cities_sub:
        sp.loc[c, "nome_drs"] = name_sub
# Add a column with ICU capacity
sp["icu_capacity"] = 1.0
for c in sp.index.unique():
    sp.loc[c, "icu_capacity"] = icu[sp.loc[c, "nome_drs"][0]]

drs = sp.groupby(["nome_drs", "datahora"]).sum()[["casos", "casos_novos", "obitos", "obitos_novos"]]
# Add a column with ICU capacity
drs["icu_capacity"] = 1.0
for c in drs.index.unique():
    drs.loc[c, "icu_capacity"] = icu[c[0]]

state_inf = sp[sp["datahora"] == valid_date]
sp_cities = pd.Series(state_inf.index)

# Read the mobility matrix, if available.
if path.isfile(mobility_name):
    mobility_matrix = pd.read_csv(mobility_name, header=None, sep=" ")
    cities_names = pd.read_csv(mobility_cities, header=None)
    # Cut the matrix to see only the desired cities
    cities_names = [i.title() for i in cities_names[0]]
    mobility_matrix.index = cities_names
    mobility_matrix.columns = cities_names
    mobility_matrix = mobility_matrix.loc[sp_cities, sp_cities]
else:
    n_cities = len(sp_cities)
    mobility_matrix = pd.DataFrame(np.zeros((n_cities, n_cities)), index=sp_cities,
        columns=sp_cities)

# Adjust the mobility matrix grouping by DRS
# Multiply the columns by polulation
population = state_inf["pop"].copy()
mobility_matrix = mobility_matrix.mul(population, axis=1)

# Group by DRS
mobility_matrix["nome_drs"] = state_inf["nome_drs"]
mobility_matrix = mobility_matrix.groupby("nome_drs").sum()
mobility_matrix = mobility_matrix.T
mobility_matrix["nome_drs"] = state_inf["nome_drs"]
mobility_matrix = mobility_matrix.groupby("nome_drs").sum()
mobility_matrix = mobility_matrix.T

# Divide back each column by the DRS population
pop_drs = state_inf.groupby("nome_drs").sum()["pop"]
mobility_matrix = mobility_matrix.div(pop_drs, axis=1)

# Adapt the format to the one use by robot dance
sp.reset_index(inplace=True)
sp = sp.loc[: ,["nome_munic", "datahora", "casos", "pop", "icu_capacity"]]
sp["state"] = "SP"
sp.columns = ["city", "date", "confirmed", "estimated_population_2019", "icu_capaciy", "state"]
sp.to_csv("../covid_with_cities.csv")

for d in pop_drs.index:
    drs.loc[d, "pop"] = pop_drs[d]
drs.reset_index(inplace=True)
drs = drs.loc[: ,["nome_drs", "datahora", "casos", "pop", "icu_capacity"]]
drs["state"] = "SP"
drs.columns = ["city", "date", "confirmed", "estimated_population_2019", "icu_capacity", "state"]
drs.to_csv("../covid_with_drs.csv")
mobility_matrix.to_csv("../drs_mobility.csv")

