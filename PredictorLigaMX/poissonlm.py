###
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LigaMXDataset:
    """
    Carga todas las temporadas de Liga MX contenidas en la carpeta
    world/north-america/mexico de OpenFootball.

    Devuelve un DataFrame con:

    temporada
    torneo
    jornada
    fecha
    hora
    local
    visitante
    goles_local
    goles_visitante
    """

    def __init__(self, folder):

        self.folder = Path(folder)

    def load(self):

        partidos = []

        # Busca únicamente los archivos de Primera División
        archivos = sorted(self.folder.glob("*_mx1.txt"))

        for archivo in archivos:

            print(f"Leyendo {archivo.name}")

            partidos.extend(self._parse_file(archivo))

        df = pd.DataFrame(partidos)

        return df

    def _parse_file(self, archivo):

        temporada = None
        torneo = None
        jornada = None
        fecha = None

        partidos = []

        patron_partido = re.compile(
            r"""
            ^\s*
            (\d{1,2}:\d{2})?        # Hora (opcional)
            \s*
            (.*?)                   # Equipo local
            \s+v\s+
            (.*?)                   # Equipo visitante
            \s+
            (\d+)-(\d+)             # Marcador
            """,
            re.VERBOSE,
        )

        with open(archivo, encoding="utf8") as f:

            for linea in f:

                linea = linea.rstrip()

                if not linea:
                    continue

                
                # Temporada
                
                if linea.startswith("="):

                    temporada = linea.replace("= Liga MX", "").strip()

                    continue

              
                # Ignorar comentarios
                
                if linea.startswith("#"):

                    continue

                
                # Jornada
                
                m = re.match(r"▪\s+(.*), Matchday (\d+)", linea)

                if m:

                    torneo = m.group(1).strip()

                    jornada = int(m.group(2))

                    continue

                
                # Fecha
                
                m = re.match(
                    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+",
                    linea
                )

                if m:

                    fecha = linea.strip()

                    continue

                
                # Partido
               
                m = patron_partido.match(linea)

                if m:

                    partidos.append({

                        "temporada": temporada,

                        "torneo": torneo,

                        "jornada": jornada,

                        "fecha": fecha,

                        "hora": m.group(1),

                        "local": m.group(2).strip(),

                        "visitante": m.group(3).strip(),

                        "goles_local": int(m.group(4)),

                        "goles_visitante": int(m.group(5))

                    })

        return partidos
    
#%%
liga = LigaMXDataset(r"C:\Users\Pablo Duran\OneDrive\Documentos\PartidosWorld\world\north-america\mexico")

df = liga.load()
#%%
print(df.head(10))
print(df.shape)
print(df.isnull().sum())
print(df.temporada.nunique())
#%%
df["fecha"] = pd.to_datetime(
    df["fecha"],
    format="%a %b %d %Y"
)
df = df.sort_values("fecha").reset_index(drop=True)

#%%
print(df.local.nunique())
print("\n")
#%%
df.info()
#%%
df.describe()
#%%
equipos = sorted(set(df.local) | set(df.visitante))

print(equipos)
#%%
loc=df.groupby("local")["goles_local"].mean()
vis=df.groupby("visitante")["goles_visitante"].mean()
gol_mean=pd.concat([loc,vis], axis=1)
print(gol_mean.head(10))
#%%
gol_mean["ventaja_local"]=gol_mean["goles_local"]-gol_mean["goles_visitante"]
print(gol_mean)
print("\n")
#%%
lambda_visitante=df["goles_local"].mean()-df["goles_visitante"].mean()
print(f"Ventaja de local: {lambda_visitante}")
#%%
from scipy.stats import poisson,skellam
from scipy.optimize import minimize
media = df[["goles_local", "goles_visitante"]].mean()
poisson_pred = np.column_stack([
    [poisson.pmf(i, media["goles_local"]) for i in range(8)],
    [poisson.pmf(i, media["goles_visitante"]) for i in range(8)]
])
#%%


fig, ax = plt.subplots(figsize=(9,4))

plt.hist(
    df[['goles_local','goles_visitante']].values,
    bins=range(9),
    density=True,
    alpha=0.7,
    label=['Local','Visitante'],
    color=["#FFA07A","#20B2AA"]
)

plt.plot(
    np.arange(8),
    poisson_pred[:,0],
    '-o',
    color='#CD5C5C',
    label='Poisson Local'
)

plt.plot(
    np.arange(8),
    poisson_pred[:,1],
    '-o',
    color='#006400',
    label='Poisson Visitante'
)

plt.xlabel("Goles por partido")
plt.ylabel("Probabilidad")
plt.title("Distribución de goles Liga MX")
plt.legend()

plt.show()
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf

model_gol=pd.concat([df[['local','visitante','goles_local']].assign(home=1).rename(
            columns={'local':'equipo', 'visitante':'oponente','goles_local':'goles'}),
           df[['local','visitante','goles_visitante']].assign(home=0).rename(
            columns={'visitante':'equipo', 'local':'oponente','goles_visitante':'goles'})])

poisson_model = smf.glm(formula="goles ~ home + equipo + oponente", data=model_gol, 
                        family=sm.families.Poisson()).fit()

print(poisson_model.summary())

#%%
nuevo = pd.DataFrame({
    "equipo":["Deportivo Toluca"],
    "oponente":["Pumas UNAM"],
    "home":[1]
})

print(poisson_model.predict(nuevo))
#%%

def simular_partido(modelo, local, visitante, goles_max=10):
    loc= pd.DataFrame({"equipo": [local],"oponente":[visitante],"home":[1]})
    vis=pd.DataFrame({"equipo":visitante, "oponente":local, "home":[0]})
    goles_local=modelo.predict(loc)
    goles_visitante=modelo.predict(vis)
    marcadores=[[poisson.pmf(i,goles) for i in range(0,goles_max)] for goles in [goles_local,goles_visitante]]
    return (np.outer(np.array(marcadores[0]),np.array(marcadores[1])))
#%%
def score_matrix(prob,
                      equipo_local="Local",
                      equipo_visitante="Visitante",
                      mostrar_porcentaje=True,
                      cmap="Blues"):
   
    n = prob.shape[0]

    fig, ax = plt.subplots(figsize=(8,8))

    # Heatmap
    im = ax.imshow(prob, cmap=cmap)

    # Barra de color
    cbar = plt.colorbar(im)
    cbar.set_label("Probabilidad")

    # Diagonal
    ax.plot([-0.5,n-0.5],[-0.5,n-0.5],
            color="black",
            linewidth=2)

    # Texto en cada celda
    for i in range(n):
        for j in range(n):

            if i > j:
                color = "green"
            elif i == j:
                color = "black"
            else:
                color = "red"

            if mostrar_porcentaje:
                texto = f"{100*prob[i,j]:.1f}%"
            else:
                texto = f"{prob[i,j]:.3f}"

            ax.text(j,
                    i,
                    texto,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xlabel(f"Goles {equipo_visitante}", fontsize=12)
    ax.set_ylabel(f"Goles {equipo_local}", fontsize=12)

    ax.set_title(f"{equipo_local} vs {equipo_visitante}",
                 fontsize=16,
                 fontweight="bold")

    plt.tight_layout()
    plt.show()
plt.show()

#%%
def prob_a_american(proba_individual):

    decimal = 1/proba_individual

    if decimal >= 2:
        return round((decimal-1)*100)
    else:
        return round(-100/(decimal-1))
#%%

def win_prob(prob, local="loc", visitante='vis'):
    prob_local = np.tril(prob, -1).sum()
    prob_empate = np.trace(prob)
    prob_visitante = np.triu(prob, 1).sum()
    diccionario={local:prob_local,"empate":prob_empate,visitante:prob_visitante}
    tabla=pd.DataFrame([diccionario])
    tabla[f"Momio { local}"]=tabla[local].apply(prob_a_american)
    tabla[" Momio empate"]=tabla["empate"].apply(prob_a_american)
    tabla[f"Momio {visitante}"]=tabla[visitante].apply(prob_a_american)
    return tabla
   
#%%
partido_random=simular_partido(poisson_model, "Deportivo Toluca", "Pumas UNAM", 10)
print(score_matrix(partido_random, "Toluca", "Pumas UNAM"))
probas=win_prob(partido_random, "Toluca", "Pumas UNAM")
print(probas)



