###
import re
from pathlib import Path
import pandas as pd


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
df.to_csv(
    "liga_mx_historica.csv",
    index=False,
    encoding="utf-8-sig"
)