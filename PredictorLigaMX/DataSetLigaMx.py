# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:01:06 2026

@author: Pablo Duran
"""

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
        anio_actual = None

        partidos = []

        patron_partido = re.compile(
            r"""
            ^\s*
            (\d{1,2}:\d{2})?          # Hora (opcional)
            \s*
            (.*?)                     # Equipo local
            \s+v\s+
            (.*?)                     # Equipo visitante
            \s+
            (\d+)-(\d+)               # Goles
            (?:\s+.*)?                # Cualquier información extra (HT, pen., etc.)
            \s*$
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

               
                # Comentarios
               

                if linea.startswith("#"):
                    continue

                # Jornada
                

                m = re.match(r"\s*▪\s+(.*), Matchday (\d+)", linea)

                if m:

                    torneo = m.group(1).strip()
                    jornada = int(m.group(2))
                    continue

                
                # Fecha
               

                if re.match(r"\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", linea):

                    fecha_txt = linea.strip()

                    # ¿La fecha trae año?
                    m_anio = re.search(r"\b(20\d{2})\b", fecha_txt)

                    if m_anio:

                        anio_actual = int(m_anio.group(1))

                    else:

                        if anio_actual is None:
                            raise ValueError(
                                f"No se pudo determinar el año para la fecha: {fecha_txt}"
                            )

                        fecha_txt += f" {anio_actual}"

                    # datetime
                    fecha = pd.to_datetime(
                        fecha_txt,
                        format="%a %b %d %Y"
                    )

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
df.to_csv('LigaMXHistorica.csv', index=False)