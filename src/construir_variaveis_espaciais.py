# -*- coding: utf-8 -*-
"""Construção sanitizada de variáveis espaciais dos ATDs.

Entradas públicas esperadas:
- pontos dos analisadores em GeoPackage ou Shapefile;
- Malha Municipal Digital do IBGE;
- Áreas Urbanizadas do Brasil, do IBGE;
- CSV municipal com indicadores socioeconômicos.

Nenhum caminho local, credencial ou endereço institucional está incorporado.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

CRS_METRICO = "EPSG:31983"  # SIRGAS 2000 / UTM 23S
RAIO_MAX_KM = 200.0
DISTANCIA_BASE_KM = 5.0
BETA = 1.0


def potencial(valores: np.ndarray, dist_km: np.ndarray,
              raio_km: float = RAIO_MAX_KM, log: bool = False) -> np.ndarray:
    x = np.log1p(valores) if log else valores
    pesos = np.where(dist_km <= raio_km, 1.0 / np.power(dist_km + DISTANCIA_BASE_KM, BETA), 0.0)
    return (pesos * x[None, :]).sum(axis=1)


def normalizar_01(s: pd.Series) -> pd.Series:
    minimo, maximo = s.min(), s.max()
    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - minimo) / (maximo - minimo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--atds", type=Path, required=True)
    p.add_argument("--municipios", type=Path, required=True)
    p.add_argument("--manchas", type=Path, required=True)
    p.add_argument("--indicadores", type=Path, required=True)
    p.add_argument("--saida", type=Path, default=Path("resultados/variaveis_espaciais.gpkg"))
    a = p.parse_args()

    atds = gpd.read_file(a.atds).to_crs(CRS_METRICO)
    municipios = gpd.read_file(a.municipios).to_crs(CRS_METRICO)
    manchas = gpd.read_file(a.manchas).to_crs(CRS_METRICO)
    indicadores = pd.read_csv(a.indicadores, sep=None, engine="python")

    if "CD_MUN" not in municipios.columns:
        raise ValueError("A malha municipal deve conter CD_MUN")
    indicadores["CD_MUN"] = indicadores["CD_MUN"].astype(str).str.zfill(7)
    municipios["CD_MUN"] = municipios["CD_MUN"].astype(str).str.zfill(7)

    # Associação de cada ATD ao município.
    atds = gpd.sjoin(atds, municipios[["CD_MUN", "NM_MUN", "geometry"]], predicate="within", how="left")

    # Seleção da maior mancha urbana por município.
    manchas_mun = gpd.sjoin(manchas, municipios[["CD_MUN", "geometry"]], predicate="intersects", how="left")
    manchas_mun["area_m2"] = manchas_mun.geometry.area
    maiores = manchas_mun.sort_values("area_m2").groupby("CD_MUN", as_index=False).tail(1)
    centros = maiores[["CD_MUN", "geometry"]].copy()
    centros["geometry"] = centros.geometry.representative_point()
    centros = centros.merge(indicadores, on="CD_MUN", how="left")

    # Matriz de distâncias ATD x centro urbano municipal.
    coord_atd = np.column_stack([atds.geometry.x, atds.geometry.y])
    coord_mun = np.column_stack([centros.geometry.x, centros.geometry.y])
    dist_km = np.sqrt(((coord_atd[:, None, :] - coord_mun[None, :, :]) ** 2).sum(axis=2)) / 1000.0

    mapa = {
        "POPULACAO_ESTIMADA": "pop",
        "PIB": "pib",
        "FROTA_CARGA": "fcarga",
        "FROTA_MISTO": "fmisto",
        "FROTA_PASSAGEIRO": "fpass",
        "ESTAB_PASSAGEIROS": "epass",
        "ESTAB_CARGA": "ecarga",
        "ATIVIDADES_AUXILIARES": "eaux",
        "ORGANIZACAO_CARGA": "eorgc",
    }
    for coluna, prefixo in mapa.items():
        if coluna not in centros.columns:
            continue
        valores = pd.to_numeric(centros[coluna], errors="coerce").fillna(0).to_numpy()
        bruto = potencial(valores, dist_km, raio_km=RAIO_MAX_KM, log=False)
        logar = potencial(valores, dist_km, raio_km=RAIO_MAX_KM, log=True)
        atds[f"{prefixo}_pot200_br"] = bruto
        atds[f"{prefixo}_pot200_01"] = normalizar_01(pd.Series(bruto)).to_numpy()
        atds[f"{prefixo}_log200_br"] = logar
        atds[f"{prefixo}_log200_01"] = normalizar_01(pd.Series(logar)).to_numpy()

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    atds.to_file(a.saida, layer="atds_variaveis_espaciais", driver="GPKG")
    atds.drop(columns="geometry").to_csv(a.saida.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"Concluído: {a.saida}")


if __name__ == "__main__":
    main()
