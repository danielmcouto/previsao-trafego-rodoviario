# -*- coding: utf-8 -*-
"""Modelagem de previsão de tráfego rodoviário com horizonte de 15 minutos.

Versão pública e sanitizada do código do TCC.

Divisão temporal:
- treinamento: 2022 e 2023;
- validação e seleção de hiperparâmetros: 2024;
- teste final: 2025.

O script compara persistência, referências diária e semanal, regressão linear
e HistGradientBoosting em três configurações cumulativas de atributos (A, B e C),
separadamente para veículos de passeio e comerciais.

A base de entrada deve estar previamente consolidada em CSV ou Parquet e conter
as colunas mínimas descritas em dados/README.md. Nenhuma credencial, caminho
pessoal ou endereço interno está incorporado neste arquivo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEMENTE = 42
ANOS_TREINO = (2022, 2023)
ANO_VALIDACAO = 2024
ANO_TESTE = 2025
HORIZONTE_MIN = 15

LIMITES_POR_ANALISADOR = {
    "treino": 100_000,
    "validacao": 40_000,
    "teste": 60_000,
}
LIMITES_GLOBAIS = {
    "treino": 3_000_000,
    "validacao": 1_000_000,
    "teste": 1_500_000,
}

COLUNAS_IDENTIFICACAO = ["analisador", "faixa"]
COLUNAS_CALENDARIO_BASICO = [
    "hora", "dia_semana", "mes", "fim_de_semana",
    "hora_sen", "hora_cos", "dia_semana_sen", "dia_semana_cos",
    "mes_sen", "mes_cos",
]
COLUNAS_CALENDARIO_AMPLIADO = [
    "periodo_15min", "dia_mes", "semana_ano", "trimestre", "sabado",
    "domingo", "dia_util", "feriado", "nome_feriado", "tipo_feriado",
    "vespera_feriado", "pos_feriado", "potencial_ponte", "ferias",
    "fim_de_ano", "inicio_mes", "fim_mes", "periodo_dia",
]

# A lista histórica segue o delineamento descrito no TCC. O script cria essas
# colunas a partir das observações reais de cada analisador/faixa.
COLUNAS_HISTORICAS = [
    "valor_atual", "lag_15min", "lag_30min", "lag_45min", "lag_60min",
    "lag_2h", "lag_3h", "lag_6h", "lag_1dia", "lag_1semana",
    "lag_2semanas", "media_30min", "media_1h", "media_2h", "media_3h",
    "media_6h", "desvio_1h", "desvio_3h", "dif_15min", "dif_1h",
]

# Nomes públicos esperados para os atributos territoriais selecionados.
# Se a sua base usar nomes diferentes, altere somente estas listas.
ESPACIAIS_PASSEIO = [
    "pot_urb01", "dist_urb", "area_urb", "em_urb", "loc_pibpc",
    "pop_pot200_01", "fpass_pot100_01", "epass_pot100_01",
]
ESPACIAIS_COMERCIAL = [
    "pot_urb01", "dist_urb", "area_urb", "em_urb", "loc_pibpc",
    "pop_pot200_01", "fcarga_pot200_01", "fmisto_pot200_01",
    "ecarga_pot200_01", "eaux_pot200_01", "eorgc_pot200_01",
]

ESPACO_HIPERPARAMETROS = {
    "learning_rate": [0.03, 0.05, 0.07, 0.10],
    "max_iter": [200, 350, 500],
    "max_leaf_nodes": [31, 47, 63],
    "min_samples_leaf": [20, 40, 80],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0],
    "max_bins": [63, 127, 255],
}


def ler_base(caminho: Path) -> pd.DataFrame:
    """Lê CSV ou Parquet e padroniza os nomes essenciais."""
    if caminho.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(caminho)
    else:
        df = pd.read_csv(caminho, sep=None, engine="python", low_memory=False)
    renomear = {
        "Analisador": "analisador", "Faixa": "faixa",
        "DataHora": "data_hora", "datahora": "data_hora",
        "Passeio": "passeio", "Comercial": "comercial",
        "Dado_Projetado": "dado_projetado",
    }
    df = df.rename(columns={c: renomear.get(c, c) for c in df.columns})
    obrigatorias = {"analisador", "faixa", "data_hora", "passeio", "comercial", "dado_projetado"}
    faltantes = obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df = df.dropna(subset=["data_hora", "analisador", "faixa"])
    return df.sort_values(["analisador", "faixa", "data_hora"]).reset_index(drop=True)


def adicionar_calendario(df: pd.DataFrame, coluna_data: str) -> pd.DataFrame:
    """Cria calendário referente ao instante previsto, não ao instante atual."""
    out = df.copy()
    dt = out[coluna_data]
    out["hora"] = dt.dt.hour
    out["dia_semana"] = dt.dt.dayofweek
    out["mes"] = dt.dt.month
    out["fim_de_semana"] = (dt.dt.dayofweek >= 5).astype("int8")
    out["hora_sen"] = np.sin(2 * np.pi * (dt.dt.hour * 60 + dt.dt.minute) / 1440)
    out["hora_cos"] = np.cos(2 * np.pi * (dt.dt.hour * 60 + dt.dt.minute) / 1440)
    out["dia_semana_sen"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["dia_semana_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["mes_sen"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    out["mes_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
    out["periodo_15min"] = dt.dt.hour * 4 + dt.dt.minute // 15
    out["dia_mes"] = dt.dt.day
    out["semana_ano"] = dt.dt.isocalendar().week.astype(int)
    out["trimestre"] = dt.dt.quarter
    out["sabado"] = (dt.dt.dayofweek == 5).astype("int8")
    out["domingo"] = (dt.dt.dayofweek == 6).astype("int8")
    out["inicio_mes"] = (dt.dt.day <= 5).astype("int8")
    out["fim_mes"] = (dt.dt.day >= dt.dt.days_in_month - 4).astype("int8")
    minutos = dt.dt.hour * 60 + dt.dt.minute
    out["periodo_dia"] = pd.cut(
        minutos, bins=[-1, 359, 719, 1079, 1439],
        labels=["madrugada", "manha", "tarde", "noite"]
    ).astype("string")
    # Estas colunas dependem de um calendário externo. Se já estiverem na base,
    # os valores são preservados; caso contrário, são inicializadas de forma neutra.
    padroes = {
        "feriado": 0, "nome_feriado": "nenhum", "tipo_feriado": "nenhum",
        "vespera_feriado": 0, "pos_feriado": 0, "potencial_ponte": 0,
        "ferias": 0, "fim_de_ano": 0,
    }
    for coluna, valor in padroes.items():
        if coluna not in out.columns:
            out[coluna] = valor
    out["dia_util"] = ((out["fim_de_semana"] == 0) & (out["feriado"] == 0)).astype("int8")
    return out


def _shift_exato(grupo: pd.DataFrame, coluna: str, passos: int) -> pd.Series:
    """Defasagem válida somente quando a diferença temporal é exatamente a esperada."""
    valor = grupo[coluna].shift(passos)
    data_anterior = grupo["data_hora"].shift(passos)
    esperado = pd.Timedelta(minutes=15 * passos)
    valido = (grupo["data_hora"] - data_anterior) == esperado
    return valor.where(valido)


def criar_atributos_historicos(df: pd.DataFrame, alvo: str) -> pd.DataFrame:
    """Constrói atributos sem usar valores projetados como observações reais."""
    out = df.copy()
    out["observado"] = out[alvo].where(out["dado_projetado"].eq(0))
    partes = []
    for _, g in out.groupby(["analisador", "faixa"], sort=False):
        g = g.sort_values("data_hora").copy()
        g["valor_atual"] = g["observado"]
        for nome, passos in {
            "lag_15min": 1, "lag_30min": 2, "lag_45min": 3, "lag_60min": 4,
            "lag_2h": 8, "lag_3h": 12, "lag_6h": 24, "lag_1dia": 96,
            "lag_1semana": 672, "lag_2semanas": 1344,
        }.items():
            g[nome] = _shift_exato(g, "observado", passos)
        passado = g["observado"].shift(1)
        g["media_30min"] = passado.rolling(2, min_periods=2).mean()
        g["media_1h"] = passado.rolling(4, min_periods=4).mean()
        g["media_2h"] = passado.rolling(8, min_periods=8).mean()
        g["media_3h"] = passado.rolling(12, min_periods=12).mean()
        g["media_6h"] = passado.rolling(24, min_periods=24).mean()
        g["desvio_1h"] = passado.rolling(4, min_periods=4).std()
        g["desvio_3h"] = passado.rolling(12, min_periods=12).std()
        g["dif_15min"] = g["valor_atual"] - g["lag_15min"]
        g["dif_1h"] = g["valor_atual"] - g["lag_60min"]
        # Alvo observado exatamente 15 minutos adiante.
        g["alvo"] = g["observado"].shift(-1)
        g["data_hora_alvo"] = g["data_hora"].shift(-1)
        g["intervalo_alvo_valido"] = (g["data_hora_alvo"] - g["data_hora"]) == pd.Timedelta(minutes=15)
        partes.append(g)
    out = pd.concat(partes, ignore_index=True)
    out = out[out["intervalo_alvo_valido"] & out["alvo"].notna()].copy()
    return adicionar_calendario(out, "data_hora_alvo")


def amostrar_por_analisador(df: pd.DataFrame, limite: int, semente: int) -> pd.DataFrame:
    partes = []
    for _, g in df.groupby("analisador", sort=False):
        if len(g) > limite:
            g = g.sample(n=limite, random_state=semente)
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


def separar_temporalmente(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ano = df["data_hora_alvo"].dt.year
    partes = {
        "treino": df[ano.isin(ANOS_TREINO)].copy(),
        "validacao": df[ano.eq(ANO_VALIDACAO)].copy(),
        "teste": df[ano.eq(ANO_TESTE)].copy(),
    }
    for nome, dados in partes.items():
        dados = amostrar_por_analisador(dados, LIMITES_POR_ANALISADOR[nome], SEMENTE)
        limite_global = LIMITES_GLOBAIS[nome]
        if len(dados) > limite_global:
            dados = dados.sample(n=limite_global, random_state=SEMENTE)
        partes[nome] = dados.sort_values("data_hora_alvo").reset_index(drop=True)
    return partes


def colunas_configuracao(alvo: str, configuracao: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numericas = list(COLUNAS_HISTORICAS + COLUNAS_CALENDARIO_BASICO)
    categoricas = list(COLUNAS_IDENTIFICACAO)
    if configuracao in {"B", "C"}:
        for c in COLUNAS_CALENDARIO_AMPLIADO:
            (categoricas if c in {"nome_feriado", "tipo_feriado", "periodo_dia"} else numericas).append(c)
    if configuracao == "C":
        espaciais = ESPACIAIS_PASSEIO if alvo == "passeio" else ESPACIAIS_COMERCIAL
        numericas.extend([c for c in espaciais if c in df.columns])
    numericas = [c for c in numericas if c in df.columns]
    categoricas = [c for c in categoricas if c in df.columns]
    return numericas, categoricas


def preprocessador(numericas: Sequence[str], categoricas: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), list(numericas)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), list(categoricas)),
    ])


def pipeline_linear(numericas: Sequence[str], categoricas: Sequence[str]) -> Pipeline:
    return Pipeline([
        ("pre", preprocessador(numericas, categoricas)),
        ("modelo", LinearRegression(n_jobs=-1)),
    ])


def pipeline_hgb(numericas: Sequence[str], categoricas: Sequence[str], parametros: dict) -> Pipeline:
    return Pipeline([
        ("pre", preprocessador(numericas, categoricas)),
        ("modelo", HistGradientBoostingRegressor(
            loss="squared_error", early_stopping=False,
            random_state=SEMENTE, **parametros
        )),
    ])


def metricas(y: Iterable[float], previsao: Iterable[float]) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(previsao, dtype=float), 0, None)
    denominador = np.abs(y).sum()
    return {
        "EAM": mean_absolute_error(y, p),
        "REQM": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p),
        "EPAP": np.abs(y - p).sum() / denominador if denominador else np.nan,
    }


def selecionar_hgb(treino: pd.DataFrame, validacao: pd.DataFrame,
                   numericas: Sequence[str], categoricas: Sequence[str], n_iter: int = 24) -> Tuple[dict, pd.DataFrame]:
    candidatos = ParameterSampler(ESPACO_HIPERPARAMETROS, n_iter=n_iter, random_state=SEMENTE)
    historico, melhor, melhor_eam = [], None, np.inf
    Xtr, ytr = treino[list(numericas) + list(categoricas)], treino["alvo"]
    Xva, yva = validacao[list(numericas) + list(categoricas)], validacao["alvo"]
    for ordem, params in enumerate(candidatos, 1):
        modelo = pipeline_hgb(numericas, categoricas, params)
        modelo.fit(Xtr, ytr)
        met = metricas(yva, modelo.predict(Xva))
        historico.append({"ordem": ordem, **params, **met})
        if met["EAM"] < melhor_eam:
            melhor_eam, melhor = met["EAM"], dict(params)
    return melhor, pd.DataFrame(historico).sort_values("EAM")


def avaliar_referencias(teste: pd.DataFrame, alvo_nome: str) -> List[dict]:
    saida = []
    referencias = {
        "Persistência": "valor_atual",
        "Dia anterior": "lag_1dia",
        "Semana anterior": "lag_1semana",
    }
    for nome, coluna in referencias.items():
        mascara = teste[coluna].notna()
        saida.append({
            "Alvo": alvo_nome, "Modelo": nome, "Configuração": "Referência",
            "Partição": "Teste", "n": int(mascara.sum()),
            **metricas(teste.loc[mascara, "alvo"], teste.loc[mascara, coluna]),
        })
    return saida


def executar_alvo(base: pd.DataFrame, alvo: str, pasta_saida: Path) -> pd.DataFrame:
    print(f"Preparando alvo: {alvo}")
    dados = criar_atributos_historicos(base, alvo)
    partes = separar_temporalmente(dados)
    resultados = avaliar_referencias(partes["teste"], alvo)

    # Seleção de hiperparâmetros realizada apenas com o Modelo A e 2024.
    num_a, cat_a = colunas_configuracao(alvo, "A", dados)
    melhores, tuning = selecionar_hgb(partes["treino"], partes["validacao"], num_a, cat_a, 24)
    tuning.to_csv(pasta_saida / f"tuning_hgb_{alvo}.csv", index=False, encoding="utf-8-sig")
    (pasta_saida / f"hiperparametros_{alvo}.json").write_text(
        json.dumps(melhores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for configuracao in ["A", "B", "C"]:
        numericas, categoricas = colunas_configuracao(alvo, configuracao, dados)
        colunas = numericas + categoricas
        for nome_modelo, modelo in [
            ("Regressão linear", pipeline_linear(numericas, categoricas)),
            ("HistGradientBoosting", pipeline_hgb(numericas, categoricas, melhores)),
        ]:
            print(f"Treinando {nome_modelo} {configuracao} - {alvo}")
            modelo.fit(partes["treino"][colunas], partes["treino"]["alvo"])
            for nome_particao in ["validacao", "teste"]:
                bloco = partes[nome_particao]
                pred = np.clip(modelo.predict(bloco[colunas]), 0, None)
                resultados.append({
                    "Alvo": alvo, "Modelo": nome_modelo,
                    "Configuração": configuracao,
                    "Partição": nome_particao.capitalize(), "n": len(bloco),
                    **metricas(bloco["alvo"], pred),
                })
            joblib.dump(
                modelo,
                pasta_saida / f"{nome_modelo.lower().replace(' ', '_').replace('ã','a').replace('ç','c')}_{configuracao}_{alvo}.joblib"
            )
    return pd.DataFrame(resultados)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelagem pública do TCC de previsão de tráfego")
    parser.add_argument("--entrada", type=Path, required=True, help="CSV ou Parquet consolidado")
    parser.add_argument("--saida", type=Path, default=Path("resultados"), help="Pasta de saída")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    base = ler_base(args.entrada)
    resultados = []
    for alvo in ["passeio", "comercial"]:
        resultados.append(executar_alvo(base, alvo, args.saida))
    consolidado = pd.concat(resultados, ignore_index=True)
    consolidado.to_csv(args.saida / "metricas_modelagem.csv", index=False, encoding="utf-8-sig")
    print(f"Concluído. Resultados em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
