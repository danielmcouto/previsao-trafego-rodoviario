# Previsão de tráfego rodoviário com aprendizado de máquina

Código-fonte associado ao Trabalho de Conclusão de Curso **“Previsão de tráfego rodoviário com aprendizado de máquina e avaliação comparativa de modelos”**, de Daniel de Matos Couto.

## Objetivo

Prever, com 15 minutos de antecedência, os volumes de veículos de passeio e comerciais por faixa de rolamento e avaliar a contribuição incremental de atributos históricos, temporais, espaciais e socioeconômicos.

## Conteúdo

- `src/modelagem_previsao_trafego.py`: criação dos atributos temporais, divisão cronológica, busca de hiperparâmetros, treinamento e avaliação.
- `src/construir_variaveis_espaciais.py`: construção pública e parametrizada dos indicadores espaciais.
- `dados/README.md`: estrutura mínima das entradas.

## Divisão temporal

- Treinamento: 2022 e 2023
- Validação e seleção de hiperparâmetros: 2024
- Teste final: 2025

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python src/modelagem_previsao_trafego.py --entrada dados/base_consolidada.parquet --saida resultados
```

## Dados

Os dados brutos não são distribuídos neste repositório.

## Observação de reprodutibilidade

A reprodução numérica integral depende da mesma base utilizada no estudo, do calendário de feriados e das variáveis territoriais descritas no TCC.
