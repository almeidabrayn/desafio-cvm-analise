# 📊 Análise Preditiva de Captação em Fundos de Investimento

## 🎯 Objetivo do Projeto
Desenvolver um modelo preditivo para identificar os principais fatores que influenciam a captação líquida futura em fundos de investimento brasileiros, utilizando dados públicos da Comissão de Valores Mobiliários (CVM).

![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📈 Principais Resultados
- **Acurácia do modelo**: 70.7%
- **ROC-AUC**: 0.787 (capacidade preditiva moderada)
- **Principal fator**: Tamanho do fundo (70.4% de importância)
- **Período analisado**: Janeiro a Junho de 2025
- **Fundos analisados**: 100 fundos com maior histórico

## 🏗️ Estrutura do Projeto
Desafio CVM/
├── src/
│ └── desafio.cvm.py # Pipeline completo de ETL e modelagem
├──   dados processados/ # Resultados e visualizações
│ ├── dados_processados.csv # Dataset com features e target
│ ├── graficos_*.png # Visualizações da análise
│ └── importancia_features.csv # Importância das variáveis
├── Estudos/ # Documentação e relatórios
├── README.md # Este arquivo
├── requirements.txt # Dependências do projeto
├── .gitattributes # Configuração Git LFS - Usado após o arquivo do vídeo passar de 100 MB
└── .gitignore # Arquivos ignorados pelo Git

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8 ou superior
- 4GB de RAM mínimo
- Conexão com internet para download dos dados

### Instalação
```bash
# Clone o repositório
git clone https://github.com/almeidabrayn/desafio-cvm-analise.git
cd desafio-cvm-analise

# Instale as dependências
pip install -r requirements.txt
