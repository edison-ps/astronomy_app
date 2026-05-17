# AstroPy App — Computação Astronômica com Testes Unitários

Aplicação científica em Python voltada para **astronomia computacional**, desenvolvida como base para demonstração de boas práticas de **engenharia de software aplicada à ciência**: arquitetura modular, tipagem estática, tratamento de exceções e cobertura de testes unitários com `pytest`.

---

## Objetivo

Demonstrar como aplicações de software científico podem garantir:

- **Precisão matemática** via testes determinísticos contra valores de referência
- **Confiabilidade científica** com tolerâncias numéricas explícitas (`pytest.approx`)
- **Estabilidade do código** com cobertura de *edge cases* e inversibilidade
- **Testabilidade de integrações externas** com *mocks* de APIs HTTP

---

## Estrutura do Projeto

```
astronomy_app/
│
├── core/                        # Lógica científica central
│   ├── coordinates.py           # Conversões entre sistemas de coordenadas
│   ├── julian_date.py           # Datas Julianas e tempo sideral
│   ├── distance.py              # Distâncias estelares e módulo de distância
│   ├── magnitude.py             # Fotometria e lei de Pogson
│   └── orbital.py               # Mecânica orbital (leis de Kepler)
│
├── services/
│   └── nasa_api.py              # Cliente HTTP para APIs públicas da NASA
│
├── utils/
│   └── conversions.py           # Conversões de unidades e ângulos
│
├── tests/
│   ├── test_conversions.py      # Testes para utils/conversions.py
│   ├── test_coordinates.py      # Testes para core/coordinates.py
│   ├── test_julian_date.py      # Testes para core/julian_date.py
│   ├── test_distance.py         # Testes para core/distance.py
│   ├── test_magnitude.py        # Testes para core/magnitude.py
│   ├── test_orbital.py          # Testes para core/orbital.py
│   └── test_nasa_api.py         # Testes para services/nasa_api.py
│
├── conftest.py                  # Configuração global do pytest
├── pytest.ini                   # Configuração do pytest (cobertura, markers)
└── requirements.txt             # Dependências do projeto
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/astronomy_app.git
cd astronomy_app

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Executando os Testes

```bash
# Executar toda a suíte de testes
pytest

# Executar com relatório de cobertura
pytest --cov=. --cov-report=term-missing

# Executar um módulo específico
pytest tests/test_orbital.py -v

# Executar apenas testes com determinada marcação
pytest -m "not slow" -v
```

---

## Módulos

### `utils/conversions.py`
Primitivas de conversão reutilizáveis: graus/radianos, H:M:S, D:M:S, parsecs, anos-luz, UA.

### `core/coordinates.py`
Transformações entre sistemas de coordenadas celestes:
- Equatorial (AR, Dec) ↔ Horizontal (Alt/Az)
- Equatorial ↔ Eclíptico
- Equatorial ↔ Galáctico
- Separação angular (fórmula de Vincenty)

### `core/julian_date.py`
- Conversão Gregoriano → Data Juliana (algoritmo de Meeus)
- `datetime` Python ↔ JD
- Tempo Sideral Local (TSL)
- Épocas J2000.0 e MJD

### `core/distance.py`
- Paralaxe trigonométrica → distância em parsecs
- Módulo de distância (μ = m − M)
- Correção por extinção interestelar
- Distância 3D Euclidiana entre estrelas
- Processamento vetorizado com NumPy

### `core/magnitude.py`
- Lei de Pogson: fluxo ↔ magnitude
- Magnitude aparente ↔ absoluta
- Razão de fluxos e magnitude combinada
- Luminosidade em unidades solares
- Brilho superficial
- Funções vetorizadas com NumPy

### `core/orbital.py`
- Período orbital (3ª lei de Kepler)
- Equação vis-viva (velocidade orbital)
- Velocidade de escape
- Equação de Kepler (iteração Newton-Raphson)
- Anomalias excêntrica e verdadeira
- Esfera de Hill

### `services/nasa_api.py`
- Cliente HTTP para a API da NASA (APOD e NeoWs)
- Injeção de dependência via `requests.Session`
- Tratamento de erros HTTP e de rede
- Dataclasses tipadas para as respostas

---

## Práticas de Testes Demonstradas

| Técnica | Onde encontrar |
|---|---|
| `pytest.approx` (tolerância numérica) | Todos os módulos de teste |
| `@pytest.mark.parametrize` | `test_conversions`, `test_orbital`, `test_magnitude` |
| `pytest.fixture` | `test_coordinates`, `test_distance`, `test_nasa_api` |
| `unittest.mock.MagicMock` | `test_nasa_api` |
| Testes de inversibilidade (*round-trip*) | `test_conversions`, `test_julian_date` |
| Testes de valores de referência científicos | `test_orbital`, `test_magnitude`, `test_distance` |
| Validação de *edge cases* | `test_magnitude` (lista vazia), `test_distance` (paralaxe zero) |
| Testes de exceções | `test_orbital`, `test_distance`, `test_nasa_api` |
| Testes NumPy vetorizados | `test_magnitude`, `test_distance` |

---

## Referências

- Meeus, J. *Astronomical Algorithms*, 2nd ed. Willmann-Bell, 1998.
- Pogson, N. R. (1856). *Magnitudes of Thirty-Six of the Minor Planets*. MNRAS.
- IAU Working Group on Nomenclature for Fundamental Astronomy (2006).
- NASA APIs: [api.nasa.gov](https://api.nasa.gov)
