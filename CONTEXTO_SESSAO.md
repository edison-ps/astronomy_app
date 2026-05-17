# Contexto da Sessão — Astronomy App

**Data:** Sábado, 9 de maio de 2026  
**Projeto:** `c:\_Pessoais\astronomy_app`  
**Ambiente:** Python 3.7.6 · Windows 10 · PowerShell

---

## Objetivo

Desenvolver uma aplicação científica em Python voltada para **astronomia computacional**, com foco em cálculos astronômicos precisos e arquitetura adequada para implementação de testes unitários — servindo como base para um artigo técnico sobre testes unitários em software científico.

---

## O que foi construído

### Estrutura de arquivos

```
astronomy_app/
├── core/
│   ├── __init__.py
│   ├── coordinates.py
│   ├── julian_date.py
│   ├── distance.py
│   ├── magnitude.py
│   └── orbital.py
├── services/
│   ├── __init__.py
│   └── nasa_api.py
├── utils/
│   ├── __init__.py
│   └── conversions.py
├── tests/
│   ├── __init__.py
│   ├── test_conversions.py
│   ├── test_coordinates.py
│   ├── test_julian_date.py
│   ├── test_distance.py
│   ├── test_magnitude.py
│   ├── test_orbital.py
│   └── test_nasa_api.py
├── __init__.py
├── conftest.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Módulos implementados

### `utils/conversions.py`
Primitivas de conversão reutilizáveis:
- `degrees_to_radians` / `radians_to_degrees`
- `hms_to_degrees` / `degrees_to_hms` — Ascensão Reta em H:M:S
- `dms_to_degrees` / `degrees_to_dms` — Declinação em D:M:S
- `normalize_angle` → [0, 360) / `normalize_angle_signed` → [-180, 180)
- `parsec_to_light_years` / `light_years_to_parsecs`
- `au_to_km` / `km_to_au` / `parsec_to_au`

**Constantes:** `AU_TO_KM`, `PARSEC_TO_LY`, `PARSEC_TO_AU`, `LY_TO_KM`, `SPEED_OF_LIGHT_KMS`

---

### `core/coordinates.py`
Transformações entre sistemas de coordenadas celestes:

| Função | Descrição |
|--------|-----------|
| `equatorial_to_horizontal` | Equatorial (AR, Dec) → Horizontal (Alt, Az) |
| `equatorial_to_ecliptic` | Equatorial → Eclíptico |
| `ecliptic_to_equatorial` | Eclíptico → Equatorial |
| `equatorial_to_galactic` | Equatorial J2000.0 → Galáctico |
| `angular_separation` | Separação angular (fórmula de Vincenty) |

**Dataclasses:** `EquatorialCoord`, `HorizontalCoord`, `EclipticCoord`, `GalacticCoord`  
**Constantes:** `OBLIQUITY_J2000 = 23.439291111°`, polo galáctico norte IAU

---

### `core/julian_date.py`
Cálculos de tempo astronômico:

| Função | Descrição |
|--------|-----------|
| `gregorian_to_julian` | Gregoriano → Data Juliana (algoritmo de Meeus) |
| `julian_to_gregorian` | Data Juliana → Gregoriano |
| `datetime_to_julian` | `datetime` Python → JD |
| `julian_to_datetime` | JD → `datetime` UTC |
| `local_sidereal_time` | Tempo Sideral Local (fórmula USNO) |
| `julian_centuries_j2000` | Séculos Julianos desde J2000.0 |

**Dataclass:** `JulianDate` com propriedades `.mjd` e `.j2000`  
**Épocas:** J2000.0 = JD 2 451 545.0 · MJD = JD − 2 400 000.5

---

### `core/distance.py`
Indicadores de distância astronômica:

| Função | Descrição |
|--------|-----------|
| `parallax_to_distance_parsec` | Paralaxe (arcsec) → distância em pc |
| `parallax_to_distance_ly` | Paralaxe → distância em anos-luz |
| `distance_modulus` | μ = m − M |
| `distance_from_modulus` | μ → distância em pc |
| `modulus_from_distance` | Distância em pc → μ |
| `extinction_corrected_distance` | Distância com correção de extinção |
| `stellar_distance_3d` | Distância 3D Euclidiana entre duas estrelas |
| `photometric_distances` | Distâncias fotométricas vetorizadas (NumPy) |

---

### `core/magnitude.py`
Fotometria estelar (lei de Pogson):

| Função | Descrição |
|--------|-----------|
| `flux_to_magnitude` | Fluxo (Jy) → magnitude |
| `magnitude_to_flux` | Magnitude → fluxo (Jy) |
| `absolute_magnitude` | M = m − 5·log₁₀(d) + 5 − A |
| `apparent_magnitude` | m = M + 5·log₁₀(d) − 5 + A |
| `magnitude_difference_to_flux_ratio` | Δm → F₁/F₂ |
| `combined_magnitude` | Magnitude combinada de múltiplas fontes |
| `luminosity_solar` | Luminosidade em unidades solares |
| `surface_brightness` | Brilho superficial em mag/arcsec² |
| `magnitudes_to_flux_array` | Conversão vetorizada (NumPy) |
| `flux_array_to_magnitudes` | Conversão vetorizada (NumPy) |

**Constantes:** `ZERO_POINT_AB_JY = 3631.0`, `ZERO_POINT_FLUX_VEGA` (bandas U·B·V·R·I·J·H·K), `SUN_ABSOLUTE_MAG_V = 4.83`

---

### `core/orbital.py`
Mecânica orbital (leis de Kepler):

| Função | Descrição |
|--------|-----------|
| `orbital_period` | 3ª lei de Kepler: T² = a³/M |
| `orbital_velocity` | Equação vis-viva: v² = GM(2/r − 1/a) |
| `escape_velocity` | Velocidade de escape: v = √(2GM/r) |
| `solve_kepler_equation` | Equação de Kepler M = E − e·sin(E) (Newton-Raphson) |
| `true_anomaly_from_eccentric` | Anomalia excêntrica → anomalia verdadeira |
| `mean_anomaly_at_time` | Anomalia média em função do tempo |
| `perihelion_distance` / `aphelion_distance` | q = a(1−e) / Q = a(1+e) |
| `orbital_energy` | Energia orbital específica ε = −GM/(2a) |
| `hill_sphere_radius` | Raio da esfera de Hill de um planeta |

**Dataclass:** `OrbitalElements` (elementos Keplerianos com validação)  
**Constantes SI:** `G`, `M_SUN`, `M_EARTH`, `R_EARTH`, `R_SUN`, `AU_TO_M`

---

### `services/nasa_api.py`
Cliente HTTP para APIs públicas da NASA:

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `get_apod(date)` | `/planetary/apod` | Astronomy Picture of the Day |
| `get_near_earth_objects(start, end)` | `/neo/rest/v1/feed` | Asteroides próximos à Terra |
| `get_asteroid_by_id(id)` | `/neo/rest/v1/neo/{id}` | Dados detalhados de asteroide |

**Dataclasses de resposta:** `ApodData`, `NearEarthObject`  
**Exceções customizadas:** `NASAAPIError`, `NASAAPIConnectionError`, `NASAAPIHTTPError`, `NASAAPIParseError`  
**Injeção de dependência:** `session: requests.Session` — permite mock total sem tráfego HTTP real

---

## Suíte de testes

### Resultado final
```
318 passed, 0 failed  (6.04s)
```

### Distribuição por módulo

| Arquivo de teste | Nº de testes | Classes de teste |
|------------------|:------------:|------------------|
| `test_conversions.py` | 75 | `TestDegreesRadians`, `TestHMSConversions`, `TestDMSConversions`, `TestNormalizeAngle`, `TestDistanceConversions` |
| `test_coordinates.py` | 37 | `TestEquatorialCoordValidation`, `TestEquatorialToHorizontal`, `TestEquatorialEclipticTransform`, `TestEquatorialToGalactic`, `TestAngularSeparation` |
| `test_julian_date.py` | 33 | `TestJulianDateDataclass`, `TestGregorianToJulian`, `TestJulianToGregorian`, `TestDatetimeJulian`, `TestLocalSiderealTime`, `TestJulianCenturies` |
| `test_distance.py` | 37 | `TestParallaxToDistance`, `TestDistanceModulus`, `TestExtinctionCorrectedDistance`, `TestStellarDistance3D`, `TestPhotometricDistances` |
| `test_magnitude.py` | 46 | `TestFluxMagnitudeConversion`, `TestAbsoluteApparentMagnitude`, `TestFluxRatioCombinedMagnitude`, `TestLuminositySolar`, `TestSurfaceBrightness`, `TestNumpyVectorised` |
| `test_orbital.py` | 49 | `TestOrbitalElementsValidation`, `TestOrbitalPeriod`, `TestOrbitalVelocity`, `TestEscapeVelocity`, `TestSolveKeplerEquation`, `TestTrueAnomaly`, `TestOrbitalGeometry` |
| `test_nasa_api.py` | 41 | `TestApodDataFromDict`, `TestNearEarthObjectFromDict`, `TestGetApod`, `TestGetNearEarthObjects`, `TestAPIErrorHandling`, `TestClientConfiguration` |

---

## Técnicas de teste demonstradas

| Técnica pytest | Onde é demonstrada |
|----------------|-------------------|
| `pytest.approx` com `abs=` e `rel=` | Todos os módulos |
| `@pytest.mark.parametrize` | Órbitas planetárias (Mercúrio→Saturno), conversões H:M:S, distâncias |
| `pytest.fixture` com scope de classe | Sirius, Canopus, Proxima Centauri, órbita da Terra, Cometa Halley |
| `unittest.mock.MagicMock` | `test_nasa_api.py` — mock de `requests.Session` |
| Testes de inversibilidade (*round-trip*) | Equatorial↔Eclíptico, JD↔datetime, magnitude↔fluxo, parsec↔anos-luz |
| Valores de referência científicos | Meeus (JD), Hipparcos (paralaxe), USNO (tempo sideral), IAU |
| *Edge cases* matemáticos | paralaxe zero, fluxo negativo, órbita ilimitada (r > 2a), tan(π) em ponto flutuante |
| Testes NumPy (`np.testing.assert_allclose`) | `test_magnitude.py`, `test_distance.py` |
| `pytest.raises` com `match=` regex | Todos os módulos de validação |
| Testes de monotonicidade | Distâncias fotométricas, fluxos vs magnitude |
| Testes de simetria | Separação angular A→B == B→A |

---

## Problemas encontrados e resolvidos

| Problema | Causa | Solução |
|----------|-------|---------|
| `dict[str, float]` sintaxe | Python 3.7 não suporta genéricos built-in | Substituído por `Dict[str, float]` com `from typing import Dict` |
| Azimute retornando `360.0` | Acumulação de ponto flutuante em `acos` | `azimuth = azimuth % 360.0` ao final |
| `true_anomaly` retornando `360.0` | `math.tan(π) ≈ -1.22e-16` → artefato de ponto flutuante | Substituído `atan(factor * tan(E/2))` por `atan2(sin, cos)` |
| MJD de J2000.0 | Teste esperava `51545.0`; valor correto é `51544.5` | Corrigido o valor esperado no teste |
| GMST em J2000.0 | Fórmula retornava GMST às 0h UT, não no JD fornecido | Substituída por `280.46 + 360.9856 * D` (fórmula USNO direta) |
| Luminosidade de Sírius | Teste comparava com luminosidade bolométrica (~25 L☉) | Corrigido para luminosidade em banda V (~23.1 L☉) |
| `normalize_angle_signed(180°)` | Expectativa errada no teste; range é [-180, 180) | Corrigido valor esperado para `-180.0` |
| `&&` no PowerShell | PowerShell 5 não suporta `&&` como separador | Substituído por `;` e `Set-Location` separado |

---

## Dependências instaladas

```
numpy==1.21.6         (já presente no ambiente)
requests>=2.31.0
pytest==7.4.4
pytest-cov==4.1.0
coverage==7.2.7
```

> `scipy` e `astropy` não foram instalados por falha de SSL no pip; os módulos foram desenvolvidos usando apenas `math` e `numpy`, sem dependência de `scipy`.

---

## Como executar

```powershell
# Ativar ambiente virtual (se houver)
.venv\Scripts\activate

# Executar toda a suíte
python -m pytest tests/ -v

# Com relatório de cobertura
python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-omit="tests/*,conftest.py"

# Apenas um módulo
python -m pytest tests/test_orbital.py -v

# Excluir testes lentos (quando marcados)
python -m pytest -m "not slow" -v
```

---

## Referências científicas utilizadas

- Meeus, J. *Astronomical Algorithms*, 2nd ed. Willmann-Bell, 1998. (Caps. 7, 12, 13)
- Pogson, N. R. (1856). *Magnitudes of Thirty-Six of the Minor Planets*. MNRAS, 17, 12.
- Bessell, M. S. (1998). *Standard Photometric Systems*. PASP, 110, 863.
- Willmer, C. N. A. (2018). *The Absolute Magnitude of the Sun in Several Filters*. ApJS, 236, 47.
- IAU Working Group on NFAS (2006). Galactic coordinates in J2000.0.
- USNO Circular 179 (2010). GMST polynomial formula.
- NASA APIs: [api.nasa.gov](https://api.nasa.gov)
