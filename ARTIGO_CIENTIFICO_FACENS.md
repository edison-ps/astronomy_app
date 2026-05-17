# TESTES UNITÁRIOS EM SOFTWARE CIENTÍFICO: UMA ABORDAGEM ARQUITETURAL PARA APLICAÇÕES DE ASTRONOMIA COMPUTACIONAL EM PYTHON

**UNIT TESTING IN SCIENTIFIC SOFTWARE: AN ARCHITECTURAL APPROACH FOR COMPUTATIONAL ASTRONOMY APPLICATIONS IN PYTHON**

---

Edison Silva¹  
João Souza²  
Murilo Oliveira³

---

> ¹ Pós-graduando em Arquitetura de Software — Facens Centro Universitário. E-mail: edison@facens.br  
> ² Pós-graduando em Arquitetura de Software — Facens Centro Universitário. E-mail: joao@facens.br  
> ³ Pós-graduando em Arquitetura de Software — Facens Centro Universitário. E-mail: murilo@facens.br

---

## RESUMO

O desenvolvimento de software científico impõe desafios arquiteturais específicos: cálculos de precisão numérica elevada, dependência de constantes físicas estabelecidas por organismos internacionais e integração com serviços externos de dados. Este artigo apresenta a concepção e implementação de uma aplicação de astronomia computacional em Python, denominada *Astronomy App*, estruturada segundo princípios de arquitetura de software que favorecem a testabilidade. A aplicação compreende módulos para transformação de sistemas de coordenadas celestes, cálculo de datas Julianas, distâncias astronômicas, fotometria estelar e mecânica orbital, além de um cliente HTTP para APIs públicas da NASA. Uma suíte com 318 testes unitários foi construída utilizando o framework *pytest*, cobrindo os sete módulos implementados. São demonstradas técnicas como parametrização de casos, *fixtures* com escopo de classe, *mocking* de sessões HTTP via injeção de dependência, verificação de precisão numérica com `pytest.approx` e testes de inversibilidade (*round-trip*). Os resultados evidenciam que a adoção de uma arquitetura orientada à testabilidade — com separação clara de responsabilidades, uso de *dataclasses* para encapsulamento e injeção de dependências para serviços externos — não apenas eleva a confiabilidade do código científico, mas também reduz o custo de manutenção e facilita a evolução do sistema. O ambiente Python 3.7.6 com *numpy* como único pacote científico externo demonstrou ser suficiente para implementar algoritmos de precisão referenciados por Meeus (1998), pela IAU e pelo USNO.

**Palavras-chave:** Testes unitários. Arquitetura de software. Astronomia computacional. Python. pytest.

---

## ABSTRACT

The development of scientific software poses specific architectural challenges: high numerical precision computations, dependence on physical constants established by international bodies, and integration with external data services. This paper presents the design and implementation of a computational astronomy application in Python, called *Astronomy App*, structured according to software architecture principles that promote testability. The application comprises modules for celestial coordinate system transformations, Julian date computation, astronomical distances, stellar photometry, and orbital mechanics, as well as an HTTP client for NASA public APIs. A suite of 318 unit tests was built using the *pytest* framework, covering all seven implemented modules. Techniques demonstrated include test parametrization, class-scoped fixtures, HTTP session mocking via dependency injection, numerical precision verification with `pytest.approx`, and round-trip invertibility tests. Results show that adopting a testability-oriented architecture — with clear separation of concerns, dataclass encapsulation, and dependency injection for external services — not only increases the reliability of scientific code but also reduces maintenance costs and facilitates system evolution. The Python 3.7.6 environment with *numpy* as the sole external scientific package proved sufficient to implement precision algorithms referenced by Meeus (1998), the IAU, and the USNO.

**Keywords:** Unit testing. Software architecture. Computational astronomy. Python. pytest.

---

## 1 INTRODUÇÃO

A confiabilidade de resultados computacionais em ciências exatas depende diretamente da qualidade do software que os produz. Algoritmos astronômicos envolvem operações trigonométricas encadeadas, conversões de unidades entre sistemas distintos e implementações de equações físicas derivadas de leis fundamentais — contexto em que erros numéricos podem se propagar silenciosamente e invalidar conclusões científicas inteiras (MILI; TCHIER, 2015).

O campo da engenharia de software científico (*research software engineering*) tem reconhecido crescentemente a necessidade de práticas de garantia de qualidade equivalentes às adotadas pela indústria de software comercial. Wilson et al. (2014) identificam testes automatizados como uma das práticas de maior impacto para a reprodutibilidade científica, ao lado do controle de versão e da modularização do código. No entanto, a adoção efetiva dessas práticas permanece heterogênea na comunidade científica (PENG, 2011).

No domínio da astronomia computacional, a precisão dos algoritmos é frequentemente validada contra valores tabelados em obras de referência como *Astronomical Algorithms* (MEEUS, 1998), circulares do United States Naval Observatory (USNO) e resoluções da International Astronomical Union (IAU). Essa disponibilidade de valores de referência canônicos representa uma oportunidade única para a construção de suítes de testes robustas, nas quais os *oráculos de teste* são externamente verificáveis.

Este artigo investiga como decisões arquiteturais tomadas durante a concepção de uma aplicação científica — em particular, a separação de módulos por domínio funcional, o uso de *dataclasses* tipadas, a injeção de dependências em serviços externos e a adoção de constantes nomeadas — influenciam diretamente a capacidade de se construir, manter e evoluir uma suíte de testes unitários abrangente.

O trabalho está estruturado da seguinte forma: a Seção 2 apresenta a fundamentação teórica sobre arquitetura orientada à testabilidade e testes em software científico; a Seção 3 descreve a metodologia e a arquitetura da aplicação implementada; a Seção 4 apresenta os resultados da suíte de testes e discute as técnicas empregadas; e a Seção 5 tece as conclusões e indica trabalhos futuros.

---

## 2 FUNDAMENTAÇÃO TEÓRICA

### 2.1 Arquitetura de Software e Testabilidade

Testabilidade é uma propriedade de qualidade de software que expressa o grau de facilidade com que um sistema pode ser testado (ISO/IEC 25010, 2011). Do ponto de vista arquitetural, a testabilidade é favorecida por características como baixo acoplamento entre módulos, alta coesão interna, observabilidade dos estados do sistema e controlabilidade das suas entradas (BINDER, 1999).

A separação de responsabilidades (*Separation of Concerns*) é o princípio fundamental que sustenta a testabilidade. Quando cada módulo possui uma única responsabilidade bem definida, é possível testá-lo de forma isolada, sem dependência de outros componentes do sistema (MARTIN, 2017). Em aplicações científicas, isso se traduz na separação entre lógica de cálculo puro — que depende apenas de parâmetros matemáticos — e lógica de integração com sistemas externos, como APIs e bancos de dados.

A injeção de dependência (*Dependency Injection*) é outra técnica central para a testabilidade. Ao invés de instanciar colaboradores internamente, um componente recebe suas dependências por parâmetro. Isso permite que, durante os testes, dependências reais sejam substituídas por *test doubles* — objetos que simulam o comportamento esperado sem produzir efeitos colaterais externos (FREEMAN; PRYCE, 2009).

### 2.2 Testes Unitários: Fundamentos e Técnicas

Um teste unitário é uma verificação automatizada e isolada do comportamento de uma unidade de código — tipicamente uma função ou método — frente a entradas específicas (BECK, 2002). As propriedades desejáveis de bons testes unitários são frequentemente resumidas pelo acrônimo FIRST: *Fast* (rápidos), *Isolated* (isolados), *Repeatable* (repetíveis), *Self-validating* (autovalidados) e *Timely* (oportunos) (MARTIN; MARTIN, 2006).

Em software científico, surgem requisitos adicionais. A comparação de resultados numéricos de ponto flutuante exige tolerâncias explícitas, pois a representação binária de números reais introduz erros de arredondamento inerentes (GOLDBERG, 1991). O framework *pytest* oferece o utilitário `pytest.approx`, que permite especificar tanto tolerâncias absolutas (adequadas para valores próximos a zero) quanto relativas (adequadas para valores de grande magnitude).

A parametrização de testes — técnica pela qual uma mesma lógica de verificação é executada contra múltiplos conjuntos de dados de entrada — é particularmente valiosa em astronomia computacional, onde algoritmos devem produzir resultados corretos para os oito planetas do Sistema Solar, para estrelas em diferentes posições do céu e para épocas astronômicas distintas (HUNT; THOMAS, 1999).

### 2.3 Mocking e Testes de Integração com APIs Externas

Aplicações que consomem dados de serviços externos enfrentam o desafio de testar comportamentos sem efetuar chamadas de rede reais. O padrão *mock* substitui o colaborador externo por um objeto que simula respostas predefinidas, tornando os testes determinísticos e independentes de disponibilidade de rede (FOWLER, 2007).

A injeção de uma `requests.Session` como parâmetro do cliente HTTP — em vez de instanciá-la internamente — exemplifica como uma decisão arquitetural simples tem impacto direto na testabilidade: o mesmo parâmetro que recebe uma sessão real em produção pode receber um `MagicMock` nos testes, sem nenhuma alteração no código de produção (PERCIVAL; GREGORY, 2020).

### 2.4 Software Científico em Python

Python consolidou-se como a linguagem dominante em astronomia computacional, impulsionado pela legibilidade do código, pelo ecossistema científico maduro e pela facilidade de integração com bibliotecas de álgebra linear como *NumPy* (HARRIS et al., 2020). Projetos institucionais como *Astropy* (ASTROPY COLLABORATION, 2013) demonstram que é possível construir bibliotecas científicas de alta precisão com cobertura de testes abrangente em Python.

A adoção de *type hints* e *dataclasses* a partir do Python 3.7 trouxe benefícios significativos para a legibilidade e manutenibilidade de código científico: a estrutura de dados fica auto-documentada, os parâmetros de entrada e saída de funções tornam-se explícitos, e erros de tipo são detectáveis estaticamente antes da execução (VAN ROSSUM; LEHTOSALO; LANGA, 2015).

---

## 3 METODOLOGIA

### 3.1 Descrição do Projeto

A *Astronomy App* foi desenvolvida como uma aplicação Python 3.7.6 de astronomia computacional, com o duplo propósito de fornecer cálculos astronômicos precisos e servir como objeto de estudo para a aplicação de testes unitários em software científico. O ambiente de desenvolvimento foi composto por Windows 10, PowerShell e um ambiente virtual Python com as bibliotecas `numpy==1.21.6`, `pytest==7.4.4`, `pytest-cov==4.1.0`, `coverage==7.2.7` e `requests>=2.31.0`. As bibliotecas `scipy` e `astropy` foram intencionalmente excluídas para demonstrar a implementação de algoritmos a partir de fontes primárias.

### 3.2 Arquitetura da Aplicação

A aplicação foi estruturada em três camadas funcionais:

**Camada de utilitários** (`utils/`): primitivas de conversão reutilizáveis, sem dependências externas além da biblioteca padrão. Esta camada é a base matemática sobre a qual os demais módulos são construídos.

**Camada de domínio** (`core/`): módulos de cálculo astronômico puro, organizados por domínio funcional — coordenadas celestes, datas Julianas, distâncias, fotometria e mecânica orbital. Cada módulo depende apenas de `math`, `numpy` e das primitivas de `utils/`, mantendo zero acoplamento com serviços externos.

**Camada de serviços** (`services/`): cliente HTTP para a API pública da NASA, único ponto de contato com sistemas externos. O isolamento desta camada assegura que todos os módulos de domínio possam ser testados sem tráfego de rede.

A estrutura de diretórios adotada reflete essa separação:

```
astronomy_app/
├── core/
│   ├── coordinates.py
│   ├── julian_date.py
│   ├── distance.py
│   ├── magnitude.py
│   └── orbital.py
├── services/
│   └── nasa_api.py
├── utils/
│   └── conversions.py
└── tests/
    ├── test_conversions.py
    ├── test_coordinates.py
    ├── test_julian_date.py
    ├── test_distance.py
    ├── test_magnitude.py
    ├── test_orbital.py
    └── test_nasa_api.py
```

### 3.3 Decisões de Design para Testabilidade

Quatro decisões de design foram deliberadamente tomadas com foco na testabilidade:

**Uso de dataclasses tipadas**: As estruturas de dados (`EquatorialCoord`, `HorizontalCoord`, `JulianDate`, `OrbitalElements`, etc.) foram implementadas como `@dataclass`, o que elimina código boilerplate, torna os campos explícitos e facilita a construção de instâncias nos testes sem dependência de construtores complexos.

**Validação na instanciação**: Classes como `OrbitalElements` validam invariantes físicos no método `__post_init__` (e.g., excentricidade deve ser ≥ 0; semieixo maior deve ser positivo). Isso garante que objetos inválidos jamais sejam criados, simplificando a verificação de erros com `pytest.raises`.

**Constantes físicas nomeadas**: Valores como `OBLIQUITY_J2000 = 23.439291111` (graus), `PARSEC_TO_LY`, `ZERO_POINT_AB_JY` e `G` foram definidos como constantes de módulo com nomes descritivos. Isso elimina literais numéricos nos testes e torna os valores rastreáveis às fontes primárias.

**Injeção de dependência para serviços externos**: O cliente `NASAAPIClient` recebe um parâmetro opcional `session: requests.Session`. Em produção, uma sessão real é utilizada; nos testes, um `MagicMock` é injetado, permitindo simular respostas HTTP com precisão sem realizar chamadas reais.

### 3.4 Construção da Suíte de Testes

Os testes foram organizados em classes por tema dentro de cada arquivo de teste, seguindo o padrão `TestNomeDoConceito`. Três níveis de testes foram sistematicamente implementados:

- **Casos nominais**: entradas válidas produzem saídas corretas dentro das tolerâncias especificadas.
- **Casos limite** (*edge cases*): paralaxe zero, fluxo negativo, órbita hiperbólica, ângulo de 360°, etc.
- **Casos de erro**: entradas inválidas devem gerar as exceções corretas com mensagens informativas.

*Fixtures* com escopo de classe (`@pytest.fixture(scope="class")`) foram utilizadas para instanciar objetos de teste representativos — Sírius, Canopus, Próxima Centauri, Terra, Cometa Halley — que são compartilhados entre múltiplos métodos de teste sem custo de recriação.

---

## 4 RESULTADOS E DISCUSSÃO

### 4.1 Resultados da Suíte de Testes

A suíte de testes final resultou em **318 testes aprovados, 0 falhas**, com tempo de execução de 6,04 segundos. A distribuição por módulo é apresentada no Quadro 1.

**Quadro 1 — Distribuição dos testes unitários por módulo**

| Arquivo de teste      | Nº de testes | Classes de teste                                                                                        |
|-----------------------|:------------:|---------------------------------------------------------------------------------------------------------|
| `test_conversions.py` | 75           | `TestDegreesRadians`, `TestHMSConversions`, `TestDMSConversions`, `TestNormalizeAngle`, `TestDistanceConversions` |
| `test_coordinates.py` | 37           | `TestEquatorialCoordValidation`, `TestEquatorialToHorizontal`, `TestEquatorialEclipticTransform`, `TestEquatorialToGalactic`, `TestAngularSeparation` |
| `test_julian_date.py` | 33           | `TestJulianDateDataclass`, `TestGregorianToJulian`, `TestJulianToGregorian`, `TestDatetimeJulian`, `TestLocalSiderealTime`, `TestJulianCenturies` |
| `test_distance.py`    | 37           | `TestParallaxToDistance`, `TestDistanceModulus`, `TestExtinctionCorrectedDistance`, `TestStellarDistance3D`, `TestPhotometricDistances` |
| `test_magnitude.py`   | 46           | `TestFluxMagnitudeConversion`, `TestAbsoluteApparentMagnitude`, `TestFluxRatioCombinedMagnitude`, `TestLuminositySolar`, `TestSurfaceBrightness`, `TestNumpyVectorised` |
| `test_orbital.py`     | 49           | `TestOrbitalElementsValidation`, `TestOrbitalPeriod`, `TestOrbitalVelocity`, `TestEscapeVelocity`, `TestSolveKeplerEquation`, `TestTrueAnomaly`, `TestOrbitalGeometry` |
| `test_nasa_api.py`    | 41           | `TestApodDataFromDict`, `TestNearEarthObjectFromDict`, `TestGetApod`, `TestGetNearEarthObjects`, `TestAPIErrorHandling`, `TestClientConfiguration` |

Fonte: elaborado pelos autores (2026).

### 4.2 Técnicas de Teste e Sua Aplicação

**Precisão numérica com `pytest.approx`**: A comparação de resultados astronômicos requer tolerâncias explícitas. Tolerâncias absolutas (`abs=1e-6`) foram empregadas para ângulos próximos a zero; tolerâncias relativas (`rel=1e-4`) para distâncias em parsecs e velocidades orbitais. Esta distinção — frequentemente negligenciada em software científico — previne tanto falsos positivos (tolerância excessivamente larga) quanto falsos negativos (tolerância incompatível com a precisão do algoritmo).

**Parametrização planetária**: O módulo `test_orbital.py` valida a Terceira Lei de Kepler com dados dos planetas Mercúrio, Vênus, Terra, Marte, Júpiter e Saturno. Um único método de teste com `@pytest.mark.parametrize` substitui seis testes individuais, mantendo a cobertura sem duplicação de código:

```python
@pytest.mark.parametrize("planet,semi_major_axis_au,period_years", [
    ("Mercury", 0.387,  0.241),
    ("Venus",   0.723,  0.615),
    ("Earth",   1.000,  1.000),
    ("Mars",    1.524,  1.881),
    ("Jupiter", 5.203, 11.862),
    ("Saturn",  9.537, 29.457),
])
def test_orbital_period_planets(self, planet, semi_major_axis_au, period_years):
    ...
```

**Testes de inversibilidade (*round-trip*)**: Para cada par de funções de conversão inversas — Equatorial↔Eclíptico, JD↔datetime, magnitude↔fluxo, parsec↔anos-luz — foram implementados testes que aplicam a transformação direta seguida da inversa e verificam que o resultado original é recuperado. Esta categoria de teste detecta implementações inconsistentes entre funções aparentemente corretas isoladamente.

**Mocking de sessão HTTP**: O `NASAAPIClient` recebe `session: requests.Session = None`. Quando `None`, instancia uma sessão real; nos testes, recebe um `MagicMock` configurado para retornar JSONs de exemplo:

```python
mock_session = MagicMock()
mock_response = MagicMock()
mock_response.json.return_value = {"date": "2026-05-09", ...}
mock_response.status_code = 200
mock_session.get.return_value = mock_response
client = NASAAPIClient(api_key="TEST", session=mock_session)
```

Esta abordagem garante 100% de cobertura do código de integração sem nenhuma chamada HTTP real, tornando os testes determinísticos e executáveis em ambientes sem conectividade.

**Testes de casos-limite matemáticos**: Foram identificados e testados fenômenos de ponto flutuante que poderiam silenciosamente corromper resultados astronômicos. Dois exemplos ilustrativos:

- *Azimute retornando 360.0°*: a acumulação de erros de arredondamento em `acos` produzia azimute exatamente 360.0° em vez de 0.0°. A correção `azimuth = azimuth % 360.0` foi validada por teste.
- *Anomalia verdadeira em órbita circular*: `math.tan(π) ≈ −1.22 × 10⁻¹⁶` em vez de 0, causando desvio na anomalia verdadeira. A substituição de `atan(factor × tan(E/2))` por `atan2(sin, cos)` eliminou o artefato, verificada por teste parametrizado.

### 4.3 Problemas Arquiteturais Identificados e Resolvidos

O processo de desenvolvimento evidenciou um conjunto de incompatibilidades entre a implementação inicial e o ambiente de execução Python 3.7.6, resumidas no Quadro 2.

**Quadro 2 — Problemas encontrados e soluções arquiteturais adotadas**

| Problema                          | Causa                                        | Solução                                               |
|-----------------------------------|----------------------------------------------|-------------------------------------------------------|
| Sintaxe `dict[str, float]`        | Python 3.7 não suporta genéricos built-in    | Substituído por `Dict[str, float]` com `from typing import Dict` |
| Azimute retornando 360.0°         | Acumulação de ponto flutuante em `acos`      | `azimuth = azimuth % 360.0`                           |
| Anomalia verdadeira retornando 360.0° | `math.tan(π) ≈ −1.22 × 10⁻¹⁶`           | Substituído por `atan2(sin, cos)`                     |
| MJD de J2000.0 incorreto          | Teste esperava 51545.0; correto é 51544.5    | Corrigido valor esperado no teste                     |
| GMST em J2000.0 incorreto         | Fórmula calculava GMST às 0h UT              | Substituída pela fórmula USNO direta: `280.46 + 360.9856 × D` |
| Luminosidade de Sírius             | Comparação com luminosidade bolométrica       | Corrigido para luminosidade em banda V (~23,1 L☉)     |
| `normalize_angle_signed(180°)`    | Expectativa errada; range é [−180, 180)      | Corrigido valor esperado para −180.0                  |

Fonte: elaborado pelos autores (2026).

Notavelmente, todos esses problemas foram detectados pela própria suíte de testes — não por inspeção manual do código. Isso demonstra o valor dos testes como mecanismo de detecção de erros sutis, especialmente em código que manipula constantes e fórmulas físicas.

### 4.4 Impacto das Decisões Arquiteturais na Testabilidade

A análise retrospectiva da relação entre decisões de design e facilidade de teste revelou padrões consistentes:

**Separação de camadas**: A total ausência de lógica de cálculo na camada de serviços e de chamadas de rede na camada de domínio eliminou a necessidade de mocks em 277 dos 318 testes (87%). Apenas os 41 testes de `test_nasa_api.py` requereram uso de `MagicMock`.

**Dataclasses com validação**: A validação no `__post_init__` de `OrbitalElements` permitiu que todos os testes de entrada inválida fossem escritos com `pytest.raises` de forma concisa, sem necessidade de lógica de tratamento nos próprios testes.

**Constantes nomeadas**: A rastreabilidade das constantes às fontes primárias (IAU, USNO, Meeus) permitiu verificar independentemente os valores utilizados nos testes de referência, aumentando a confiança nos oráculos de teste.

---

## 5 CONCLUSÃO

Este trabalho demonstrou que a adoção de princípios de arquitetura de software — separação de responsabilidades, injeção de dependências, encapsulamento por dataclasses e definição explícita de constantes — tem impacto direto e mensurável na testabilidade de aplicações científicas. A suíte de 318 testes unitários construída para a *Astronomy App* não apenas verificou a correção dos algoritmos implementados, como também identificou e permitiu corrigir sete categorias distintas de erros, incluindo artefatos de ponto flutuante que seriam imperceptíveis na ausência de testes.

A decisão de implementar os algoritmos a partir de fontes primárias (Meeus, IAU, USNO), sem dependência de bibliotecas científicas especializadas como *astropy* ou *scipy*, demonstrou ser viável no contexto de um ambiente restrito (Python 3.7.6, NumPy como único pacote científico externo) e produziu resultados verificáveis contra os valores canônicos da literatura.

Do ponto de vista da arquitetura de software, o projeto evidencia que *testabilidade não é uma propriedade emergente*, mas uma qualidade deliberadamente construída através de decisões de design tomadas desde as fases iniciais do desenvolvimento. A resistência à criação de acoplamentos desnecessários — especialmente com serviços externos — e a organização modular por domínio funcional foram os fatores que mais contribuíram para a produtividade no desenvolvimento da suíte de testes.

Como trabalhos futuros, propõe-se: (a) a implementação de testes de integração com a API real da NASA, utilizando *fixtures* com `vcr.py` para gravação e reprodução de respostas HTTP; (b) a adição de testes de *property-based testing* com a biblioteca *Hypothesis* para exploração de espaços de entrada em algoritmos trigonométricos; e (c) a medição de cobertura de código por condição (*branch coverage*), complementando a cobertura por linha já obtida.

---

## REFERÊNCIAS

ASTROPY COLLABORATION. **Astropy: A community Python package for astronomy**. *Astronomy & Astrophysics*, v. 558, p. A33, 2013. Disponível em: https://doi.org/10.1051/0004-6361/201322068. Acesso em: 9 maio 2026.

BECK, K. **Test-Driven Development: By Example**. Boston: Addison-Wesley, 2002.

BESSELL, M. S. Standard Photometric Systems. **Publications of the Astronomical Society of the Pacific**, v. 110, n. 750, p. 863–878, 1998.

BINDER, R. V. **Testing Object-Oriented Systems: Models, Patterns, and Tools**. Boston: Addison-Wesley, 1999.

FOWLER, M. **Mocks aren't stubs**. 2007. Disponível em: https://martinfowler.com/articles/mocksArentStubs.html. Acesso em: 9 maio 2026.

FREEMAN, S.; PRYCE, N. **Growing Object-Oriented Software, Guided by Tests**. Boston: Addison-Wesley, 2009.

GOLDBERG, D. What every computer scientist should know about floating-point arithmetic. **ACM Computing Surveys**, v. 23, n. 1, p. 5–48, 1991.

HARRIS, C. R. et al. Array programming with NumPy. **Nature**, v. 585, p. 357–362, 2020.

HUNT, A.; THOMAS, D. **The Pragmatic Programmer: From Journeyman to Master**. Boston: Addison-Wesley, 1999.

IAU WORKING GROUP ON NFAS. **Galactic coordinates in J2000.0**. 2006.

ISO/IEC 25010. **Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE)**. Genebra: ISO, 2011.

MARTIN, R. C. **Clean Architecture: A Craftsman's Guide to Software Structure and Design**. Boston: Prentice Hall, 2017.

MARTIN, R. C.; MARTIN, M. **Agile Principles, Patterns, and Practices in C#**. Boston: Prentice Hall, 2006.

MEEUS, J. **Astronomical Algorithms**. 2. ed. Richmond: Willmann-Bell, 1998.

MILI, A.; TCHIER, F. **Software Testing: Concepts and Operations**. Hoboken: Wiley-IEEE Press, 2015.

NASA. **NASA Open APIs**. Disponível em: https://api.nasa.gov. Acesso em: 9 maio 2026.

PENG, R. D. Reproducible research in computational science. **Science**, v. 334, n. 6060, p. 1226–1227, 2011.

PERCIVAL, H.; GREGORY, B. **Architecture Patterns with Python**. Sebastopol: O'Reilly Media, 2020.

POGSON, N. R. Magnitudes of thirty-six of the minor planets for the first day of each month of the year 1857. **Monthly Notices of the Royal Astronomical Society**, v. 17, p. 12–15, 1856.

USNO. **Circular 179: The IAU Resolutions on Astronomical Reference Systems, Time Scales, and Earth Rotation Models**. Washington: United States Naval Observatory, 2010.

VAN ROSSUM, G.; LEHTOSALO, J.; LANGA, Ł. **PEP 484 — Type Hints**. 2015. Disponível em: https://peps.python.org/pep-0484/. Acesso em: 9 maio 2026.

WILLMER, C. N. A. The Absolute Magnitude of the Sun in Several Filters. **The Astrophysical Journal Supplement Series**, v. 236, n. 2, p. 47, 2018.

WILSON, G. et al. Best practices for scientific computing. **PLOS Biology**, v. 12, n. 1, p. e1001745, 2014.

---

*Artigo elaborado como requisito parcial para aprovação na disciplina de Arquitetura de Software — Pós-Graduação Facens Centro Universitário, Sorocaba, 2026.*

---

> **Instruções de formatação para o Word (padrão Facens/ABNT NBR 6022:2018):**
> - Papel A4, orientação retrato
> - Margens: superior 3 cm, inferior 2 cm, esquerda 3 cm, direita 2 cm
> - Fonte: Times New Roman 12 pt (corpo); 10 pt (resumo, notas de rodapé, citações longas)
> - Espaçamento: 1,5 entre linhas no corpo; simples no resumo, abstract e referências
> - Título do artigo: centralizado, negrito, caixa alta, Times New Roman 12 pt
> - Títulos das seções: numerados, negrito, alinhados à esquerda
> - Parágrafo: recuo de 1,25 cm na primeira linha
> - Quadros: título acima (Quadro N — Descrição), fonte abaixo
> - Páginas: numeradas no canto superior direito a partir da primeira página do texto
