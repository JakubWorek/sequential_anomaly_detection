# Wykrywanie anomalii w danych sekwencyjnych: przepływy normalizujące, modele dyfuzji i LSTM
**Dawid Zawiślak · Jakub Worek**

---

## 1. Sformułowanie problemu

Zadaniem jest nienadzorowane wykrywanie anomalii w wielowymiarowych szeregach czasowych. Modele są trenowane **wyłącznie na danych normalnych** i oceniane pod kątem zdolności do przypisywania wysokich wyników anomalii wcześniej niewidzianym anomalnym krokom czasowym. Na tym samym zestawie danych benchmark porównano trzy zasadniczo odmienne paradygmaty modelowania generatywnego i sekwencyjnego:

- **LSTM Predictor** — model predykcji sekwencja-do-sekwencji
- **Normalizing Flows (MAF)** — jawne szacowanie gęstości dla każdego kroku czasowego
- **Diffusion Model (DDPM)** — punktacja oparta na rekonstrukcji poprzez częściowe dodawanie szumu
- **Isolation Forest** — klasyczny punkt odniesienia bez modelowania temporalnego

---

## 2. Zbiór danych

### 2.1 NASA SMAP / MSL

Zbiory danych **SMAP** (Soil Moisture Active Passive) i **MSL** (Mars Science Laboratory / łazik Curiosity) to publicznie udostępnione przez NASA benchmarkowe dane telemetryczne do wykrywania anomalii w szeregach czasowych. Każda encja stanowi niezależny wielowymiarowy kanał z predefiniowanym podziałem na zbiór treningowy i testowy; etykiety anomalii są dostępne wyłącznie w części testowej.

| Właściwość | SMAP | MSL | Łącznie |
|---|---|---|---|
| Encje | 55 | 27 | **82** |
| Kanały na encję | 25 | 55 | — |
| Mediana długości zbioru treningowego | 2 851 | 2 158 | 2 690 |
| Mediana długości zbioru testowego | 8 308 | 2 277 | 7 884 |
| Typy anomalii | punktowe, kontekstowe | punktowe, kontekstowe | 62 punktowe / 43 kontekstowe |
| Zakres udziału anomalii | 0,4% – 50,1% | 0,4% – 50,1% | **0,4% – 50,1%** |

Zbiór danych stawia dwa odrębne wyzwania: **anomalie punktowe** (pojedyncze kroki czasowe o nietypowych wartościach) oraz **anomalie kontekstowe** (podsekwencje, które są anomalne w kontekście, choć nie indywidualnie). Szeroki zakres udziałów anomalii oraz różnica w liczbie kanałów między SMAP (25) a MSL (55) czynią go wymagającym benchmarkiem wielowarunkowym.

### 2.2 Oznaczone anomalie

Plik `labeled_anomalies.csv` zawiera indeksy segmentów anomalii w tablicy testowej dla każdej encji. Większość encji posiada jeden segment anomalii; trzy encje mają do trzech segmentów.

---

## 3. Przetwarzanie wstępne

Całe przetwarzanie wstępne jest zaimplementowane w `02_preprocessing.ipynb`.

### 3.1 Normalizacja

Scaler `StandardScaler` jest dopasowywany do części treningowej każdej encji **niezależnie** (dopasowanie na zbiorze treningowym, transformacja zbiorów treningowego/walidacyjnego/testowego). Zapobiega to wyciekowi danych i uwzględnia różne skale pomiarowe kanałów i encji.

### 3.2 Podział na zbiór treningowy i walidacyjny

Predefiniowany zbiór treningowy jest dzielony temporalnie: pierwsze 85% kroków czasowych tworzy partycję treningową, ostatnie 15% — partycję walidacyjną. Żadna z partycji nie zawiera anomalii (z założenia benchmarku), zatem zbiór walidacyjny służy wyłącznie do doboru hiperparametrów i kalibracji progu bez wycieku etykiet.

### 3.3 Okienkowanie

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| Rozmiar okna | 100 kroków czasowych | Odpowiada typowej długości anomalii; powszechny w literaturze dotyczącej SMAP/MSL |
| Krok treningowy | 10 | 10-krotna augmentacja danych przy ograniczeniu nadmiernej redundancji |
| Krok testowy | 1 | Wymagany do wyrównania wyników z krokami czasowymi |
| Krok walidacyjny | 10 | Spójny z treningiem |

Przy kroku testowym równym 1, indeks okna *i* odpowiada krokom czasowym *[i, i+100)*, a wynik przypisywany jest ostatniemu krokowi: `score_idx = i → test_timestep = i + 100`. To odwzorowanie jest stosowane w notebooku ewaluacyjnym do wyrównania wyników z prawdziwymi etykietami.

**Wynikowe liczby okien dla encji A-1 (SMAP, 25 kanałów):**

| Podział | Okna | Kształt |
|---|---|---|
| Treningowy | 235 | (235, 100, 25) |
| Walidacyjny | 34 | (34, 100, 25) |
| Testowy | 8 541 | (8541, 100, 25) |

---

## 4. Modele

### 4.1 LSTM Predictor

**Implementacja.** Własna implementacja od podstaw w **PyTorch** (`torch.nn.LSTM`, `torch.nn.Linear`). Brak zewnętrznych bibliotek modelowania sekwencyjnego — pełna kontrola nad architekturą, funkcją straty i obliczaniem wyniku anomalii.

**Architektura.** W odróżnieniu od autoenkodera rekonstrukcyjnego, predyktor jest trenowany do prognozowania ostatnich `K = 5` kroków czasowych każdego okna na podstawie poprzednich `W − K = 95` kroków. Jest to trudniejszy cel: model nie może kopiować wejścia i musi nauczyć się rzeczywistej dynamiki czasowej.

```
Wejście:  window[:, :95, :]   →  (B, 95, C)
Wyjście:  window[:, 95:, :]   →  (B, 5, C)   prognozowane
```

Model składa się z jednowarstwowego enkodera LSTM, po którym następuje liniowa głowica projekcyjna:

```
LSTM(C → hidden_dim) → h_last → Linear(hidden_dim, C × 5) → reshape (B, 5, C)
```

**Punktacja anomalii.** Dla każdego okna obliczane jest MSE per kanał między prognozowanymi a rzeczywistymi ostatnimi 5 krokami, a następnie jako wynik okna przyjmowane jest **maksimum po kanałach**. Użycie maksimum zamiast średniej zapobiega uśrednieniu anomalnych kanałów przez (zazwyczaj liczniejszy) zbiór kanałów normalnych.

```python
mse_per_channel = ((pred - target) ** 2).mean(dim=1)   # (B, C)
score = mse_per_channel.max(dim=1).values               # (B,)
```

**Hiperparametry (Optuna, 50 prób × 10 encji SMAP + 10 encji MSL):**

| Parametr | Przestrzeń przeszukiwań | SMAP | MSL |
|---|---|---|---|
| `hidden_dim` | {32, 64, 128, 256} | **128** | **128** |
| `num_layers` | 1–3 | **1** | **1** |
| `lr` | log-jednostajny [1e-4, 1e-2] | **0,00371** | **0,00142** |
| `dropout` | jednostajny [0,0; 0,3] | **0,131** | **0,208** |

Oba podzbiory wybrały tę samą architekturę (hidden_dim=128, 1 warstwa), lecz odmienne tempo uczenia — MSL preferuje wolniejsze uczenie, co odpowiada większej złożoności 55-kanałowych sekwencji.

### 4.2 Normalizing Flows (MAF)

**Implementacja.** Wykorzystano bibliotekę **`normflows`** (≥ 1.7) zamiast implementacji od podstaw. Biblioteka dostarcza gotowe bloki przepływów (`AutoregressiveRationalQuadraticSpline`, `LULinearPermute`) i klasę `NormalizingFlow` zarządzającą log-jakobianem. Własny kod obejmuje wyłącznie pętlę treningową, obliczanie wyników anomalii i wygładzanie temporalne. Wybór biblioteki był podyktowany złożonością numeryczną warstw autoregresyjnych — ręczna implementacja wymagałaby precyzyjnego zarządzania log-jakobianami, co jest podatne na błędy. Istotnym ograniczeniem jest brak wsparcia `normflows` dla akceleracji MPS (Apple Silicon), przez co trening wykonywany jest na CPU, co wydłuża go do ~1 godz. 20 min (wobec ~12 sek dla LSTM na MPS).

**Architektura.** Masked Autoregressive Flow (MAF) zbudowany z warstw Neural Spline Flow (`AutoregressiveRationalQuadraticSpline`). Model jest stosowany **per krok czasowy**: każdy indywidualny wektor obserwacji o kształcie `(C,)` traktowany jest jako niezależna próbka. Dzięki temu wymiar wejściowy pozostaje mały (25 lub 55), a trening wykonalny.

```
Dla każdego kroku czasowego t:
    score(t) = -log p_θ(x_t)
```

Przepływy są układane w stos z warstwami `LULinearPermute` między nimi, umożliwiając różne porządki w autoregresyjnej faktoryzacji.

**Dlaczego podejście per krok czasowy.** Zastosowanie przepływu do spłaszczonych okien wymagałoby modelowania rozkładu o wymiarze 100 × C = 2 500 lub 5 500, co jest niewykonalne dla MAF. Podejście per krok czasowy rezygnuje z kontekstu temporalnego w modelu gęstości, lecz odzyskuje go poprzez wygładzanie wyników post-hoc.

**Wygładzanie temporalne.** Do surowych wyników log-prawdopodobieństwa per krok czasowy stosowana jest krocząca średnia w celu wymuszenia spójności temporalnej:

```python
smoothed = np.convolve(raw_scores, np.ones(20) / 20, mode="same")
```

**Hiperparametry (Optuna, 50 prób × 10 encji SMAP + 10 encji MSL):**

| Parametr | Przestrzeń przeszukiwań | SMAP | MSL |
|---|---|---|---|
| `num_flows` | {2, 3, 4, 6} | **2** | **2** |
| `hidden_units` | {32, 64, 128} | **64** | **64** |
| `lr` | log-jednostajny [1e-4, 1e-2] | **0,00142** | **0,00193** |
| `smooth_window` | {5, 10, 20, 40} | **20** | **10** |

Strategia wygładzania różni się między podzbiorami: SMAP (anomalie krótsze, punktowe) korzysta z dłuższego okna 20, natomiast MSL (anomalie dłuższe i wyraźniejsze) z okna 10. Sugeruje to, że Optuna poprawnie wykryła tę różnicę.

### 4.3 Diffusion Model (DDPM z ConvDenoiser)

**Implementacja.** Własna implementacja od podstaw w **PyTorch**, bez użycia bibliotek typu `diffusers` (które są ukierunkowane na dane obrazowe). Zaimplementowano: harmonogram szumu (`DiffusionScheduler` z liniowym harmonogramem β), cel epsilon-prediction oraz deterministyczny proces odwrotny w stylu DDIM. Sieć odszumiająca (`ConvDenoiser`) to własna architektura 1D-Conv z osadzeniami kroków czasowych — celowo prosta, aby zachować czytelność kodu i kontrolę nad procesem scoringu.

**Architektura.** Minimalny DDPM trenowany na spłaszczonych oknach o kształcie `(W × C,)` z jednowymiarowym konwolucyjnym odszumiaczem. Odszumiacz wewnętrznie przekształca wejście do postaci `(B, C, W)` na potrzeby operacji Conv1d, zachowując strukturę temporalną, którą sieć MLP by pominęła.

```
x_flat → reshape (B, C, W) → [t_embed broadcast] → stos Conv1d → (B, C, W) → spłaszczenie
```

Osadzenie czasu jest rzutowane na wymiar kanałów i rozgłaszane wzdłuż osi czasowej, umożliwiając odszumiaczowi warunkowanie swojego działania na bieżącym poziomie szumu.

**Proces wprzód (cel treningowy).** Standardowe przewidywanie szumu ε w DDPM: dla losowego `t ∈ [0, T-1]`, dodaj szum na poziomie `t` i przewiduj szum:

```
x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε,   ε ~ N(0, I)
L = MSE(model(x_t, t), ε)
```

**Punktacja anomalii.** Częściowe dodawanie szumu + deterministyczne odwrócenie (w stylu DDIM):

1. Dodaj szum na kroku `t_eval`, uzyskując `x_{t_eval}`
2. Uruchom wyuczony odszumiacz deterministycznie od `t_eval` do `t=0`
3. Wynik anomalii = `MSE(x₀, x̂₀)`

Dane normalne są wiernie rekonstruowane; anomalne wzorce są odszumiane w kierunku normalnej rozmaitości, zwiększając błąd rekonstrukcji.

**Hiperparametry (Optuna, 50 prób × 10 encji SMAP + 10 encji MSL):**

| Parametr | Przestrzeń przeszukiwań | SMAP | MSL |
|---|---|---|---|
| `hidden_dim` | {64, 128, 256} | **128** | **128** |
| `lr` | log-jednostajny [1e-4, 1e-2] | **0,000417** | **0,000968** |
| `t_eval` | int [10, 50] | **10** | **10** |

Stratyfikacja nie zmieniła optymalnego `t_eval` — 10 jest konsekwentnie wybierane dla obu grup. Potwierdza to, że problem nie leży w zakresie przeszukiwań, ale w zdolności modelu do nauki odszumiania przy wyższych poziomach szumu.

Konsekwentny wybór `t_eval = 10` (przy maksimum wynoszącym 50) wskazuje, że model osiąga najlepsze wyniki przy minimalnym poziomie zaszumienia — działając w istocie jako lekki autoenkoder odszumiający, a nie wykorzystując pełny proces dyfuzji.

**Konfiguracja treningu:** T = 200 kroków dyfuzji, N_EPOCHS = 80, BATCH_SIZE = 256.


### 4.4 Isolation Forest (punkt odniesienia)

**Implementacja.** Wykorzystano `sklearn.ensemble.IsolationForest` z biblioteki **scikit-learn**. Żadnego własnego kodu modelowania — jedynie otoczka pętli per encja, spłaszczanie okien i negacja wyników (`-score_samples`). Szybki w implementacji i treningu (~kilka sekund dla wszystkich 82 encji), co czyni go naturalnym punktem startowym każdego projektu anomaly detection.

**Uzasadnienie wyboru.** Isolation Forest to klasyczny, ensemble'owy algorytm wykrywania anomalii, oparty na losowym partycjonowaniu przestrzeni cech. Każdą próbkę izoluje się przez rekurencyjne tworzenie losowych drzew decyzyjnych — próbki anomalne są łatwiejsze do izolacji (krótsza ścieżka w drzewie) niż próbki normalne. Algorytm nie wymaga założeń o rozkładzie danych ani modelu sekwencyjnego.

Baseline służy jako **dolna granica jakości**: modele głębokiego uczenia powinny go wyraźnie przewyższać, aby uzasadnić swój koszt obliczeniowy. Brak porównania z klasyczną metodą jest standardowym zarzutem wobec prac z zakresu wykrywania anomalii.

**Sposób działania.** Każde okno jest spłaszczane do wektora `(window_size × n_channels,)`. IsolationForest jest dopasowywany do okien treningowych, a następnie ocenia okna testowe i walidacyjne niezależnie — bez żadnego modelowania struktury temporalnej.

```python
train_flat = train_windows.reshape(n_train, window_size * n_channels)
iso.fit(train_flat)
scores = -iso.score_samples(test_flat)   # wyższy wynik = bardziej anomalne
```

**Strojone hiperparametry (Optuna, 20 prób × 10 encji SMAP + 10 MSL):**

| Parametr | Przestrzeń przeszukiwań | SMAP | MSL |
|---|---|---|---|
| `n_estimators` | {50, 100, 200, 300} | **100** | **100** |
| `max_features` | {0,50; 0,75; 1,0} | **0,75** | **0,75** |

Zbieżność parametrów SMAP i MSL sugeruje, że dla tego algorytmu różnica w liczbie kanałów (25 vs 55) nie wpływa istotnie na optymalną konfigurację — w przeciwieństwie do modeli głębokich.

### 4.5 Strojenie hiperparametrów — metodologia i czasy

**Biblioteka i algorytm.** Do strojenia hiperparametrów we wszystkich trzech modelach wykorzystano bibliotekę **Optuna** z domyślnym samplerem **TPE** (Tree-structured Parzen Estimator). TPE modeluje zależność między wartościami hiperparametrów a wynikiem walidacyjnym jako dwie oddzielne funkcje gęstości prawdopodobieństwa — jedną dla konfiguracji dających dobre wyniki i jedną dla słabych — co pozwala na efektywniejszą eksplorację przestrzeni przeszukiwań niż losowe próbkowanie przy tej samej liczbie prób. Kierunek optymalizacji we wszystkich przypadkach to **minimalizacja** metryki walidacyjnej (błąd MSE dla LSTM i Diffusion, negatywne log-prawdopodobieństwo dla Flows).

**Encje użyte do strojenia.** Strojenie przeprowadzono **osobno dla każdego podzbioru**: 10 pierwszych encji SMAP (A-1 do B-1, 25 kanałów) oraz 10 pierwszych encji MSL (C-1 do M-1, 55 kanałów). Łącznie 20 encji × 50 prób = 1 000 ocen modelu na model. Podejście stratyfikowane eliminuje błąd wynikający ze strojenia wyłącznie na encjach SMAP i aplikowania tych parametrów do strukturalnie odmiennych encji MSL.

**Liczba prób i czas pojedynczej próby.** Każdy model był strojony przez **50 prób na każdej z 20 encji** (10 SMAP + 10 MSL), co daje łącznie **1 000 ocen modelu** na model. Każda próba obejmuje: inicjalizację modelu → trening przez ograniczoną liczbę epok (30 dla LSTM, 10 dla Flows, 10 dla Diffusion przy skróconej skali czasowej T = 100) → obliczenie metryki walidacyjnej i zwrot wartości do Optuny.

**Agregacja wyników między encjami.** Po zakończeniu strojenia na wszystkich 10 encjach najlepsze parametry z każdej encji są agregowane do jednego globalnego zestawu stosowanego następnie do treningu wszystkich 82 encji:

| Parametr | Metoda agregacji |
|---|---|
| `hidden_dim`, `num_flows`, `hidden_units`, `smooth_window` | Wartość z listy dozwolonych kategorii najbliższa **średniej arytmetycznej** wyników |
| `lr` | **Średnia geometryczna** (właściwa dla parametrów w skali logarytmicznej) |
| `num_layers`, `t_eval` | Zaokrąglona **mediana** |
| `dropout` | Średnia arytmetyczna |

**Wyniki strojenia — najlepsze hiperparametry per podzbór:**

| Parametr | LSTM (SMAP/MSL) | Flows (SMAP/MSL) | Diffusion (SMAP/MSL) | IF (SMAP/MSL) |
|---|---|---|---|---|
| Architektura | `h=128, L=1` / `h=128, L=1` | `f=2, hu=64` / `f=2, hu=64` | `h=128` / `h=128` | `n=100` / `n=100` |
| Kluczowy param | `do=0,131` / `do=0,208` | `sw=20` / `sw=10` | `t=10` / `t=10` | `mf=0,75` / `mf=0,75` |
| Współ. uczenia | `0,00371` / `0,00142` | `0,00142` / `0,00193` | `0,000417` / `0,000968` | — |

*(h=hidden\_dim, L=num\_layers, f=num\_flows, hu=hidden\_units, sw=smooth\_window, t=t\_eval, n=n\_estimators, mf=max\_features, do=dropout)*

**Czasy wykonania** (sprzęt: Apple Silicon MPS / Metal Performance Shaders):

| Faza | LSTM Predictor | Normalizing Flows (MAF) | Diffusion (DDPM) |
|---|---|---|---|
| Strojenie Optuna (2 x 10 encji × 50 prób) | ~ 10 min | ~40 min | ~3.5 min |
| Trening właściwy (82 encje) | ~30 sek | ~1 godz. | ~1 min 30 sek |
| Predykcja / punktowanie (82 encje) | ~12 sek | ~50 sek | ~1 min 30 sek |
| **Łącznie** | **~11 min** | **~1 godz. 41 min** | **~7 min** |

Wyjątkowo długi czas treningu Normalizing Flows (~1 godz. dla 82 encji × 50 epok) wynika z braku pełnego wsparcia biblioteki `normflows` dla akceleracji MPS na układach Apple Silicon — obliczenia wykonywane są przez CPU zamiast GPU MPS, eliminując korzyść z akceleracji sprzętowej dostępnej dla modeli PyTorch. LSTM Predictor i Diffusion korzystają w pełni z MPS, stąd ich łączne czasy są ponad **10-krotnie krótsze**.

Pomimo dominacji LSTM i Diffusion pod względem szybkości (około 10 minut), Normalizing Flows oferują najwyższe AUC-ROC, co może uzasadniać koszt obliczeniowy w zastosowaniach, gdzie jakość rankingu jest priorytetem. W środowisku z natywnym wsparciem GPU (CUDA) dysproporcja czasów byłaby prawdopodobnie znacznie mniejsza.

---

## 5. Protokół ewaluacji

### 5.1 Odwzorowanie wyników na kroki czasowe

Okna testowe są generowane z krokiem 1. Okno o indeksie `i` zawiera kroki czasowe `[i, i+100)`, zatem jego wynik przypisywany jest ostatniemu krokowi: `score_idx = i → test_timestep = i + 100`. Indeksy segmentów anomalii z pliku `labeled_anomalies.csv` są odpowiednio przesunięte: `label_score_idx = label_timestep − 100`.

Kroki czasowe w pierwszym oknie (indeksy 0–99) nie mogą być oceniane i są wykluczone z ewaluacji.

### 5.2 Kalibracja progu

Próg decyzyjny jest ustawiany na **95. percentyl wyników walidacyjnych** dla każdej encji. Jest to kalibracja bez użycia etykiet: żadne etykiety anomalii ze zbioru testowego nie są używane do wyznaczenia progu. W przypadku nielicznych encji bez okien walidacyjnych (krótkie sekwencje treningowe), jako wartość zastępcza stosowany jest 95. percentyl wyników testowych.

### 5.3 Metryki

**AUC-ROC** — jakość rankingu niezależna od progu. Podstawowa metryka do porównania modeli.

**F1 Raw** — standardowe F1 przy skalibrowanym progu. Rygorystyczna ewaluacja per krok czasowy. Penalizuje zarówno pominięcia, jak i fałszywe alarmy.

**F1 PA (Point-Adjust)** — jeśli wykryty zostanie jakikolwiek krok czasowy wewnątrz segmentu anomalii z prawdziwych etykiet, cały segment jest zaliczany jako prawdziwy wynik pozytywny. Jest to standardowy protokół stosowany w większości publikacji dotyczących SMAP/MSL. Jest bardziej pobłażliwy, ale lepiej odzwierciedla praktyczną użyteczność: detektor, który zgłasza alarm *gdziekolwiek* w oknie anomalii, jest przydatny nawet jeśli pomija inne kroki czasowe.

Oba protokoły są raportowane, aby umożliwić porównanie z wcześniejszą literaturą oraz ilościowe określenie przepaści między precyzją temporalną a wykrywaniem na poziomie segmentów.

---

## 6. Wyniki

### 6.1 Ogólna wydajność

| Model | AUC-ROC ↑ | F1 Raw ↑ | Precyzja Raw | Czułość Raw | F1 PA ↑ |
|---|---|---|---|---|---|
| LSTM Predictor | 0,606 | 0,172 | 0,218 | 0,332 | **0,610** |
| Normalizing Flows (MAF) | **0,643** | 0,159 | 0,190 | 0,251 | 0,577 |
| Diffusion (DDPM) | 0,599 | **0,176** | 0,179 | **0,400** | 0,388 |
| Isolation Forest | 0,474 | 0,102 | 0,121 | 0,268 | 0,480 |

*Makro-średnia po 81 ocenianych encjach (1 encja pominięta — brak możliwych do oceny kroków czasowych).*

### 6.2 Porównanie SMAP i MSL

| Model | SMAP AUC | SMAP F1-PA | MSL AUC | MSL F1-PA |
|---|---|---|---|---|
| LSTM Predictor | 0,597 | 0,582 | 0,623 | **0,665** |
| Normalizing Flows (MAF) | 0,627 | 0,523 | **0,676** | 0,686 |
| Diffusion (DDPM) | 0,567 | 0,312 | 0,663 | 0,541 |
| Isolation Forest | 0,449 | 0,470 | 0,524 | 0,500 |

### 6.3 Różnica między F1 PA a F1 Raw

| Model | F1 Raw | F1 PA | Różnica |
|---|---|---|---|
| LSTM Predictor | 0,172 | 0,610 | **+0,438** |
| Normalizing Flows (MAF) | 0,159 | 0,577 | +0,417 |
| Diffusion (DDPM) | 0,176 | 0,388 | +0,211 |
| Isolation Forest | 0,102 | 0,480 | +0,377 |

---

## 7. Analiza i dyskusja

### 7.1 Normalizing Flows: najlepszy ranking, słaba precyzja na MSL

MAF osiąga najlepsze AUC-ROC ogółem (0,643) oraz najwyższy wynik na MSL w szczególności (0,676). Sugeruje to, że punktacja oparta na gęstości prawidłowo szereguje anomalne kroki czasowe względem normalnych. Jednak surowe F1 na MSL spada do **0,093** — jednego z niższych wyników, ustępując jedynie baseline. Rozbieżność ta ujawnia problem z precyzją: model generuje wiele alarmów o niskiej pewności rozsianych po zbiorze testowym, co oznacza, że próg 95. percentyla nie rozdziela czysto anomalii od normalnej zmienności. Podejście per krok czasowy, choć wykonalne, odrzuca kontekst temporalny, który mógłby wyostrzyć granicę decyzyjną.

Po stratyfikacji Optuna wybrała `smooth_window=10` dla MSL (wobec 20 dla SMAP) oraz `hidden_units=64` dla obu grup (wobec 32 w poprzednim strojeniu tylko na SMAP). Wzrost `hidden_units` przełożył się na wyższe AUC-ROC dla MSL (+0,676 vs poprzednie +0,649 bez stratyfikacji), co potwierdza wartość osobnego strojenia.

### 7.2 LSTM Predictor: najlepszy praktyczny detektor

LSTM osiąga najwyższe F1 PA (0,610) i działa konsekwentnie na obu podtypach statków kosmicznych. Cel predykcyjny — prognozowanie kolejnych 5 kroków czasowych na podstawie 95 kroków kontekstu — stanowi silniejszy sygnał treningowy niż rekonstrukcja autoenkoderowa, wymuszając na modelu naukę rzeczywistych zależności temporalnych zamiast odwzorowań tożsamościowych.

Strategia punktowania maksimum po kanałach przyczyniła się do poprawy w stosunku do wcześniejszego autoenkodera: anomalie w SMAP/MSL często ujawniają się w małej podgrupie kanałów, a uśrednianie tłumiłoby te sygnały.

Duża różnica PA (+0,438) ujawnia wzorzec wykrywania modelu: model reaguje gdzieś w obrębie większości segmentów anomalii (wysoka czułość PA), lecz z nieprecyzyjnym wyznaczeniem czasu, generując wiele fałszywych alarmów w sąsiednich normalnych regionach (niska surowa precyzja = 0,218). Jest to częściowo konsekwencja tego, że sygnał błędu predykcji jest zaszumiony w pobliżu granic anomalii.

### 7.3 Diffusion Model: najwyższa czułość, najniższa precyzja temporalna

Model dyfuzji osiąga najwyższą surową czułość (0,415), ale najniższe F1 PA (0,376) — nietypowe połączenie. Reaguje szeroko i wykrywa wiele prawdziwie anomalnych kroków czasowych indywidualnie, ale jego wykrywanie na poziomie segmentów jest słabsze niż w przypadku pozostałych modeli. Może to wydawać się nieintuicyjne — wysoka czułość powinna przekładać się na wysokie F1 PA — lecz metryka PA wymaga wykrycia co najmniej jednego punktu *wewnątrz* każdego oznaczonego segmentu. Jeśli wyniki dyfuzji są ogólnie podwyższone w całym zbiorze testowym (zamiast koncentrować się w segmentach anomalii), czułość może być wysoka, podczas gdy F1 PA pozostaje niskie.

Konsekwentny wybór `t_eval = 10` przy maksimum wynoszącym 50 jest najbardziej informatywnym sygnałem z tego modelu. Przy T = 200 całkowitych kroków, `t_eval = 10` odpowiada zaledwie 5% maksymalnego poziomu szumu — model stosuje prawie zerowe zaszumienie przed próbą rekonstrukcji. Sugeruje to, że ConvDenoiser nauczył się odwzorowania bliskiego tożsamościowemu dla danych normalnych (co jest pożądane), ale nie rozwinął znaczącej zdolności odszumiania przy wyższych poziomach szumu. Zwiększenie liczby epok treningowych lub rozszerzenie przestrzeni przeszukiwań `t_eval` na wyższe wartości (np. do T//3 = 66) może pomóc modelowi odróżniać anomalie poprzez bardziej istotne zniszczenie sygnału.

### 7.4 Isolation Forest: potwierdzenie wartości modelowania temporalnego

Wyniki Isolation Forest dostarczają istotnych spostrzeżeń, wykraczających poza samą rolę punktu odniesienia.

**AUC-ROC poniżej 0,5 na SMAP (0,449).** Wynik poniżej poziomu losowego oznacza, że algorytm przypisuje *wyższe* wyniki anomalii oknom normalnym niż anomalnym — odwrotność pożądanego zachowania. Jest to klasyczny symptom **anomalii kontekstowych**: okna anomalne nie są izolowanymi punktami oddalonymi od rozkładu normalnego, lecz sekwencjami, które *lokalnie wyglądają normalnie*, a stają się anomalne dopiero w kontekście poprzedzającego i następującego wzorca. IsolationForest, operując na spłaszczonych oknach bez pamięci temporalnej, nie jest w stanie wykryć tego rodzaju odchyleń.

**Wynik na MSL (AUC=0,524) jest zaledwie marginalnie powyżej losowego.** MSL zawiera anomalie bardziej strukturalnie wyróżniające się (dłuższe, o wyższych udziałach), stąd IF osiąga tam lepszy wynik — lecz wciąż bliski zeru.

**F1-PA Isolation Forest (0,480) jest wyższe niż Diffusion (0,388).** To niepokojący wynik: klasyczny baseline bez modelowania temporalnego przewyższa na poziomie segmentów model oparty na głębokim uczeniu. Wynika to z szerokiej reakcji IF — model generuje wiele alarmów rozkładając je równomiernie po zbiorze testowym, co przypadkowo trafia w niektóre segmenty anomalii. Diffusion, z niskim AUC-ROC (0,599), ma podobny problem słabego rankingowania.

**Kluczowy wniosek:** modele temporalne (LSTM, Flows) wyraźnie przewyższają IF na AUC-ROC (+0,13–0,17), co potwierdza, że modelowanie zależności czasowych jest niezbędne dla tego benchmarku. Wynik Diffusion zbliżony do baseline wskazuje na dalszy potencjał poprawy tego modelu.

### 7.5 SMAP vs MSL

Wszystkie trzy modele osiągają wyższe AUC-ROC na MSL niż na SMAP. Encje MSL charakteryzują się:
- Większą liczbą kanałów (55 vs 25), dostarczając bogatszego sygnału dla dystrybucyjnego wykrywania anomalii
- Dłuższymi i bardziej strukturalnie odrębnymi wzorcami anomalii
- Wyższymi bezwzględnymi udziałami anomalii w niektórych encjach (do 50%)

Anomalie SMAP są zazwyczaj krótsze (często punktowe) i kontekstowe, co utrudnia ich wykrycie bez silnego modelowania temporalnego. Przewaga Flows na MSL F1-PA (0,686) wobec LSTM (0,665) sugeruje, że jawne modelowanie temporalne staje się bardziej wartościowe w przypadku dłuższych sekwencji anomalii.

### 7.6 Protokół Raw vs Point-Adjust

Duże różnice PA/raw we wszystkich modelach (+0,211 do +0,438) wskazują, że choć modele z powodzeniem identyfikują *które segmenty zawierają anomalie* (wykrywanie na poziomie segmentów), nie potrafią zlokalizować anomalii do precyzyjnych kroków czasowych (surowa precyzja). Jest to znane ograniczenie wykrywania anomalii opartego na rekonstrukcji: wynik anomalii zmienia się stopniowo w pobliżu granic segmentów, co utrudnia precyzyjną klasyfikację na poziomie kroków czasowych.

W praktycznym wdrożeniu zachowanie to przekłada się na: systemy niezawodnie wyzwalają alarm, gdy zaczyna się segment anomalii, lecz generują również fałszywe alarmy na normalnych danych sąsiadujących z regionami anomalii. Dostrojenie progu do wyższego percentyla (np. 99.) poprawiłoby precyzję kosztem czułości.

---

## 8. Wnioski

### 8.1 Kluczowe ustalenia

1. **Normalizing Flows produkują najlepszy ranking anomalii** (AUC-ROC 0,643), szczególnie na MSL (0,676), lecz słabo przekładają się na precyzyjne wykrycia przy stałym progu. Stratyfikowane strojenie podniosło `hidden_units` z 32 do 64, co poprawiło wyniki na MSL.

2. **LSTM Predictor jest najlepiej praktycznie użytecznym detektorem** (F1-PA 0,610). Przejście od rekonstrukcji autoenkodera do wielokrokowej predykcji, w połączeniu z punktowaniem maksimum po kanałach, przyniosło największą pojedynczą poprawę w ramach eksperymentu.

3. **Model dyfuzji osiąga wyniki poniżej swojego potencjału**. Wynik `t_eval = 10` wskazuje, że ConvDenoiser nauczył się minimalnego odszumiania — działając w istocie jako lekki model rekonstrukcyjny. Architektura jest poprawna, lecz wymaga albo dłuższego treningu, albo wymuszonej eksploracji wyższych wartości `t_eval`.

4. **MSL jest łatwiejszym benchmarkiem niż SMAP** dla wszystkich modeli. Dłuższe, bardziej strukturalnie odrębne anomalie i bogatsze informacje kanałowe (55 vs 25) prowadzą do konsekwentnie wyższego AUC-ROC.

5. **Isolation Forest (AUC-ROC 0,474) potwierdza konieczność modelowania temporalnego.** Wynik poniżej poziomu losowego na SMAP (0,449) dowodzi, że anomalie w tym zbiorze są kontekstowe i niemożliwe do wykrycia bez pamięci temporalnej. Każdy z modeli DL wyraźnie przewyższa baseline na AUC-ROC (+0,13–0,17).

6. **Różnica PA/raw jest duża we wszystkich modelach** (+0,211 do +0,438). Precyzyjna lokalizacja na poziomie kroków czasowych pozostaje otwartym wyzwaniem dla wszystkich czterech podejść.

### 8.2 Rekomendacje modeli według przypadku użycia

| Przypadek użycia | Zalecany model | Uzasadnienie |
|---|---|---|
| Ranking / priorytetyzacja anomalii | Normalizing Flows | Najwyższe AUC-ROC (0,643) |
| Wyzwalanie alertów (poziom segmentów) | LSTM Predictor | Najwyższe F1-PA (0,610), najniższy wskaźnik pominięć |
| Ograniczone zasoby obliczeniowe | LSTM Predictor | Trening w ~12 sek, pełny pipeline w ~9 min |
| Szybki prototyp / punkt startowy | Isolation Forest | Zero czasu GPU, lecz słaby ranking (AUC 0,474) |

### 8.3 Kierunki przyszłych prac

- **LSTM:** Dostrajanie percentyla progu per encja zamiast stałego 95. percentyla. Eksploracja okien predykcji wielokrokowej (K = 10, 20) w celu wyostrzenia wykrywania granic.
- **Przepływy:** Zastosowanie okienkowanego MAF na reprezentacjach skompresowanych przez PCA (32 komponenty) w celu przywrócenia kontekstu temporalnego przy zachowaniu wykonalnej wymiarowości.
- **Dyfuzja:** Wymuszenie eksploracji wyższych wartości `t_eval` w Optunie (zakres [30, T//2]) i wydłużenie epok treningowych w celu umożliwienia rzeczywistego odszumiania na pośrednich poziomach szumu. Alternatywnie, ocena czy wariacyjny autoenkoder (VAE) osiąga podobną punktację opartą na rekonstrukcji przy ułamku nakładu obliczeniowego.
- **Wszystkie modele:** Per-spacecraft tuning jest już zaimplementowany. Następny krok: osobna kalibracja progu percentylowego per podzbór (np. 97. percentyl dla SMAP, 93. dla MSL) zamiast stałego 95. percentyla.

---

## 9. Odtwarzalność

| Notebook | Przeznaczenie |
|---|---|
| `01_eda.ipynb` | Eksploracja i wizualizacja zbioru danych |
| `02_preprocessing.ipynb` | Normalizacja, okienkowanie, podział na zbiory treningowy/walidacyjny/testowy |
| `03_lstm.ipynb` | LSTM Predictor: trening, osobne Optuna SMAP/MSL, punktowanie |
| `03b_baseline.ipynb` | Isolation Forest: Optuna, dopasowanie, punktowanie |
| `04_flows.ipynb` | Normalizing Flows (MAF): trening, osobne Optuna SMAP/MSL, punktowanie |
| `05_diffusion.ipynb` | DDPM ConvDenoiser: trening, osobne Optuna SMAP/MSL, punktowanie |
| `06_evaluation.ipynb` | Metryki, wykresy porównawcze, wnioski |

**Zależności:** PyTorch ≥ 2.2, normflows ≥ 1.7, Optuna ≥ 4.0, NumPy, pandas, scikit-learn, Matplotlib, Seaborn. Zarządzane przez `uv` (`pyproject.toml`).

**Sprzęt:** Apple Silicon MPS (Metal Performance Shaders). Wybór urządzenia automatycznie przechodzi na CUDA, a następnie CPU.

**Kolejność wykonania:** `02` → `03b` / `03` / `04` / `05` (niezależnie) → `06`.
