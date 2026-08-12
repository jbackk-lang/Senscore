# Senscore — przegląd kodu i testy

Przegląd repozytorium [jbackk-lang/Senscore](https://github.com/jbackk-lang/Senscore): pipeline filtracji sygnałów z detektorów (`senscoreAll.py`), oparty na pięciu warstwach — SENSCORE, TRM, TIMDR, GIA, FIELDCORE.

## Znaleziony błąd

**`GIAFilter` odrzucał prawdziwy tor, a zachowywał szum.**

Filtr liczył oś toru zwykłym PCA na wszystkich hitach naraz. Zwykłe PCA nie jest odporne na odstające punkty — jeden hit daleko od toru potrafi zdominować wariancję i obrócić główną oś PCA w swoim kierunku.

Przykład testowy: linia 10 hitów wzdłuż osi X + jeden odstający punkt `(5, 50, 0)`.

| Wersja | Zachowane hity | Wynik |
|---|---|---|
| Przed poprawką | 3 (2 hity blisko środka toru + sam szum) | 7 poprawnych hitów z końców toru odrzuconych, szum zachowany |
| Po poprawce | 10 (cały tor) | szum poprawnie odrzucony |

**Poprawka:** oś toru jest teraz szacowana metodą zbliżoną do RANSAC — losowane są pary punktów jako kandydackie proste, wybierany kandydat z największą liczbą „inlierów” w zasięgu `max_residual`, a finalna oś dopasowywana PCA tylko do tych inlierów. Dodatkowo `np.linalg.eig` zamieniono na `np.linalg.eigh` (macierz kowariancji jest symetryczna, więc `eigh` jest szybsze i zawsze zwraca wartości rzeczywiste).

Przykład z README repozytorium działa bez zmian po poprawce.

## Uwaga (nie błąd, ale warto wiedzieć)

Domyślny próg `FIELDCOREFilter` (10 sigma) jest znacznie luźniejszy niż próg `TIMDRFilter` (3 sigma), który działa wcześniej w pipeline. W efekcie `FIELDCOREFilter` przy domyślnej konfiguracji praktycznie nigdy nic nie usuwa — wszystko, co wystarczająco odstaje, zostało już odrzucone przez TIMDR. Udokumentowane testem `test_default_threshold_rarely_triggers`.

## Wyniki testów

26 testów, `python3 -m pytest test_senscoreAll.py -v` — **26 passed**.

| # | Warstwa | Test | Co sprawdza | Wynik |
|---|---|---|---|---|
| 1 | SENSCORE | `test_weight_formula` | poprawność wzoru wagi (czułość × wiarygodność / (1 + szum)) | ✅ PASSED |
| 2 | SENSCORE | `test_missing_calibration_defaults_to_weight_one` | brak kalibracji → waga domyślna 1.0 | ✅ PASSED |
| 3 | SENSCORE | `test_empty_event` | pusty event nie wywołuje błędu | ✅ PASSED |
| 4 | SENSCORE | `test_zero_reliability_zeroes_signal` | zerowa wiarygodność zeruje sygnał | ✅ PASSED |
| 5 | TRM | `test_isolated_hit_removed` | odległe, izolowane hity są odrzucane | ✅ PASSED |
| 6 | TRM | `test_clustered_hits_kept` | hity blisko siebie w czasie/przestrzeni są zachowane | ✅ PASSED |
| 7 | TRM | `test_single_hit_always_dropped` | pojedynczy hit w evencie zawsze traktowany jako szum | ✅ PASSED |
| 8 | TRM | `test_empty_event` | pusty event nie wywołuje błędu | ✅ PASSED |
| 9 | TIMDR | `test_energy_outlier_removed` | skrajny outlier energetyczny jest odrzucany | ✅ PASSED |
| 10 | TIMDR | `test_time_tails_trimmed_when_span_too_large` | zbyt rozciągnięte w czasie zdarzenie jest przycinane do środkowych 80% | ✅ PASSED |
| 11 | TIMDR | `test_time_not_trimmed_when_span_small` | krótkie zdarzenie nie jest przycinane | ✅ PASSED |
| 12 | TIMDR | `test_empty_event` | pusty event nie wywołuje błędu | ✅ PASSED |
| 13 | TIMDR | `test_constant_energy_no_div_by_zero` | stała energia (std=0) nie powoduje dzielenia przez zero | ✅ PASSED |
| 14 | GIA | `test_fewer_than_three_hits_passthrough` | poniżej 3 hitów filtr nic nie robi | ✅ PASSED |
| 15 | GIA | `test_linear_track_kept` | idealnie liniowy tor jest w całości zachowany | ✅ PASSED |
| 16 | GIA | `test_off_axis_point_removed` | **regresja dla naprawionego błędu** — tor zachowany, szum odrzucony | ✅ PASSED |
| 17 | GIA | `test_identical_points_no_crash` | identyczne punkty (macierz kowariancji zerowa) nie wywołują błędu | ✅ PASSED |
| 18 | GIA | `test_residuals_are_real_not_complex` | 200 losowych konfiguracji — brak wyjątków, wyniki liczbowe poprawne | ✅ PASSED |
| 19 | GIA | `test_empty_event` | pusty event nie wywołuje błędu | ✅ PASSED |
| 20 | FIELDCORE | `test_default_threshold_rarely_triggers` | dokumentuje: domyślny próg (10σ) praktycznie nic nie usuwa | ✅ PASSED |
| 21 | FIELDCORE | `test_explicit_low_threshold_removes_hot_pixel` | niższy próg poprawnie usuwa „gorący piksel” | ✅ PASSED |
| 22 | FIELDCORE | `test_empty_event` | pusty event nie wywołuje błędu | ✅ PASSED |
| 23 | Pipeline | `test_readme_example_runs` | przykład z README repozytorium działa bez błędów | ✅ PASSED |
| 24 | Pipeline | `test_empty_event_through_pipeline` | pusty event przechodzi przez cały pipeline bez błędu | ✅ PASSED |
| 25 | Pipeline | `test_single_hit_event_ends_up_empty` | pojedynczy hit kończy jako pusty event (efekt TRM) | ✅ PASSED |
| 26 | Pipeline | `test_large_random_event_no_crash` | 200 losowych hitów, 5 czujników — cały pipeline bez błędów, brak NaN | ✅ PASSED |

## Pliki

- `senscoreAll.py` — kod źródłowy z poprawką w `GIAFilter`
- `test_senscoreAll.py` — zestaw 26 testów (`pytest`)

## Uruchomienie testów

```bash
pip install pytest numpy
python3 -m pytest test_senscoreAll.py -v
```
