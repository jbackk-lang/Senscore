# Senscore
warstwa korelacji z czułością / charakterystyką czujników
SENSCORE Pipeline — README
Opis projektu
senscoreAll.py implementuje kompletny pipeline filtracji sygnałów z detektorów, oparty na pięciu warstwach:

SENSCORE — filtr czułości czujników

TRM — filtr topologiczny

TIMDR — filtr dynamiki

GIA — filtr fizyczno‑semantyczny

FIELDCORE — filtr stabilizujący

Pipeline został zaprojektowany do redukcji fałszywych sygnałów (false positives) w danych z detektorów cząstek, czujników przemysłowych lub systemów pomiarowych o wysokiej gęstości sygnałów.

Struktura pliku
Plik senscoreAll.py zawiera:

definicje struktur danych (Hit, Event, SensorCalibration),

implementacje pięciu filtrów,

klasę FullPipeline, która łączy wszystkie filtry w jedną sekwencję,

przykładowy kod uruchomieniowy.

Pipeline filtracji
1. SENSCORE — filtr czułości czujników
Warstwa wejściowa.
Skalibrowane wagi sygnałów na podstawie:

czułości kanału,

poziomu szumu własnego,

wiarygodności czujnika.

Cel:  
odrzucić lub osłabić sygnały z kanałów o niskiej jakości zanim trafią do dalszej analizy.

2. TRM — filtr topologiczny
Buduje lokalną topologię zdarzenia:

analizuje sąsiedztwo przestrzenne i czasowe,

odrzuca izolowane hity,

zachowuje struktury spójne.

Cel:  
usunąć szum, który nie tworzy żadnej fizycznej lub geometrycznej struktury.

3. TIMDR — filtr dynamiki
Analiza dynamiki sygnału:

spójność energetyczna,

spójność czasowa,

odrzucanie outlierów.

Cel:  
wyłapać sygnały niezgodne z globalną dynamiką zdarzenia.

4. GIA — filtr fizyczno‑semantyczny
Analiza zgodności z modelem fizycznym:

dopasowanie toru (PCA),

odrzucanie punktów o dużej resztowej odległości od osi toru.

Cel:  
zachować tylko sygnały zgodne z fizyką zdarzenia.

5. FIELDCORE — filtr stabilizujący
Końcowa stabilizacja:

wygładzanie energii,

usuwanie „gorących pikseli”,

finalne czyszczenie.

Cel:  
uzyskać stabilny, oczyszczony zestaw hitów.

Uruchomienie
Minimalny przykład:

python
from senscoreAll import FullPipeline, Event, Hit, SensorCalibration

calibration = {
    1: SensorCalibration(1, sensitivity=1.0, noise_level=0.1, reliability=0.9),
    2: SensorCalibration(2, sensitivity=0.8, noise_level=0.5, reliability=0.7),
}

hits = [
    Hit(sensor_id=1, x=0, y=0, z=0, t=0, energy=10, raw_value=1),
    Hit(sensor_id=1, x=1, y=0.1, z=0, t=1, energy=11, raw_value=1.1),
    Hit(sensor_id=2, x=50, y=50, z=0, t=0.5, energy=0.5, raw_value=0.2),
]

event = Event(hits=hits)
pipeline = FullPipeline(calibration_map=calibration)

filtered = pipeline.process(event)
print(filtered)
Zastosowania
detektory cząstek (CERN‑style),

systemy sensorów przemysłowych,

systemy pomiarowe o dużej gęstości sygnałów,

filtrowanie danych z kamer ToF / LiDAR,

analiza sygnałów w systemach robotycznych.

Cele projektu
redukcja fałszywych sygnałów o 40–50%,

stabilizacja danych wejściowych,

modularna architektura filtrów,

możliwość podmiany filtrów na wersje ML/GNN.

Plan rozwoju
dodanie wersji GNN dla TRM i GIA,

dodanie dynamicznej kalibracji SENSCORE,

integracja z realnymi danymi z detektorów,

wersja GPU.
