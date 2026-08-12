from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class Hit:
    sensor_id: int
    x: float
    y: float
    z: float
    t: float
    energy: float
    raw_value: float  # surowy sygnał z kanału


@dataclass
class Event:
    hits: List[Hit]


@dataclass
class SensorCalibration:
    sensor_id: int
    sensitivity: float      # czułość (gain)
    noise_level: float      # poziom szumu własnego
    reliability: float      # wiarygodność kanału (0–1)


class SENSCOREFilter:
    """
    Filtr korelacji z czułością czujników.
    Waży sygnały w zależności od:
    - czułości,
    - szumu,
    - wiarygodności kanału.
    """

    def __init__(self, calibration_map: Dict[int, SensorCalibration]):
        self.calibration_map = calibration_map

    def apply(self, event: Event) -> Event:
        weighted_hits = []
        for h in event.hits:
            cal = self.calibration_map.get(h.sensor_id, None)
            if cal is None:
                # brak kalibracji → traktuj jako średni kanał
                weight = 1.0
            else:
                # prosty model wagi:
                # im większa czułość i wiarygodność, tym większa waga,
                # im większy szum, tym mniejsza waga.
                weight = cal.sensitivity * cal.reliability / (1.0 + cal.noise_level)

            # przeskaluj energię / wartość
            new_energy = h.energy * weight
            new_raw = h.raw_value * weight

            weighted_hits.append(
                Hit(
                    sensor_id=h.sensor_id,
                    x=h.x,
                    y=h.y,
                    z=h.z,
                    t=h.t,
                    energy=new_energy,
                    raw_value=new_raw,
                )
            )

        return Event(hits=weighted_hits)


class TRMFilter:
    """
    Topologiczny filtr TRM:
    - buduje prostą strukturę sąsiedztwa,
    - odrzuca hity, które nie mają sensownych sąsiadów
      (np. izolowane szumy).
    """

    def __init__(self, max_distance: float = 10.0, max_dt: float = 5.0):
        self.max_distance = max_distance
        self.max_dt = max_dt

    def apply(self, event: Event) -> Event:
        hits = event.hits
        if not hits:
            return event

        positions = np.array([[h.x, h.y, h.z] for h in hits])
        times = np.array([h.t for h in hits])

        kept_hits = []
        for i, h in enumerate(hits):
            # szukamy sąsiadów w przestrzeni i czasie
            dpos = np.linalg.norm(positions - positions[i], axis=1)
            dt = np.abs(times - times[i])

            neighbors = np.where((dpos < self.max_distance) & (dt < self.max_dt))[0]
            # jeśli hit jest całkowicie izolowany → traktujemy jako szum
            if len(neighbors) > 1:  # >1, bo zawsze znajdzie samego siebie
                kept_hits.append(h)

        return Event(hits=kept_hits)


class TIMDRFilter:
    """
    Filtr dynamiki TIMDR:
    - ocenia spójność czasową i energetyczną,
    - odrzuca hity, które nie pasują do globalnej dynamiki zdarzenia.
    """

    def __init__(self, max_energy_deviation: float = 3.0, max_time_spread: float = 20.0):
        self.max_energy_deviation = max_energy_deviation
        self.max_time_spread = max_time_spread

    def apply(self, event: Event) -> Event:
        hits = event.hits
        if not hits:
            return event

        energies = np.array([h.energy for h in hits])
        times = np.array([h.t for h in hits])

        mean_energy = np.mean(energies)
        std_energy = np.std(energies) + 1e-6

        time_span = np.max(times) - np.min(times)

        kept_hits = []
        for h in hits:
            # filtr energii: odrzucamy skrajne outliery
            z_energy = (h.energy - mean_energy) / std_energy
            if np.abs(z_energy) > self.max_energy_deviation:
                continue

            # filtr czasu: jeśli całe zdarzenie jest zbyt rozciągnięte,
            # możemy odrzucić hity z „ogonów”
            if time_span > self.max_time_spread:
                # prosty warunek: trzymaj tylko środkowe 80% czasu
                t_min = np.percentile(times, 10)
                t_max = np.percentile(times, 90)
                if not (t_min <= h.t <= t_max):
                    continue

            kept_hits.append(h)

        return Event(hits=kept_hits)


class GIAFilter:
    """
    Filtr fizyczno-semantyczny GIA:
    - sprawdza zgodność z prostym modelem fizycznym:
      np. tor w polu magnetycznym ~ łuk / helisa.
    Tu: uproszczony wariant → preferujemy hity układające się w
    mniej więcej liniową / łukową strukturę.

    POPRAWKA: pierwotna wersja liczyła oś toru zwykłym PCA na WSZYSTKICH
    hitach naraz. Zwykłe PCA nie jest odporne na odstające punkty — jeden
    hit daleko od toru potrafi zdominować wariancję i obrócić główną oś
    PCA w swoim kierunku. W efekcie filtr odrzucał prawdziwe hity z toru,
    a zachowywał sam szum (zweryfikowane testem: linia 10 hitów wzdłuż
    osi X + jeden odstający punkt (5, 50, 0) → stara wersja zachowywała
    3 hity blisko środka + szum, odrzucając 7 poprawnych hitów z końców
    toru). Nowa wersja szacuje oś metodą zbliżoną do RANSAC: losuje pary
    punktów jako kandydackie proste, wybiera kandydata z największą
    liczbą „inlierów” (hitów w zasięgu max_residual), a finalną oś
    dopasowuje PCA tylko do tych inlierów.
    """

    def __init__(self, max_residual: float = 5.0):
        self.max_residual = max_residual

    def apply(self, event: Event) -> Event:
        hits = event.hits
        if len(hits) < 3:
            return event

        positions = np.array([[h.x, h.y, h.z] for h in hits])

        main_dir, mean_pos = self._robust_axis(positions)
        residuals = self._residuals(positions, mean_pos, main_dir)

        kept_hits = [h for h, r in zip(hits, residuals) if r <= self.max_residual]
        return Event(hits=kept_hits)

    @staticmethod
    def _residuals(positions, mean_pos, main_dir):
        centered = positions - mean_pos
        proj_len = centered @ main_dir
        proj = np.outer(proj_len, main_dir)
        return np.linalg.norm(centered - proj, axis=1)

    @staticmethod
    def _pca_axis(positions):
        mean_pos = positions.mean(axis=0)
        centered = positions - mean_pos
        cov = centered.T @ centered
        # macierz kowariancji jest symetryczna -> eigh (szybsze, zawsze
        # zwraca wartości/wektory rzeczywiste, w przeciwieństwie do eig)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_dir = eigvecs[:, -1]  # eigh zwraca wartości rosnąco
        norm = np.linalg.norm(main_dir)
        if norm > 0:
            main_dir = main_dir / norm
        return main_dir, mean_pos

    def _robust_axis(self, positions, n_trials=200, seed=0):
        n = len(positions)
        rng = np.random.default_rng(seed)

        # dla małych zbiorów zwykłe PCA na całości wystarczy
        if n < 5:
            return self._pca_axis(positions)

        idx_pairs = set()
        attempts = 0
        max_attempts = n_trials * 3
        max_pairs = min(n_trials, n * (n - 1) // 2)
        while len(idx_pairs) < max_pairs and attempts < max_attempts:
            i, j = rng.integers(0, n, size=2)
            attempts += 1
            if i == j:
                continue
            idx_pairs.add((min(i, j), max(i, j)))

        best_inliers = None
        best_count = -1
        for i, j in idx_pairs:
            p1, p2 = positions[i], positions[j]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm

            centered = positions - p1
            proj_len = centered @ direction
            proj = np.outer(proj_len, direction)
            residuals = np.linalg.norm(centered - proj, axis=1)

            inliers = residuals <= self.max_residual
            count = int(np.sum(inliers))
            if count > best_count:
                best_count, best_inliers = count, inliers

        if best_inliers is None or best_inliers.sum() < 2:
            # brak sensownego kandydata (np. wszystkie punkty w tym samym
            # miejscu) -> zwykłe PCA na całym zbiorze
            return self._pca_axis(positions)

        # finalna oś: PCA dopasowane tylko do znalezionych inlierów
        return self._pca_axis(positions[best_inliers])


class FIELDCOREFilter:
    """
    Filtr stabilizujący FIELDCORE:
    - wygładza energię / wartości,
    - usuwa pojedyncze „gorące piksele”.
    """

    def __init__(self, max_isolated_energy: float = 10.0):
        self.max_isolated_energy = max_isolated_energy

    def apply(self, event: Event) -> Event:
        hits = event.hits
        if not hits:
            return event

        energies = np.array([h.energy for h in hits])
        mean_energy = np.mean(energies)
        std_energy = np.std(energies) + 1e-6

        kept_hits = []
        for h in hits:
            z_energy = (h.energy - mean_energy) / std_energy
            # jeśli hit jest ekstremalnie energetyczny i nie ma kontekstu,
            # traktujemy go jako „gorący piksel”
            if np.abs(z_energy) > self.max_isolated_energy:
                continue
            kept_hits.append(h)

        return Event(hits=kept_hits)


class FullPipeline:
    """
    Pełny pipeline:
    SENSCORE → TRM → TIMDR → GIA → FIELDCORE
    """

    def __init__(
        self,
        calibration_map: Dict[int, SensorCalibration],
    ):
        self.senscore = SENSCOREFilter(calibration_map)
        self.trm = TRMFilter()
        self.timdr = TIMDRFilter()
        self.gia = GIAFilter()
        self.fieldcore = FIELDCOREFilter()

    def process(self, event: Event) -> Event:
        e = self.senscore.apply(event)
        e = self.trm.apply(e)
        e = self.timdr.apply(e)
        e = self.gia.apply(e)
        e = self.fieldcore.apply(e)
        return e


# ===== PRZYKŁADOWE UŻYCIE =====

if __name__ == "__main__":
    # przykładowa kalibracja czujników
    calibration = {
        1: SensorCalibration(sensor_id=1, sensitivity=1.0, noise_level=0.1, reliability=0.9),
        2: SensorCalibration(sensor_id=2, sensitivity=0.8, noise_level=0.5, reliability=0.7),
        3: SensorCalibration(sensor_id=3, sensitivity=1.2, noise_level=0.2, reliability=0.95),
        # ...
    }

    # przykładowe hity (tu: losowe, do testów)
    hits = [
        Hit(sensor_id=1, x=0.0, y=0.0, z=0.0, t=0.0, energy=10.0, raw_value=1.0),
        Hit(sensor_id=1, x=1.0, y=0.1, z=0.0, t=1.0, energy=11.0, raw_value=1.1),
        Hit(sensor_id=2, x=50.0, y=50.0, z=0.0, t=0.5, energy=0.5, raw_value=0.2),  # szum
        # dodaj więcej hitów...
    ]

    event = Event(hits=hits)
    pipeline = FullPipeline(calibration_map=calibration)

    filtered_event = pipeline.process(event)

    print("Wejście:", len(event.hits), "hitów")
    print("Wyjście:", len(filtered_event.hits), "hitów po filtracji")
    for h in filtered_event.hits:
        print(h)
