import numpy as np
import random

from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor as pool


class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int):

        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread
        self._points_in_template = points_in_template
        self.x_dim: int = max_template_spread ** (points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template                               # сколько зубчиков в каждом шаблоне

        # сами шаблоны
        templates = (np.repeat(0, self.x_dim).reshape(-1, 1), )

        # непонятный код, который заполняет шаблоны нужными значениями. Пытаться вникнуть бесполезно.
        for i in range(1, points_in_template):
            col = (np.repeat(
                np.arange(1, max_template_spread + 1, dtype=int), max_template_spread ** (points_in_template - (i + 1))
            ) + templates[i - 1][::max_template_spread ** (points_in_template - i)]).reshape(-1, 1)

            templates += (col, )  # don't touch

        self._templates: np.ndarray = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: np.ndarray = self._templates[:, 1:] - self._templates[:, :-1]

    def fit(self, time_series: np.ndarray) -> None:
        '''Обучить класс на конкретном ряду.'''

        self._time_series   = time_series
        self.y_dim          = self._time_series.size - self._templates[0][-1]
        self._original_size = self._time_series.size

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        self._training_vectors: np.ndarray = \
            np.full(shape=(self.x_dim, self.y_dim, self.z_dim), fill_value=np.inf, dtype=float)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )

            self._training_vectors[i, :template_data.shape[0]] = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )

    def pull(self, steps: int, eps: float, n_trajectories: int, noise_amp: float, prev_preds: np.array=None) -> np.ndarray:
        '''
        Основной метод пулла, который использовался в статье.

        Parameters
        ----------
        steps : int
            На сколько шагов прогнозируем.
        eps : float
            Минимальное Евклидово расстояние от соответствующего шаблона, в пределах которого должны находиться
            вектора наблюдений, чтобы считаться "достаточно похожими".
        n_trajectories : int
            Сколько траекторий использовать. Чем больше, тем дольше время работы и потенциально точнее результат.
        noise_amp : float
            Максимальная амплитуда шума, используемая при расчете траекторий.
        prev_preds: np.array - вектор предыдущих предсказаний (для запуска в цикле для self-healing)

        Возвращает матрицу размером steps x n_trajectories, где по горизонтали идут шаги, а по вертикали - прогнозы
        каждой из траекторий на этом шаге.
        '''
        # прибавляем к тренировочному датасету steps пустых векторов, которые будем заполнять значениями на ходу
        # только при первом проходе, для self-healing это не повторяется
        if prev_preds is None or self._training_vectors.shape[1] != self.y_dim + steps:
            self._training_vectors = np.hstack([self._training_vectors,
                                                np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])

        # удлиняем изначальый ряд на значение steps + delta
        # delta - "запас места" справа для использования всех возмжных шаблонов
        # к концу тестовой выборки 
        # (проблемы были бы когда пронозируемая точка - непоследняя)
        delta = self._max_template_spread * self._points_in_template
        self._time_series = np.resize(
            self._time_series,
            self._original_size + steps + delta
        )
        if prev_preds is None:
            prev_preds = np.array([np.nan])

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        forecast_sets = np.full((steps, n_trajectories), np.nan)
        for i in range(n_trajectories):
            self._time_series[self._original_size: self._original_size + steps] = prev_preds
            self._time_series[self._original_size + steps:] = np.nan
            for j in range(steps):
                points = np.array([])
                if len(prev_preds) == 1:
                    rng = [self._points_in_template - 1]
                else:
                    rng = range(self._points_in_template - 1, 0, -1)
                # цикл по всем возможным позициям пронозируемой точки в векторе
                for pos_in_vec in rng:
                    if pos_in_vec < self._points_in_template - 1:
                        # переставка шаблона так, чтобы прогнозируемая точка оказалась последней
                        # был шаблон 1, 2, 3 [n, n + 1, n + 3, n + 6]
                        # хотим прогнозировать предполследнюю точку
                        # получаем 1, 5, -3 [n, n + 1, n + 6, n + 3]
                        template_shapes = np.hstack([
                            self._template_shapes[:, : pos_in_vec - 1],
                            (
                                self._template_shapes[:, pos_in_vec - 1].reshape(-1, 1)
                                + self._template_shapes[:, pos_in_vec].reshape(-1, 1)
                            ),
                            self._template_shapes[:, pos_in_vec + 1:],
                            -np.sum(self._template_shapes[:, pos_in_vec:], axis=1).reshape(-1, 1)
                        ])
                    else:
                        template_shapes = self._template_shapes

                    # print(pos_in_vec, template_shapes[10], self._template_shapes[10])
                    # тестовые вектора, которые будем сравнивать с тренировочными
                    last_vectors = (self._time_series[:self._original_size + j + delta]
                                    [np.cumsum(-template_shapes[:, ::-1], axis=1)[:, ::-1] - delta])
                    if np.mean(np.isnan(last_vectors).any(axis=1)) == 1:
                        continue

                    # last_vectors = last_vectors[~np.isnan(last_vectors).any(axis=1), :]
                    last_vectors = np.nan_to_num(last_vectors, True, np.inf, np.inf, np.inf)
                    if len(last_vectors) == 0:
                        continue
                    distance_matrix = self._calc_distance_matrix(
                        last_vectors,
                        np.repeat(True, self.x_dim),
                        steps,
                        pos_in_vec,
                        self._points_in_template
                    )

                    # последние точки тренировочных векторов, оказавшихся в пределах eps
                    if points.size == 0:
                        points = self._training_vectors[distance_matrix < eps][:, pos_in_vec]
                    else:
                        points = np.concatenate(
                            [points, self._training_vectors[distance_matrix < eps][:, pos_in_vec]],
                            axis=0
                        )
                # теперь нужно выбрать финальное прогнозное значение из возможных
                # я выбираю самое часто встречающееся значение, но тут уже можно на свое усмотрение

                forecast_point                             = self._freeze_point(points, 'mf') \
                    + random.uniform(-noise_amp, noise_amp)
                forecast_sets[j, i]                        = forecast_point
                self._time_series[self._original_size + j] = forecast_point

                # у нас появилась новая точка в ряду, последние вектора обновились, добавим их в обучающие
                new_training_vectors = (
                    self._time_series[:self._original_size + 1 + j]
                    [np.hstack((np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                    - 1, np.repeat(-1, self.x_dim).reshape(-1, 1)))]
                )
                self._training_vectors[:, self.y_dim + j, :] = new_training_vectors

            # честно говоря, я не помню, зачем нужен код дальше

            # delete added vectors after each run
            self._training_vectors[:, self.y_dim:] = np.inf

            # delete added points after each run
            # self._time_series[-steps - delta:] = np.nan
        return forecast_sets

    def cluster_sets(self, forecast_sets: np.ndarray, dbs_eps: float, dbs_min_samples: int):
        '''
        Скластеризировать полученные в результате пулла множества прогнозных значений.
        Возвращает центр самого большого кластера на каждом шаге.
        '''

        predictions = np.full(shape=[forecast_sets.shape[0], ], fill_value=np.nan)
        dbs         = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

        for i in range(len(forecast_sets)):
            curr_set = forecast_sets[i]
            dbs.fit(curr_set.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = curr_set[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                predictions[i] = biggest_cluster_center

        return predictions

    def _calc_distance_matrix(self, test_vectors: np.ndarray, mask: np.ndarray, steps: int, pos_in_vec, vec_len) -> np.ndarray:
        '''
        По необъяснимым причинам считать матрицу расстояний между тестовыми векторами и тренировочными быстрее вот так.
        '''
        # переставка шаблона так, чтобы прогнозируемая точка оказалась последней
        # был шаблон 1, 2, 3 [n, n + 1, n + 3, n + 6]
        # хотим прогнозировать предполследнюю точку
        # получаем 1, 5, -3 [n, n + 1, n + 6, n + 3]
        training_vectors = np.concatenate(
            [
                self._training_vectors[mask, :, :pos_in_vec],
                self._training_vectors[mask, :, pos_in_vec + 1:],
                self._training_vectors[mask, :, pos_in_vec].reshape(len(mask), -1, 1)
            ],
            axis=2
        )
        distance_matrix = ((training_vectors[mask, :, 0] - np.repeat(test_vectors[:, 0], self.y_dim + steps)
                            .reshape(-1, self.y_dim + steps)) ** 2)
        for i in range(1, vec_len - 1):
            distance_matrix += ((training_vectors[mask, :, i] - np.repeat(test_vectors[:, i], self.y_dim + steps)
                                .reshape(-1, self.y_dim + steps)) ** 2)
        distance_matrix = distance_matrix ** 0.5

        return distance_matrix

    def _freeze_point(self, points_pool: np.ndarray, how: str, dbs_eps: float = 0.0, dbs_min_samples: int = 0) -> float:
        '''
        Выбрать финальный прогноз в данной точке из множества прогнозных значений.

        "How" варианты:
            "mean" = "mean"
            "mf"   = "most frequent"
            "cl"   = "cluster", нужны dbs_eps и dbs_min_samples
        '''

        if points_pool.size == 0:
            result = np.nan
        else:
            if how == 'mean':
                result = float(points_pool.mean())

            elif how == 'mf':
                points, counts = np.unique(points_pool, return_counts=True)
                result         = points[counts.argmax()]

            elif how == 'cl':
                dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
                dbs.fit(points_pool.reshape(-1, 1))

                cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

                if (cluster_labels.size > 0
                        and np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                    biggest_cluster_center = points_pool[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                    result                 = biggest_cluster_center
                else:
                    result = np.nan

        return result

    def _fish(self, steps: int, eps: float, current_step: int) -> list:
        '''
        "Закидываем удочку". Из текущей точки закидываем все шаблоны вперед, насколько позволяет их длина, и сохраняем
        последние точки соответствующих обучающих векторов в массив points, которые затем аггрегируем в "общий" список
        forecasted_points.
        '''

        forecasted_points = []

        # длина удочки = максимальное расстояние между соседними зубчиками в шаблоне
        for i in range(min(self._max_template_spread, steps - current_step)):

            # вектора, последняя точка которых "висит в воздухе"
            last_vectors = (
                self._time_series[:self._original_size + current_step]
                [np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                 + (self._template_shapes[:, -1] - 1).reshape(-1, 1)]
            )

            # отсеиваем "недостаточно длинные" шаблоны
            length_mask = self._template_shapes[:, -1] > i

            # убираем вектора, в которые попадают непрогнозируемые точки
            non_nan_mask = ~np.isnan(last_vectors).any(axis=1)

            total_mask = length_mask & non_nan_mask

            last_vectors    = last_vectors[total_mask]
            distance_matrix = self._calc_distance_matrix(last_vectors, total_mask, steps)

            points = self._training_vectors[total_mask][distance_matrix < eps, -1]
            forecasted_points.append(points)

        return forecasted_points

    def push(self, steps, eps, dbs_eps, dbs_min_samples):
        self._training_vectors     = np.hstack([self._training_vectors,
                                                np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan
        point_pools                = [[] for _ in range(steps)]

        for back_point in range(steps):  # двигаем задний порог
            front_point = back_point

            # двигаем передний порог
            while front_point < min(front_point + self._max_template_spread, steps):
                if np.isnan(self._time_series[self._original_size - 1 + front_point]):
                    front_point += 1
                    continue

                forecasted_points = self._fish(steps, eps, front_point)

                for i in range(len(forecasted_points)):
                    point_pools[front_point + i].extend(forecasted_points[i])

                    self._time_series[self._original_size + front_point + i] =\
                        self._freeze_point(np.array(forecasted_points[i]), 'cl', dbs_eps, dbs_min_samples)

                front_point += 1

            # добавляем только что полученные новые вектора к обучающим
            self._training_vectors[:, -steps] = self._time_series[:self._original_size + 1][
                np.hstack([
                    np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1] - 1,
                    np.repeat(-1, self.x_dim).reshape(-1, 1)
                ])
            ]

            back_point += 1

        return self._time_series[-steps:], point_pools
