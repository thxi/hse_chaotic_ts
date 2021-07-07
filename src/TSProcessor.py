import numpy as np

from joblib import Parallel, delayed
import time

from sklearn.cluster import DBSCAN


class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int):

        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread

        self.x_dim: int = max_template_spread**(
            points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template  # сколько зубчиков в каждом шаблоне

        # сами шаблоны
        templates = (np.repeat(0, self.x_dim).reshape(-1, 1), )

        # непонятный код, который заполняет шаблоны нужными значениями. Пытаться вникнуть бесполезно.
        for i in range(1, points_in_template):
            col = (
                np.repeat(np.arange(1, max_template_spread + 1, dtype=int),
                          max_template_spread**(points_in_template -
                                                (i + 1))) +
                templates[i - 1][::max_template_spread**(points_in_template -
                                                         i)]).reshape(-1, 1)

            templates += (col, )  # don't touch

        self._templates: np.ndarray = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: np.ndarray = self._templates[:,
                                                            1:] - self._templates[:, :
                                                                                  -1]  # k1, k2, ...

    def fit(self, time_series: np.ndarray, refit=False) -> None:
        """Обучить класс на конкретном ряду."""

        # TODO: add refit

        self._time_series = time_series
        # if not refit:
        #     self.y_dim = self._time_series.size - self._templates[0][-1]
        self.y_dim = self._time_series.size - self._templates[0][-1]
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
                self._time_series[self._templates[i] +
                                  np.arange(self._time_series.size -
                                            self._templates[i][-1])[:, None]])

            self._training_vectors[i, :template_data.shape[0]] = (
                self._time_series[self._templates[i] +
                                  np.arange(self._time_series.size -
                                            self._templates[i][-1])[:, None]])

    def pull(self,
             steps: int,
             eps: float,
             n_trajectories: int,
             noise_amp: float,
             prev_result: np.ndarray = None,
             random_seed=1,
             n_jobs: int = -1) -> np.ndarray:
        """
        Основной метод пулла, который использовался в статье.

        :param steps: На сколько шагов прогнозируем.
        :param eps: Минимальное Евклидово расстояние от соответствующего шаблона, в пределах которого должны находиться
            вектора наблюдений, чтобы считаться "достаточно похожими".
        :param n_trajectories: Сколько траекторий использовать.
            Чем больше, тем дольше время работы и потенциально точнее результат.
        :param noise_amp: Максимальная амплитуда шума, используемая при расчете траекторий.
        :param n_jobs: максимальное количество процессов для вычисления каждой траектории
        :param prev_result: предыдущие предсказания
        :return: Возвращает матрицу размером steps x n_trajectories, где по горизонтали идут шаги, а по вертикали - прогнозы
        каждой из траекторий на этом шаге.
        """

        if prev_result is None:
            # прибавляем к тренировочному датасету steps пустых векторов, которые будем заполнять значениями на ходу
            self._training_vectors = np.hstack([
                self._training_vectors,
                np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)
            ])

            # удлиняем изначальый ряд на значение steps
            self._time_series = np.resize(self._time_series,
                                          self._original_size + steps)
            self._time_series[-steps:] = np.nan
        else:
            # update the time series with predicted values
            self._time_series[self._original_size:self._original_size +
                              steps] = prev_result
            # update the _training_vectors
            # self.fit(self._time_series, refit=True)

        def get_trajectory_forecast(i: int, time_series: np.ndarray,
                                    training_vectors: np.ndarray,
                                    template_shapes: np.ndarray,
                                    original_size: int, x_dim: int,
                                    y_dim: int) -> np.ndarray:
            print(f"{i} start")
            np.random.seed(random_seed * i)
            # if prev_result is not None:
            #     np.random.seed(2 * i)
            time_series = time_series.copy()
            training_vectors = training_vectors.copy()
            forecast_set = np.full((steps, ), np.nan)
            for j in range(steps):
                if prev_result is not None and not np.isnan(prev_result[j]):
                    forecast_set[j] = prev_result[j]
                    continue
                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (time_series[:original_size + j][np.cumsum(
                    -template_shapes[:, ::-1],
                    axis=1)[:, ::-1]])  # invert templates

                distance_matrix = _calc_distance_matrix(
                    training_vectors, last_vectors, y_dim,
                    np.repeat(True, x_dim), steps)

                # последние точки тренировочных векторов, оказавшихся в пределах eps

                points = training_vectors[distance_matrix < eps][:, -1]
                # TODO: some trajectories can die (i.e. len(points)=0 which results in a nan)
                # if len(points) == 0:
                #     new_eps = np.quantile(distance_matrix, q=0.01)
                #     points = training_vectors[distance_matrix < new_eps][:, -1]

                # теперь нужно выбрать финальное прогнозное значение из возможных
                # я выбираю самое часто встречающееся значение, но тут уже можно на свое усмотрение
                # sometimes trajectories might contain NaNs
                forecast_point = _freeze_point(
                    points, 'mean') + np.random.normal(0, noise_amp)
                forecast_set[j] = forecast_point
                time_series[original_size + j] = forecast_point

                # у нас появилась новая точка в ряду, последние вектора обновились, добавим их в обучающие
                new_training_vectors = (
                    time_series[:original_size + 1 + j][np.hstack((
                        np.cumsum(-template_shapes[:, ::-1], axis=1)[:, ::-1] -
                        1, np.repeat(-1, x_dim).reshape(-1, 1)))])

                training_vectors[:, y_dim + j, :] = new_training_vectors

            print(f"{i} end")
            return forecast_set

        start = time.time()
        results = Parallel(n_jobs=n_jobs)(delayed(get_trajectory_forecast)(
            i, self._time_series, self._training_vectors,
            self._template_shapes, self._original_size, self.x_dim, self.y_dim)
                                          for i in range(n_trajectories))
        end = time.time()
        print('{:.2f}s'.format(end - start))

        results = np.array(results).T

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        return results

    def cluster_sets(self, forecast_sets: np.ndarray, dbs_eps: float,
                     dbs_min_samples: int):
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param forecast_sets:
        :param dbs_eps:
        :param dbs_min_samples:
        :return: Возвращает центр самого большого кластера на каждом шаге.
        """

        predictions = np.full(shape=[
            forecast_sets.shape[0],
        ],
                              fill_value=np.nan)
        dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

        for i in range(len(forecast_sets)):
            curr_set = forecast_sets[i]
            curr_set = curr_set[
                ~np.isnan(curr_set)]  # filter nans for trajectories
            if len(curr_set) == 0:  # only nans left
                continue

            dbs.fit(curr_set.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(
                dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = curr_set[
                    dbs.labels_ == cluster_labels[
                        cluster_sizes.argmax()]].mean()
                predictions[i] = biggest_cluster_center

        return predictions

    def get_quantile_prediction(self,
                                forecast_sets: np.ndarray,
                                max_err: float,
                                alpha=0.05):
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param forecast_sets:
        :param max_err: max forecasting error
        :param alpha:
        :return: Возвращает среднее предсказание в [alpha, 1-alpha] квантилях
        """

        predictions = np.full(shape=[
            forecast_sets.shape[0],
        ],
                              fill_value=np.nan)
        for i in range(len(forecast_sets)):
            q1, q2 = np.quantile(forecast_sets[i], q=[alpha, 1 - alpha])
            print(i, q1, q2, q2 - q1)
            if q2 - q1 <= max_err:
                predictions[i] = np.mean(
                    forecast_sets[i][(forecast_sets[i] > q1)
                                     & (forecast_sets[i] < q2)])

        return predictions


def _calc_distance_matrix(training_vectors: np.ndarray,
                          test_vectors: np.ndarray, y_dim: int,
                          mask: np.ndarray, steps: int) -> np.ndarray:
    """
    По необъяснимым причинам считать матрицу расстояний между тестовыми векторами и тренировочными быстрее вот так.
    """

    # print(training_vectors[mask, :, 0].shape,
    #       np.repeat(test_vectors[:, 0], y_dim + steps).reshape(-1, y_dim + steps).shape)
    distance_matrix = (
        (training_vectors[mask, :, 0] - np.repeat(
            test_vectors[:, 0], y_dim + steps).reshape(-1, y_dim + steps))**2 +
        (training_vectors[mask, :, 1] - np.repeat(
            test_vectors[:, 1], y_dim + steps).reshape(-1, y_dim + steps))**2 +
        (training_vectors[mask, :, 2] - np.repeat(
            test_vectors[:, 2], y_dim + steps).reshape(-1, y_dim + steps))**
        2)**0.5

    return distance_matrix


def _freeze_point(points_pool: np.ndarray,
                  how: str,
                  dbs_eps: float = 0.0,
                  dbs_min_samples: int = 0) -> float:
    """
    Выбрать финальный прогноз в данной точке из множества прогнозных значений.

    "How" варианты:
        "mean" = "mean"
        "mf"   = "most frequent"
        "cl"   = "cluster", нужны dbs_eps и dbs_min_samples
    """
    result = None
    if points_pool.size == 0:
        result = np.nan
    else:
        if how == 'mean':
            result = float(points_pool.mean())

        elif how == 'mf':
            points, counts = np.unique(points_pool, return_counts=True)
            result = points[counts.argmax()]

        elif how == 'cl':
            dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
            dbs.fit(points_pool.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(
                dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if (cluster_labels.size > 0 and np.count_nonzero(
                ((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                biggest_cluster_center = points_pool[
                    dbs.labels_ == cluster_labels[
                        cluster_sizes.argmax()]].mean()
                result = biggest_cluster_center
            else:
                result = np.nan

    return result
