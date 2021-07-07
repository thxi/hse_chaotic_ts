import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int):

        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread
        self._points_in_template = points_in_template

        self.x_dim: int = max_template_spread**(
            points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template  # сколько зубчиков в каждом шаблоне

        # сами шаблоны
        templates = (np.repeat(0, self.x_dim).reshape(-1, 1), )
        # код, который заполняет шаблоны нужными значениями
        for i in range(1, points_in_template):
            col = (
                np.repeat(np.arange(1, max_template_spread + 1, dtype=int),
                          max_template_spread**(points_in_template -
                                                (i + 1))) +
                templates[i - 1][::max_template_spread**(points_in_template -
                                                         i)]).reshape(-1, 1)

            templates += (col, )

        self._templates: np.ndarray = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: np.ndarray = self._templates[:,1:] \
            - self._templates[:, :-1]  # k1, k2, ...

    def fit(self, time_series: np.ndarray) -> None:
        """ Fill training vectors from time_series """

        # self._time_series = time_series
        self.y_dim = time_series.size - self._templates[0][-1]

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
            template_data = (time_series[
                self._templates[i] +
                np.arange(time_series.size - self._templates[i][-1])[:, None]])

            self._training_vectors[i, :template_data.shape[0]] = (time_series[
                self._templates[i] +
                np.arange(time_series.size - self._templates[i][-1])[:, None]])

    def predict(self,
                X_start: np.ndarray,
                X_test: np.ndarray,
                method: str,
                eps: float = 0.01,
                n_trajectories: int = 1,
                priori_eps=0.01,
                noise_amp=0.05,
                dbs_eps=0.01,
                dbs_min_samples=4):
        # TODO: rewrite comment
        assert (X_start.shape[0] >= self._max_template_spread *
                (self._points_in_template - 1)), "X_start should be bigger"
        # TODO: remove h_max
        h_max = X_test.shape[0]

        trajectories_prediction = None
        unified_prediction = None

        if method == 'no-np':  # no non predictable
            assert n_trajectories == 1, 'n_trajectories should be 1'
            trajectories_prediction = self.pull(
                X_start,
                steps=h_max,
                eps=eps,
                n_trajectories=1,
                noise_amp=noise_amp,
                handle_first_type_non_pred=True,
                n_jobs=-1)  # [steps x 1]
            unified_prediction = trajectories_prediction[:, 0]  # [steps]
        elif method == 'priori':
            assert X_test is not None, 'X_test should be specified'
            assert (~np.isnan(X_test)).all(), \
                 'all X_test points should be non nan'
            trajectories_prediction = self.pull(
                X_start,
                steps=h_max,
                eps=eps,
                n_trajectories=n_trajectories,
                noise_amp=noise_amp,
                handle_first_type_non_pred=True,
                use_priori=True,
                X_test=X_test,
                n_jobs=-1,
            )  # [steps x n_trajectories]
            unified_prediction = self.cluster_sets(trajectories_prediction,
                                                   dbs_eps,
                                                   dbs_min_samples)  # [steps]

            unified_prediction[np.abs(unified_prediction -
                                      X_test) > eps] = np.nan

        elif method == 'cluster':
            trajectories_prediction = self.pull(
                X_start,
                steps=h_max,
                eps=eps,
                n_trajectories=n_trajectories,
                noise_amp=noise_amp,
                n_jobs=-1)  # [steps x n_trajectories]
            unified_prediction = self.cluster_sets(trajectories_prediction,
                                                   dbs_eps,
                                                   dbs_min_samples)  # [steps]
        else:
            raise ValueError(f"unknown method '{method}'")

        return trajectories_prediction, unified_prediction

    def pull(self,
             X_start: np.ndarray,
             steps: int,
             eps: float,
             n_trajectories: int,
             noise_amp: float,
             X_test: np.ndarray = None,
             handle_first_type_non_pred=False,
             use_priori=False,
             priori_eps=0.1,
             random_seed=1,
             n_jobs: int = -1) -> np.ndarray:
        """
        Get trajectories' predictions from X_start to X_test

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

        if use_priori:
            assert X_test is not None, 'X_test should be specified'

        # doign this to fill X_start
        original_size = X_start.shape[0]
        X_start = np.resize(X_start, X_start.shape[0] + steps)
        X_start[-steps:] = np.nan

        def get_trajectory_forecast(
            i: int,
            X_start: np.ndarray,
            training_vectors: np.ndarray,
            template_shapes: np.ndarray,
        ) -> np.ndarray:
            print(f"{i} start")
            np.random.seed(random_seed * i)
            X_start = X_start.copy()
            training_vectors = training_vectors.copy()
            forecast_set = np.full((steps, ), np.nan)
            for j in range(steps):
                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (X_start[:original_size + j][np.cumsum(
                    -template_shapes[:, ::-1],
                    axis=1)[:, ::-1]])  # invert templates

                distance_matrix = _calc_distance_matrix(
                    training_vectors, last_vectors)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = training_vectors[distance_matrix < eps][:, -1]
                if handle_first_type_non_pred:
                    # some trajectories can die (i.e. len(points)=0 which results in a nan)
                    if len(points) == 0:
                        # TODO: add quantile parameter
                        new_eps = np.quantile(distance_matrix, q=0.01)
                        points = training_vectors[
                            distance_matrix < new_eps][:, -1]

                # теперь нужно выбрать финальное прогнозное значение из возможных
                # TODO: add parameter to select the method
                forecast_point = _choose_trajectory_point(points, 'mean') \
                    + np.random.normal(0, noise_amp)

                if use_priori:
                    if abs(forecast_point - X_test[j]) < priori_eps:
                        forecast_set[j] = forecast_point
                        X_start[original_size + j] = forecast_point
                else:
                    forecast_set[j] = forecast_point
                    X_start[original_size + j] = forecast_point

            print(f"{i} end")
            return forecast_set

        start = time.time()
        X_traj_pred = Parallel(n_jobs=n_jobs)(delayed(get_trajectory_forecast)(
            i, X_start, self._training_vectors, self._template_shapes)
                                              for i in range(n_trajectories))
        end = time.time()
        print('{:.2f}s'.format(end - start))

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_traj_pred = np.array(X_traj_pred).T

        return X_traj_pred

    def cluster_sets(self, X_traj_pred: np.ndarray, dbs_eps: float,
                     dbs_min_samples: int):
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param forecast_sets:
        :param dbs_eps:
        :param dbs_min_samples:
        :return: Возвращает центр самого большого кластера на каждом шаге.
        """

        X_pred = np.full(shape=[
            X_traj_pred.shape[0],
        ], fill_value=np.nan)
        dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

        for i in range(len(X_traj_pred)):
            curr_set = X_traj_pred[i]
            curr_set = curr_set[
                ~np.isnan(curr_set)]  # filter nans for trajectories
            if len(curr_set) == 0:  # only nans left
                continue

            dbs.fit(curr_set.reshape(-1, 1))

            # TODO: handle case when there are 2 big clusters
            cluster_labels, cluster_sizes = np.unique(
                dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = curr_set[
                    dbs.labels_ == cluster_labels[
                        cluster_sizes.argmax()]].mean()
                X_pred[i] = biggest_cluster_center

        return X_pred

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


def _calc_distance_matrix(
    training_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> np.ndarray:
    """
    calculate the distance matrix between training_vectors and test_vectors
    """

    # drop last point from training vectors
    vectors_in_template = training_vectors.shape[-2]
    training_vectors = training_vectors[:, :, :-1]

    # reshaping test_vectors to efficiently calculate the distances
    test_vectors = test_vectors[:, np.newaxis, :]
    test_vectors = np.repeat(test_vectors, vectors_in_template, axis=1)

    distances = np.sqrt(((training_vectors - test_vectors)**2).sum(axis=-1))

    return distances


def _choose_trajectory_point(points_pool: np.ndarray,
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
            raise Exception("Not implemented")
            # points, counts = np.unique(points_pool, return_counts=True)
            # result = points[counts.argmax()]

        elif how == 'cl':
            raise Exception("Not implemented")
            # dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
            # dbs.fit(points_pool.reshape(-1, 1))

            # cluster_labels, cluster_sizes = np.unique(
            #     dbs.labels_[dbs.labels_ > -1], return_counts=True)

            # if (cluster_labels.size > 0 and np.count_nonzero(
            #     ((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
            #     biggest_cluster_center = points_pool[
            #         dbs.labels_ == cluster_labels[
            #             cluster_sizes.argmax()]].mean()
            #     result = biggest_cluster_center
            # else:
            #     result = np.nan

    return result
