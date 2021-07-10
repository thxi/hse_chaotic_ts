import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

# TODO: fix docstrings


class TSProcessor:
    def __init__(self, *, points_in_template: int, max_template_spread: int):

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

    def heal(
        self,
        X_pred,
    ):
        raise NotImplementedError()

    def predict_unified(
        self,
        X_traj_pred: np.ndarray,
        method: str,
        use_priori=False,
        X_test: np.ndarray = None,
        priori_eps: float = None,
        # min number of trajectories to make
        # a prediction at a point, otherwise the point is non predictable
        dbs_min_trajectories: int = None,
        min_trajectories: int = None,
        max_err: float = 0.1,
        alpha: float = 0.3,
        dbs_eps=0.01,
        dbs_min_samples=4,
    ) -> np.ndarray:
        """
        get a unified prediction for each point
        """
        X_pred = None
        res = {}
        if use_priori:
            assert X_test is not None
            assert X_test.shape[0] == X_traj_pred.shape[0]
            assert priori_eps is not None

        if method == 'cluster':
            traj_alive, cluster_centers, X_pred = self.cluster_sets(
                X_traj_pred, dbs_min_trajectories, dbs_eps, dbs_min_samples)
            res['traj_alive'] = traj_alive
            res['cluster_centers'] = cluster_centers
        elif method == 'quantile':
            qs, traj_alive, X_pred = self.get_quantile_prediction(
                X_traj_pred, min_trajectories, max_err, alpha)
            res['qs'] = qs
            res['traj_alive'] = traj_alive
        else:
            raise ValueError(f"unknown method '{method}'")

        if use_priori:
            X_pred[np.abs(X_pred - X_test) > priori_eps] = np.nan

        res['X_pred'] = X_pred

        return res

    def predict_trajectories(
        self,
        X_start: np.ndarray,  # forecast after X_start
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        use_priori=False,
        X_test: np.ndarray = None,
        priori_eps=0.1,
        save_first_type_non_pred=False,  # save distance matrix
        random_seed=1,
        n_jobs: int = -1,
        print_time=True,
    ) -> np.ndarray:
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

        assert (X_start.shape[0] >= self._max_template_spread *
                (self._points_in_template - 1)), "X_start should be bigger"

        if use_priori:
            assert X_test is not None, 'X_test should be specified'

        # doign this to fill X_start
        original_size = X_start.shape[0]
        X_start = np.resize(X_start, X_start.shape[0] + h_max)
        X_start[-h_max:] = np.nan

        def get_trajectory_forecast(
            i: int,
            X_start: np.ndarray,
            training_vectors: np.ndarray,
            template_shapes: np.ndarray,
        ) -> np.ndarray:
            if print_time:
                print(f"{i} start")
            np.random.seed(random_seed * i)
            X_start = X_start.copy()
            training_vectors = training_vectors.copy()
            forecast_set = np.full((h_max, ), np.nan)
            for j in range(h_max):
                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (X_start[:original_size + j][np.cumsum(
                    -template_shapes[:, ::-1],
                    axis=1)[:, ::-1]])  # invert templates

                distance_matrix = _calc_distance_matrix(
                    training_vectors, last_vectors)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = training_vectors[distance_matrix < eps][:, -1]
                if save_first_type_non_pred:
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

            if print_time:
                print(f"{i} end")
            return forecast_set

        start = time.time()
        X_traj_pred = Parallel(n_jobs=n_jobs)(delayed(get_trajectory_forecast)(
            i, X_start, self._training_vectors, self._template_shapes)
                                              for i in range(n_trajectories))
        end = time.time()
        if print_time:
            print('{:.2f}s'.format(end - start))

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_traj_pred = np.array(X_traj_pred).T

        return X_traj_pred

    def cluster_sets(
        self,
        X_traj_pred: np.ndarray,
        min_trajectories,
        dbs_eps: float,
        dbs_min_samples: int,
    ):
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param X_traj_pred:
        :param dbs_eps:
        :param dbs_min_samples:
        :return: Возвращает центр самого большого кластера на каждом шаге.
        """

        X_pred = np.full(shape=[
            X_traj_pred.shape[0],
        ], fill_value=np.nan)
        dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

        cluster_centers = []
        traj_alive = []
        for i in range(len(X_traj_pred)):
            traj_pred = X_traj_pred[i]
            traj_pred = traj_pred[
                ~np.isnan(traj_pred)]  # filter nans for trajectories
            traj_alive.append(len(traj_pred))
            if len(traj_pred) == 0:  # only nans left
                continue

            dbs.fit(traj_pred.reshape(-1, 1))

            # TODO: handle case when there are 2 big clusters
            cluster_labels, cluster_sizes = np.unique(
                dbs.labels_[dbs.labels_ > -1], return_counts=True)
            # print(i, cluster_sizes)

            step_preds = []
            for label in cluster_labels:
                pred = traj_pred[label == dbs.labels_].mean()
                step_preds.append(pred)
            cluster_centers.append(step_preds)

            if cluster_labels.size > 0:
                max_cluster_size = cluster_sizes.max()
                max_cluster_label = cluster_sizes.argmax()
                if max_cluster_size < min_trajectories:
                    continue
                biggest_cluster_center = traj_pred[
                    dbs.labels_ == cluster_labels[max_cluster_label]].mean()
                X_pred[i] = biggest_cluster_center

        return traj_alive, cluster_centers, X_pred

    def get_quantile_prediction(
        self,
        X_traj_pred: np.ndarray,
        min_trajectories,
        max_err: float = 0.1,
        alpha=0.3,
    ):
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param X_traj_pred:
        :param max_err: max forecasting error
        :param alpha:
        :return: Возвращает среднее предсказание в [alpha, 1-alpha] квантилях
        """

        X_pred = np.full(shape=[
            X_traj_pred.shape[0],
        ], fill_value=np.nan)

        qs = []
        traj_alive = []
        for i in range(len(X_traj_pred)):
            traj_pred = X_traj_pred[i]
            traj_pred = traj_pred[~np.isnan(traj_pred)]
            traj_alive.append(len(traj_pred))
            if len(traj_pred) == 0:
                # no predictions for trajectories on i-th step
                qs.append((np.nan, np.nan))
                continue
            q1, q2 = np.quantile(traj_pred, q=[alpha, 1 - alpha])
            qs.append((q1, q2))
            # print(i, q1, q2, q2 - q1, len(traj_pred))
            if len(traj_pred) < min_trajectories:
                # only a few trajectories left
                continue
            if q2 - q1 <= max_err:
                X_pred[i] = np.mean(traj_pred[(traj_pred > q1)
                                              & (traj_pred < q2)])

        qs = np.array(qs)
        traj_alive = np.array(traj_alive)
        return qs, traj_alive, X_pred


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


def _choose_trajectory_point(
    points_pool: np.ndarray,
    how: str,
    dbs_eps: float = 0.0,
    dbs_min_samples: int = 0,
) -> float:
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
