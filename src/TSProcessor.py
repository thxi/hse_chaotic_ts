from typing import Union

import numpy as np
import cupy as cp
from sklearn.cluster import DBSCAN
from cuml import DBSCAN as cumlDBSCAN
from joblib import Parallel, delayed

# https://stackoverflow.com/a/55239060
from timeit import default_timer as timer
from datetime import timedelta

# TODO: fix docstrings


class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int,
                 X_train: Union[np.ndarray, cp.ndarray]):

        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread
        self._points_in_template = points_in_template

        self.x_dim: int = max_template_spread**(
            points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template  # сколько зубчиков в каждом шаблоне

        xp = cp.get_array_module(X_train)
        # сами шаблоны
        templates = (xp.zeros(shape=(self.x_dim, 1), dtype=int), )
        # код, который заполняет шаблоны нужными значениями
        for i in range(1, points_in_template):
            col = (
                xp.repeat(xp.arange(1, max_template_spread + 1, dtype=int),
                          max_template_spread**(points_in_template -
                                                (i + 1))) +
                templates[i - 1][::max_template_spread**(points_in_template -
                                                         i)]).reshape(-1, 1)

            templates += (col, )

        self._templates: xp.ndarray = xp.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: xp.ndarray = self._templates[:,1:] \
            - self._templates[:, :-1]  # k1, k2, ...

        self.__fit(X_train)

    def __fit(self, X_train: Union[np.ndarray, cp.ndarray]) -> None:
        """ Fill training vectors from time_series """

        xp = cp.get_array_module(X_train)

        self.y_dim = X_train.shape[0] - int(self._templates[0][-1])

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        self._training_vectors = xp.full(shape=(self.x_dim, self.y_dim,
                                                self.z_dim),
                                         fill_value=xp.inf,
                                         dtype=float)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (X_train[
                self._templates[i] +
                xp.arange(X_train.size - self._templates[i][-1])[:, None]])

            self._training_vectors[i, :template_data.shape[0]] = (X_train[
                self._templates[i] +
                xp.arange(X_train.size - self._templates[i][-1])[:, None]])

        self._last_vectors = xp.cumsum(-self._template_shapes[:, ::-1],
                                       axis=1)[:, ::-1]

    def heal(
        self,
        X_pred,
    ):
        raise NotImplementedError()

    def predict_trajectories(
        self,
        X_start: Union[cp.ndarray, np.ndarray],  # forecast after X_start
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        X_pred: Union[cp.ndarray, np.ndarray] = None,
        use_priori=False,
        X_test: Union[cp.ndarray, np.ndarray] = None,
        priori_eps=0.1,
        random_seed=1,
        n_jobs: int = -1,
        print_time=False,
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
        xp = cp.get_array_module(X_start)
        if xp.__name__ == 'numpy':
            return self._predict_trajectories_cpu(X_start, h_max, eps,
                                                  n_trajectories, noise_amp,
                                                  X_pred, use_priori, X_test,
                                                  priori_eps, random_seed,
                                                  n_jobs, print_time)
        assert X_pred is None, "X_pred is not implemented for gpu"
        return self._predict_trajectories_gpu(X_start, h_max, eps,
                                              n_trajectories, noise_amp,
                                              use_priori, X_test, priori_eps)

    def _predict_trajectories_gpu(
        self,
        X_start: cp.ndarray,  # forecast after X_start
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        use_priori=False,
        X_test: cp.ndarray = None,
        priori_eps=0.1,
    ) -> cp.ndarray:

        assert (X_start.shape[0] == self._max_template_spread *
                (self._points_in_template - 1)), "X_start should be bigger"

        if use_priori:
            assert X_test is not None, 'X_test should be specified'

        # doign this to fill X_start
        original_size = X_start.shape[0]
        X_start = cp.resize(X_start, X_start.shape[0] + h_max)
        X_start[-h_max:] = cp.nan

        X_start = cp.repeat(X_start[cp.newaxis, :], n_trajectories, axis=0)
        training_for_dist = cp.repeat(self._training_vectors[cp.newaxis, :, :,
                                                             -1],
                                      n_trajectories,
                                      axis=0)
        noise = cp.random.normal(0, noise_amp, size=(n_trajectories, h_max))
        for i in range(h_max):
            test_vectors = X_start[:, :original_size + i][:,
                                                          self._last_vectors]
            dist = _calc_distance_matrix_gpu(self._training_vectors,
                                             test_vectors)

            # see https://stackoverflow.com/a/29046530
            predictions = cp.nanmean(cp.where(dist < eps, training_for_dist,
                                              cp.nan).reshape(
                                                  n_trajectories, -1),
                                     axis=1) + noise[:, i]
            if use_priori:
                predictions = cp.where(
                    cp.abs(predictions - X_test[i]) < priori_eps, predictions,
                    cp.nan)
            X_start[:, original_size + i] = predictions

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_start = X_start[:, original_size:]
        X_start = X_start.T
        return X_start

    def _predict_trajectories_cpu(
        self,
        X_start: np.ndarray,  # forecast after X_start
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        X_pred: np.ndarray = None,
        use_priori=False,
        X_test: np.ndarray = None,
        priori_eps=0.1,
        random_seed=1,
        n_jobs: int = -1,
        print_time=False,
    ) -> np.ndarray:
        assert (X_start.shape[0] == self._max_template_spread *
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
        ) -> np.ndarray:
            np.random.seed(random_seed * i)
            X_start = X_start.copy()
            for j in range(h_max):
                # тестовые вектора, которые будем сравнивать с тренировочными
                if X_pred is not None and not np.isnan(X_pred[j]):
                    X_start[original_size + j] = X_pred[j]
                    continue

                test_vectors = X_start[:original_size + j][self._last_vectors]

                # TODO: might optimize
                # if np.mean(np.isnan(last_vectors).any(axis=1)) == 1:
                #     continue

                distance_matrix = _calc_distance_matrix_cpu(
                    self._training_vectors, test_vectors)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = self._training_vectors[distance_matrix < eps][:, -1]
                if np.all(points == np.nan):
                    continue

                # теперь нужно выбрать финальное прогнозное значение из возможных
                forecast_point = points.mean() + np.random.normal(0, noise_amp)

                if use_priori:
                    if np.abs(forecast_point - X_test[j]) < priori_eps:
                        X_start[original_size + j] = forecast_point
                else:
                    X_start[original_size + j] = forecast_point
            return X_start[original_size:]

        start = timer()
        X_traj_pred = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(get_trajectory_forecast)(i, X_start)
            for i in range(n_trajectories))
        end = timer()
        if print_time:
            print(timedelta(seconds=end - start))

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_traj_pred = np.array(X_traj_pred).T

        return X_traj_pred

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
    ) -> cp.ndarray:
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
            traj_alive, cluster_centers, X_pred = self._cluster_sets(
                X_traj_pred, dbs_min_trajectories, dbs_eps, dbs_min_samples)
            res['traj_alive'] = traj_alive
            res['cluster_centers'] = cluster_centers
        elif method == 'quantile':
            qs, traj_alive, X_pred = self._get_quantile_prediction(
                X_traj_pred, min_trajectories, max_err, alpha)
            res['qs'] = qs
            res['traj_alive'] = traj_alive
        else:
            raise ValueError(f"unknown method '{method}'")

        if use_priori:
            X_pred[np.abs(X_pred - X_test) > priori_eps] = np.nan

        res['X_pred'] = X_pred

        return res

    def _cluster_sets(
        self,
        X_traj_pred: Union[np.ndarray, cp.ndarray],
        min_trajectories,
        dbs_eps: float,
        dbs_min_samples: int,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param X_traj_pred:
        :param dbs_eps:
        :param dbs_min_samples:
        :return: Возвращает центр самого большого кластера на каждом шаге.
        """

        xp = cp.get_array_module(X_traj_pred)

        X_pred = xp.full(shape=[
            X_traj_pred.shape[0],
        ], fill_value=xp.nan)
        if xp.__name__ == 'numpy':
            dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
        else:
            dbs = cumlDBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

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
            if xp.all(dbs.labels_ == -1):
                continue
            cluster_labels, cluster_sizes = xp.unique(
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

    def _get_quantile_prediction(
        self,
        X_traj_pred: cp.ndarray,
        min_trajectories,
        max_err: float = 0.1,
        alpha=0.3,
    ) -> Union[np.ndarray, cp.ndarray]:
        """
        Скластеризировать полученные в результате пулла множества прогнозных значений.

        :param X_traj_pred:
        :param max_err: max forecasting error
        :param alpha:
        :return: Возвращает среднее предсказание в [alpha, 1-alpha] квантилях
        """

        xp = cp.get_array_module(X_traj_pred)

        if xp.__name__ != 'numpy':
            X_traj_pred = cp.asnumpy(X_traj_pred)

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

        if xp.__name__ != 'numpy':
            X_traj_pred = cp.asarray(X_traj_pred)
        return qs, traj_alive, X_pred


def _calc_distance_matrix_gpu(
    training_vectors: cp.ndarray,
    test_vectors: cp.ndarray,
) -> cp.ndarray:
    """
    calculate the distance matrix between training_vectors and test_vectors
    """

    # drop last point from training vectors
    training_vectors = training_vectors[:, :, :-1]
    training_vectors = training_vectors[cp.newaxis, :, :, :]

    # reshaping test_vectors to efficiently calculate the distances
    test_vectors = test_vectors[:, :, cp.newaxis, :]

    distances = cp.sqrt(((training_vectors - test_vectors)**2).sum(axis=-1))

    return distances


def _calc_distance_matrix_cpu(
    training_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> np.ndarray:
    """
    calculate the distance matrix between training_vectors and test_vectors
    """

    # drop last point from training vectors
    training_vectors = training_vectors[:, :, :-1]

    # reshaping test_vectors to efficiently calculate the distances
    test_vectors = test_vectors[:, cp.newaxis, :]

    distances = np.sqrt(((training_vectors - test_vectors)**2).sum(axis=-1))

    return distances