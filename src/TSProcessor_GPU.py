import gc

import numpy as np
import torch
from sklearn.cluster import DBSCAN

# https://stackoverflow.com/a/55239060
from timeit import default_timer as timer
from datetime import timedelta
from tqdm.auto import tqdm

# TODO: fix docstrings


class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int,
                 X_train: torch.Tensor):

        self._device = X_train.device
        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread
        self._points_in_template = points_in_template

        self.x_dim: int = max_template_spread**(
            points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template  # сколько зубчиков в каждом шаблоне

        # сами шаблоны
        templates = (np.zeros(shape=(self.x_dim, 1), dtype=int), )
        # код, который заполняет шаблоны нужными значениями
        for i in range(1, points_in_template):
            col = (
                np.repeat(np.arange(1, max_template_spread + 1, dtype=int),
                          max_template_spread**(points_in_template -
                                                (i + 1))) +
                templates[i - 1][::max_template_spread**(points_in_template -
                                                         i)]).reshape(-1, 1)

            templates += (col, )

        self._templates = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes= self._templates[:,1:] \
            - self._templates[:, :-1]  # k1, k2, ...

        self.__fit(X_train)

    def __fit(self, X_train: torch.Tensor) -> None:
        """ Fill training vectors from time_series """

        self.y_dim = X_train.shape[0] - int(self._templates[0][-1])

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        self._training_vectors = torch.full(size=(self.x_dim, self.y_dim,
                                                  self.z_dim),
                                            fill_value=np.inf,
                                            dtype=X_train.dtype,
                                            device=self._device)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (X_train[
                self._templates[i] +
                np.arange(X_train.shape[0] - self._templates[i][-1])[:, None]])

            self._training_vectors[i, :template_data.shape[0]] = (X_train[
                self._templates[i] +
                np.arange(X_train.shape[0] - self._templates[i][-1])[:, None]])

        self._last_vectors = np.cumsum(-self._template_shapes[:, ::-1],
                                       axis=1)[:, ::-1]

        # self._training_vectors = torch.from_numpy(self._training_vectors).to(
        #     self._device)
        # TODO: negative strides are not supported, see https://github.com/facebookresearch/InferSent/issues/99
        self._last_vectors = torch.from_numpy(self._last_vectors.copy()).to(
            self._device)

    def heal(
        self,
        X_start: torch.Tensor,
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        X_pred: torch.Tensor,
        random_seed=1,
    ):
        # print(X_start.shape, self._max_template_spread, self._points_in_template)
        assert (X_start.shape[0] == self._max_template_spread *
                (self._points_in_template - 1)), "X_start should be bigger"

        # doing this to fill X_start
        original_size = X_start.shape[0]
        X_start = X_start.detach().clone()
        X_start = X_start.resize_(X_start.shape[0] + h_max)
        X_start[-h_max:] = X_pred

        X_start = torch.repeat_interleave(X_start[np.newaxis, :],
                                          n_trajectories,
                                          dim=0)

        # training_for_dist = torch.repeat_interleave(
        #     self._training_vectors[None, :, :, -1], n_trajectories, dim=0)

        torch.manual_seed(random_seed)
        noise = torch.normal(0, noise_amp, size=(n_trajectories, h_max))

        delta = self._max_template_spread * (self._points_in_template - 1)

        # a dirty hack to use torch.where with nan
        # TODO: make prettier
        torchnan = torch.tensor([np.nan]).type(X_start.dtype).to(self._device)

        for i in tqdm(range(h_max)):
            # if not torch.isnan(X_pred[i]).item():
            #     continue
            predictions_mat = torch.full(size=(n_trajectories,
                                               self._points_in_template),
                                         fill_value=np.nan,
                                         dtype=X_start.dtype)
            for point_in_template in range(0, self._points_in_template):

                last_vectors = None
                if point_in_template != self._points_in_template - 1:
                    last_vectors = np.concatenate([
                        np.cumsum(-self._template_shapes[:, ::-1]
                                  [:, :point_in_template],
                                  axis=1)[:, ::-1],
                        np.cumsum(self._template_shapes[:, point_in_template:],
                                  axis=1)
                    ],
                                                  axis=1) - delta
                else:
                    last_vectors = self._last_vectors

                test_vectors = X_start[:, :original_size + i +
                                       delta][:, last_vectors]

                dist = _calc_distance_matrix_gpu(self._training_vectors,
                                                 test_vectors,
                                                 point_in_template)

                training_for_dist = torch.repeat_interleave(
                    self._training_vectors[None, :, :, -1],
                    n_trajectories,
                    dim=0)
                wh = torch.where(dist < eps, training_for_dist,
                                 torchnan).reshape(n_trajectories, -1)
                predictions = _nanmean(wh, inplace=True, dim=1)
                predictions_mat[:, point_in_template] = predictions

                gc.collect()
                torch.cuda.empty_cache()

            # print(X_pred[i], predictions_mat)

            predictions = predictions_mat.mean(dim=1) + noise[:, i]
            predictions = predictions.to(self._device)
            # X_start[:, original_size + i] = predictions
            X_start[:, original_size + i] = (predictions_mat[:, 0] +
                                             noise[:, i]).to(self._device)

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_start = X_start[:, original_size:]
        X_start = X_start.T
        return X_start

    def predict_trajectories(
        self,
        X_start: torch.Tensor,  # forecast after X_start
        h_max: int,  # forecasting horizon
        eps: float,  # eps for distance matrix
        n_trajectories: int,
        noise_amp: float,
        X_pred: torch.Tensor = None,
        use_priori=False,
        X_test: torch.Tensor = None,
        priori_eps=0.1,
        random_seed=1,
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

        assert (X_start.shape[0] == self._max_template_spread *
                (self._points_in_template - 1)), "X_start should be bigger"

        if use_priori:
            assert X_test is not None, 'X_test should be specified'

        # doing this to fill X_start
        original_size = X_start.shape[0]
        X_start = X_start.detach().clone()
        X_start = X_start.resize_(X_start.shape[0] + h_max)
        X_start[-h_max:] = np.nan

        X_start = torch.repeat_interleave(X_start[np.newaxis, :],
                                          n_trajectories,
                                          dim=0)

        training_for_dist = torch.repeat_interleave(
            self._training_vectors[None, :, :, -1], n_trajectories, dim=0)

        torch.manual_seed(random_seed)
        noise = torch.normal(0, noise_amp,
                             size=(n_trajectories, h_max)).to(self._device)

        # a dirty hack to use torch.where with nan
        # TODO: make prettier
        torchnan = torch.tensor([np.nan]).type(X_start.dtype).to(self._device)

        for i in range(h_max):
            if X_pred is not None and not np.isnan(X_pred[i]):
                X_start[:, original_size + i] = X_pred[i]
                continue
            test_vectors = X_start[:, :original_size + i][:,
                                                          self._last_vectors]
            dist = _calc_distance_matrix_gpu(self._training_vectors,
                                             test_vectors)

            # see https://stackoverflow.com/a/29046530
            wh = torch.where(dist < eps, training_for_dist,
                             torchnan).reshape(n_trajectories, -1)
            predictions = _nanmean(wh, inplace=True, dim=1) + noise[:, i]
            if use_priori:
                predictions = torch.where(
                    torch.abs(predictions - X_test[i]) < priori_eps,
                    predictions, torchnan)
            X_start[:, original_size + i] = predictions

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        # размер: steps x n_trajectories
        X_start = X_start[:, original_size:]
        X_start = X_start.T
        return X_start

    def predict_unified(
        self,
        X_traj_pred: np.ndarray,
        method: str,
        use_priori=False,
        X_test: np.ndarray = None,
        priori_eps: float = None,
        # quantile params
        min_trajectories: int = None,
        max_err: float = None,
        alpha: float = None,
        # cluster params
        dbs_min_trajectories: int = None,
        dbs_eps: float = None,
        dbs_min_samples: int = None,
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
        X_traj_pred: np.ndarray,
        min_trajectories,
        dbs_eps: float,
        dbs_min_samples: int,
    ) -> np.ndarray:
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
            if np.all(dbs.labels_ == -1):
                continue
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

    def _get_quantile_prediction(
        self,
        X_traj_pred: np.ndarray,
        min_trajectories,
        max_err: float,
        alpha,
    ) -> np.ndarray:
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
            if len(traj_pred) == 0:
                # no predictions for trajectories on i-th step
                traj_alive.append(0)
                qs.append((np.nan, np.nan))
                continue
            q1, q2 = np.quantile(traj_pred, q=[alpha, 1 - alpha])
            qs.append((q1, q2))
            # print(i, q1, q2, q2 - q1, len(traj_pred))
            traj_pred = traj_pred[(traj_pred > q1) & (traj_pred < q2)]
            traj_alive.append(len(traj_pred))
            if len(traj_pred) < min_trajectories:
                # only a few trajectories left
                continue
            if q2 - q1 <= max_err:
                X_pred[i] = np.mean(traj_pred)

        qs = np.array(qs)
        traj_alive = np.array(traj_alive)

        return qs, traj_alive, X_pred


def _calc_distance_matrix_gpu(
    training_vectors: torch.Tensor,
    test_vectors: torch.Tensor,
    point_in_template: int = None,
) -> torch.Tensor:
    """
    calculate the distance matrix between training_vectors and test_vectors
    """

    if point_in_template is not None:
        training_vectors = torch.cat([
            training_vectors[:, :, :point_in_template],
            training_vectors[:, :, point_in_template + 1:]
        ],
                                     dim=2)
    else:
        # drop last point from training vectors
        training_vectors = training_vectors[:, :, :-1]
    training_vectors = training_vectors[np.newaxis, :, :, :]

    # reshaping test_vectors to efficiently calculate the distances
    test_vectors = test_vectors[:, :, np.newaxis, :]

    distances = torch.sqrt(((training_vectors - test_vectors)**2).sum(axis=-1))

    return distances


# see https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
def _nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


# def _swap_training_vec(vec, a, b):
#     # swap a and b without c:
#     # a = a + b
#     # b = a - b
#     # a = a - b
#     vec[:, :, a] = vec[:, :, a] + vec[:, :, b]
#     vec[:, :, b] = vec[:, :, a] - vec[:, :, b]
#     vec[:, :, a] = vec[:, :, a] - vec[:, :, b]
