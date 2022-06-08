from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse


class Adjacency:
    def __init__(
        self,
        spin_side: int = None,
        dimensions: int = 2,
        adja_dict: Optional[Dict[tuple, float]] = None,
    ):
        self.AdjaDict: Dict[Tuple(int, int), float] = {}
        self.Dimensions = dimensions
        self.SpinSide = spin_side

        if adja_dict is not None:
            self.AdjaDict = adja_dict

        self.NeighboursCouplings: Optional[np.ndarray] = None
        self.Connectivity: Optional[Union[int, List[int]]] = None
        self.MaxNeighbours: Optional[int] = None
        self.AdjaMatrix: Optional[np.ndarray] = None

    def create_adjacency(self, connectivity: Union[int, List[int]], seed: int = 12345):
        assert connectivity <= 7, "Not implemented for connectivity greater than 7"
        assert self.Dimensions == 2, "Random adjacency only implemented for dim=2"
        self.Connectivity = connectivity
        self._create_adjacency(connectivity, seed=seed)
        self._create_neighbours()

    def get_adjadict(self) -> Dict[tuple, int]:
        return self.AdjaDict

    def get_neighbours(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.NeighboursCouplings is None:
            self._create_neighbours()
        return self.NeighboursCouplings[..., 0], self.NeighboursCouplings[..., 1]

    def get_adjamatrix(self) -> np.ndarray:
        if self.AdjaMatrix is None:
            self._create_adjamatrix()
        return self.AdjaMatrix.toarray()

    def get_sparse(self) -> sparse.coo.coo_matrix:
        if self.AdjaMatrix is None:
            self._create_adjamatrix()
        return self.AdjaMatrix

    def savetxt(self) -> None:
        assert bool(
            self.AdjaDict
        ), "The connectivity dictionary is empty, instantiate first."
        txtarr = []
        for (i, j), coupling in self.AdjaDict.items():
            # see http://mcsparse.uni-bonn.de/spinglass/ for the format style
            # see loadtxt as well
            txtarr.append([i + 1, j + 1, coupling])
        np.savetxt(f"couplings-{self.SpinSide}spins", txtarr)

    def loadtxt(self, txt_path: str) -> None:
        txt_file = np.loadtxt(txt_path)
        # see http://spinglass.uni-bonn.de/ for the input format
        # for 2D lattice the elements are numbered row-wise
        # for 3D lattice the elements are numbered sequentially
        # layer by layer starting with index 1
        # and so on
        adjadict = {}
        for i in range(txt_file.shape[0]):
            adjadict.update(
                {(int(txt_file[i, 0] - 1), int(txt_file[i, 1] - 1)): txt_file[i, 2]}
            )
        self.AdjaDict = adjadict

    def _create_neighbours(self) -> None:
        if self.MaxNeighbours is None:
            self.MaxNeighbours = self.SpinSide ** 2
        self.NeighboursCouplings = np.zeros((self.SpinSide ** 2, self.MaxNeighbours, 2))
        for spins, coupling in self.AdjaDict.items():
            num_nghb = np.where(self.NeighboursCouplings[spins[0], :, 1] == 0)[0]
            self.NeighboursCouplings[spins[0], num_nghb[0]] = spins[1], coupling
            num_nghb = np.where(self.NeighboursCouplings[spins[1], :, 1] == 0)[0]
            self.NeighboursCouplings[spins[1], num_nghb[0]] = spins[0], coupling
        mask = np.where((self.NeighboursCouplings == 0).all(0).all(1))
        # the max number of neighbours is alway the minimum of zeros
        # elements in the matrix
        self.MaxNeighbours = mask[0]
        self.NeighboursCouplings = np.delete(self.NeighboursCouplings, mask, axis=1)

    def _create_adjacency(
        self, connectivity: Union[int, List[int]], seed: int = 12345
    ) -> None:
        if isinstance(connectivity, int):
            connectivity = np.arange(connectivity) + 1
        # get the number of neighbours
        neighbs = np.zeros(connectivity[-1], dtype=int)
        neighbs[connectivity - 1] = 4
        # if connectivity is 4 or 7 we have 4 more neighbours
        neighbs[np.logical_and(connectivity % 3 == 0, connectivity != 0)] *= 2
        self.MaxNeighbours = neighbs.sum()
        # set a seed to sample couplings
        np.random.seed(seed)
        for i in range(self.SpinSide):
            for j in range(self.SpinSide):
                spin_num = j + i * self.SpinSide
                # fill Adjacency dictionary
                self._create_couplings(np.asarray([i, j]), spin_num, connectivity, seed)

    def _create_couplings(
        self,
        idxs: np.ndarray,
        spin_num: int,
        connectivity: Union[int, List[int]],
        seed: int,
    ) -> None:
        for connect in connectivity:
            if connect == 1:
                # right spin coupling
                if idxs[1] + 1 < self.SpinSide:
                    self.AdjaDict.update({(spin_num, spin_num + 1): np.random.normal()})
                # down spin coupling
                if idxs[0] + 1 < self.SpinSide:
                    self.AdjaDict.update(
                        {(spin_num, spin_num + self.SpinSide): np.random.normal()}
                    )
            if connect == 2:
                # up-right spin coupling
                if (idxs[0] - 1 > -1) and (idxs[1] + 1 < self.SpinSide):
                    self.AdjaDict.update(
                        {(spin_num, spin_num - self.SpinSide + 1): np.random.normal()}
                    )
                if (idxs + [1, 1] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {(spin_num, spin_num + self.SpinSide + 1): np.random.normal()}
                    )
            if connect == 3:
                # 2-right spin coupling
                if idxs[1] + 2 < self.SpinSide:
                    self.AdjaDict.update({(spin_num, spin_num + 2): np.random.normal()})
                # 2-down spin coupling
                if idxs[0] + 2 < self.SpinSide:
                    self.AdjaDict.update(
                        {(spin_num, spin_num + 2 * self.SpinSide): np.random.normal()}
                    )
            if connect == 4:
                if (idxs[0] - 2 > -1) and (idxs[1] + 1 < self.SpinSide):
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num - 2 * self.SpinSide + 1,
                            ): np.random.normal()
                        }
                    )
                if (idxs[0] - 1 > -1) and (idxs[1] + 2 < self.SpinSide):
                    self.AdjaDict.update(
                        {(spin_num, spin_num - self.SpinSide + 2): np.random.normal()}
                    )
                if (idxs + [1, 2] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {(spin_num, spin_num + self.SpinSide + 2): np.random.normal()}
                    )
                if (idxs + [2, 1] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num + 2 * self.SpinSide + 1,
                            ): np.random.normal()
                        }
                    )
            if connect == 5:
                if idxs[1] + 3 < self.SpinSide:
                    self.AdjaDict.update({(spin_num, spin_num + 3): np.random.normal()})
                if idxs[0] + 3 < self.SpinSide:
                    self.AdjaDict.update(
                        {(spin_num, spin_num + 3 * self.SpinSide): np.random.normal()}
                    )
            if connect == 6:
                if (idxs[0] - 2 > -1) and (idxs[1] + 2 < self.SpinSide):
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num - 2 * self.SpinSide + 2,
                            ): np.random.normal()
                        }
                    )
                if (idxs + [2, 2] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num + 2 * self.SpinSide + 2,
                            ): np.random.normal()
                        }
                    )
            if connect == 7:
                if (idxs[0] - 3 > -1) and (idxs[1] - 3 < self.SpinSide):
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num - 3 * self.SpinSide + 1,
                            ): np.random.normal()
                        }
                    )
                if (idxs[0] - 1 > -1) and (idxs[1] + 3 < self.SpinSide):
                    self.AdjaDict.update(
                        {(spin_num, spin_num - self.SpinSide + 3): np.random.normal()}
                    )
                if (idxs + [1, 3] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {(spin_num, spin_num + self.SpinSide + 3): np.random.normal()}
                    )
                if (idxs + [3, 1] < [self.SpinSide, self.SpinSide]).all():
                    self.AdjaDict.update(
                        {
                            (
                                spin_num,
                                spin_num + 3 * self.SpinSide + 1,
                            ): np.random.normal()
                        }
                    )

    def _create_adjamatrix(self) -> None:
        i_vec, j_vec = [], []
        couplings = []
        for (i, j), coupling in self.AdjaDict.items():
            i_vec.extend([i, j])
            j_vec.extend([j, i])
            couplings.extend([coupling, coupling])
        self.AdjaMatrix = sparse.coo_matrix((couplings, (i_vec, j_vec)))
