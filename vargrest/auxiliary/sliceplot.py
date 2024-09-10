import matplotlib.figure as mpl_f
import numpy as np
from vargrest.variogramestimation.parametricvariogram import VariogramType, AnisotropicVariogram
from vargrest.variogramestimation.variogramestimation import ParametricVariogramEstimate, NonparametricVariogramEstimate
from vargrest.auxiliary import variogram


_VARIOGRAM_MAP = {
    VariogramType.Exponential: variogram.ExponentialVariogram,
    VariogramType.Spherical: variogram.SphericalVariogram,
    VariogramType.Gaussian: variogram.GaussianVariogram,
    VariogramType.GeneralExponential: lambda rx, ry, rz, azi: variogram.GeneralExponentialVariogram(
        rx, ry, rz, azi, power=AnisotropicVariogram.GEN_EXP_PWR
    )
}


class SlicePlot:
    """ Class for generating a slice plot of a grid """
    def __init__(self, nx, ny, nz, dx, dy, dz) -> None:
        # Generate figure
        self._shape = nx, ny, nz
        self._resolution = dx, dy, dz
        self._fig = mpl_f.Figure(figsize=(15, 10))
        self._axes = self._fig.subplots(2, 2)

    def save(self, filename):
        self._fig.savefig(filename)

    def add_parametric_estimate(self, pe: ParametricVariogramEstimate):
        pp = pe.polished_parameters()
        nx, ny, nz = self._shape
        dx, dy, dz = self._resolution
        var = _VARIOGRAM_MAP[pe.family](
            pp['r_major']['value'],
            pp['r_minor']['value'],
            pp['r_vertical']['value'],
            pe.raw_parameters()['azi'],
        )
        par_v = var.create_corr_array(nx, dx, ny, dy, nz, dz)
        par_v = par_v[
            nx // 2 + 1:nx // 2 + nx + 1,
            ny // 2 + 1:ny // 2 + ny + 1,
            nz // 2 + 1:nz // 2 + nz + 1,
        ]
        par_v = (1.0 - par_v) * pp['sigma']['value'] ** 2
        self._add_slices(par_v, '--', label=f'Param. ({pe.family.value})')

    def add_non_parametric_estimate(self, ne: NonparametricVariogramEstimate):
        self._add_slices(ne.variogram_map_values(), label='Empirical')

    def _add_slices(self, grid, *args, scale=False, **kwargs):
        def _filter_nan(xx, yy):
            return xx[~np.isnan(yy)], yy[~np.isnan(yy)]

        if scale:
            def _scale(xx):
                return xx / np.nanmax(xx)
        else:
            def _scale(xx):
                return xx

        assert self._shape == grid.shape
        nx, ny, nz = self._shape
        dx, dy, dz = self._resolution
        # X slice
        xax = np.arange(0, nx // 2 + 1) * dx
        yax = _scale(grid[nx // 2:, ny // 2, nz // 2])
        self._axes[0, 0].plot(*_filter_nan(xax, yax), *args, **kwargs)
        self._axes[0, 0].legend()
        self._axes[0, 0].set_title('X-slice')
        self._axes[0, 0].grid()

        # Y slice
        xax = np.arange(0, ny // 2 + 1) * dy
        yax = _scale(grid[nx // 2, ny // 2:, nz // 2])
        self._axes[0, 1].plot(*_filter_nan(xax, yax), *args, **kwargs)
        self._axes[0, 1].legend()
        self._axes[0, 1].set_title('Y-slice')
        self._axes[0, 1].grid()

        # Z slice
        xax = np.arange(0, nz // 2 + 1) * dz
        yax = _scale(grid[nx // 2, ny // 2, nz // 2:])
        self._axes[1, 0].plot(*_filter_nan(xax, yax), *args, **kwargs)
        self._axes[1, 0].legend()
        self._axes[1, 0].set_title('Z-slice')
        self._axes[1, 0].grid()
