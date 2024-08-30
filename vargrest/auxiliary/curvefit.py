import dataclasses
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import trapezoid


@dataclasses.dataclass
class QualityMeasure:
    full: float
    x_slice: float
    y_slice: float
    z_slice: float

    @staticmethod
    def nan():
        return QualityMeasure(np.nan, np.nan, np.nan, np.nan)


def fit_3d_field(
    func, jac, array, resolution, counts, par_guess, bounds, sigma_wt
) -> Tuple[np.ndarray, QualityMeasure]:
    """
    Fits a grid-evaluated function to observations.

    :param func: Callable
        The grid-evaluated function. The first argument should be a 3xN array as input, representing the x, y, and z
        coordinates on where to evaluate the function. The returned valued should be an N-length array representing the
        values of the function at the given coordinates
    :param jac: Optional[Callable]
        Jacobian of the above func. Can be None if scipy should estimate the Jacobian instead.
    :param array: np.ndarray {shape (nx, ny, nz)}
        An array of observations (evaluations of func). array[i, j, k] represent the evaluation of func at
            [(-nx / 2 + i) * dx, (-ny / 2 + j) * dy, (-nz / 2 + k) * dz]
        np.nan may be provided for coordinates without data.
    :param resolution: Tuple[float, float, float]
        Length-3 tuple giving the spatial resolution (currently not in use)
    :param counts: np.ndarray {shape (nx, ny, nz)}
        The number of observations behind array, element-wise (currently not in use)
    :param par_guess: np.ndarray {shape (M,)}
        The initial parameter guess for the parameters of func (past the first argument)
    :param bounds: Tuple[np.ndarray {shape (M,)}, np.ndarray {shape (M,)}]
        The bounds for the parameters of func (past the first argument)
    :param sigma_wt: float
        A weighting parameter to adjust the weight of observations away from the center xyz = (0, 0, 0). A smaller value
        means more weight is put on observations close to the center. The scale should be considered as a number of grid
        cells.
    :return: Tuple[np.ndarray {shape (M,)}, float]
        First, an array of the optimal parameter set for func, fit to the data in array. Second, a scalar value
        representing the accuracy of the fit. This 'quality' value is 0.0 if the fit is equal (in terms of L2) to the
        fit of using baseline_parameters. It is 1.0 if the fit is 100%.

    """
    nx, ny, nz = array.shape
    dx, dy, dz = resolution

    xmin = -int(nx / 2)
    xmax = -xmin
    ymin = -int(ny / 2)
    ymax = -ymin
    zmin = -int(nz / 2)
    zmax = -zmin
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    zv = np.linspace(zmin, zmax, nz)
    xm, ym, zm = np.meshgrid(xv, yv, zv, indexing='ij')

    indep_data = np.vstack((xm.ravel(), ym.ravel(), zm.ravel()))
    dep_data = array.ravel()
    counts = counts.ravel()

    # Filter out nan entries
    not_nan = np.ones_like(dep_data, dtype=bool)
    if np.all(np.isnan(dep_data)):
        # Not possible to calculate a proper estimate
        return np.full_like(par_guess, fill_value=np.nan), QualityMeasure.nan()
    elif np.any(np.isnan(dep_data)):
        not_nan = np.squeeze(np.nonzero(~np.isnan(dep_data)))
        indep_data = indep_data[:, not_nan]
        dep_data = dep_data[not_nan]
        counts = counts[not_nan]
    xmv = xm.ravel()[not_nan]
    ymv = ym.ravel()[not_nan]
    zmv = zm.ravel()[not_nan]

    # Prepare weights, w_b ~ 1/\sigma_b^2
    if sigma_wt is not None:
        wt_coef = 1.0 / (2.0 * sigma_wt**2)
        dsq = xmv**2 + ymv**2 + zmv**2
        wts = np.exp(-wt_coef * dsq)
        wts = _transform_array(wts, 2)
        #wts = wts / wts.sum()
        sig = np.divide(1.0, wts, out=np.full_like(wts, np.inf), where=wts != 0.0)
    else:
        sig = None

    # Least-squares fit
    popt = curve_fit(func, indep_data, dep_data, sigma=sig, p0=par_guess, bounds=bounds, jac=jac)[0]

    # Calculate quality of solution
    quality = _calculate_quality_1(lambda x: func(x, *popt), indep_data, dep_data, not_nan, array)

    return popt, quality


def _calculate_quality_1(
    func: Callable[[np.ndarray], float],  # (3, N) float -> (N,) float
    indep_data: np.ndarray,  # (3, N) float
    dep_data: np.ndarray,  # (N,) float
    not_nan: np.ndarray,  # (N,) boolean
    array,  # (K,L,M) float, K * L * M == N
):
    def _to_3d(a):
        f1 = np.full(array.size, fill_value=np.nan)
        f1[not_nan] = a
        return f1.reshape(array.shape)

    dep_data_3d = _to_3d(dep_data)
    par_est_3d = _to_3d(func(indep_data))
    sigma_est = np.median(dep_data)

    top_err = _calculate_quality_contribution_1(dep_data_3d, par_est_3d, sigma_est)
    sub_err = _calculate_quality_contribution_2(dep_data_3d, par_est_3d, sigma_est)

    return QualityMeasure(
        _aggregate_contributions(top_err.full, sub_err.full),
        _aggregate_contributions(top_err.x_slice, sub_err.x_slice),
        _aggregate_contributions(top_err.y_slice, sub_err.y_slice),
        _aggregate_contributions(top_err.z_slice, sub_err.z_slice),
    )


def _calculate_quality_contribution_1(empirical_3d, parametric_3d, sigma_estimate):
    contrib1_cells = _find_contrib1_cells(empirical_3d, parametric_3d, sigma_estimate)
    return _calculate_quality_measure(
        empirical_3d, parametric_3d, sigma_estimate, contrib1_cells, _contribution_1
    )


def _calculate_quality_contribution_2(empirical_3d, parametric_3d, sigma_estimate):
    contrib2_cells = _find_contrib2_cells(empirical_3d)
    return _calculate_quality_measure(
        empirical_3d, parametric_3d, sigma_estimate, contrib2_cells, _contribution_2
    )


def _aggregate_contributions(top, sub):
    return 1.0 - (top * 0.75 + min(sub, 0.25))


def _contribution_1(empirical, parametric, sigma_estimate):
    diff = empirical - parametric
    worst_case = empirical - sigma_estimate
    return np.abs(diff).sum() / np.abs(worst_case).sum()


def _contribution_2(empirical, parametric, sigma_estimate):
    return np.median(np.abs(empirical - parametric)) / sigma_estimate


def _calculate_quality_measure(empirical_3d, parametric_3d, sigma_estimate, index, measure):
    full = measure(empirical_3d[index], parametric_3d[index], sigma_estimate)

    mx, my, mz = np.array(index.shape) // 2

    ix = np.zeros_like(index, dtype=bool)
    ix[:, my, mz] = 1
    ix &= index
    x = measure(empirical_3d[ix], parametric_3d[ix], sigma_estimate)

    iy = np.zeros_like(index, dtype=bool)
    iy[mx, :, mz] = 1
    iy &= index
    y = measure(empirical_3d[iy], parametric_3d[iy], sigma_estimate)

    iz = np.zeros_like(index, dtype=bool)
    iz[mx, my, :] = 1
    iz &= index
    z = measure(empirical_3d[iz], parametric_3d[iz], sigma_estimate)

    return QualityMeasure(full, x, y, z)


def _find_contrib1_cells(empirical_3d, parametric_3d, sigma_estimate, pc=0.5):
    center = np.zeros_like(empirical_3d, dtype=bool)
    cx, cy, cz = center.shape
    px, py, pz = cx // 4, cy // 4, cz // 4
    center[px:-px, py:-py, pz:-pz] = True
    return (
            ~np.isnan(empirical_3d)
            & center
            & (empirical_3d < pc * sigma_estimate)
            & (parametric_3d < pc * sigma_estimate)
    )


def _find_contrib2_cells(empirical):
    return ~np.isnan(empirical)


def _center_slice(array):
    nx, ny, nz = array.shape
    x_slice = array[:, ny // 2, nz // 2]
    y_slice = array[nx // 2, :, nz // 2]
    z_slice = array[nx // 2, ny // 2, :]
    return x_slice, y_slice, z_slice


def _transform_array(x, n):
    x_below = 2**(n-1) * np.power(x, n)
    x_above = 1.0 - 2**(n-1) * np.power(1 - x, n)
    return np.where(x < 0.5, x_below, x_above)


def find_dominant_direction(variogram_map, grid_resolution):
    n_angles = 24   # Number of angles to check
    n_dists = 21    # Number of lag distance points to evaluate integrand at

    dx = grid_resolution[0]
    dy = grid_resolution[1]
    nx = variogram_map.shape[0]
    ny = variogram_map.shape[1]

    if variogram_map.ndim == 3:     # If 3D, extract middle horizontal slice
        nz = variogram_map.shape[2]
        if nz % 2 == 1:
            g = variogram_map[:, :, nz // 2]
        else:
            v0 = variogram_map[:, :, nz // 2 - 1]
            v1 = variogram_map[:, :, nz // 2]
            v01 = np.where(np.isnan(v0), 0.0, v0) + np.where(np.isnan(v1), 0.0, v1)
            w = ~np.isnan(v0) + ~np.isnan(v1)
            g = np.where(w > 0, w * v01, np.nan)
    elif variogram_map.ndim == 2:
        g = variogram_map
    else:
        raise NotImplementedError('Dominant direction not implemented for other than 2D and 3D')

    if np.all(np.isnan(g)):
        return 0.0

    xlength = (nx - 1) * dx
    ylength = (ny - 1) * dy
    h_max = 0.5 * min(xlength, ylength)    # Limiting distance (upper limit of integration)
    hh = np.linspace(0.0, h_max, n_dists)
    hxx = np.linspace(-0.5 * xlength, 0.5 * xlength, nx)
    hyy = np.linspace(-0.5 * ylength, 0.5 * ylength, ny)

    hxx, hyy = np.meshgrid(hxx, hyy, indexing='ij')

    hxx = hxx.ravel()  # Flatten input
    hyy = hyy.ravel()
    gg = g.ravel()

    ggnans = np.isnan(gg.astype(float))
    hxx = hxx[~ggnans]   # Filter out nans
    hyy = hyy[~ggnans]
    gg = gg[~ggnans]
    hxxyy = np.vstack((hxx, hyy)).transpose()

    azimuths = np.linspace(0.0, np.pi, n_angles)
    integrals = np.empty_like(azimuths)

    interpolator = LinearNDInterpolator(hxxyy, gg)
    for i, azi in enumerate(azimuths):      # Loop over angles
        hxx_i = np.cos(azi) * hh
        hyy_i = np.sin(azi) * hh
        gg_i = interpolator(hxx_i, hyy_i)
        integrals[i] = trapezoid(y=gg_i, x=hh)   # Compute approximate integral from 0 to h_max

    if np.all(np.isnan(integrals)):     # Get index of smallest integral
        return 0.0
    elif np.any(np.isnan(integrals)):
        i_min = np.nanargmin(integrals)
    else:
        i_min = np.argmin(integrals)

    return azimuths[i_min]              # Return corresponding azimuth
