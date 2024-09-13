import json
from enum import Enum

import numpy as np
import os
import pickle
from typing import Union, Dict, List

from vargrest.auxiliary.sliceplot import SlicePlot
from vargrest.auxiliary.variogramplot import VariogramPlot
from vargrest.variogramdata.variogramdata import VariogramDataInterface
from vargrest.variogramestimation.variogramestimation import\
    ParametricVariogramEstimate, NonparametricVariogramEstimate, VariogramEstimator


ExecutionSummary = Dict[str, Union[str, int, float]]


class SummaryDataType(Enum):
    # Programmatic labels for data that is used in the summary
    Identifier = 'identifier'
    Family = 'family'
    ArchelFilter = 'archel_filter'
    Indicator = 'indicator'
    Box = 'box'
    Attribute = 'attribute'
    Quality = 'quality[<1.0]'
    RMajor = 'r_major[m]'
    RMinor = 'r_minor[m]'
    Azimuth = 'azimuth[deg]'
    RVertical = 'r_vertical[m]'
    Sigma = 'sigma[N/A]'
    QualityX = 'quality_x[<1.0]'
    QualityY = 'quality_y[<1.0]'
    QualityZ = 'quality_z[<1.0]'


def summarize(pe: ParametricVariogramEstimate, meta_data: Dict[SummaryDataType, Union[str, int, float]]
              ) -> ExecutionSummary:
    polished = pe.polished_parameters()
    polished_flat = {f'{k}[{polished[k]["unit"]}]': polished[k]['value'] for k in polished.keys()}
    summary = {
        d.value: polished_flat[d.value]
        for d in SummaryDataType
        if d.value in polished_flat
    }
    summary[SummaryDataType.Quality.value] = pe.quality.full
    summary[SummaryDataType.QualityX.value] = pe.quality.x_slice
    summary[SummaryDataType.QualityY.value] = pe.quality.y_slice
    summary[SummaryDataType.QualityZ.value] = pe.quality.z_slice
    # Include meta data
    summary.update({m.value: v for m, v in meta_data.items()})
    return summary


def conclude(ve: VariogramEstimator,
             pe: ParametricVariogramEstimate,
             ne: NonparametricVariogramEstimate,
             qc_dir: str,
             fn_template: str,
             full_qc: bool
             ):

    if full_qc is True:
        # Pickle variogram estimation data
        pickle.dump((ve, ne), open(os.path.join(qc_dir, fn_template + '_data_.pkl'), 'wb'))

        # Generate slice file
        ve.generate_3d_slice_image(qc_dir, fn_template + '_slices_')

    # Save variogram map plots to file
    dump_variogram_plot(ve, ne, pe, qc_dir, fn_template)


def dump_variogram_plot(ve: VariogramEstimator,
                        ne: NonparametricVariogramEstimate,
                        pe: ParametricVariogramEstimate,
                        qc_dir: str,
                        fn_template: str):
    clims = (0.0, 1.5 * np.nanvar(ve.data()))
    vp = VariogramPlot(ne, pe, clims, 0.95)
    fn = fn_template + "_variograms_2d_.png"
    p = os.path.join(qc_dir, fn)
    vp.fig.savefig(p)


def dump_summaries_to_csv(summaries: List[ExecutionSummary], csv_file: str):
    csv_values = [s for s in SummaryDataType if s != SummaryDataType.Box]
    with open(csv_file, 'w') as writer:
        # Header
        header = ''.join([f'{f.value:<18.18}' for f in csv_values])
        writer.write(header.strip() + '\n')

        # Content
        def _formatter(_s):
            if isinstance(_s, float):
                return f'{_s: <17.5}'
            else:
                return f'{str(_s):<17.17}'

        for r in summaries:
            line = ' '.join([_formatter(r[f.value]) for f in csv_values])
            writer.write(line.strip())
            writer.write('\n')


def dump_summaries_to_json(summaries: List[ExecutionSummary], json_file: str):
    json.dump(summaries, open(json_file, 'w'), indent=2)
    # Replace 'NaN' with 'null' to comply with JSON specification. This is not a bullet-proof replacement, but
    # should not be a problem since we have full control of the content of the output file
    new_jf = open(json_file, 'r').read().replace('NaN', 'null')
    open(json_file, 'w').write(new_jf)
