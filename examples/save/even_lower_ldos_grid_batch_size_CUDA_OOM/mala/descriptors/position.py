"""Position descriptor class."""
import os

import ase
import ase.io

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor


class Position(Descriptor):
    """Class for parsing of Position descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(Position, self).__init__(parameters)

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "Position"

    @property
    def feature_size(self):
        """Get the feature dimension of this data."""
        return 3

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a position descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.
        """
        if in_units == "None" or in_units is None:
            return array
        else:
            raise Exception("Unsupported unit for position descriptors.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a position descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "None" or out_units is None:
            return array
        else:
            raise Exception("Unsupported unit for position descriptors.")

    def _calculate(self, atoms, outdir, grid_dimensions, **kwargs):
        """Perform actual position extraction."""
        positions_descriptors_np = atoms.get_positions()
        return positions_descriptors_np, len(positions_descriptors_np)

