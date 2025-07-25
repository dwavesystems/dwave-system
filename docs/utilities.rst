.. _system_utilities:

=========
Utilities
=========

.. todo:: add dwave_sampler.qpu_graph post https://github.com/dwavesystems/dwave-system/pull/585

.. automodule:: dwave.system.utilities

.. currentmodule:: dwave.system

.. autosummary::
   :toctree: generated/

   ~utilities.anneal_schedule_with_offset
   ~utilities.common_working_graph
   ~coupling_groups.coupling_groups
   ~utilities.energy_scales_custom_schedule

Temperature and Unit-Conversion Utilities
-----------------------------------------

.. automodule:: dwave.system.temperatures

.. currentmodule:: dwave.system.temperatures

.. autosummary::
   :toctree: generated/

   background_susceptibility_bqm
   background_susceptibility_ising
   effective_field
   fast_effective_temperature
   fluxbias_to_h
   freezeout_effective_temperature
   h_to_fluxbias
   Ip_in_units_of_B
   maximum_pseudolikelihood
   maximum_pseudolikelihood_temperature


.. [Chat2007]
    Chatterjee, Sourav.
    “Estimation in Spin Glasses: A First Step.”
    The Annals of Statistics 35, no. 5 (2007): 1931-46.
    http://www.jstor.org/stable/25464568