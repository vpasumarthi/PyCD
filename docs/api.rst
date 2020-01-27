API Documentation
=================

.. autoclass:: PyCD.Material
   :members: generate_sites

.. autoclass:: PyCD.Neighbors
   :members: get_system_element_index, get_quantum_indices, get_coordinates, compute_distance, hop_neighbor_sites, get_pairwise_min_image_vector_data, generate_neighbor_list

.. autoclass:: PyCD.System
   :members: pot_r_ewald, get_effective_k_vectors, get_cosine_data, pot_k_ewald, pot_k_ewald_with_k_vector_data, benchmark_ewald, base_charge_config_for_accuracy_analysis, minimize_real_space_cutoff_error, minimize_fourier_space_cutoff_error, compute_cutoff_errors, convergence_check_with_r_cut, get_energy_profile_with_r_cut, convergence_check_with_k_cut, get_energy_profile_with_k_cut, check_for_k_cut_step_energy_convergence, get_convergence_rcut, get_simulation_cell_real_space_parameters, get_step_change_analysis_with_k_cut, plot_energy_profile_in_bounded_k_cut, get_new_k_vectors, get_k_vector_energy_contribution, get_k_vector_based_energy_contribution, get_precise_step_change_data, get_k_cut_choices, get_optimized_r_cut, get_cutoff_parameters, get_ewald_parameters, get_precomputed_array_real, get_precomputed_array_fourier, get_precomputed_array_fourier_with_k_vector_data, get_precomputed_array

.. autoclass:: PyCD.Run
   :members: get_element_type_element_index, get_process_attributes, get_process_rates, compute_drift_mobility, generate_random_doping_distribution, get_doping_distribution, get_doping_analysis, get_shell_based_neighbors, get_system_shell_based_neighbors, get_site_wise_shell_indices, generate_initial_occupancy, base_charge_config, charge_config, preproduction, do_kmc_steps

.. autoclass:: PyCD.Analysis
   :members: compute_msd, generate_msd_plot

.. automodule:: PyCD.material_setup
   :members: material_setup

.. automodule:: PyCD.material_preprod
   :members: material_preprod

.. automodule:: PyCD.material_run
   :members: material_run

.. automodule:: PyCD.material_msd
   :members: material_msd

.. automodule:: PyCD.io
   :members: read_poscar, write_poscar, generate_report

