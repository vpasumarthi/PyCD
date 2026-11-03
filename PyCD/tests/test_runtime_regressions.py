"""Regression coverage for configuration entry points and mixed doping modes."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from PyCD.core import Run


ENTRY_POINTS = [
    ("PyCD.material_setup", "material_setup"),
    ("PyCD.material_preprod", "material_preprod"),
    ("PyCD.material_run", "material_run"),
    ("PyCD.material_msd", "material_msd"),
    ("PyCD.material_coc_msd", "materialCOCMSD"),
    ("PyCD.scripts.coc_msd", "compute_coc_msd"),
    ("PyCD.scripts.mean_distance", "compute_mean_distance"),
    ("PyCD.scripts.generate_transition_prob_matrix", "generate_transition_prob_matrix"),
    ("PyCD.scripts.generate_species_site_sd_list", "generate_species_site_sd_list"),
    ("PyCD.scripts.generate_msd_analytical_data", "generate_msd_analytical_data"),
]


class MaterialReached(Exception):
    """Stop before expensive simulation work, after real configuration loading."""


@pytest.mark.parametrize("module_name,function_name", ENTRY_POINTS)
@pytest.mark.parametrize("example", ["BVO", "Hematite"])
def test_entry_point_loads_example_configuration(
    module_name, function_name, example, tmp_path, monkeypatch
):
    module = importlib.import_module(module_name)
    source = Path(__file__).resolve().parents[2] / "examples" / example
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = yaml.safe_load((source / "InputFiles" / "sys_config.yml").read_text())
    for directory in ("InputFiles", "ConfigurationFiles"):
        target = tmp_path / directory
        target.mkdir()
        (target / "sys_config.yml").write_text(yaml.safe_dump(config))
    simulation = yaml.safe_load((source / "simulation_parameters.yml").read_text())
    simulation.update(work_dir_depth=1, input_file_directory_name="InputFiles", compute_mode="serial")
    (run_dir / "simulation_parameters.yml").write_text(yaml.safe_dump(simulation))

    def inspect_material(parameters):
        # Verify nested configuration survives parsing and argument conversion.
        for key, value in config.items():
            assert getattr(parameters, key) == value
        assert Path(parameters.input_coord_file_location).name == "POSCAR"
        raise MaterialReached

    monkeypatch.setattr(module, "Material", inspect_material)
    function = getattr(module, function_name)
    with pytest.raises(MaterialReached):
        if function_name == "material_setup":
            function(tmp_path / "InputFiles", [1, 1, 1], [1, 1, 1], False, False, False, False, False)
        elif function_name == "materialCOCMSD":
            function(tmp_path, 0, [1, 1, 1], [1, 1, 1], 3, 300, "full", "full",
                     [1, 0], 1, 1, 1, 1, 0, False, "s", "angstrom", False)
        elif function_name in ("generate_transition_prob_matrix", "generate_species_site_sd_list"):
            function([], run_dir)
        elif function_name == "generate_msd_analytical_data":
            function([], [], [], 1, 1, run_dir)
        else:
            function(run_dir)


@pytest.mark.parametrize("modes", [("manual", "random"), ("random", "manual"), ("random", "random")])
def test_random_doping_is_independent_of_insertion_order(modes):
    run = Run.__new__(Run)
    run.doping = {
        "num_dopants": [1, 1],
        "insertion_type": list(modes),
        "dopant_site_indices": [[0], [2]],
        "min_shell_separation": [0, 0],
    }
    run.dopant_element_types = ["A", "B"]
    run.substitution_element_types = ["X", "X" if modes == ("random", "random") else "Y"]
    run.system = SimpleNamespace(
        system_size=np.array([1, 1, 1]), system_class_index_list=[], hop_neighbor_list=[]
    )
    run.material = SimpleNamespace(
        element_types=["X", "Y"], total_elements_per_unit_cell=4,
        n_elements_per_unit_cell=np.array([2, 2]),
    )
    # A zero-shell exclusion removes only the selected site. Keep the actual
    # random selection and available-site bookkeeping under test.
    run.get_shell_based_neighbors = lambda site, *args: [[site]]
    result = run.get_doping_distribution()
    assert set(result) == {"A", "B"}
    assert len(result["A"]) == len(result["B"]) == 1
    assert result["A"][0] in (0, 1)
    assert result["B"][0] in ((0, 1) if modes == ("random", "random") else (2, 3))
    assert result["A"][0] != result["B"][0]
