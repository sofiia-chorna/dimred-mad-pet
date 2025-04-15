import os

import ase.io

mptraj = ase.io.read("data/mad-test-mptrj-settings.xyz", ":")

dataset_dict = {
    origin: [] for origin in set([atoms.info.get("origin") for atoms in mptraj])
}

for atoms in mptraj:
    origin = atoms.info.get("origin")
    dataset_dict[origin].append(atoms)

for origin, values in dataset_dict.items():
    dataset_folder_path = os.path.join("data", "mptraj")
    os.makedirs(dataset_folder_path, exist_ok=True)

    print(origin, len(values))

    ase.io.write(os.path.join(dataset_folder_path, f"{origin}.xyz"), values)
