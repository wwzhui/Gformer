from matformer.train_props import train_prop_model 
props = [
    "formation_energy_peratom",#0
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",#4
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",#10
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",##27
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]
# train_prop_model(learning_rate=0.001,name="matformer", prop=props[27], pyg_input=True, n_epochs=500, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="./Ehull", use_angle=False, save_dataloader=False)
# train_prop_model(learning_rate=0.001,name="matformer", prop=props[4], pyg_input=True, n_epochs=300, max_neighbors=16, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="./mbj", use_angle=False, save_dataloader=False)
# train_prop_model(learning_rate=0.001,name="matformer", prop=props[1], pyg_input=True, n_epochs=300, max_neighbors=12, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="./OPT", use_angle=False, save_dataloader=False)
train_prop_model(learning_rate=0.001,name="matformer", prop=props[10], pyg_input=True, n_epochs=500, max_neighbors=16, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="./total", use_angle=False, save_dataloader=False)



