#!/usr/bin/env python
"""
Make drug_dose optional in MLPModel.forward so use_drug_structure=False trains
without KeyError. Patches the ORIGINAL Squidiff package (GBM trains from the
clean baseline, not the changeB copy).

Verified: prints whether the match landed; re-grep after to confirm.
"""
P = "/scratch/ni2071/Diffusion/Squidiff_gbm/Squidiff/MLPModel.py"
s = open(P).read()

before = """        if self.use_encoder:
            cell_id = model_kwargs.get('cell_id', None)
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                z_sem = self.encoder(model_kwargs['x_start'], label=None,
                                     drug_dose=model_kwargs['drug_dose'],
                                     cell_id=cell_id)
            else:
                z_sem = self.encoder(model_kwargs['x_start'], label=model_kwargs['group'],
                                     drug_dose=model_kwargs['drug_dose'],
                                     cell_id=cell_id)"""

after = """        if self.use_encoder:
            cell_id   = model_kwargs.get('cell_id', None)
            drug_dose = model_kwargs.get('drug_dose', None)   # GBM: optional (no fingerprint)
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                z_sem = self.encoder(model_kwargs['x_start'], label=None,
                                     drug_dose=drug_dose,
                                     cell_id=cell_id)
            else:
                z_sem = self.encoder(model_kwargs['x_start'], label=model_kwargs['group'],
                                     drug_dose=drug_dose,
                                     cell_id=cell_id)"""

if before in s:
    open(P, "w").write(s.replace(before, after, 1))
    print("RESULT: patch applied")
elif "drug_dose = model_kwargs.get('drug_dose', None)" in s:
    print("RESULT: already patched (idempotent no-op)")
else:
    print("RESULT: MATCH FAILED -- raw region for inspection:")
    i = s.find("if self.use_encoder")
    print(repr(s[i:i+520]))
    raise SystemExit(1)   # release fix (21 Oct 2026): exit non-zero so run_prep_gbm.sh cannot continue past a failed patch
