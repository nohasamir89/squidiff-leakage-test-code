#!/usr/bin/env python3
"""Average predictions over independent draws. PREREG amendment 9.

Each draw is a SEPARATE process invocation of the wrapper, so nothing is shared
between them. Running the harness in a loop via exec() failed five times on
namespace collisions - it reassigns TAG, _k and other names into the caller's
scope. Separate processes remove that class of failure entirely.
"""
import sys, os, glob, re
import numpy as np
tag = sys.argv[1]
# The glob must EXCLUDE the per-cell files: vectors_..._draw0_pc.npz also
# matches vectors_..._draw*.npz, which silently doubled the file count to 32
# and threw on the ragged per-cell arrays. Sorted numerically so draw 0 is
# first, since its per-cell file is the one retained.
files = [f for f in glob.glob('leakage_results/vectors_%s_draw*.npz' % tag)
         if not f.endswith('_pc.npz')]
files.sort(key=lambda f: int(re.search(r'_draw(\d+)\.npz$', f).group(1)))
print('  %d draw files (per-cell files excluded)' % len(files))
if not files: sys.exit('FATAL: no draw files for %s' % tag)
print('averaging %d draws for %s' % (len(files), tag))
acc, cond, spread = None, None, []
for f in files:
    v = np.load(f, allow_pickle=True)
    if acc is None:
        acc = {k: np.asarray(v[k], dtype=np.float64) for k in ('true','A','B','C')}
        cond = v['cond']
    else:
        if not np.array_equal(v['cond'], cond):
            sys.exit('FATAL: condition order differs in %s' % f)
        for k in acc: acc[k] += np.asarray(v[k], dtype=np.float64)
    spread.append(np.asarray(v['B'], dtype=np.float64))
for k in acc: acc[k] /= len(files)
sd = np.std(np.stack(spread), axis=0)
print('  mean per-gene SD of predictions across draws: %.5f' % sd.mean())
if sd.mean() < 1e-9:
    sys.exit('FATAL: predictions identical across draws - the stochastic encoder '
             'is not varying and the averaging is meaningless')
np.savez_compressed('leakage_results/vectors_%s.npz' % tag, cond=cond,
                    draw_sd=sd.astype(np.float32), **acc)
# The post-run provenance check asserts vectors_<TAG>_pc.npz exists. Per-cell
# predictions are not averaged - the energy-distance endpoint needs a real cell
# population, not a mean of populations - so draw 0's file is retained under the
# final tag and that is disclosed here.
pc0 = files[0].replace('.npz', '_pc.npz')
if os.path.exists(pc0):
    os.replace(pc0, 'leakage_results/vectors_%s_pc.npz' % tag)
    print('  per-cell file: draw 0 retained (populations are not averaged)')
for f in files:
    os.remove(f)
    pc = f.replace('.npz', '_pc.npz')
    if os.path.exists(pc): os.remove(pc)
# CONTROL_MAGNITUDE: the fingerprint control must reproduce Steps 6 and 9.
# Weight comparison cannot detect a broken treated-control pairing - Step 9
# seeds 0 and 1 on identical data differ by 133.8, and the broken Step 10 run
# differed by 133.6. Prediction magnitude can: broken gave 0.0295 against the
# correct 0.0076.
if 'fingerprint_s10' in tag:
    _m = float(np.abs(acc['B'] - acc['C']).mean())
    print('  control prediction |delta| %.5f  (Steps 6 and 9: 0.00757)' % _m)
    if abs(_m - 0.00757) > 0.002:
        sys.exit('FATAL: control does not reproduce the Step 6 and 9 magnitude; '
                 'the treated-control pairing may be broken')
print('  wrote leakage_results/vectors_%s.npz' % tag)
