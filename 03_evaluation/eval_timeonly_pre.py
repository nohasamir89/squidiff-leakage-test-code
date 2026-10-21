import sys, os, json, torch
ckpt, tag, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sd = torch.load(ckpt, map_location='cpu', weights_only=True)
nmod = sum(1 for k in sd if 'adaln_mod' in k)
n = int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel')))
print('[precheck] %s: %d state_dict entries, %d adaln_mod tensors' % (tag, n, nmod))
if nmod != 12: sys.exit('FATAL: expected 12 adaln_mod tensors, got %d' % nmod)
if n != 58186374: sys.exit('FATAL: expected 58186374 entries, got %d' % n)
side = ckpt + '.mode.json'
if mode == 'timeonly':
    if not os.path.exists(side):
        sys.exit('FATAL: time-only checkpoint has no mode sidecar; it would be '
                 'evaluated in SEMANTIC mode: ' + side)
    m = json.load(open(side))
    if not (m.get('use_adaln') is True and m.get('use_adaln_timeonly') is True):
        sys.exit('FATAL: sidecar does not declare the time-only arm: %r' % m)
    print('[precheck] sidecar declares use_adaln=True use_adaln_timeonly=True')
else:
    if os.path.exists(side):
        sys.exit('FATAL: semantic checkpoint unexpectedly has a mode sidecar: ' + side)
    print('[precheck] no sidecar, so the sampler will route as semantic adaLN')
print('[precheck] PASS (capacity confirmed; the arm itself is NOT identifiable from weights)')
