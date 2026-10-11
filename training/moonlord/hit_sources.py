"""Damage taken by source (which Moon Lord attack landed each hit), from fight logs:
  python training/moonlord/hit_sources.py "logs/moonlord/20261011_132722_fly_fight_ep*.jsonl"
A hit = HP drop > 25 between ticks; source = nearest projectile (160 px) or boss part
(edge within 120 px) the tick before, else a deathray beam passing within 220 px."""
import json,glob,math,collections,sys
NAMES={452:'homing eye',454:'sphere',455:'deathray',456:'tongue',462:'bolt'}
PART={396:'body: head',397:'body: hand',398:'body: core',400:'body: true eye'}
def attribute(pp, st, st2):
    best=None
    for q in st.get('projectiles',[]):
        if q['type']==455: continue
        dd=math.hypot(q['cx']-pp['cx'],q['cy']-pp['cy'])
        if dd<160 and (best is None or dd<best[0]): best=(dd,NAMES.get(q['type'],str(q['type'])))
    for m in st.get('npcs',[]):
        if m.get('type') in PART:
            dd=math.hypot(m.get('cx',m['x'])-pp['cx'],m.get('cy',m['y'])-pp['cy'])-60
            if dd<120 and (best is None or dd<best[0]): best=(dd,PART[m['type']])
    if best: return best[1]
    for q in st.get('projectiles',[])+st2.get('projectiles',[]):
        if q['type']==455:
            ux,uy=q['vx'],q['vy']; nn=math.hypot(ux,uy) or 1
            qx,qy=pp['cx']-q['cx'],pp['cy']-q['cy']
            if (qx*ux+qy*uy)/nn>0 and abs(-qx*uy+qy*ux)/nn<220: return 'deathray'
    return 'unclear'
dmg=collections.Counter(); cnt=collections.Counter(); tot=0; trip=collections.Counter()
for f in sorted(glob.glob(sys.argv[1])):
    prev=None
    for l in open(f):
        r=json.loads(l)
        if r.get('phase')!='play': continue
        st=r.get('state') or {}; p=st.get('player') or {}
        if 'cx' not in p: continue
        if prev and prev['p']['hp']-p['hp']>25:
            d=prev['p']['hp']-p['hp']; k=attribute(prev['p'],prev['st'],st)
            dmg[k]+=d; cnt[k]+=1; tot+=d
            if prev['nurse']: trip[k]+=d
        prev={'p':p,'st':st,'nurse':r.get('nurse')}
print('total damage',tot,' during Nurse trips',sum(trip.values()))
for k,v in dmg.most_common():
    print('%-16s %5d HP  %4.1f%%  %3d hits  avg %3.0f'%(k,v,100*v/tot,cnt[k],v/cnt[k]))
