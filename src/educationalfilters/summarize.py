"""
summarize.py – Aggregations for VRF / IF / RF coverage.
"""

from __future__ import annotations
import pandas as pd

def _song_base(dfc: pd.DataFrame) -> pd.DataFrame:
    return dfc.drop_duplicates('metadata_filename') if 'metadata_filename' in dfc.columns else dfc.copy()

def prepare_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your original “percentages by corpus”, using range_check / interval_jumps_check / rhythm_check.
    """
    corpora = {"Ciciban": df[df['corpus']=='Ciciban'], "SLP": df[df['corpus']=='SLP']}
    all_results = {}
    for label, sub in corpora.items():
        base = _song_base(sub)
        total = len(base) or 1
        vrf = base.get('range_check', pd.Series(index=base.index, dtype=object)) \
                  .map({'PRE':'VRF1','PRE_PLUS':'VRF2'}).fillna('X')
        iff = base.get('interval_jumps_check', pd.Series(index=base.index, dtype=object)) \
                  .map({'PRE':'IF1','PRE_PLUS':'IF2'}).fillna('X')
        rf  = base.get('rhythm_check', pd.Series(index=base.index, dtype=object)).fillna('X')

        res = {
            "VRF1": (vrf == 'VRF1').mean()*100,
            "VRF2": vrf.isin(['VRF1','VRF2']).mean()*100,
            "IF1" : (iff == 'IF1').mean()*100,
            "IF2" : iff.isin(['IF1','IF2']).mean()*100,
            "VRF1+IF1": ((vrf == 'VRF1') & (iff == 'IF1')).mean()*100,
            "VRF2+IF2": ((vrf == 'VRF2') & (iff == 'IF2')).mean()*100,
            "ANY (VRF+IF)": ((vrf!='X') & (iff!='X')).mean()*100,

            "RF1": (rf == 'RF1').sum()/total*100,
            "RF2": rf.isin(['RF1','RF2']).sum()/total*100,
            "RF3": rf.isin(['RF1','RF2','RF3']).sum()/total*100,
            "RF4": rf.isin(['RF1','RF2','RF3','RF4']).sum()/total*100,

            "VRF2+IF2+RF3": (((vrf=='VRF2')&(iff=='IF2')&rf.isin(['RF1','RF2','RF3'])).mean()*100),
            "VRF2+IF2+RF4": (((vrf=='VRF2')&(iff=='IF2')&rf.isin(['RF1','RF2','RF3','RF4'])).mean()*100),
            "ANY (VRF+IF+RF)": ((vrf!='X')&(iff!='X')&rf.isin(['RF1','RF2','RF3','RF4'])).mean()*100,
        }
        all_results[label] = res
    return pd.DataFrame(all_results)

def prepare_all_filters_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as above but uses VRF_label / IF_label / RF_label columns for clarity.
    """
    corpora = {"Ciciban": df[df['corpus']=='Ciciban'], "SLP": df[df['corpus']=='SLP']}
    out = {}

    for label, sub in corpora.items():
        base = _song_base(sub)
        vrf = base.get('VRF_label', pd.Series(index=base.index, dtype=object)).fillna('X')
        iff = base.get('IF_label',  pd.Series(index=base.index, dtype=object)).fillna('X')
        rf  = base.get('RF_label',  pd.Series(index=base.index, dtype=object)).fillna('X')

        vrf1 = (vrf=='VRF1'); vrf2_cum = vrf.isin(['VRF1','VRF2'])
        if1  = (iff=='IF1');  if2_cum  = iff.isin(['IF1','IF2'])
        rf1  = (rf=='RF1');   rf2_c = rf.isin(['RF1','RF2'])
        rf3_c = rf.isin(['RF1','RF2','RF3']); rf4_c = rf.isin(['RF1','RF2','RF3','RF4'])

        p = (lambda s: float(s.mean()) * 100.0) if len(base) else (lambda s: 0.0)

        out[label] = {
            "VRF1": p(vrf1), "IF1": p(if1), "VRF2": p(vrf2_cum), "IF2": p(if2_cum),
            "VRF1 + IF1": p(vrf1 & if1), "VRF2+IF2": p((vrf=='VRF2') & (iff=='IF2')),
            "ANY (VRF+IF)": p(vrf2_cum & if2_cum),
            "RF1": p(rf1), "RF2": p(rf2_c), "RF3": p(rf3_c), "RF4": p(rf4_c),
            "VRF2+IF2+RF3": p((vrf=='VRF2') & (iff=='IF2') & rf3_c),
            "VRF2+IF2+RF4": p((vrf=='VRF2') & (iff=='IF2') & rf4_c),
            "ANY (VRF+IF+RF)": p(vrf2_cum & if2_cum & rf4_c),
        }

    order_top = ["VRF1","IF1","VRF2","IF2","VRF1 + IF1","VRF2+IF2","ANY (VRF+IF)"]
    order_bot = ["RF1","RF2","RF3","RF4","VRF2+IF2+RF3","VRF2+IF2+RF4","ANY (VRF+IF+RF)"]
    return pd.DataFrame(out).reindex(order_top + order_bot)