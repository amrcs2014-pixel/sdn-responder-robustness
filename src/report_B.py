"""Paper B: tables, figures, mixed-effects models and hypothesis verdicts."""
import os, json, math, warnings
import numpy as np, pandas as pd
import analysis as A
import figures as F
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
OUT = os.path.join(A.ROOT, "paper_outputs"); os.makedirs(OUT, exist_ok=True)
ORDER = ["portdrop", "template", "vtemplate", "vtemplate_D", "llm", "verimit", "verimit_D"]
LBL = dict(F.LBL, portdrop="R1 Port-drop", template="R2 Template", vtemplate="R2v V-Template",
           vtemplate_D="R2v+D V-Template+D", llm="R3 LLM", verimit="R4 VeriMitigate", verimit_D="R5 VeriMitigate+D")
SHORT = {"portdrop": "R1", "template": "R2", "vtemplate": "R2v", "vtemplate_D": "R2v+D", "llm": "R3", "verimit": "R4",
         "verimit_D": "R5"}
OBJ = ["O1", "O2", "O3", "O4"]
SW = {"portdrop": "Pd", "template": "Tp", "vtemplate": "Vt", "vtemplate_D": "VtD", "llm": "Llm", "verimit": "Vm",
      "verimit_D": "VmD", "vm_prov": "Prov", "vm_thr": "Thr", "vm_cost": "Cost", "vm_hyst": "Hyst"}
OW = {"O1": "One", "O2": "Two", "O3": "Three", "O4": "Four"}
MAC = {}


def mac(k, v): MAC["".join(ch for ch in k if ch.isalpha())] = v


def pct(x, d=1): return f"{100 * x:.{d}f}"


def load():
    b = A.flat_B(A.load("B_main"))
    return b


def table_objectives(b):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Harm induced by the adversary, per responder and objective (means over 3 budgets $\times$ 6 topologies $\times$ 5 seeds; 3 seeds for R3--R5). Amp.: legitimate bytes lost to adversary-triggered responder rules per adversary byte; Down: legitimate-host downtime (s) per 1{,}000 adversary packets; Prot.: protected-service availability (\%); Coll.: share of benign flows broken by any responder rule, including rules triggered by detector false positives (\%).}",
         r"\label{tab:obj_res}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{l" + "cccc" * 2 + "}", r"\toprule",
         r" & \multicolumn{4}{c}{O1 Collateral induction} & \multicolumn{4}{c}{O2 Protected-service isolation} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Responder & Amp. & Down & Prot. & Coll. & Amp. & Down & Prot. & Coll. \\", r"\midrule"]
    L2 = [r"\midrule", r" & \multicolumn{4}{c}{O3 Responder exhaustion} & \multicolumn{4}{c}{O4 Oscillation} \\",
          r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
          r"Responder & Amp. & Down & Prot. & Coll. & Amp. & Down & Prot. & Coll. \\", r"\midrule"]
    rows1, rows2 = [], []
    for r in ORDER:
        d = b[b.resp == r]
        if d.empty: continue
        cells = {}
        for o in OBJ:
            x = d[d.obj == o]
            cells[o] = [f"{x.amplification.mean():.2f}", f"{x.downtime_per_kpkt.mean():.2f}", pct(x.prot_avail.mean()),
                        pct(x.collateral.mean(), 2)]
            mac(f"amp{SW[r]}{OW[o]}", f"{x.amplification.mean():.2f}")
            mac(f"down{SW[r]}{OW[o]}", f"{x.downtime_per_kpkt.mean():.2f}")
            mac(f"prot{SW[r]}{OW[o]}", pct(x.prot_avail.mean()))
            mac(f"coll{SW[r]}{OW[o]}", pct(x.collateral.mean(), 2))
        rows1.append(f"{LBL[r]} & " + " & ".join(cells["O1"] + cells["O2"]) + r" \\")
        rows2.append(f"{LBL[r]} & " + " & ".join(cells["O3"] + cells["O4"]) + r" \\")
    L += rows1 + L2 + rows2 + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_obj.tex"), "w").write("\n".join(L) + "\n")


def fig_budget(b):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    bud = ["low", "med", "high"]
    for ax, o in zip(axes, OBJ):
        for r in ORDER:
            d = b[(b.resp == r) & (b.obj == o)]
            if d.empty: continue
            s = d.groupby("budget").amplification.mean().reindex(bud)
            ax.plot(range(3), np.maximum(s.values, 1e-4), "-o", color=F.C[r], lw=1.6, ms=3.5, label=LBL[r])
        ax.set_yscale("log"); ax.set_title(o, fontsize=8, color=F.INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(["low", "med", "high"])
        ax.axhline(1.0, color=F.INK2, lw=0.8, ls="--")
        ax.set_xlabel("Adversary budget")
    axes[0].set_ylabel("Amplification (bytes lost / byte sent)")
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.axhspan(5e-5, 1.5e-4, color=F.GRID, lw=0, zorder=0)
    axes[-1].text(2.0, 1.9e-4, "zero (drawn at $10^{-4}$)", fontsize=5.5, color=F.INK2, ha="right")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    F.save(fig, os.path.join(OUT, "fig_budget.pdf"))


def table_exhaustion(b):
    o3 = b[b.obj == "O3"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{O3 responder exhaustion (means over budgets, topologies and seeds). The genuine DDoS starts 20\,s after the adversary; its TTM is censored at 60\,s.}",
         r"\label{tab:o3}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Responder & Incidents & Rules inst. & Peak table & Loop p95 (s) & Genuine TTM (s) & Genuine resid.\ (\%) \\",
         r"\midrule"]
    for r in ORDER:
        d = o3[o3.resp == r]
        if d.empty: continue
        L.append(f"{LBL[r]} & {d.incidents.mean():.1f} & {d.rules_installed.mean():.1f} & {d.rules_max.mean():.1f} & "
                 f"{np.nanmean(d.loop_p95):.1f} & {np.median(d.gen_ttm_c):.1f} & {pct(d.gen_residual.mean())} \\\\")
        mac(f"oThreeRules{SW[r]}", f"{d.rules_installed.mean():.1f}")
        mac(f"oThreeTtm{SW[r]}", f"{np.median(d.gen_ttm_c):.1f}")
        mac(f"oThreeLoop{SW[r]}", f"{np.nanmean(d.loop_p95):.1f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_o3.tex"), "w").write("\n".join(L) + "\n")


def table_defense_ablation(b):
    try:
        ab = A.flat_B(A.load("B_abl"))
    except FileNotFoundError:
        return
    ref = b[(b.budget == "med") & (b.seed < 3) & b.resp.isin(["verimit", "verimit_D"])]
    d = pd.concat([ab, ref])
    names = [("verimit", "R4 (no defense)"), ("vm_prov", "R4 + D1 provenance"), ("vm_thr", "R4 + D2 threshold"),
             ("vm_cost", "R4 + D3 cost gating"), ("vm_hyst", "R4 + D4 hysteresis/budget"), ("verimit_D", "R5 = R4 + D1--D4")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Defense ablation on VeriMitigate (medium budget, 6 topologies $\times$ 3 seeds). Adversary-induced amplification was zero for every variant and objective at this budget; the columns show where the variants differ: protected-service availability under O2 and the fate of the concurrent genuine DDoS under exhaustion (O3; TTM censored at 60\,s).}",
         r"\label{tab:dabl}", r"\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Variant & O2 prot.\ (\%) & O3 genuine TTM (s) & O3 genuine residual (\%) & O3 loop p95 (s) \\", r"\midrule"]
    for r, lab in names:
        x = d[d.resp == r]
        if x.empty: continue
        o3 = x[x.obj == "O3"]
        L.append(f"{lab} & {pct(x[x.obj == 'O2'].prot_avail.mean())} & {np.median(o3.gen_ttm_c):.1f} & "
                 f"{pct(o3.gen_residual.mean())} & {np.nanmean(o3.loop_p95):.1f} " + r"\\")
        mac(f"dablTtm{SW[r]}", f"{np.median(o3.gen_ttm_c):.1f}"); mac(f"dablRes{SW[r]}", pct(o3.gen_residual.mean()))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_dabl.tex"), "w").write("\n".join(L) + "\n")


def table_cost():
    try:
        c = A.flat_A(A.load("B_cost"))
    except FileNotFoundError:
        return
    m = A.flat_A(A.load("A_main"))
    d = pd.concat([m[m.resp.isin(["vtemplate", "verimit"])], c])
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Defense cost under genuine attacks (companion M0 grid: 5 classes $\times$ 6 topologies $\times$ 5 seeds, no adversary).}",
         r"\label{tab:cost}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Responder & Mitig.\ (\%) & TTM (s) & Residual (\%) & Collateral (\%) & FP incidents/run \\", r"\midrule"]
    for r in ["vtemplate", "vtemplate_D", "verimit", "verimit_D"]:
        x = d[d.resp == r]
        if x.empty: continue
        L.append(f"{LBL[r]} & {pct(x.mitigated.mean())} & {np.median(x.ttm_c):.1f} & {pct(x.residual.mean())} & "
                 f"{pct(x.collateral.mean(), 2)} & {x.fp_incidents.mean():.2f} \\\\")
        mac(f"cost{SW[r]}Ttm", f"{np.median(x.ttm_c):.1f}"); mac(f"cost{SW[r]}Res", pct(x.residual.mean()))
        mac(f"cost{SW[r]}Coll", pct(x.collateral.mean(), 2)); mac(f"cost{SW[r]}Mit", pct(x.mitigated.mean()))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_cost.tex"), "w").write("\n".join(L) + "\n")
    return d


def mixed(b):
    d = b.copy()
    d["lamp"] = np.log10(d.amplification + 1e-3)
    d["resp"] = pd.Categorical(d.resp, categories=["verimit"] + [r for r in ORDER if r != "verimit"])
    res = {}
    try:
        md = smf.mixedlm("lamp ~ C(resp) * C(obj)", d, groups=d["topo"]).fit(reml=True)
        res["lamp"] = md
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{Linear mixed-effects model of $\log_{10}$(amplification $+10^{-3}$): responder main effects relative to R4 VeriMitigate (reference objective O1) with a random intercept per topology. Interaction terms are in the supplementary log.}",
             r"\label{tab:lmm}", r"\footnotesize", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Term & Estimate & 95\% CI & $p$ \\", r"\midrule"]
        ci = md.conf_int()
        for name in md.params.index:
            if name.startswith("C(resp)") and ":" not in name:
                r = name.split("T.")[1].rstrip("]")
                p = md.pvalues[name]
                L.append(f"{LBL[r]} & {md.params[name]:+.2f} & [{ci.loc[name, 0]:+.2f}, {ci.loc[name, 1]:+.2f}] & "
                         f"{'$<$0.001' if p < 1e-3 else f'{p:.3f}'} \\\\")
        L += [r"\midrule", f"Topology variance & {float(md.cov_re.iloc[0, 0]):.3f} & & \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(OUT, "tab_lmm.tex"), "w").write("\n".join(L) + "\n")
        open(os.path.join(A.RES, "B_lmm_summary.txt"), "w").write(str(md.summary()))
    except Exception as e:
        print("mixed model failed", e)
    return res


def pairwise(b):
    """Holm-corrected paired tests of R5 vs R4 and R4 vs R3 on amplification, per objective."""
    out, ps = [], []
    for a, c in [("verimit_D", "verimit"), ("verimit", "llm"), ("verimit", "template"), ("vtemplate_D", "vtemplate")]:
        for o in OBJ:
            t = A.paired(b[b.obj == o], a, c, "amplification", keys=("budget", "topo", "seed"))
            out.append((a, c, o, t)); ps.append(t["p"])
    adj = A.holm(ps)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired Wilcoxon tests on amplification (matched budget, topology, seed), Holm-adjusted; $r$: matched-pairs rank-biserial correlation (negative: first responder lower).}",
         r"\label{tab:pair}", r"\footnotesize", r"\begin{tabular}{llrrcr}", r"\toprule",
         r"Comparison & Obj. & Mean A & Mean B & $p_\mathrm{Holm}$ & $r$ \\", r"\midrule"]
    for (a, c, o, t), pa in zip(out, adj):
        L.append(f"{SHORT[a]} vs {SHORT[c]} & {o} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & "
                 f"{'$<$0.001' if pa < 1e-3 else f'{pa:.3f}'} & {t['rbc']:+.2f} \\\\")
        mac(f"pw{SW[a]}{SW[c]}{OW[o]}p", "<0.001" if pa < 1e-3 else f"{pa:.3f}")
        mac(f"pw{SW[a]}{SW[c]}{OW[o]}r", f"{t['rbc']:+.2f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_pair.tex"), "w").write("\n".join(L) + "\n")


def hypotheses(b, cost):
    v = {}
    # H1: unverified responders R1-R3 have amplification > 1 for at least one objective (mean over budgets)
    h1 = {}
    for r in ["portdrop", "template", "llm"]:
        d = b[b.resp == r]
        if d.empty: continue
        h1[r] = {o: float(d[d.obj == o].amplification.mean()) for o in OBJ}
    v["H1"] = {r: max(x.values()) > 1 for r, x in h1.items()}
    # H2: R4 reduces O2 to near zero (prot availability >= 99% and amp < 0.05) but does not eliminate O1 (amp O1 > 0.05)
    d = b[b.resp == "verimit"]
    if not d.empty:
        o2p = d[d.obj == "O2"].prot_avail.mean(); o2a = d[d.obj == "O2"].amplification.mean()
        o1a = d[d.obj == "O1"].amplification.mean()
        v["H2"] = dict(o2_prot=o2p, o2_amp=o2a, o1_amp=o1a, verdict=bool(o2p >= 0.99 and o2a < 0.05 and o1a > 0.05))
    # H3: R5 reduces O1 below a practical threshold (amp < 0.01) at a TTM cost < 2x
    d5 = b[b.resp == "verimit_D"]
    if not d5.empty and cost is not None:
        o1a5 = d5[d5.obj == "O1"].amplification.mean()
        t4 = np.median(cost[cost.resp == "verimit"].ttm_c); t5 = np.median(cost[cost.resp == "verimit_D"].ttm_c)
        v["H3"] = dict(o1_amp=o1a5, ttm_ratio=t5 / t4, verdict=bool(o1a5 < 0.01 and t5 / t4 < 2))
    json.dump(dict(h1=h1, verdicts=v), open(os.path.join(A.RES, "B_hypotheses.json"), "w"), indent=1, default=float)
    for k, x in v.items():
        if isinstance(x, dict) and "verdict" in x:
            mac("verdict" + {"H1": "One", "H2": "Two", "H3": "Three"}[k], "supported" if x["verdict"] else "not supported")
    if "H3" in v: mac("ttmRatioRFive", f"{v['H3']['ttm_ratio']:.2f}")
    return v


def auditability(b):
    for r in ["verimit_D", "vtemplate_D"]:
        d = b[b.resp == r]
        tp, fp, fn = d.spoof_tp.sum(), d.spoof_fp.sum(), d.spoof_fn.sum()
        mac(f"spoofP{SW[r]}", f"{tp / max(tp + fp, 1):.3f}"); mac(f"spoofR{SW[r]}", f"{tp / max(tp + fn, 1):.3f}")
        mac(f"spoofN{SW[r]}", str(int(tp + fp + fn)))


def table_fpharm(b):
    """Harm caused by incidents that detector false positives triggered (not attributable to the adversary)."""
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Responder harm caused by incidents triggered by detector false positives (benign records), per run, over the whole adversarial grid. It is excluded from the adversary-attributed metrics above.}",
         r"\label{tab:fpharm}", r"\footnotesize", r"\begin{tabular}{lcc}", r"\toprule",
         r"Responder & Legitimate bytes lost (MB/run) & Host downtime (s/run) \\", r"\midrule"]
    for r in ORDER:
        d = b[b.resp == r]
        if d.empty: continue
        L.append(f"{LBL[r]} & {d.fp_harm_bytes.mean() / 1e6:.2f} & {d.fp_down.mean():.1f} " + r"\\")
        mac(f"fpBytes{SW[r]}", f"{d.fp_harm_bytes.mean() / 1e6:.2f}"); mac(f"fpDown{SW[r]}", f"{d.fp_down.mean():.1f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_fpharm.tex"), "w").write("\n".join(L) + "\n")


def table_freq(b):
    """How often an adversary run caused any responder harm (amplification > 1e-3), and the worst case."""
    d = b[b.seed < 3]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Frequency and severity of successful abuse: number of adversarial runs (of 54 per objective; seeds 0--2 for all responders) in which adversary-triggered rules destroyed legitimate traffic (amplification $>10^{-3}$), and the largest amplification observed over all objectives.}",
         r"\label{tab:freq}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Responder & O1 & O2 & O3 & O4 & Worst amplification \\", r"\midrule"]
    for r in ORDER:
        x = d[d.resp == r]
        if x.empty: continue
        cnt = [int((x[x.obj == o].amplification > 1e-3).sum()) for o in OBJ]
        L.append(f"{LBL[r]} & " + " & ".join(str(c) for c in cnt) + f" & {x.amplification.max():.1f} " + r"\\")
        mac(f"freq{SW[r]}", str(sum(cnt))); mac(f"worst{SW[r]}", f"{x.amplification.max():.1f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_freq.tex"), "w").write("\n".join(L) + "\n")


def table_dabl_t(b):
    """Single defenses on the verified template ladder (R2v), all budgets, 5 seeds."""
    try:
        t = A.flat_B(A.load("B_abl_t"))
    except FileNotFoundError:
        return
    d = pd.concat([t, b[b.resp.isin(["vtemplate", "vtemplate_D"])]])
    names = [("vtemplate", "R2v (no defense)"), ("vt_prov", "R2v + D1 provenance"), ("vt_thr", "R2v + D2 threshold"),
             ("vt_cost", "R2v + D3 cost gating"), ("vt_hyst", "R2v + D4 hysteresis/budget"), ("vtemplate_D", "R2v + D1--D4")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Which defense does what: single defenses added to the verified template ladder (all budgets, 6 topologies $\times$ 5 seeds). Mean amplification per objective; O3 columns give rules installed per run and the genuine DDoS's median TTM.}",
         r"\label{tab:dablt}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Variant & O1 amp. & O2 amp. & O3 amp. & O4 amp. & O3 rules & O3 genuine TTM (s) \\", r"\midrule"]
    for r, lab in names:
        x = d[d.resp == r]
        if x.empty: continue
        o3 = x[x.obj == "O3"]
        L.append(f"{lab} & " + " & ".join(f"{x[x.obj == o].amplification.mean():.3f}" for o in OBJ)
                 + f" & {o3.rules_installed.mean():.1f} & {np.median(o3.gen_ttm_c):.1f} " + r"\\")
        for o in OBJ: mac(f"dablt{SW.get(r, r)}{OW[o]}", f"{x[x.obj == o].amplification.mean():.3f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_dablt.tex"), "w").write("\n".join(L) + "\n")


def table_stale():
    try:
        rows = A.load("B_stale")
    except FileNotFoundError:
        return
    adv = [d for d in rows if d["scen"] != "M0"]; gen = [d for d in rows if d["scen"] == "M0"]
    b = A.flat_B(adv); b["stale"] = [d["stale"] for d in adv]
    base = A.flat_B([d for d in A.load("B_main") if d["resp"] == "vtemplate_D" and d["budget"] == "med"]); base["stale"] = 0.0
    bb = pd.concat([base, b])
    g = A.flat_A(gen); g["stale"] = [d["stale"] for d in gen for _ in d["out"]["attacks"]]
    g0 = A.flat_A([d for d in A.load("B_cost") if d["resp"] == "vtemplate_D"]); g0["stale"] = 0.0
    gg = pd.concat([g0, g])
    allr = [(d.get("stale", 0.0), i) for d in rows for i in d["incidents"]]
    allr += [(0.0, i) for d in A.load("B_main") + A.load("B_cost") if d["resp"] == "vtemplate_D" and d.get("budget", "med") == "med" for i in d["incidents"]]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Sensitivity of provenance-aware scoping (R2v+D) to stale IP bindings (a fraction of clients whose binding points to a wrong port). Adversarial rows: medium budget, 6 topologies $\times$ 5 seeds; genuine-attack rows: companion M0 grid.}",
         r"\label{tab:stale}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
         r"Stale bindings & 0\% & 10\% & 30\% \\", r"\midrule"]
    for o in OBJ:
        L.append(f"{o} amplification & " + " & ".join(f"{bb[(bb.stale == s) & (bb.obj == o)].amplification.mean():.4f}" for s in [0.0, 0.1, 0.3]) + r" \\")
    L.append(r"\midrule")
    L.append(r"Genuine: mitigated (\%) & " + " & ".join(pct(gg[gg.stale == s].mitigated.mean()) for s in [0.0, 0.1, 0.3]) + r" \\")
    L.append(r"Genuine: residual (\%) & " + " & ".join(pct(gg[gg.stale == s].residual.mean()) for s in [0.0, 0.1, 0.3]) + r" \\")
    L.append(r"Genuine: collateral (\%) & " + " & ".join(pct(gg[gg.stale == s].collateral.mean(), 2) for s in [0.0, 0.1, 0.3]) + r" \\")
    cells_p, cells_r = [], []
    for s in [0.0, 0.1, 0.3]:
        inc = [i for st, i in allr if st == s]
        tp = sum(1 for i in inc if i["spoof_flag"] and i["spoof_truth"]); fp = sum(1 for i in inc if i["spoof_flag"] and not i["spoof_truth"])
        fn = sum(1 for i in inc if not i["spoof_flag"] and i["spoof_truth"])
        cells_p.append(f"{tp / max(tp + fp, 1):.2f}"); cells_r.append(f"{tp / max(tp + fn, 1):.2f}")
        mac(f"staleP{['zero', 'ten', 'thirty'][[0.0, 0.1, 0.3].index(s)]}", f"{tp / max(tp + fp, 1):.2f}")
    L.append("Spoof-flag precision & " + " & ".join(cells_p) + r" \\")
    L.append("Spoof-flag recall & " + " & ".join(cells_r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tab_stale.tex"), "w").write("\n".join(L) + "\n")


def main():
    b = load()
    table_stale()
    table_fpharm(b); table_freq(b); table_dabl_t(b)
    table_objectives(b); fig_budget(b); table_exhaustion(b); table_defense_ablation(b)
    cost = table_cost(); mixed(b); pairwise(b); v = hypotheses(b, cost); auditability(b)
    mac("nRunsB", str(len(b)))
    for o in OBJ:
        for bud in ["low", "med", "high"]:
            x = b[(b.obj == o) & (b.budget == bud) & (b.resp == "template")]
            if not x.empty: mac(f"ampTp{OW[o]}{bud}", f"{x.amplification.mean():.2f}")
    with open(os.path.join(OUT, "numbers_B.tex"), "w") as f:
        for k, val in sorted(MAC.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    print(json.dumps(v, default=float, indent=1))


if __name__ == "__main__":
    main()
