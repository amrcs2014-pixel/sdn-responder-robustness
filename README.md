# Spoofed evidence and starvation: robustness of automated SDN responders

Code and complete experiment logs for the paper:

> A. H. Abdelhaliem, *Spoofed evidence and starvation turn automated SDN responders against their own networks* (submitted).

Automated responders turn intrusion alerts into blocking rules. Because they act on the identifiers carried by the evidence, an adversary who forges those identifiers can aim the response at legitimate hosts. This study measures mitigation-induced denial of service against seven responders, from port blocking and classic templates to a language-model responder with and without pre-deployment verification (VeriMitigate). The adversary controls one compromised host and spoofs the IPv4 source address, pursuing four objectives:

- **O1** collateral induction,
- **O2** protected-service isolation,
- **O3** responder exhaustion,
- **O4** oscillation.

Four defenses are evaluated:

- **D1** provenance-aware scoping,
- **D2** evidence thresholding,
- **D3** cost-sensitive gating,
- **D4** hysteresis with rule budgets.

**Companion repository** (VeriMitigate, real Containernet/Open vSwitch/Ryu validation, synthesizer training): [https://github.com/amrcs2014-pixel/verimitigate-sdn](https://github.com/amrcs2014-pixel/verimitigate-sdn).

**Ethics.** All adversary traffic is generated inside the emulated testbed by parameterized generators. No exploit code against real controllers or devices is included.

## Key results (reproduced by the logs in `results/`)

- **The classic "block the source" template is easily abused.** It was turned against legitimate hosts in 144 of 216 adversarial runs, destroyed up to 174 bytes of legitimate traffic per adversary byte, and cut protected-service availability to 65.8%.
- **The fine-tuned LLM responder is rarely fooled, but severely when it is.** It failed in 7 of 216 runs, with amplification up to 67. Pre-deployment verification removed these cases.
- **Starvation:** rotating spoofed identities keeps a slow synthesizer busy, so a concurrent genuine DDoS was never mitigated. Hysteresis with rule budgets halved the damage.
- **Provenance-aware scoping alone removed collateral induction and oscillation.** All four defenses together cost 1.08× in time-to-mitigate against genuine attacks and tolerated 30% stale bindings.

## Contents

| Path | What it is |
|---|---|
| `src/` | Emulator, scenarios (`adversary()` in `scenarios.py`), trace-driven detector, verifier, responders R1–R5 and defenses D1–D4 (`responder.py`), cached LLM server, experiment runner, `report_B.py` (statistics, mixed-effects model, hypotheses, tables, figures) |
| `results/B_main.jsonl` | 4 objectives × 3 budgets × 6 topologies × 7 responders |
| `results/B_none.jsonl` | No-response reference |
| `results/B_abl.jsonl`, `results/B_abl_t.jsonl` | Single-defense ablations on VeriMitigate and on the verified template ladder |
| `results/B_cost.jsonl` | Defense cost under genuine attacks |
| `results/A_main.jsonl` | Baseline grid reused for the defense-cost comparison |
| `results/B_stale.jsonl` | Stale-binding sensitivity |
| `results/B_hypotheses.json`, `results/B_lmm_summary.txt` | Pre-registered hypotheses and the mixed-effects model |
| `results/llm_cache.sqlite` | Every LLM completion used, keyed by prompt (reproduces model outputs without a GPU) |
| `docs/LAB_NOTEBOOK.md` | Full lab notebook, including the per-incident harm-attribution fix |

## Reproducing

1. **Setup.** `pip install -r requirements.txt` (Python 3.11).
2. **Data.** Download `FULL_SDN_NIDS.parquet` (LAN-SDN-NIDS, CC BY 4.0, https://doi.org/10.21950/QMAXKP) into `data/`, then run `python src/detector.py`.
3. **Model.** Download `Qwen/Qwen2.5-1.5B-Instruct` into `models/qwen1.5b/`, and get the adapter `sft.pt` from the Release of [https://github.com/amrcs2014-pixel/verimitigate-sdn](https://github.com/amrcs2014-pixel/verimitigate-sdn), placing it in `models/adapters/`.
4. **Experiments.** Run `bash src/run_all.sh`; runs are deterministic, and LLM calls are served from the cache.
5. **Tables and figures.** `python src/report_B.py`.

## Licence and citation

- **Code:** MIT. **Logs:** CC BY 4.0.
- **LAN-SDN-NIDS** and **Qwen2.5** are under their own licences and are not redistributed here.
- See `CITATION.cff`.
