"""
Knowledge base for the hardware support copilot.

A compact corpus of hardware manufacturing and bring-up support documents.
Each document carries a stable id, a topic tag used for optional pre-filtering,
and text written the way a real technical note reads. A handful of near-duplicate
"distractor" documents share vocabulary with a canonical answer but resolve a
different problem -- without them, retrieval evaluation is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    topic: str
    title: str
    text: str
    is_distractor: bool = False

    def searchable(self) -> str:
        return f"{self.title}. {self.text}"


DOCUMENTS: list[KBDocument] = [
    KBDocument(
        "SMT-REFLOW-01", "smt",
        "Reflow profile tuning for tombstoning",
        "Tombstoning on small passives is driven by uneven heating that lifts one "
        "termination before the other wets. Reduce the ramp rate into the reflow "
        "zone to below 1.5 C per second and balance pad thermal mass by adjusting "
        "the stencil aperture ratio. Verify with a profiled board that both "
        "terminations reach liquidus within 3 seconds of each other.",
    ),
    KBDocument(
        "SMT-PASTE-02", "smt",
        "Solder paste print defects and stencil design",
        "Insufficient solder is most often a print problem rather than a reflow "
        "problem. Check stencil aperture area ratio, which should exceed 0.66 for "
        "reliable release, and inspect for clogged apertures. Increasing print "
        "pressure to compensate for poor release smears paste and creates bridging "
        "downstream; fix the aperture instead.",
    ),
    KBDocument(
        "SPI-AOI-03", "inspection",
        "Reducing SPI and AOI false calls",
        "A high false-call rate wastes operator review time and trains operators "
        "to ignore real defects. Tighten inspection thresholds against a "
        "characterized golden board rather than against nominal CAD, and set "
        "process control limits from measured process capability. Re-qualify "
        "thresholds after any stencil or paste change.",
    ),
    KBDocument(
        "THERMAL-04", "thermal",
        "Edge compute thermal throttling under load",
        "Junction temperature above the throttle threshold with utilization pinned "
        "high indicates inadequate heat extraction, commonly intake filter "
        "blockage or degraded thermal interface material. Clear the intake path "
        "and verify TIM contact pressure before suspecting the silicon. Sustained "
        "operation should hold junction temperature below 85 C.",
    ),
    KBDocument(
        "SENSOR-CAL-05", "sensor",
        "Sensor calibration drift over temperature",
        "Calibration drift that appears only over temperature indicates the "
        "compensation table does not cover the operating range. Re-run thermal "
        "compensation at the temperature extremes seen in the field and extend the "
        "table bounds. Drift that persists after recompensation points to a "
        "mechanical mounting issue rather than a calibration one.",
    ),
    KBDocument(
        "SIGNAL-06", "signal_integrity",
        "High-speed signal integrity on dense boards",
        "Eye closure on high-speed lanes in a dense stackup usually traces to "
        "impedance discontinuities at vias and connector transitions. Simulate the "
        "channel before layout freeze, back-drill unused via stubs, and validate "
        "with an eye diagram at final trace length rather than on a test coupon.",
    ),
    KBDocument(
        "YIELD-07", "yield",
        "First pass yield recovery method",
        "When first pass yield collapses across multiple stations, resist fixing "
        "everything at once. Pull parametric and defect data and build a Pareto "
        "first; a small number of failure modes usually dominate. Route "
        "process-driven modes to the line and design-driven modes to engineering, "
        "and run them in parallel with clear owners.",
    ),
    KBDocument(
        "NPI-08", "npi",
        "NPI build phase exit criteria",
        "Each NPI phase from EVT through PVT to mass production has an explicit "
        "exit gate. Do not advance a build phase on schedule pressure alone; a "
        "defect carried past its gate compounds in cost. Gate on first pass yield "
        "entitlement for the phase and on closure of severity-ranked defects.",
    ),
    # Distractors: shared vocabulary, different resolution.
    KBDocument(
        "SMT-REFLOW-09", "smt",
        "Reflow oven maintenance schedule",
        "Reflow ovens require periodic maintenance: clean flux condensate from the "
        "cooling zone, verify thermocouple calibration, and inspect the conveyor "
        "chain tension. This is preventive maintenance and does not address "
        "specific solder defects, which are covered separately.",
        is_distractor=True,
    ),
    KBDocument(
        "THERMAL-10", "thermal",
        "Chassis fan control configuration",
        "Fan curves map temperature to duty cycle. A fan that never spins up may "
        "indicate a tachometer fault rather than a thermal problem. This page "
        "documents fan control configuration only; compute-module throttling under "
        "load is addressed in the thermal management section.",
        is_distractor=True,
    ),
    KBDocument(
        "SENSOR-CAL-11", "sensor",
        "Sensor factory calibration procedure",
        "Factory calibration establishes the initial intrinsic parameters at "
        "nominal temperature and stores them in device memory. This is the "
        "one-time bringup procedure and does not cover field drift over "
        "temperature, which requires thermal recompensation.",
        is_distractor=True,
    ),
]

# Ground truth for the eval harness: question intent -> the doc that answers it.
GROUND_TRUTH: dict[str, str] = {
    "why are small resistors standing up on end after reflow": "SMT-REFLOW-01",
    "not enough solder on the pads after printing": "SMT-PASTE-02",
    "too many false calls on our aoi machine wasting operator time": "SPI-AOI-03",
    "compute board is throttling and running hot under load": "THERMAL-04",
    "sensor reading drifts when it gets hot but is fine at room temp": "SENSOR-CAL-05",
    "eye diagram is closing on the fast lanes of a dense pcb": "SIGNAL-06",
    "first pass yield dropped across several stations how do i recover": "YIELD-07",
    "when can i move from evt to the next build phase": "NPI-08",
}


def build_documents(include_distractors: bool = True) -> list[KBDocument]:
    return [d for d in DOCUMENTS if include_distractors or not d.is_distractor]


def doc_index() -> dict[str, KBDocument]:
    return {d.doc_id: d for d in DOCUMENTS}


def validate() -> None:
    ids = {d.doc_id for d in DOCUMENTS}
    if len(ids) != len(DOCUMENTS):
        raise ValueError("duplicate document ids")
    missing = set(GROUND_TRUTH.values()) - ids
    if missing:
        raise ValueError(f"ground truth references missing docs: {sorted(missing)}")
