#!/usr/bin/env bash
#
# Run LeafMachine3 over the Global Greening GBIF image set -- one species subdir at a time.
#
#   input   /datab/Global_Greening/GBIF/images/<species>/
#   output  /datab/Global_Greening/GBIF/LM3/<species>/
#
# Every species uses the SAME settings file; only the input dir, the output dir and the run name
# change. LM3 composes its run directory as <project.output.dir>/<project.run_name>, and the CLI
# has --input/--output but no --run-name, so the run name is set by writing a one-line-patched copy
# of the settings per species into <OUT_ROOT>/_configs/. Those copies are kept, not deleted: they
# are the record of exactly what each run was given.
#
# Runs are SEQUENTIAL by design. A single LM3 run already saturates the box (its own GPU worker
# pool plus a CPU process pool), so overlapping species would just contend for VRAM.
#
# Resuming: species whose pipeline already ran to the end are SKIPPED by default, and species with
# an unfinished run are RE-ENTERED -- LM3 resumes those from its own SQLite project DB whenever the
# output dir exists and `project.run_mode.overwrite` is false. So the plain no-argument invocation
# is the resume command: it finishes what is half-done and runs what was never started.
#
# "Finished" is read from the project DB, not from the presence of the file: a species counts as
# complete when no stage is left pending, none errored, and the LAST stage is done. A stage can sit
# in 'running' forever when some images errored -- n_done never reaches n_total -- which is why
# 'running' alone is deliberately not disqualifying. asclepias_syriaca is exactly that case: 96 of
# its 5131 sheets failed inside ONNX at archival_detector, so that stage stays 'running' even though
# every later stage, reporter and ect included, completed over the other 5035.
#
# Usage
#   ./run_global_greening.sh                      # resume: skip complete species, run the rest
#   ./run_global_greening.sh acer_rubrum          # just these species
#   ./run_global_greening.sh --status             # print each species' state and exit
#   ./run_global_greening.sh --dry-run            # print what would run, touch nothing
#   ./run_global_greening.sh --redo-complete      # re-enter complete species too
#   LM3_CONFIG=/path/to/other.yaml ./run_global_greening.sh
#
set -uo pipefail

LM3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${LM3_PYTHON:-$LM3_DIR/.venv_LM3/bin/python}"
CONFIG="${LM3_CONFIG:-$LM3_DIR/LM3_settings_global_greening.yaml}"

IMAGES_ROOT="${GG_IMAGES_ROOT:-/datab/Global_Greening/GBIF/images}"
OUT_ROOT="${GG_OUT_ROOT:-/datab/Global_Greening/GBIF/LM3}"

# Not species: a test dir and an unidentified bucket.
SKIP_DIRS=(tg unknown_species)

DRY_RUN=0
REDO_COMPLETE=0
STATUS_ONLY=0
ONLY=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1 ;;
        --redo-complete) REDO_COMPLETE=1 ;;
        --status)        STATUS_ONLY=1 ;;
        # Skipping complete species is the default now, so the old flag is a no-op. It is kept
        # rather than rejected so older invocations and notes do not break.
        --skip-existing) ;;
        -h|--help)       sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)              echo "unknown option: $1" >&2; exit 2 ;;
        *)               ONLY+=("$1") ;;
    esac
    shift
done

# ---- preflight: fail before a multi-day batch, not during it -------------------------------- #
for path in "$PY" "$CONFIG" "$IMAGES_ROOT"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
"$PY" -c "import leafmachine3" 2>/dev/null || { echo "cannot import leafmachine3 with $PY" >&2; exit 1; }
grep -qE '^[[:space:]]*run_name:' "$CONFIG" || { echo "no run_name: line in $CONFIG" >&2; exit 1; }

mkdir -p "$OUT_ROOT/_configs" "$OUT_ROOT/_logs"

# ---- completeness probe ---------------------------------------------------------------------- #
# Ask the project DB whether a species' pipeline reached the end. Anything other than "complete"
# means re-enter the species; LM3 either resumes it from the DB or starts it fresh, both of which
# are safe. An unreadable DB is treated the same way ON PURPOSE -- a run killed mid-write leaves a
# hot -wal that a read-only open cannot recover, and that is precisely a run worth re-entering.
species_state() {
    local db="$1"
    [[ -f "$db" ]] || { echo absent; return; }
    "$PY" - "$db" <<'PYPROBE'
import sqlite3, sys
try:
    con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    rows = con.execute("SELECT stage_order, state FROM project_status").fetchall()
    con.close()
except Exception:
    print("unreadable"); raise SystemExit(0)
if not rows:
    print("incomplete"); raise SystemExit(0)
last    = max(o for o, _ in rows)
blocked = any(st in ("pending", "error") for _, st in rows)
ended   = all(st == "done" for o, st in rows if o == last)
print("complete" if ended and not blocked else "incomplete")
PYPROBE
}

# ---- species list ---------------------------------------------------------------------------- #
species=()
for dir in "$IMAGES_ROOT"/*/; do
    name="$(basename "$dir")"
    skip=0
    for s in "${SKIP_DIRS[@]}"; do [[ "$name" == "$s" ]] && skip=1; done
    [[ $skip -eq 1 ]] && continue
    if [[ ${#ONLY[@]} -gt 0 ]]; then
        keep=0
        for o in "${ONLY[@]}"; do [[ "$name" == "$o" ]] && keep=1; done
        [[ $keep -eq 0 ]] && continue
    fi
    species+=("$name")
done

# A typo in an explicit species name should stop the run, not silently do nothing -- and so should
# naming a dir that the skip list drops, which is otherwise indistinguishable from a typo.
if [[ ${#ONLY[@]} -gt 0 && ${#species[@]} -ne ${#ONLY[@]} ]]; then
    echo "requested species that will not run:" >&2
    for o in "${ONLY[@]}"; do
        in_skip=0
        for s in "${SKIP_DIRS[@]}"; do [[ "$o" == "$s" ]] && in_skip=1; done
        if [[ $in_skip -eq 1 ]]; then
            echo "  $o -- in the skip list (SKIP_DIRS)" >&2
        elif [[ ! -d "$IMAGES_ROOT/$o" ]]; then
            echo "  $o -- no such dir under $IMAGES_ROOT" >&2
        fi
    done
    exit 1
fi
[[ ${#species[@]} -eq 0 ]] && { echo "no species to run" >&2; exit 1; }

echo "config     : $CONFIG"
echo "python     : $PY"
echo "images     : $IMAGES_ROOT"
echo "output     : $OUT_ROOT/<species>"
echo "skipping   : ${SKIP_DIRS[*]}"
echo "species    : ${#species[@]}"
echo

if [[ $STATUS_ONLY -eq 1 ]]; then
    printf '%-32s %-12s %s\n' SPECIES STATE ACTION
    for sp in "${species[@]}"; do
        st="$(species_state "$OUT_ROOT/$sp/${sp}.sqlite")"
        if [[ "$st" == complete && $REDO_COMPLETE -eq 0 ]]; then act=skip; else act=run; fi
        printf '%-32s %-12s %s\n' "$sp" "$st" "$act"
    done
    exit 0
fi

# ---- run ------------------------------------------------------------------------------------- #
started="$(date +%Y%m%d-%H%M%S)"
summary="$OUT_ROOT/_logs/summary_${started}.tsv"
[[ $DRY_RUN -eq 0 ]] && printf 'species\tstatus\tseconds\timages\tlog\n' > "$summary"

ok=(); failed=(); skipped=()

for i in "${!species[@]}"; do
    sp="${species[$i]}"
    in_dir="$IMAGES_ROOT/$sp"
    run_dir="$OUT_ROOT/$sp"
    cfg="$OUT_ROOT/_configs/${sp}.yaml"
    log="$OUT_ROOT/_logs/${sp}.log"
    n_img="$(find "$in_dir" -maxdepth 1 -type f | wc -l)"

    state="$(species_state "$run_dir/${sp}.sqlite")"
    if [[ "$state" == complete && $REDO_COMPLETE -eq 0 ]]; then
        echo "[$((i + 1))/${#species[@]}] $sp -- already complete, skipping"
        skipped+=("$sp")
        continue
    fi

    case "$state" in
        incomplete) note="  [resuming an unfinished run]" ;;
        unreadable) note="  [project DB unreadable -- re-entering]" ;;
        complete)   note="  [complete, but --redo-complete was given]" ;;
        *)          note="" ;;
    esac
    echo "[$((i + 1))/${#species[@]}] $sp  ($n_img files)  -> $run_dir$note"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "        $PY -m leafmachine3 --config $cfg --input $in_dir --output $OUT_ROOT"
        echo "        (run_name would be set to '$sp')"
        continue
    fi

    # Per-species config: the shared settings with run_name swapped, so the run lands in
    # <OUT_ROOT>/<species> rather than <OUT_ROOT>/<the shared run_name>.
    sed -E "s|^([[:space:]]*run_name:).*|\1 ${sp}|" "$CONFIG" > "$cfg"

    t0=$SECONDS
    "$PY" -m leafmachine3 \
        --config "$cfg" \
        --input  "$in_dir" \
        --output "$OUT_ROOT" \
        > >(tee -a "$log") 2>&1
    rc=$?
    elapsed=$((SECONDS - t0))

    if [[ $rc -eq 0 ]]; then
        echo "        done in ${elapsed}s"
        ok+=("$sp")
        printf '%s\tok\t%d\t%d\t%s\n' "$sp" "$elapsed" "$n_img" "$log" >> "$summary"
    else
        # Keep going: one bad species should not cost the other twenty.
        echo "        FAILED (exit $rc) after ${elapsed}s -- see $log" >&2
        failed+=("$sp")
        printf '%s\tfail(%d)\t%d\t%d\t%s\n' "$sp" "$rc" "$elapsed" "$n_img" "$log" >> "$summary"
    fi
done

# ---- summary ---------------------------------------------------------------------------------- #
echo
echo "ok      : ${#ok[@]}"
[[ ${#skipped[@]} -gt 0 ]] && echo "skipped : ${#skipped[@]}  (${skipped[*]})"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed  : ${#failed[@]}  (${failed[*]})"
    [[ $DRY_RUN -eq 0 ]] && echo "summary : $summary"
    exit 1
fi
[[ $DRY_RUN -eq 0 ]] && echo "summary : $summary"
# Explicit: without this the script inherits the status of the test above, so a clean dry run
# (where that test is false) would exit 1.
exit 0
