#!/bin/bash
#
# Aug 2026 RADE V2 port timing bug test
#
# Usage: ./test/v2_c_timing.sh delay [end_delay n_steps]
#   Single point:  ./test/v2_c_timing.sh 0.012
#   Sweep:         ./test/v2_c_timing.sh 0 0.02 4
#                  (5 points: 0, 0.005, 0.01, 0.015, 0.02)
#
# RESULTS=file.txt   write (delay_ms, py_loss, c_loss) to file.txt (sweep or single point)
# PLOT=basename       generate basename.png and basename.eps/.tex (LaTeX-includable) from
#                     RESULTS -- requires a sweep (end_delay/n_steps given), e.g.:
#   RESULTS=260831_delay_loss.txt PLOT=260831_delay_loss ./test/v2_c_timing.sh 0 0.02 20

RADE_C=${HOME}/rade_c/build/src
PATH=${PATH}:${RADE_C}
WAV=${WAV:-wav/all.wav}
RX_OPTS=${RX_OPTS:-}
PY_OPTS=${PY_OPTS:-}
INF_OPTS=${INF_OPTS:-}
RESULTS=${RESULTS:-}
PLOT=${PLOT:-}

is_sweep=0
if [ -n "$2" ] && [ -n "$3" ]; then
    is_sweep=1
    delays=$(python3 -c "
import numpy as np
for d in np.linspace($1, $2, int($3) + 1):
    print(f'{d:.6f}')
")
else
    delays=$1
fi

declare -a summary

for delay in $delays; do
    echo "--- WAV=$WAV delay=$delay ---" >&2

    ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth $WAV /dev/null \
    --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 \
    --auxdata --w1_dec 128 --write_rx rx_v2_nopy.f32 --prepend_noise $delay &>/dev/null $INF_OPTS

    ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync rx_v2_nopy.f32 /dev/null \
        $PY_OPTS >/dev/null 2>py_debug.txt
    py_delta_hat=$(awk '/delta_hat:/{for(i=1;i<=NF;i++) if($i=="delta_hat:") val=$(i+1)} END{print val}' \
        py_debug.txt)

    cat rx_v2_nopy.f32 | radae_rx --v2 -v 3 $RX_OPTS > features_rx_v2_c.f32 2>c_debug.txt
    c_delta_hat=$(grep "extract_symbol:" c_debug.txt | tail -1 | sed 's/.*delta_hat=\([0-9.]*\).*/\1/')

    loss_out=$(python3 loss.py features_in.f32 features_out_rx2.f32 \
        --features_hat2 features_rx_v2_c.f32 --clip_start 100 --clip_end 300)
    py_loss=$(echo "$loss_out" | grep "^ *loss:" | head -1 | awk '{print $2}')
    c_loss=$(echo  "$loss_out" | grep "^ *loss:" | tail -1 | awk '{print $2}')

    summary+=("$delay $py_delta_hat $c_delta_hat $py_loss $c_loss")
done

rm -f py_debug.txt c_debug.txt

echo ""
printf "%-12s %-12s %-12s %-10s %-10s\n" "delay" "py_dhat" "c_dhat" "py_loss" "c_loss"
printf "%-12s %-12s %-12s %-10s %-10s\n" "------------" "------------" "------------" "----------" "----------"
for row in "${summary[@]}"; do
    read -r d pd cd pl cl <<< "$row"
    printf "%-12s %-12s %-12s %-10s %-10s\n" "$d" "$pd" "$cd" "$pl" "$cl"
done

declare -a py_losses c_losses
for row in "${summary[@]}"; do
    read -r d pd cd pl cl <<< "$row"
    py_losses+=("$pl")
    c_losses+=("$cl")
done
py_csv=$(IFS=,; echo "${py_losses[*]}")
c_csv=$(IFS=,; echo "${c_losses[*]}")
stddevs=$(python3 -c "
import numpy as np
py = np.array([$py_csv])
c = np.array([$c_csv])
print(f'{py.std():.4f} {c.std():.4f}')
")
read -r py_std c_std <<< "$stddevs"
printf "%-12s %-12s %-12s %-10s %-10s\n" "------------" "------------" "------------" "----------" "----------"
printf "%-12s %-12s %-12s %-10s %-10s\n" "" "" "stddev:" "$py_std" "$c_std"

if [ -n "$RESULTS" ]; then
    rm -f "$RESULTS"
    for row in "${summary[@]}"; do
        read -r d pd cd pl cl <<< "$row"
        delay_ms=$(python3 -c "print(f'{$d*1000:.3f}')")
        printf "%s\t%s\t%s\n" "$delay_ms" "$pl" "$cl" >> "$RESULTS"
    done
fi

if [ -n "$PLOT" ] && [ $is_sweep -eq 1 ]; then
    echo "radae_plots; loss_delay_plot('${PLOT}','',\"${RESULTS}\"); quit" | octave-cli -qf
    echo "radae_plots; loss_delay_plot('','${PLOT}',\"${RESULTS}\"); quit" | octave-cli -qf
fi
