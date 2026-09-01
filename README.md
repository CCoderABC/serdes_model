# ADC-Based SerDes Model

The executable baseline models the linear frequency response of the differential electrical channel and receiver analog front end (AFE), then evaluates an ADC/DSP/FEC receiver.

```text
payload -> RS(544,514) encoder -> Gray PAM4 ZOH -> optional TX edge filter
    -> channel Sdd21 -> AFE Av_port
    -> flat VGA/AGC -> AFE noise + ADC noise + aperture jitter -> effective ADC
    -> code centering -> symbol-spaced FFE
    -> postcursor DFE -> PAM4 slicer -> Gray demapper -> RS decoder -> payload

Current linear reference: channel + AFE + AFE noise -> MMSE FFE -> dpSNR / DER
```

## Conventions

- A quoted `100 ohm` port is differential, realized as `50 ohm` per side.
- A Touchstone channel is converted from ports `[in+, in-, out+, out-]` to generalized mixed mode. The extracted transfer is `Sdd21`, with power waves referenced to the differential port impedance.
- The AFE transfer is `Av_port = Vout,diff / Vin,diff`. It is not `S21`, transducer gain, available gain, or power gain.
- The simple cascade `H_total = Sdd21 * Av_port` assumes the channel is terminated in its reference impedance and the AFE voltage response already represents the intended input loading. A later network-level model should be used when AFE input return loss or channel/AFE impedance interaction matters.
- Phase is retained. Group delay, impulse response, and pulse response are derived from the complex cascade rather than magnitude-only data.
- TX swing is loaded differential outer peak-to-peak voltage across the matched `100 ohm` channel input. It is not unloaded Thevenin-generator swing.
- TX pulse shaping filters the signal only. Source-port noise is injected at the channel input, while AFE noise is injected at the AFE input; neither is incorrectly passed through the TX edge filter.
- The configured AFE noise density is differential, AFE-input-port referred, and one-sided. The configuration explicitly states whether termination and ADC-input load noise are included.
- A symbol-spaced MMSE FFE is an analysis reference receiver. A real implementation can use different tap counts and coefficient constraints.
- The stochastic code path includes AFE noise, additional ADC noise, aperture jitter, clipping, quantization, and DFE error propagation. The separate linear-reference dpSNR/DER path still excludes ADC effects, jitter, CDR, DFE, FEC, and adaptation transients.

The parameterized AFE uses

\[
H_{AFE}(s)=A_0\frac{\prod_i(1+s/\omega_{z,i})}{\prod_j(1+s/\omega_{p,j})}.
\tag{Eq. 1}
\]

The analytic demonstration channel uses

\[
H_{ch}(f)=10^{-IL(f)/20}e^{-j2\pi f\tau},
\quad
IL(f)=IL_0+a\sqrt{f/f_r}+b(f/f_r).
\tag{Eq. 2}
\]

The phase associated with the analytic loss is reconstructed as a minimum-phase response before the explicit propagation delay is applied. This prevents a magnitude-only, zero-phase placeholder from creating a noncausal symmetric impulse response.

The `analytic_high_loss` option specifies insertion loss directly at Nyquist. Loss above the configured DC value is divided between square-root skin-effect loss and linear dielectric loss. The default example targets `36 dB` at `28 GHz`. This is a controllable baseline; measured mixed-mode Touchstone data remains preferred for channel signoff.

```yaml
channel:
  kind: analytic_high_loss
  differential_z0_ohm: 100.0
  delay_ps: 500.0
  loss_dc_db: 0.2
  insertion_loss_db_at_nyquist: 36.0
  skin_effect_fraction: 0.45
```

## TX pulse shaping

The one-UI rectangular pulse always represents the PAM4 zero-order hold. The additional TX shaping option is selected in YAML:

```yaml
transmitter:
  differential_outer_pp_v: 0.9
  pulse_shaping:
    kind: gaussian_rise_time  # or ideal_zoh
    rise_time_20_80_ps: 7.5
```

`ideal_zoh` applies no extra bandwidth limit. `gaussian_rise_time` uses the COM-style response

\[
H_{TX}(f)=\exp\left[-\left(\frac{\pi fT_r}{1.6832}\right)^2\right],
\tag{Eq. 3}
\]

where `Tr` is the 20%–80% edge time. Its DC gain is one, so the configured `900 mVpp` steady-state outer swing remains `[-450, -150, 150, 450] mV`.

## TX and channel stage plots

Before applying the AFE, noise, or receiver equalization, the model exports the TX-output and channel-output symbol responses. For one normalized PAM4 unit (`150 mV` in the default configuration), the plotted spectra are

\[
P_{TX}(f)=H_{ZOH}(f)H_{TX}(f),
\tag{Eq. 4}
\]

\[
P_{CH,out}(f)=H_{ZOH}(f)H_{TX}(f)S_{dd21}(f).
\tag{Eq. 5}
\]

Both are transformed to one-symbol time-domain pulses. Frequency magnitudes are referenced to the TX pulse DC value; time-domain voltages are loaded differential voltage per PAM unit. The channel output is the matched differential channel reference plane before the AFE.

The same deterministic random PAM4 sequence is convolved with those two pulse
responses to produce side-by-side two-UI eye diagrams. Each eye is aligned to
the main pulse cursor at its own reference plane. Plot interpolation only
smooths the lines; reported center samples remain on the configured simulation
grid. These first-stage eyes contain TX pulse shaping and channel ISI only—AFE,
noise, jitter, ADC effects, and RX equalization are excluded.

## CTLE/AFE complex response

The next reference plane applies the configured differential CTLE voltage gain
`Av_port = Vout,diff / Vin,diff`. Each real pole and zero contributes its full
complex factor, so the model retains both amplitude and phase:

\[
H_{AFE}(j2\pi f)=A_0
\frac{\prod_i(1+jf/f_{z,i})}{\prod_j(1+jf/f_{p,j})}.
\tag{Eq. 6}
\]

The AFE-output spectrum, symbol pulse, and eye are calculated with this complex
response rather than a magnitude-only approximation. The cascade assumes a
matched `100 ohm` differential channel termination and that the intended AFE
input loading is already represented by `Av_port`. AFE noise, ADC behavior, and
digital RX equalization remain excluded from these staged plots.

## Effective ADC stage

The ADC input range is a differential `400 mVpp`, or `-200 mV` to `+200 mV`.
An ideal frequency-flat VGA precedes the ADC. Its scalar `Av_port` is selected
from the bounded worst-case PAM waveform peak and a configurable full-scale
target:

\[
G_{VGA}=\frac{0.95\,V_{FS,peak}}{V_{AFE,worst\ peak}}.
\tag{Eq. 7}
\]

For the default pulse response, the CTLE worst-case bound is `877.43 mV peak`,
so the applied VGA gain is `-13.289 dB` and the bounded ADC target is
`190 mV peak`. The ideal VGA adds no frequency emphasis, phase shift, group
delay, bandwidth limit, or noise.

The configured `6 ENOB` is interpreted as an ideal-equivalent 64-level uniform
mid-rise quantizer:

\[
q_{eff}=\frac{V_{FS,pp}}{2^{ENOB}}
=\frac{400\ \mathrm{mV}}{64}=6.25\ \mathrm{mV}.
\tag{Eq. 8}
\]

Its equivalent in-range quantization-noise RMS is

\[
\sigma_q=\frac{q_{eff}}{\sqrt{12}}=1.804\ \mathrm{mV}.
\tag{Eq. 9}
\]

The ADC stage samples the VGA output once per UI at the AFE main-cursor timing.
Samples outside the differential full-scale range saturate to code `0` or `63`.
The deterministic ADC transfer plot excludes stochastic impairments, while the
codes consumed by the FFE include the configured AFE noise, additional ADC
input noise, and aperture jitter.

The default AFE noise density is a one-sided, differential, AFE-input-port
referred `2 nV/√Hz`; receiver-termination noise is declared included. Its
complex output shaping is retained and the ideal VGA then scales it to the ADC
input, producing `3.91 mV RMS` in the current run. A separate `0.75 mV RMS`
differential ADC-input noise term is added. This term is explicitly assumed to
be beyond the effective quantization model; it would double count noise if the
quoted ENOB already included that same circuit noise.

Independent Gaussian aperture jitter of `150 fs RMS` (`0.0084 UI`) produces a
first-order voltage error from the local noiseless signal slope. Its observed
RMS voltage contribution is `0.35 mV`. ADC DNL/INL and correlated or bounded
clock jitter remain excluded.

## Code-domain FFE

Unsigned ADC codes are centered by subtracting `31.5`, then processed by a
17-tap symbol-spaced FFE:

\[
y[n]=\sum_{k=-8}^{8}w_k\left(\mathrm{code}[n+k]-31.5\right).
\tag{Eq. 10}
\]

The symmetric analysis offsets are implementable causally by adding eight UI
of output latency. Coefficients are trained by regularized least squares on the
first half of the sequence. The second half is held out, with an additional
half-tap-span guard on each side of the split so training and test observation
windows do not overlap.

For held-out output `y`, a training-derived gain `g` and offset `b` define

\[
\mathrm{dpSNR}_{code}
=10\log_{10}\left(
\frac{5g^2}{\operatorname{mean}\left[(y-b-ga)^2\right]}
\right).
\tag{Eq. 11}
\]

With AFE noise, additional ADC noise, aperture jitter, quantization, and an
actually FEC-encoded source stream, the default run improves held-out dpSNR
from `4.61 dB` before the FFE to `16.43 dB` after it. The held-out FFE records
19 symbol errors in 7,664 decisions, for an empirical DER of `2.48e-3`; its
Gaussian DER estimate is `2.28e-3`.

## Code-domain DFE and BER interpretation

A 12-tap symbol-spaced DFE follows the FFE. Known symbols are used only to fit
the postcursor coefficients on the training segment. The held-out sequence is
then evaluated both genie-aided and causally decision-directed:

\[
y_{DFE}[n]=y_{FFE}[n]-\sum_{k=1}^{12}b_k\hat a[n-k].
\tag{Eq. 12}
\]

Only previously detected symbols appear in the feedback sum, so the DFE cannot
cancel precursor ISI. The first 12 held-out symbols warm up the feedback history
and are excluded from the reported metrics. The decision-directed result must
be used for receiver performance; the genie-aided result is a diagnostic upper
reference that reveals decision-error propagation when the two modes differ.

The stochastic default run improves variance-based held-out dpSNR slightly,
from `16.43 dB` at the FFE output to `16.50 dB` after decision-directed DFE.
The measured result changes from 19 FFE symbol/bit errors to 18 DFE symbol/bit
errors in 7,652 symbols, or 15,304 Gray-coded bit opportunities. The empirical
DFE DER is `2.35e-3` and BER is `1.18e-3`. Genie-aided and decision-directed
results are identical in this realization, so no DFE error propagation is
observed. The two-sided 95% Wilson BER interval is `7.44e-4` to `1.86e-3`.

For a zero-error experiment to demonstrate an error probability below `p` with
95% confidence, the required opportunity count is

\[
N \ge \frac{\ln(0.05)}{\ln(1-p)} \approx \frac{2.996}{p}.
\tag{Eq. 13}
\]

Thus direct evidence for `BER < 1e-12` needs about `3e12` independent bit
opportunities. The model also reports a Gaussian-tail BER extrapolation derived
from held-out residual variance. With the newly injected impairments, the
current DFE estimate is `1.05e-3`, compared with the measured `1.18e-3`. The
difference is expected for a short run with residual non-Gaussian ISI,
quantization, and possible DFE error propagation.

## Hard-decision RS(544,514) FEC

The evaluated payload codewords are now systematically encoded before PAM4
modulation, and the hard decisions after the DFE are decoded with a bit-exact
RS(544,514) codec. Random prefix and suffix filler align complete codewords
inside the finite held-out interval and are not included in FEC results. The
reference uses GF(2^10), 514 payload symbols, 30 parity symbols, and corrects up
to 15 erroneous 10-bit symbols per codeword. Its primitive polynomial and
generator coefficients match the public IEEE 802.3bj encoder C-model
contribution.

Each codeword contains 5,440 coded bits or 2,720 PAM4 symbols. The code rate is
`514/544 = 0.94485`, corresponding to `5.84%` parity overhead relative to the
payload. At 56 GBd PAM4, the raw 112 Gb/s becomes `105.82 Gb/s` of payload rate
before other PCS overhead. One codeword spans `48.57 ns`; implementation and
pipeline decoder latency are not yet modeled.

Two complete held-out codewords are currently evaluated. They contain 5 and 10
erroneous RS symbols, both below the 15-symbol correction limit, so the decoder
corrects all 15 received bit errors and observes zero errors in 10,280 decoded
payload bits. This direct result only establishes a 95% zero-error upper bound
of `2.91e-4`, not a very low post-FEC BER.

Under an independent-bit-error approximation, the measured codeword-subset
pre-FEC BER of `1.38e-3` gives an uncorrectable-codeword probability of
`4.08e-3` and an estimated failed-codeword-passthrough BER of `1.26e-5`.
Reaching a modeled post-FEC BER of `1e-12` requires pre-FEC BER at or below
approximately `3.64e-4`. The present operating point is about 3.8 times too
high, so this FEC improves BER materially but does not close the target.

## Decision-point evaluation

PAM4 symbols are normalized to `[-3, -1, 1, 3]`, so their variance is `5`. After sampling-phase optimization and the reference FFE, let `g0` be the main cursor, `gk` the other cursors, and `sigma_n^2` the FFE-output noise variance. The reported decision-point SNR is

\[
\mathrm{dpSNR}=10\log_{10}
\left(
\frac{5|g_0|^2}{5\sum_{k\ne0}|g_k|^2+\sigma_n^2}
\right).
\tag{Eq. 14}
\]

The fast Gaussian estimate uses

\[
\mathrm{DER}_{Gaussian}=\frac{3}{2}Q\left(
\sqrt{\frac{10^{\mathrm{dpSNR}/10}}{5}}
\right).
\tag{Eq. 15}
\]

The pattern-conditioned result enumerates the configured number of strongest ISI cursors and treats only the remaining weak ISI plus AFE noise as Gaussian. The present slicer uses symmetric midpoint thresholds. Both DER values describe an ideal detector before FEC and do not include DFE error propagation.

The output keeps three quantities separate:

- `dp_snr_db`: result of the modeled signal, channel, AFE, noise, and ideal FFE.
- `estimated_dp_snr_db`: `dp_snr_db` minus the configured implementation loss.
- `dp_snr_margin_db`: `estimated_dp_snr_db` minus the AWGN SNR required for the configured target DER.

The default rise time, channel loss, and noise density are illustrative assumptions, not interface specifications.

## PAPR at each reference plane

Peak-to-average power ratio is reported using the same deterministic PAM4
sequence at every node:

\[
\mathrm{PAPR}
=\frac{\max_n\left(v_n^2\right)}{\operatorname{mean}_n\left(v_n^2\right)},
\qquad
\mathrm{PAPR}_{dB}=10\log_{10}(\mathrm{PAPR}).
\tag{Eq. 16}
\]

The mean is not removed. For a fixed impedance, resistance cancels from the
ratio, so differential voltage samples can be used directly. Analog-node PAPR
uses the continuous 8-sample/UI grid; ADC input and reconstructed-output PAPR
use one main-cursor sample/UI. Rows through the ideal ADC output are signal-only;
the final impaired ADC-input and ADC-output rows include configured noise and
jitter. The first and last 512 symbols are discarded to avoid convolution
boundary effects. A scalar VGA changes RMS and peak voltage by the same factor,
so it does not change signal-only PAPR.

## Run the demonstration

```bash
PYTHONPATH=src python3 scripts/run_channel_afe_demo.py \
  --config configs/channel_afe_demo.yaml
```

Useful one-run overrides are:

```bash
# Remove the additional finite-edge filter but retain the PAM4 ZOH.
PYTHONPATH=src python3 scripts/run_channel_afe_demo.py --tx-pulse-shaping ideal_zoh

# Sweep a harder high-loss channel point.
PYTHONPATH=src python3 scripts/run_channel_afe_demo.py --channel-loss-db-at-nyquist 40
```

Outputs are written under `runs/channel_afe_demo/`:

- `frequency_response.csv`
- `tx_channel_frequency_response.csv`
- `tx_channel_pulse_response.csv`
- `tx_channel_eye_center_samples.csv`
- `ctle_afe_frequency_response.csv`
- `ctle_afe_pulse_response.csv`
- `tx_channel_afe_eye_center_samples.csv`
- `vga_frequency_response.csv`
- `adc_center_samples.csv`
- `adc_impairment_samples.csv`
- `papr_summary.csv`
- `code_domain_ffe_taps.csv`
- `code_domain_ffe_test_samples.csv`
- `code_domain_dfe_taps.csv`
- `code_domain_dfe_test_samples.csv`
- `fec_codeword_summary.csv`
- `decision_point.csv`
- `metrics.json`
- `channel_afe_summary.png`
- `decision_point_summary.png`
- `tx_channel_stage_summary.png`
- `tx_channel_eye_summary.png`
- `ctle_afe_summary.png`
- `tx_channel_afe_eye_summary.png`
- `adc_summary.png`
- `adc_impairment_summary.png`
- `papr_summary.png`
- `code_domain_ffe_summary.png`
- `code_domain_dfe_summary.png`
- `fec_summary.png`

## Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Using a measured channel

Change the channel section of the YAML configuration:

```yaml
channel:
  kind: touchstone
  path: channels/example.s4p
  port_order: [0, 1, 2, 3]  # in+, in-, out+, out-
  expected_differential_z0_ohm: 100.0
```

The requested analysis bandwidth must not exceed the measured Touchstone bandwidth. A DC point is extrapolated from the low-frequency magnitude and phase slope and is reported in the output metadata.
